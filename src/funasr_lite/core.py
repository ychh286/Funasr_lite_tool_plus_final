from __future__ import annotations

import inspect
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from funasr import AutoModel

from .config import DecodeConfig, ModelConfig, ObsConfig, OutputConfig


# -----------------------------
# In-process model cache (per-process)
# -----------------------------


_MODEL_CACHE: Dict[tuple, AutoModel] = {}


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs based on a callable's signature, but be conservative.

    Rationale:
    - FunASR APIs evolve; filtering can avoid unexpected keyword errors.
    - However, many FunASR entrypoints accept **kwargs, and (importantly) some
      class signatures are not introspectable via ``inspect.signature(Class)``.
      Over-filtering can silently drop required fields (e.g., ``model``), causing
      downstream assertion errors.
    """

    try:
        target = callable_obj.__init__ if inspect.isclass(callable_obj) else callable_obj
        sig = inspect.signature(target)

        # If the callable accepts **kwargs, do NOT filter (pipelines often rely on this).
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)

        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _model_cache_key(model_cfg: ModelConfig, decode_cfg: DecodeConfig) -> tuple:
    """Cache key that affects model init / weights / component wiring."""

    return (
        model_cfg.model_id,
        model_cfg.vad_model_id,
        model_cfg.punc_model_id,
        model_cfg.device or "auto",
        bool(model_cfg.disable_update),
        int(decode_cfg.max_single_segment_ms),
    )


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class ModelInitInfo:
    model_id: str
    vad_model_id: str
    punc_model_id: Optional[str]
    device: str
    init_time_s: float
    cache_hit: bool


@dataclass
class TimingInfo:
    model_init_s: float
    asr_s: float
    postprocess_s: float


@dataclass
class TranscribeResult:
    text: str
    meta: Dict[str, Any]


# -----------------------------
# Model init / get
# -----------------------------


def get_or_init_model(model_cfg: ModelConfig, decode_cfg: DecodeConfig) -> Tuple[AutoModel, ModelInitInfo]:
    """Return an AutoModel, reusing a cached instance if possible."""

    disable_cache = os.getenv("FUNASR_LITE_DISABLE_MODEL_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    key = _model_cache_key(model_cfg, decode_cfg)
    if (not disable_cache) and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], ModelInitInfo(
            model_id=model_cfg.model_id,
            vad_model_id=model_cfg.vad_model_id,
            punc_model_id=model_cfg.punc_model_id,
            device=model_cfg.device or "auto",
            init_time_s=0.0,
            cache_hit=True,
        )

    t0 = time.perf_counter()

    vad_kwargs: Dict[str, Any] = {
        # FunASR VAD commonly uses this parameter name; keep it here and filter by signature.
        "max_single_segment_time": int(decode_cfg.max_single_segment_ms),
    }

    init_kwargs: Dict[str, Any] = {
        "model": model_cfg.model_id,
        "vad_model": model_cfg.vad_model_id,
        "vad_kwargs": vad_kwargs,
        "device": model_cfg.device,
        "disable_update": model_cfg.disable_update,
    }

    # Optional punctuation model (if your FunASR build supports it)
    if model_cfg.punc_model_id:
        init_kwargs["punc_model"] = model_cfg.punc_model_id

    # Keep only kwargs that AutoModel accepts
    filtered = _filter_kwargs_for_callable(AutoModel, init_kwargs)

    model = AutoModel(**filtered)
    _MODEL_CACHE[key] = model

    init_time = time.perf_counter() - t0
    return model, ModelInitInfo(
        model_id=model_cfg.model_id,
        vad_model_id=model_cfg.vad_model_id,
        punc_model_id=model_cfg.punc_model_id,
        device=model_cfg.device or "auto",
        init_time_s=float(init_time),
        cache_hit=False,
    )


# -----------------------------
# Transcription
# -----------------------------


def _extract_text(generate_result: Any) -> str:
    """Normalize FunASR generate() output into plain text."""

    # Common pattern: list[dict(text=...)]
    if isinstance(generate_result, list) and generate_result:
        first = generate_result[0]
        if isinstance(first, dict) and "text" in first:
            return str(first.get("text") or "")

    # Sometimes it's a dict
    if isinstance(generate_result, dict) and "text" in generate_result:
        return str(generate_result.get("text") or "")

    # Fallback
    return str(generate_result)


def transcribe_file(
    wav_path: str | None,
    model_cfg: ModelConfig,
    decode_cfg: DecodeConfig,
    output_cfg: OutputConfig,
    obs_cfg: ObsConfig,
    *,
    # Compat alias: some callers (WebUI/CLI) historically used `audio_path=`.
    audio_path: str | None = None,
) -> TranscribeResult:
    """Transcribe one audio file path.

    Note:
        Prefer passing the file path as the first positional argument.
        `audio_path=` is accepted as an alias for compatibility.
    """
    if wav_path is None and audio_path is not None:
        wav_path = audio_path
    if wav_path is None:
        raise ValueError("Missing audio file path: provide wav_path (positional) or audio_path=...")

    # FunASR models generally expect a decodable audio file. Many common sources
    # (e.g., yt-dlp outputs) are .m4a/.mp4 which `soundfile` cannot read.
    # Convert non-wav inputs to 16k mono wav via ffmpeg.
    try:
        from pathlib import Path

        p = Path(wav_path)
        if p.suffix.lower() != ".wav":
            from .audio import ensure_wav_16k_mono

            wav_path = str(ensure_wav_16k_mono(p))
    except Exception:
        # If conversion fails, fall back to the original path and let downstream
        # raise a more informative error.
        pass

    overall_t0 = time.perf_counter()

    model, model_init_info = get_or_init_model(model_cfg, decode_cfg)

    # generate()
    t_asr0 = time.perf_counter()
    gen_kwargs: Dict[str, Any] = {
        "input": [wav_path],
        "batch_size_s": int(decode_cfg.batch_size_s),
        "language": decode_cfg.language,
        "use_itn": bool(decode_cfg.use_itn),
        "merge_vad": bool(decode_cfg.merge_vad),
        "merge_length_s": int(decode_cfg.merge_length_s),
    }

    # Filter kwargs according to the model.generate signature if possible
    filtered_gen = _filter_kwargs_for_callable(model.generate, gen_kwargs)
    raw = model.generate(**filtered_gen)
    asr_time = time.perf_counter() - t_asr0

    # post-process
    t_post0 = time.perf_counter()
    text = _extract_text(raw)

    if output_cfg.split_lines:
        # Keep this lightweight; quality is not guaranteed.
        from .formatting import split_lines_by_punct

        text = split_lines_by_punct(text)

    post_time = time.perf_counter() - t_post0

    overall = time.perf_counter() - overall_t0

    meta: Dict[str, Any] = {
        "model_init": asdict(model_init_info),
        "timing": asdict(
            TimingInfo(
                model_init_s=float(model_init_info.init_time_s),
                asr_s=float(asr_time),
                postprocess_s=float(post_time),
            )
        ),
        "overall_s": float(overall),
    }

    if obs_cfg.verbose:
        meta["config"] = {
            "model": asdict(model_cfg),
            "decode": asdict(decode_cfg),
            "output": asdict(output_cfg),
        }

    return TranscribeResult(text=text, meta=meta)

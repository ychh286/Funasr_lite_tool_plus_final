from __future__ import annotations

import inspect
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DecodeConfig,
    ModelConfig,
    ObsConfig,
    OutputConfig,
    speaker_diarization_enabled,
    validate_speaker_diarization_config,
)


_MODEL_CACHE: Dict[tuple, Any] = {}


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs conservatively to survive FunASR API changes."""

    try:
        target = callable_obj.__init__ if inspect.isclass(callable_obj) else callable_obj
        sig = inspect.signature(target)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _model_cache_key(model_cfg: ModelConfig, decode_cfg: DecodeConfig) -> tuple:
    return (
        model_cfg.model_id,
        model_cfg.vad_model_id,
        model_cfg.punc_model_id,
        model_cfg.spk_model_id,
        model_cfg.device or "auto",
        bool(model_cfg.disable_update),
        int(decode_cfg.max_single_segment_ms),
        decode_cfg.speaker_mode,
    )


@dataclass
class ModelInitInfo:
    model_id: str
    vad_model_id: str
    punc_model_id: Optional[str]
    spk_model_id: Optional[str]
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
    segments: List[Dict[str, Any]]
    meta: Dict[str, Any]


def _automodel_class() -> Any:
    """延迟导入 FunASR AutoModel。

    功能：
        允许测试、GUI 配置页和非转写命令在未安装 FunASR 时仍可导入本模块。
        真正开始转写时才要求运行环境具备 FunASR。

    返回值：
        `funasr.AutoModel` 类。
    """

    try:
        from funasr import AutoModel
    except ImportError as exc:  # pragma: no cover - 取决于本地环境
        raise RuntimeError("FunASR is not installed. Install project dependencies before transcription.") from exc
    return AutoModel


def get_or_init_model(model_cfg: ModelConfig, decode_cfg: DecodeConfig) -> Tuple[Any, ModelInitInfo]:
    disable_cache = os.getenv("FUNASR_LITE_DISABLE_MODEL_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    if speaker_diarization_enabled(model_cfg):
        validate_speaker_diarization_config(model_cfg)

    key = _model_cache_key(model_cfg, decode_cfg)
    if (not disable_cache) and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], ModelInitInfo(
            model_id=model_cfg.model_id,
            vad_model_id=model_cfg.vad_model_id,
            punc_model_id=model_cfg.punc_model_id,
            spk_model_id=model_cfg.spk_model_id,
            device=model_cfg.device or "auto",
            init_time_s=0.0,
            cache_hit=True,
        )

    t0 = time.perf_counter()
    AutoModel = _automodel_class()

    init_kwargs: Dict[str, Any] = {
        "model": model_cfg.model_id,
        "vad_model": model_cfg.vad_model_id,
        "vad_kwargs": {
            "max_single_segment_time": int(decode_cfg.max_single_segment_ms),
        },
        "device": model_cfg.device,
        "disable_update": model_cfg.disable_update,
    }

    if model_cfg.punc_model_id:
        init_kwargs["punc_model"] = model_cfg.punc_model_id
    if model_cfg.spk_model_id:
        init_kwargs["spk_model"] = model_cfg.spk_model_id
        init_kwargs["spk_mode"] = decode_cfg.speaker_mode

    model = AutoModel(**_filter_kwargs_for_callable(AutoModel, init_kwargs))
    _MODEL_CACHE[key] = model

    init_time = time.perf_counter() - t0
    return model, ModelInitInfo(
        model_id=model_cfg.model_id,
        vad_model_id=model_cfg.vad_model_id,
        punc_model_id=model_cfg.punc_model_id,
        spk_model_id=model_cfg.spk_model_id,
        device=model_cfg.device or "auto",
        init_time_s=float(init_time),
        cache_hit=False,
    )


def _extract_primary_result(generate_result: Any) -> Dict[str, Any]:
    if isinstance(generate_result, list) and generate_result:
        first = generate_result[0]
        if isinstance(first, dict):
            return first
    if isinstance(generate_result, dict):
        return generate_result
    return {}


def _extract_text(generate_result: Any) -> str:
    primary = _extract_primary_result(generate_result)
    if "text" in primary:
        return str(primary.get("text") or "")
    return str(generate_result)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _extract_segments(generate_result: Any) -> List[Dict[str, Any]]:
    primary = _extract_primary_result(generate_result)
    sentence_info = primary.get("sentence_info")
    if not isinstance(sentence_info, list):
        return []

    segments: List[Dict[str, Any]] = []
    for idx, item in enumerate(sentence_info):
        if not isinstance(item, dict):
            continue

        text = str(item.get("sentence") or item.get("text") or item.get("raw_text") or "").strip()
        if not text:
            continue

        speaker_id = _safe_int(item.get("spk"))
        segments.append(
            {
                "index": idx,
                "speaker_id": speaker_id,
                "speaker": f"spk{speaker_id}" if speaker_id is not None else None,
                "text": text,
                "start_ms": _safe_int(item.get("start")),
                "end_ms": _safe_int(item.get("end")),
                "timestamp": item.get("timestamp"),
                "ts_list": item.get("ts_list"),
            }
        )

    return segments


def transcribe_file(
    wav_path: str | None,
    model_cfg: ModelConfig,
    decode_cfg: DecodeConfig,
    output_cfg: OutputConfig,
    obs_cfg: ObsConfig,
    *,
    audio_path: str | None = None,
) -> TranscribeResult:
    if wav_path is None and audio_path is not None:
        wav_path = audio_path
    if wav_path is None:
        raise ValueError("Missing audio file path: provide wav_path (positional) or audio_path=...")

    try:
        from pathlib import Path

        p = Path(wav_path)
        if p.suffix.lower() != ".wav":
            from .audio import ensure_wav_16k_mono

            wav_path = str(ensure_wav_16k_mono(p))
    except Exception:
        pass

    overall_t0 = time.perf_counter()
    model, model_init_info = get_or_init_model(model_cfg, decode_cfg)

    t_asr0 = time.perf_counter()
    gen_kwargs: Dict[str, Any] = {
        "input": [wav_path],
        "batch_size_s": int(decode_cfg.batch_size_s),
        "language": decode_cfg.language,
        "use_itn": bool(decode_cfg.use_itn),
        "merge_vad": bool(decode_cfg.merge_vad),
        "merge_length_s": int(decode_cfg.merge_length_s),
    }
    if decode_cfg.preset_spk_num is not None:
        gen_kwargs["preset_spk_num"] = int(decode_cfg.preset_spk_num)

    raw = model.generate(**_filter_kwargs_for_callable(model.generate, gen_kwargs))
    asr_time = time.perf_counter() - t_asr0

    t_post0 = time.perf_counter()
    text = _extract_text(raw)
    segments = _extract_segments(raw)

    if output_cfg.split_lines:
        from .formatting import split_lines_by_punct

        text = split_lines_by_punct(text)

    if speaker_diarization_enabled(model_cfg) and not segments:
        raise RuntimeError(
            "Speaker diarization was enabled, but FunASR did not return sentence_info. "
            "Use a timestamp-capable Paraformer model with punc + cam++."
        )

    post_time = time.perf_counter() - t_post0
    overall = time.perf_counter() - overall_t0

    speaker_ids = {
        seg["speaker_id"]
        for seg in segments
        if seg.get("speaker_id") is not None
    }

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
        "speaker_diarization": {
            "enabled": speaker_diarization_enabled(model_cfg),
            "segment_count": len(segments),
            "speaker_count": len(speaker_ids),
        },
    }

    if obs_cfg.verbose:
        meta["config"] = {
            "model": asdict(model_cfg),
            "decode": asdict(decode_cfg),
            "output": asdict(output_cfg),
        }

    return TranscribeResult(text=text, segments=segments, meta=meta)

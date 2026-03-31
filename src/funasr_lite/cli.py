from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_SCENARIO,
    DecodeConfig,
    ModelConfig,
    ObsConfig,
    OutputConfig,
    SCENARIO_PRESETS,
    apply_speaker_diarization_defaults,
    validate_speaker_diarization_config,
)
from .core import transcribe_file
from .save import render_speaker_content, save_json, save_speaker_text, save_text


def _speaker_sidecar_paths(out_path: Path) -> tuple[Path, Path]:
    speaker_text = out_path.with_name(f"{out_path.stem}_speakers{out_path.suffix}")
    speaker_json = out_path.with_name(f"{out_path.stem}_speakers.json")
    return speaker_text, speaker_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funasr-lite",
        description="Local transcription helper built on FunASR.",
    )
    p.add_argument("input", type=str, help="Path to an audio/video file")
    p.add_argument("--out", type=str, default="", help="Output path (.md/.txt). If omitted, prints to stdout.")
    p.add_argument("--format", choices=["md", "txt"], default="md")
    p.add_argument("--split-lines", action="store_true", help="Split lines by punctuation (best-effort).")
    p.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=sorted(SCENARIO_PRESETS.keys()),
        help="High-level preset that sets reasonable defaults.",
    )
    p.add_argument("--model", type=str, default="", help="Custom model id")
    p.add_argument("--vad", type=str, default="", help="Custom VAD model id")
    p.add_argument("--punc", type=str, default="", help="Custom punctuation model id")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--language", type=str, default="", help="Language hint (e.g., auto/zh/en).")
    p.add_argument("--use-itn", action="store_true", help="Enable ITN")
    p.add_argument("--no-itn", action="store_true", help="Disable ITN")
    p.add_argument("--batch-size-s", type=int, default=0)
    p.add_argument("--merge-vad", action="store_true")
    p.add_argument("--no-merge-vad", action="store_true")
    p.add_argument("--merge-length-s", type=int, default=0)
    p.add_argument("--max-single-segment-ms", type=int, default=0)
    p.add_argument("--speaker-diarization", action="store_true")
    p.add_argument("--speaker-count", type=int, default=0, help="Known speaker count. 0 means auto.")
    p.add_argument("--verbose", action="store_true", help="Print metadata as JSON.")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    model_cfg = ModelConfig(device=None if args.device == "auto" else args.device)
    if args.model:
        model_cfg = ModelConfig(
            model_id=args.model,
            vad_model_id=model_cfg.vad_model_id,
            punc_model_id=model_cfg.punc_model_id,
            spk_model_id=model_cfg.spk_model_id,
            device=model_cfg.device,
            disable_update=model_cfg.disable_update,
        )
    if args.vad:
        model_cfg = ModelConfig(
            model_id=model_cfg.model_id,
            vad_model_id=args.vad,
            punc_model_id=model_cfg.punc_model_id,
            spk_model_id=model_cfg.spk_model_id,
            device=model_cfg.device,
            disable_update=model_cfg.disable_update,
        )
    if args.punc:
        model_cfg = ModelConfig(
            model_id=model_cfg.model_id,
            vad_model_id=model_cfg.vad_model_id,
            punc_model_id=args.punc,
            spk_model_id=model_cfg.spk_model_id,
            device=model_cfg.device,
            disable_update=model_cfg.disable_update,
        )
    if args.speaker_diarization:
        model_cfg = apply_speaker_diarization_defaults(
            model_cfg,
            allow_auto_model_swap=not bool(args.model),
        )
        validate_speaker_diarization_config(model_cfg)

    decode_cfg = SCENARIO_PRESETS[args.scenario].decode
    if args.language:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "language": args.language})
    if args.use_itn:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "use_itn": True})
    if args.no_itn:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "use_itn": False})
    if args.batch_size_s > 0:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "batch_size_s": args.batch_size_s})
    if args.merge_vad:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "merge_vad": True})
    if args.no_merge_vad:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "merge_vad": False})
    if args.merge_length_s > 0:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "merge_length_s": args.merge_length_s})
    if args.max_single_segment_ms > 0:
        decode_cfg = DecodeConfig(
            **{**decode_cfg.__dict__, "max_single_segment_ms": args.max_single_segment_ms}
        )
    if args.speaker_count > 0:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "preset_spk_num": args.speaker_count})

    out_cfg = OutputConfig(fmt=args.format, split_lines=bool(args.split_lines))
    obs_cfg = ObsConfig(verbose=bool(args.verbose))

    result = transcribe_file(
        wav_path=str(inp),
        model_cfg=model_cfg,
        decode_cfg=decode_cfg,
        output_cfg=out_cfg,
        obs_cfg=obs_cfg,
    )

    if args.out:
        out_path = Path(args.out)
        save_text(
            text=result.text,
            out_path=out_path,
            fmt=out_cfg.fmt,
            split_lines=out_cfg.split_lines,
            source_name=inp.stem,
        )
        if args.speaker_diarization and result.segments:
            speaker_text_path, speaker_json_path = _speaker_sidecar_paths(out_path)
            save_speaker_text(result.segments, speaker_text_path, out_cfg.fmt, source_name=inp.stem)
            save_json(result.segments, speaker_json_path)
        print(str(out_path))
    else:
        if args.speaker_diarization and result.segments:
            print(render_speaker_content(result.segments, fmt=out_cfg.fmt, source_name=inp.stem))
        else:
            print(result.text)

    if obs_cfg.verbose:
        print("\n--- META ---")
        print(json.dumps(result.meta, ensure_ascii=False, indent=2))

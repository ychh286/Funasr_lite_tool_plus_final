#!/usr/bin/env python3
"""Local wrapper around funasr_lite with optional speaker diarization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from funasr_lite.config import (
    DEFAULT_SCENARIO,
    DecodeConfig,
    ModelConfig,
    ObsConfig,
    OutputConfig,
    SCENARIO_PRESETS,
    apply_speaker_diarization_defaults,
    validate_speaker_diarization_config,
)
from funasr_lite.core import transcribe_file
from funasr_lite.save import render_speaker_content, save_json, save_speaker_text, save_text


def _stem(p: Path) -> str:
    return p.stem


def _speaker_sidecar_paths(out_path: Path) -> tuple[Path, Path]:
    speaker_text = out_path.with_name(f"{out_path.stem}_speakers{out_path.suffix}")
    speaker_json = out_path.with_name(f"{out_path.stem}_speakers.json")
    return speaker_text, speaker_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="audio/video file")
    p.add_argument("--out", default="", help="output path (.md/.txt). If omitted, prints transcript")
    p.add_argument("--format", choices=["md", "txt"], default="md")
    p.add_argument("--split-lines", action="store_true")
    p.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=sorted(SCENARIO_PRESETS.keys()))
    p.add_argument("--model", default="")
    p.add_argument("--vad", default="")
    p.add_argument("--punc", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--language", default="")
    p.add_argument("--use-itn", action="store_true")
    p.add_argument("--no-itn", action="store_true")
    p.add_argument("--batch-size-s", type=int, default=0)
    p.add_argument("--merge-vad", action="store_true")
    p.add_argument("--no-merge-vad", action="store_true")
    p.add_argument("--merge-length-s", type=int, default=0)
    p.add_argument("--max-single-segment-ms", type=int, default=0)
    p.add_argument("--speaker-diarization", action="store_true")
    p.add_argument(
        "--speaker-count",
        type=int,
        default=0,
        help="Known speaker count. 0 means automatic clustering.",
    )
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()

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
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "batch_size_s": int(args.batch_size_s)})
    if args.merge_vad:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "merge_vad": True})
    if args.no_merge_vad:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "merge_vad": False})
    if args.merge_length_s > 0:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "merge_length_s": int(args.merge_length_s)})
    if args.max_single_segment_ms > 0:
        decode_cfg = DecodeConfig(
            **{**decode_cfg.__dict__, "max_single_segment_ms": int(args.max_single_segment_ms)}
        )
    if args.speaker_count > 0:
        decode_cfg = DecodeConfig(**{**decode_cfg.__dict__, "preset_spk_num": int(args.speaker_count)})

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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_text(
            text=result.text,
            out_path=out_path,
            fmt=args.format,
            split_lines=bool(args.split_lines),
            source_name=_stem(inp),
        )
        if args.speaker_diarization and result.segments:
            speaker_text_path, speaker_json_path = _speaker_sidecar_paths(out_path)
            save_speaker_text(
                segments=result.segments,
                out_path=speaker_text_path,
                fmt=args.format,
                source_name=_stem(inp),
            )
            save_json(result.segments, speaker_json_path)
            result.meta["speaker_outputs"] = {
                "text": str(speaker_text_path),
                "json": str(speaker_json_path),
            }
        print(str(out_path))
    else:
        if args.speaker_diarization and result.segments:
            print(render_speaker_content(result.segments, fmt=args.format, source_name=_stem(inp)))
        else:
            print(result.text)

    if args.verbose:
        print("\n--- META ---")
        print(json.dumps(result.meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

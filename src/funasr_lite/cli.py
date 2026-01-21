from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_SCENARIO,
    DecodeConfig,
    ModelConfig,
    ObsConfig,
    OutputConfig,
    SCENARIO_PRESETS,
)
from .core import transcribe_file
from .save import save_text


def _as_jsonable(d: dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funasr-lite",
        description="Local transcription helper built on FunASR (CLI).",
    )

    p.add_argument("input", type=str, help="Path to an audio/video file")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output path (.md/.txt). If omitted, prints to stdout.",
    )
    p.add_argument(
        "--format",
        choices=["md", "txt"],
        default="md",
        help="Output format when --out is omitted.",
    )
    p.add_argument(
        "--split-lines",
        action="store_true",
        help="Split lines by punctuation (best-effort).",
    )

    # Scenario preset (recommended)
    p.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=sorted(SCENARIO_PRESETS.keys()),
        help="High-level preset that sets reasonable defaults.",
    )

    # Model
    p.add_argument("--model", type=str, default="", help="Custom model id")
    p.add_argument("--vad", type=str, default="", help="Custom VAD model id")
    p.add_argument(
        "--punc",
        type=str,
        default="",
        help="Custom punctuation model id (enable punctuation).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference.",
    )

    # Decode
    p.add_argument(
        "--language",
        type=str,
        default="",
        help="Language hint (e.g., auto/zh/en).",
    )
    p.add_argument("--use-itn", action="store_true", help="Enable ITN")
    p.add_argument("--no-itn", action="store_true", help="Disable ITN")
    p.add_argument(
        "--batch-size-s",
        type=int,
        default=0,
        help="Chunk length in seconds for decode batching.",
    )
    p.add_argument(
        "--merge-vad",
        action="store_true",
        help="Merge VAD segments before output.",
    )
    p.add_argument(
        "--no-merge-vad",
        action="store_true",
        help="Do not merge VAD segments.",
    )
    p.add_argument(
        "--merge-length-s",
        type=int,
        default=0,
        help="Max merge length in seconds.",
    )
    p.add_argument(
        "--max-single-segment-ms",
        type=int,
        default=0,
        help="Upper bound (ms) for a single VAD segment.",
    )

    # Observability
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print metadata (timings/memory) as JSON.",
    )
    return p


def _apply_overrides(
    scenario: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[ModelConfig, DecodeConfig, OutputConfig, ObsConfig]:
    # Start from scenario
    model_cfg = ModelConfig(**scenario["model"])
    decode_cfg = DecodeConfig(**scenario["decode"])
    out_cfg = OutputConfig(**scenario["output"])
    obs_cfg = ObsConfig(**scenario["obs"])

    # CLI overrides
    if args.model:
        model_cfg.model_id = args.model
    if args.vad:
        model_cfg.vad_model_id = args.vad
    if args.punc:
        model_cfg.punc_model_id = args.punc
    if args.device:
        model_cfg.device = args.device

    if args.language:
        decode_cfg.language = args.language

    if args.use_itn:
        decode_cfg.use_itn = True
    if args.no_itn:
        decode_cfg.use_itn = False

    if args.batch_size_s > 0:
        decode_cfg.batch_size_s = args.batch_size_s

    if args.merge_vad:
        decode_cfg.merge_vad = True
    if args.no_merge_vad:
        decode_cfg.merge_vad = False

    if args.merge_length_s > 0:
        decode_cfg.merge_length_s = args.merge_length_s

    if args.max_single_segment_ms > 0:
        decode_cfg.max_single_segment_time_ms = args.max_single_segment_ms

    out_cfg.output_format = args.format
    out_cfg.split_lines = bool(args.split_lines)
    obs_cfg.verbose_meta = bool(args.verbose)

    return model_cfg, decode_cfg, out_cfg, obs_cfg


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scenario not in SCENARIO_PRESETS:
        raise SystemExit(f"Unknown scenario: {args.scenario}")

    scenario = SCENARIO_PRESETS[args.scenario]
    model_cfg, decode_cfg, out_cfg, obs_cfg = _apply_overrides(scenario, args)

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    result = transcribe_file(
        wav_path=str(inp),
        model_cfg=model_cfg,
        decode_cfg=decode_cfg,
        output_cfg=out_cfg,
        obs_cfg=obs_cfg,
    )

    text = result.text
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_text(text=text, out_path=out_path, fmt=out_cfg.output_format)
        print(str(out_path))
    else:
        print(text)

    if obs_cfg.verbose_meta:
        print("\n--- META ---")
        print(_as_jsonable(result.meta))

#!/usr/bin/env python3
"""CLI 版的 Audio Workbench 工作流入口。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from funasr_lite.llm_client import LLMSettings
from funasr_lite.pipeline_workflow import (
    DEFAULT_MODEL,
    DEFAULT_PUNC,
    DEFAULT_SCENARIO,
    WorkflowRunRequest,
    run_audio_workbench,
)
from funasr_lite.workflow_profiles import get_workflow_profiles


ROOT = Path(__file__).resolve().parents[1]


def _looks_like_url(value: str) -> bool:
    """判断 CLI 输入是否是 URL。"""

    return bool(re.match(r"^https?://", value or ""))


def main() -> None:
    """解析命令行参数并执行一次 headless Audio Workbench 流程。"""

    parser = argparse.ArgumentParser(
        description="Pipeline wrapper for FunASR + yt-dlp + transcript prompt generation."
    )
    parser.add_argument("input", help="Local media path or URL.")
    parser.add_argument("--tag", default="", help="Optional tag such as #组会 / #技术 / #人文.")
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt file name or stem under prompts/. Overrides --tag mapping.",
    )
    profile_keys = sorted(get_workflow_profiles().keys())
    parser.add_argument(
        "--profile",
        default="",
        choices=[""] + profile_keys,
        help="Optional workflow profile. Profiles can pin download / prompt / note destinations.",
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model", default=os.getenv("FUNASR_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--punc", default=os.getenv("FUNASR_PUNC_MODEL_ID", DEFAULT_PUNC))
    parser.add_argument("--speaker-diarization", action="store_true")
    parser.add_argument(
        "--speaker-count",
        type=int,
        default=0,
        help="Known speaker count. 0 means automatic clustering.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Where transcript artifacts are stored. Defaults to profile output root or project outputs.",
    )
    parser.add_argument("--download-root", default="", help="Where URL media should be stored permanently.")
    parser.add_argument("--note-root", default="", help="Where final markdown summaries should be written.")
    parser.add_argument(
        "--download-mode",
        default="",
        choices=["", "audio", "video"],
        help="For URL inputs: keep audio only or full video. Defaults to video for managed profiles, otherwise audio.",
    )
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--retention-days", type=int, default=7)
    parser.add_argument("--keep-download", action="store_true")
    args, passthrough = parser.parse_known_args()

    source_url = args.input if _looks_like_url(args.input) else ""
    local_media_path = "" if source_url else args.input
    download_mode = args.download_mode or ("video" if args.profile else "audio")
    result = run_audio_workbench(
        WorkflowRunRequest(
            source_url=source_url,
            local_media_path=local_media_path,
            tag=args.tag,
            prompt_name=args.prompt,
            profile_key=args.profile,
            scenario=args.scenario,
            device=args.device,
            model=args.model,
            punc=args.punc,
            speaker_diarization=bool(args.speaker_diarization),
            speaker_count=int(args.speaker_count),
            output_root=args.output_root,
            default_output_root=str(ROOT / "outputs" / "transcripts"),
            download_root=args.download_root,
            note_root=args.note_root,
            download_mode=download_mode,
            retention_days=int(args.retention_days),
            keep_download=bool(args.keep_download),
            timestamp=args.timestamp,
            prepared_audio_name="audio.wav",
            passthrough_args=tuple(passthrough),
            llm=LLMSettings(enabled=False),
        )
    )
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

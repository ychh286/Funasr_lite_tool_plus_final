#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_PUNC = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
DEFAULT_SCENARIO = "中文会议/普通录音"
TAG_PROMPTS = {
    "#组会": "group_meeting.md",
    "#项目会": "meeting_project.md",
    "#讲座": "seminar_audio.md",
    "#研讨会": "seminar_audio.md",
    "#技术": "tech_video.md",
    "#日常": "daily_chat.md",
    "#人文": "humanities_video.md",
}


def find_cookie_file() -> Path | None:
    candidates = [
        ROOT / "secrets" / "cookies" / "all_cookies.txt",
        Path.home() / ".openclaw" / "secrets" / "cookies" / "all_cookies.txt",
        Path(r"\\wsl.localhost\Ubuntu\home\ychh_pc\.openclaw\secrets\cookies\all_cookies.txt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def clean_transcript_text(raw_text: str) -> str:
    lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"<\|[^>]*\>", "", line).strip()
        if not line:
            continue
        if line.startswith("# ASR Transcript:"):
            continue
        if line.startswith("# Speaker Transcript:"):
            continue
        if line.startswith("- Generated:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_media(input_value: str, out_dir: Path) -> Path:
    if re.match(r"^https?://", input_value):
        source_tpl = out_dir / "source.%(ext)s"
        cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0", "-o", str(source_tpl)]
        cookie_file = find_cookie_file()
        if cookie_file:
            cmd.extend(["--cookies", str(cookie_file)])
        cmd.append(input_value)
        run_command(cmd)
        candidates = sorted(out_dir.glob("source.*"))
        if not candidates:
            raise SystemExit("yt-dlp completed but no downloaded source file was found.")
        return candidates[0]

    media = Path(input_value).expanduser().resolve()
    if not media.exists():
        raise SystemExit(f"Input not found: {media}")
    return media


def resolve_prompt_name(tag: str, prompt: str) -> str:
    prompt_dir = ROOT / "prompts"
    available = {path.name.lower(): path.name for path in prompt_dir.glob("*.md")}
    available_stems = {path.stem.lower(): path.name for path in prompt_dir.glob("*.md")}

    if prompt:
        key = prompt.strip().lower()
        if key in available:
            return available[key]
        if key in available_stems:
            return available_stems[key]
        raise SystemExit(f"Prompt not found: {prompt}")

    if not tag:
        return ""

    return TAG_PROMPTS.get(tag.strip(), "")


def build_prompt(
    tag: str,
    prompt_name: str,
    transcript_txt: Path,
    out_dir: Path,
    source_value: str,
    *,
    speaker_transcript_md: Path | None = None,
) -> Path | None:
    if not prompt_name:
        return None
    template_path = ROOT / "prompts" / prompt_name
    if not template_path.exists():
        return None

    prompt_out = out_dir / "summary_prompt.md"
    transcript_body = transcript_txt.read_text(encoding="utf-8")
    payload = [
        template_path.read_text(encoding="utf-8").rstrip(),
        "",
        "# 待处理转写信息",
        f"- 标签: {tag or '未指定'}",
        f"- 模板: {prompt_name}",
        f"- 来源: {source_value}",
        f"- 转写目录: {out_dir}",
        "",
    ]

    if speaker_transcript_md and speaker_transcript_md.exists():
        payload.extend(
            [
                "# 带发言人转写（优先依据本节做总结）",
                "",
                speaker_transcript_md.read_text(encoding="utf-8").rstrip(),
                "",
            ]
        )

    payload.extend(
        [
            "# 原始转写（请严格依据以下内容，不要虚构）",
            "",
            transcript_body.rstrip(),
            "",
        ]
    )

    prompt_out.write_text("\n".join(payload), encoding="utf-8")
    return prompt_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline wrapper for FunASR + yt-dlp + transcript prompt generation.")
    parser.add_argument("input", help="Local media path or URL.")
    parser.add_argument("--tag", default="", help="Optional tag such as #组会 / #技术 / #人文.")
    parser.add_argument("--prompt", default="", help="Optional prompt file name or stem under prompts/. Overrides --tag mapping.")
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
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "transcripts"))
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--retention-days", type=int, default=7)
    parser.add_argument("--keep-download", action="store_true")
    args, passthrough = parser.parse_known_args()

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root).expanduser()
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retention_days > 0 and out_root.exists():
        cutoff = datetime.now().timestamp() - args.retention_days * 24 * 3600
        for child in out_root.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass

    media = resolve_media(args.input, out_dir)
    wav_path = out_dir / "audio.wav"
    run_command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(media), "-ac", "1", "-ar", "16000", str(wav_path)])

    transcript_md = out_dir / "transcript.md"
    transcript_txt = out_dir / "transcript.txt"
    transcript_speakers_md = out_dir / "transcript_speakers.md"
    transcript_speakers_json = out_dir / "transcript_speakers.json"

    transcribe_cmd = [
        sys.executable,
        str(ROOT / "transcribe_local.py"),
        str(wav_path),
        "--scenario",
        args.scenario,
        "--device",
        args.device,
        "--out",
        str(transcript_md),
        "--split-lines",
    ]
    model_env_override = os.getenv("FUNASR_MODEL_ID")
    punc_env_override = os.getenv("FUNASR_PUNC_MODEL_ID")
    if not (args.speaker_diarization and args.model == DEFAULT_MODEL and not model_env_override):
        transcribe_cmd.extend(["--model", args.model])
    if not (args.speaker_diarization and args.punc == DEFAULT_PUNC and not punc_env_override):
        transcribe_cmd.extend(["--punc", args.punc])
    if args.speaker_diarization:
        transcribe_cmd.append("--speaker-diarization")
    if args.speaker_count > 0:
        transcribe_cmd.extend(["--speaker-count", str(args.speaker_count)])
    transcribe_cmd.extend(passthrough)
    run_command(transcribe_cmd, cwd=ROOT)

    transcript_txt.write_text(clean_transcript_text(transcript_md.read_text(encoding="utf-8")), encoding="utf-8")

    speaker_prompt_source: Path | None = None
    if transcript_speakers_md.exists():
        speaker_prompt_source = transcript_speakers_md

    prompt_name = resolve_prompt_name(args.tag, args.prompt)
    prompt_path = build_prompt(
        args.tag,
        prompt_name,
        transcript_txt,
        out_dir,
        args.input,
        speaker_transcript_md=speaker_prompt_source,
    )

    if media.parent == out_dir and not args.keep_download and media != wav_path:
        try:
            media.unlink()
        except OSError:
            pass

    payload = {
        "tag": args.tag,
        "prompt_template": prompt_name,
        "speaker_diarization": bool(args.speaker_diarization),
        "output_dir": str(out_dir),
        "wav_path": str(wav_path),
        "transcript_md": str(transcript_md),
        "transcript_txt": str(transcript_txt),
        "transcript_speakers_md": str(transcript_speakers_md) if transcript_speakers_md.exists() else "",
        "transcript_speakers_json": str(transcript_speakers_json) if transcript_speakers_json.exists() else "",
        "summary_prompt": str(prompt_path) if prompt_path else "",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

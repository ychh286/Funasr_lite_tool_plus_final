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

from funasr_lite.workflow_profiles import (
    build_suggested_note_path,
    build_managed_media_path,
    detect_workflow_profile_from_input,
    get_workflow_profile,
    get_workflow_profiles,
)


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


def find_cookie_file(input_value: str = "") -> Path | None:
    env_candidates = [
        os.getenv("FUNASR_LITE_COOKIE_FILE", "").strip(),
        os.getenv("YTDLP_COOKIE_FILE", "").strip(),
    ]
    for raw in env_candidates:
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path

    cookies_root = Path(r"E:\Software_tools\yt-dlp\cookies")
    raw_input = (input_value or "").lower()
    site_candidates: list[Path] = []
    if "bilibili.com" in raw_input or "b23.tv" in raw_input:
        site_candidates.extend(
            [
                cookies_root / "bili_cookies.txt",
                cookies_root / "cookies.txt",
            ]
        )
    elif "youtube.com" in raw_input or "youtu.be" in raw_input:
        site_candidates.extend(
            [
                cookies_root / "youtube_cookies.txt",
                cookies_root / "cookies.txt",
            ]
        )
    elif "patreon.com" in raw_input:
        site_candidates.extend(
            [
                cookies_root / "patreon_cookies.txt",
                cookies_root / "cookies.txt",
            ]
        )
    else:
        site_candidates.append(cookies_root / "cookies.txt")

    fallback_candidates = [
        ROOT / "secrets" / "cookies" / "all_cookies.txt",
    ]

    for path in site_candidates + fallback_candidates:
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


def run_command(cmd: list[str], *, cwd: Path | None = None, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _echo_completed_output(cp: subprocess.CompletedProcess[str]) -> None:
    if cp.stdout:
        sys.stdout.buffer.write(cp.stdout.encode("utf-8", errors="replace"))
        if not cp.stdout.endswith("\n"):
            sys.stdout.buffer.write(b"\n")
    if cp.stderr:
        sys.stderr.buffer.write(cp.stderr.encode("utf-8", errors="replace"))
        if not cp.stderr.endswith("\n"):
            sys.stderr.buffer.write(b"\n")


def _extract_download_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    lowered = text.lower()
    if "premium member to download" in lowered or "1080p 高清 are missing" in lowered or "1080p 高码率" in text:
        warnings.append(
            "当前 cookies 不具备目标站点的高权限下载能力；如需更高清晰度/高码率，请更新 E:\\Software_tools\\yt-dlp\\cookies 下对应站点的 cookies 文件。"
        )
    return warnings


def _predict_ytdlp_path(cmd: list[str]) -> Path | None:
    try:
        cp = run_command(cmd, capture_output=True)
    except subprocess.CalledProcessError:
        return None

    lines = [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    return Path(lines[-1])


def resolve_media(
    input_value: str,
    out_dir: Path,
    *,
    download_root: Path | None = None,
    download_mode: str = "audio",
) -> tuple[Path, list[str]]:
    if re.match(r"^https?://", input_value):
        target_dir = download_root or out_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        source_tpl = target_dir / "%(title)s [%(id)s].%(ext)s"
        archive_path = target_dir / ".yt-dlp-archive.txt"
        predict_cmd = ["yt-dlp", "--get-filename", "-o", str(source_tpl)]
        cmd = ["yt-dlp", "--download-archive", str(archive_path), "-o", str(source_tpl)]
        if download_mode == "audio":
            predict_cmd.extend(["-x", "--audio-format", "wav", "--audio-quality", "0"])
            cmd.extend(["-x", "--audio-format", "wav", "--audio-quality", "0"])
        else:
            predict_cmd.extend(["--merge-output-format", "mp4"])
            cmd.extend(["--merge-output-format", "mp4"])
        cookie_file = find_cookie_file(input_value)
        if cookie_file:
            predict_cmd.extend(["--cookies", str(cookie_file)])
            cmd.extend(["--cookies", str(cookie_file)])
        predict_cmd.append(input_value)
        cmd.append(input_value)

        predicted = _predict_ytdlp_path(predict_cmd)
        cp = run_command(cmd, capture_output=True)
        _echo_completed_output(cp)
        warnings = _extract_download_warnings((cp.stdout or "") + "\n" + (cp.stderr or ""))

        if predicted and predicted.exists():
            return predicted, warnings

        candidates = sorted(
            [
                p
                for p in target_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".m4a", ".wav", ".mp3"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit("yt-dlp completed but no downloaded media file was found.")
        return candidates[0], warnings

    media = Path(input_value).expanduser().resolve()
    if not media.exists():
        raise SystemExit(f"Input not found: {media}")
    return media, []


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
    profile_name: str = "",
    note_target_path: Path | None = None,
    media_library_path: Path | None = None,
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
        f"- 工作流配置: {profile_name or '通用'}",
        f"- 来源: {source_value}",
        f"- 转写目录: {out_dir}",
        f"- 媒体存档路径: {media_library_path or '未指定'}",
        f"- 建议总结写入路径: {note_target_path or '未指定'}",
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
    parser.add_argument("--output-root", default="", help="Where transcript artifacts are stored. Defaults to profile output root or project outputs.")
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

    profile = get_workflow_profile(args.profile) if args.profile else detect_workflow_profile_from_input(args.input)
    default_output_root = str(ROOT / "outputs" / "transcripts")
    output_root_value = args.output_root or (profile.output_root if profile and profile.output_root else default_output_root)
    download_root = Path(args.download_root).expanduser() if args.download_root else (Path(profile.download_root) if profile else None)
    note_root = Path(args.note_root).expanduser() if args.note_root else (Path(profile.note_root) if profile else None)
    effective_tag = args.tag or (profile.tag if profile else "")
    effective_prompt_name = args.prompt or (profile.prompt_name if profile else "")
    download_mode = args.download_mode or ("video" if profile else "audio")

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_root_value).expanduser()
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

    media, download_warnings = resolve_media(
        args.input,
        out_dir,
        download_root=download_root,
        download_mode=download_mode,
    )
    if profile and download_root and media.exists():
        managed_media_path = build_managed_media_path(profile, media, download_root=download_root)
        if managed_media_path != media:
            managed_media_path.parent.mkdir(parents=True, exist_ok=True)
            if not managed_media_path.exists():
                media.rename(managed_media_path)
            media = managed_media_path

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

    prompt_name = resolve_prompt_name(effective_tag, effective_prompt_name)
    note_target_path = build_suggested_note_path(profile, media, note_root=note_root) if profile and note_root else None
    prompt_path = build_prompt(
        effective_tag,
        prompt_name,
        transcript_txt,
        out_dir,
        args.input,
        speaker_transcript_md=speaker_prompt_source,
        profile_name=profile.display_name if profile else "",
        note_target_path=note_target_path,
        media_library_path=media,
    )

    if media.parent == out_dir and not args.keep_download and media != wav_path:
        try:
            media.unlink()
        except OSError:
            pass

    payload = {
        "tag": effective_tag,
        "profile": profile.key if profile else "",
        "profile_name": profile.display_name if profile else "",
        "prompt_template": prompt_name,
        "speaker_diarization": bool(args.speaker_diarization),
        "output_dir": str(out_dir),
        "download_mode": download_mode,
        "download_root": str(download_root) if download_root else "",
        "download_warnings": download_warnings,
        "note_root": str(note_root) if note_root else "",
        "source_media_path": str(media),
        "suggested_note_path": str(note_target_path) if note_target_path else "",
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

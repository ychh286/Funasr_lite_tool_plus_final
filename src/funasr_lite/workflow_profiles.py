from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WorkflowProfile:
    key: str
    display_name: str
    tag: str
    prompt_name: str
    download_root: str
    note_root: str
    output_root: str
    note_title_prefix: str


WORKFLOW_PROFILES: dict[str, WorkflowProfile] = {
    "bohai": WorkflowProfile(
        key="bohai",
        display_name="Bilibili 渤海小吏",
        tag="#人文",
        prompt_name="humanities_video -b站-渤海小吏.md",
        download_root=r"E:\Software_tools\yt-dlp\downloads\BiliBili\渤海小吏\合集·进化缓慢的人性",
        note_root=r"E:\obsidian_files\AI_coding_obsidian\【音视频总结】\【b站】【渤海小吏】充电视频汇总\深度解读",
        output_root=r"E:\Software_tools\yt-dlp\downloads\BiliBili\渤海小吏\合集·进化缓慢的人性\_workflow_outputs",
        note_title_prefix="【深度解读】",
    ),
    "peak": WorkflowProfile(
        key="peak",
        display_name="Bilibili 峰video",
        tag="#人文",
        prompt_name="humanities_video - b站-峰video.md",
        download_root=r"E:\Software_tools\yt-dlp\downloads\BiliBili\峰video\合集·“带你升级！”系列充电专属视频",
        note_root=r"E:\obsidian_files\AI_coding_obsidian\【音视频总结】\【b站】【峰video】充电视频-01-36-精读总结汇总\【b站】【峰video】充电视频-01-36-深度精读改写",
        output_root=r"E:\Software_tools\yt-dlp\downloads\BiliBili\峰video\合集·“带你升级！”系列充电专属视频\_workflow_outputs",
        note_title_prefix="【深度精读】",
    ),
}


_VIDEO_NAME_RE = re.compile(
    r"^(?P<index>\d+)\s*-\s*(?P<title>.+?)\s*\[(?P<bvid>BV[0-9A-Za-z]+)\]$"
)


def get_workflow_profile(key: str) -> Optional[WorkflowProfile]:
    return WORKFLOW_PROFILES.get((key or "").strip().lower())


def detect_workflow_profile_from_input(input_value: str) -> Optional[WorkflowProfile]:
    try:
        candidate = Path(input_value).expanduser().resolve()
    except OSError:
        return None

    for profile in WORKFLOW_PROFILES.values():
        try:
            root = Path(profile.download_root).resolve()
        except OSError:
            continue
        if root == candidate or root in candidate.parents:
            return profile
    return None


def build_suggested_note_path(profile: WorkflowProfile, media_path: Path, *, note_root: Path | None = None) -> Path:
    stem = media_path.stem
    match = _VIDEO_NAME_RE.match(stem)
    note_root = note_root or Path(profile.note_root)

    if not match:
        return note_root / f"{_sanitize_filename(profile.note_title_prefix + stem)}.md"

    idx = int(match.group("index"))
    title = match.group("title").strip()
    bvid = match.group("bvid").strip()
    filename = f"{profile.note_title_prefix}{idx:02d} - {title} [{bvid}].md"
    return note_root / _sanitize_filename(filename)


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\\\|?*]', "_", name).strip()
    sanitized = sanitized.rstrip(". ")
    return sanitized or "untitled.md"

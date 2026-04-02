from __future__ import annotations

import json
import os
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


_VIDEO_NAME_RE = re.compile(
    r"^(?P<index>\d+)\s*-\s*(?P<title>.+?)\s*\[(?P<bvid>BV[0-9A-Za-z]+)\]$"
)

_PROFILES_CACHE: dict[str, WorkflowProfile] | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_candidates() -> list[Path]:
    env = os.getenv("FUNASR_LITE_PROFILE_CONFIG", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    root = _project_root()
    candidates.extend(
        [
            root / "workflow_profiles.local.json",
            root / "secrets" / "workflow_profiles.local.json",
        ]
    )
    return candidates


def _load_profiles_from_disk() -> dict[str, WorkflowProfile]:
    for path in _config_candidates():
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Workflow profile config must be a JSON object: {path}")

        profiles_obj = raw.get("profiles", raw)
        if not isinstance(profiles_obj, dict):
            raise ValueError(f"`profiles` must be a JSON object: {path}")

        profiles: dict[str, WorkflowProfile] = {}
        for key, value in profiles_obj.items():
            if not isinstance(value, dict):
                raise ValueError(f"Profile `{key}` must be a JSON object: {path}")
            profile_key = str(key).strip().lower()
            if not profile_key:
                raise ValueError(f"Encountered empty profile key in {path}")
            profiles[profile_key] = WorkflowProfile(
                key=profile_key,
                display_name=str(value.get("display_name") or profile_key),
                tag=str(value.get("tag") or ""),
                prompt_name=str(value.get("prompt_name") or ""),
                download_root=str(value.get("download_root") or ""),
                note_root=str(value.get("note_root") or ""),
                output_root=str(value.get("output_root") or ""),
                note_title_prefix=str(value.get("note_title_prefix") or ""),
            )
        return profiles
    return {}


def get_workflow_profiles() -> dict[str, WorkflowProfile]:
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        _PROFILES_CACHE = _load_profiles_from_disk()
    return _PROFILES_CACHE


def reload_workflow_profiles() -> dict[str, WorkflowProfile]:
    global _PROFILES_CACHE
    _PROFILES_CACHE = _load_profiles_from_disk()
    return _PROFILES_CACHE


def get_workflow_profile(key: str) -> Optional[WorkflowProfile]:
    return get_workflow_profiles().get((key or "").strip().lower())


def detect_workflow_profile_from_input(input_value: str) -> Optional[WorkflowProfile]:
    try:
        candidate = Path(input_value).expanduser().resolve()
    except OSError:
        return None

    for profile in get_workflow_profiles().values():
        if not profile.download_root:
            continue
        try:
            root = Path(profile.download_root).expanduser().resolve()
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
        prefix = profile.note_title_prefix or ""
        return note_root / f"{_sanitize_filename(prefix + stem)}.md"

    idx = int(match.group("index"))
    title = match.group("title").strip()
    bvid = match.group("bvid").strip()
    prefix = profile.note_title_prefix or ""
    filename = f"{prefix}{idx:02d} - {title} [{bvid}].md"
    return note_root / _sanitize_filename(filename)


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\\\|?*]', "_", name).strip()
    sanitized = sanitized.rstrip(". ")
    return sanitized or "untitled.md"

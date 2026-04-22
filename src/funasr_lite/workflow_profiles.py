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
    url_patterns: tuple[str, ...]


_VIDEO_NAME_RE = re.compile(
    r"^(?P<index>\d+)\s*-\s*(?P<title>.+?)\s*\[(?P<bvid>BV[0-9A-Za-z]+)\]$"
)
_BVID_RE = re.compile(r"\[(?P<bvid>BV[0-9A-Za-z]+)\]$")
_LEADING_INDEX_RE = re.compile(r"^(?P<index>\d+)\s*-\s*(?P<rest>.+)$")

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
                url_patterns=tuple(str(item) for item in (value.get("url_patterns") or []) if str(item).strip()),
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
    raw_input = (input_value or "").strip()
    for profile in get_workflow_profiles().values():
        for pattern in profile.url_patterns:
            try:
                if re.search(pattern, raw_input, re.IGNORECASE):
                    return profile
            except re.error:
                continue

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


def build_managed_media_path(
    profile: WorkflowProfile,
    media_path: Path,
    *,
    download_root: Path | None = None,
) -> Path:
    target_root = download_root or Path(profile.download_root)
    stem = media_path.stem
    if _VIDEO_NAME_RE.match(stem):
        return target_root / media_path.name

    bvid = _extract_bvid(stem)
    if bvid:
        existing = _find_existing_media_by_bvid(target_root, bvid, suffix=media_path.suffix)
        if existing:
            return existing

    title = stem
    if bvid:
        title = re.sub(rf"\s*\[{re.escape(bvid)}\]$", "", title).strip()
    leading = _LEADING_INDEX_RE.match(title)
    if leading:
        title = leading.group("rest").strip()

    next_index = _infer_next_index(target_root)
    if bvid:
        filename = f"{next_index:02d} - {title} [{bvid}]{media_path.suffix}"
    else:
        filename = f"{next_index:02d} - {title}{media_path.suffix}"
    return target_root / _sanitize_filename(filename)


def _extract_bvid(stem: str) -> str | None:
    match = _VIDEO_NAME_RE.match(stem)
    if match:
        return match.group("bvid")
    match = _BVID_RE.search(stem)
    if match:
        return match.group("bvid")
    return None


def _find_existing_media_by_bvid(root: Path, bvid: str, *, suffix: str = "") -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if suffix and path.suffix.lower() != suffix.lower():
            continue
        if f"[{bvid}]" in path.name:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[0]


def _infer_next_index(root: Path) -> int:
    max_index = 0
    if root.exists():
        for path in root.iterdir():
            if not path.is_file():
                continue
            match = _LEADING_INDEX_RE.match(path.stem)
            if not match:
                continue
            try:
                max_index = max(max_index, int(match.group("index")))
            except ValueError:
                continue
    return max_index + 1


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\\\|?*]', "_", name).strip()
    sanitized = sanitized.rstrip(". ")
    return sanitized or "untitled.md"

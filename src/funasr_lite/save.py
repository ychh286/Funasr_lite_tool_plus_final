from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .formatting import split_lines_by_punct


def _generated_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_ms(ms: int | None) -> str:
    if ms is None:
        return "unknown"
    total_ms = max(int(ms), 0)
    seconds, millis = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def render_content(text: str, fmt: str, split_lines: bool, source_name: str = "") -> str:
    fmt = fmt.lower().strip(".")
    body = split_lines_by_punct(text) if split_lines else text.strip()

    if fmt == "txt":
        return body + "\n"

    header = f"# ASR Transcript: {source_name or 'input'}\n\n- Generated: {_generated_timestamp()}\n\n"
    return header + body + "\n"


def render_speaker_content(
    segments: List[Dict[str, Any]],
    fmt: str,
    source_name: str = "",
) -> str:
    fmt = fmt.lower().strip(".")

    rows = []
    for seg in segments:
        speaker = seg.get("speaker") or "unknown-speaker"
        start = _format_ms(seg.get("start_ms"))
        end = _format_ms(seg.get("end_ms"))
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        if fmt == "txt":
            rows.append(f"[{start} -> {end}] {speaker}: {text}")
        else:
            rows.append(f"- [{start} -> {end}] **{speaker}**: {text}")

    body = "\n".join(rows).strip()
    if fmt == "txt":
        return body + ("\n" if body else "")

    header = (
        f"# Speaker Transcript: {source_name or 'input'}\n\n"
        f"- Generated: {_generated_timestamp()}\n\n"
    )
    return header + body + ("\n" if body else "")


def save_text(text: str, out_path: Path, fmt: str, split_lines: bool, source_name: str = "") -> Path:
    fmt = fmt.lower().strip(".")
    if fmt not in {"md", "txt"}:
        raise ValueError("format must be 'md' or 'txt'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_content(text, fmt=fmt, split_lines=split_lines, source_name=source_name),
        encoding="utf-8",
    )
    return out_path


def save_speaker_text(
    segments: List[Dict[str, Any]],
    out_path: Path,
    fmt: str,
    source_name: str = "",
) -> Path:
    fmt = fmt.lower().strip(".")
    if fmt not in {"md", "txt"}:
        raise ValueError("format must be 'md' or 'txt'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_speaker_content(segments, fmt=fmt, source_name=source_name),
        encoding="utf-8",
    )
    return out_path


def save_json(data: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

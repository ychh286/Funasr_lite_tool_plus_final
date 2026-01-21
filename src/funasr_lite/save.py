\
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from .formatting import split_lines_by_punct


def render_content(text: str, fmt: str, split_lines: bool, source_name: str = "") -> str:
    fmt = fmt.lower().strip(".")
    body = split_lines_by_punct(text) if split_lines else text.strip()

    if fmt == "txt":
        return body + "\n"

    # markdown default
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# ASR 转录：{source_name}\n\n- Generated: {ts}\n\n"
    return header + body + "\n"


def save_text(text: str, out_path: Path, fmt: str, split_lines: bool, source_name: str = "") -> Path:
    fmt = fmt.lower().strip(".")
    if fmt not in {"md", "txt"}:
        raise ValueError("format must be 'md' or 'txt'")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_content(text, fmt=fmt, split_lines=split_lines, source_name=source_name), encoding="utf-8")
    return out_path

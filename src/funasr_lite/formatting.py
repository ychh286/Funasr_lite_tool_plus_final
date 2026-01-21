\
from __future__ import annotations

import re
from funasr.utils.postprocess_utils import rich_transcription_postprocess


_PUNCT_SPLIT_RE = re.compile(r"([。！？；.!?;]+)")


def postprocess_text(text: str) -> str:
    """FunASR rich postprocess."""
    return rich_transcription_postprocess(text)


def split_lines_by_punct(text: str) -> str:
    """
    Experimental: split into lines by major punctuation.
    This is *formatting only*, not semantic sentence segmentation.
    """
    t = text.strip()
    if not t:
        return t

    parts = _PUNCT_SPLIT_RE.split(t)
    out = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if _PUNCT_SPLIT_RE.fullmatch(p):
            buf += p
            out.append(buf.strip())
            buf = ""
        else:
            buf += p
    if buf.strip():
        out.append(buf.strip())
    return "\n".join(out)

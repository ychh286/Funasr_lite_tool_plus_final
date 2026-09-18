from __future__ import annotations

"""转写文本的轻量后处理工具。"""

import re
from typing import Callable


_PUNCT_SPLIT_RE = re.compile(r"([。！？；.!?;]+)")
_POSTPROCESS_FN: Callable[[str], str] | None = None


def _postprocess_fn() -> Callable[[str], str]:
    """延迟加载 FunASR 的 rich_transcription_postprocess。

    在未安装 FunASR 的测试或 GUI 配置环境里，文本后处理会退回为原样返回；真正
    转写环境安装 FunASR 后仍使用官方后处理函数。
    """

    global _POSTPROCESS_FN
    if _POSTPROCESS_FN is not None:
        return _POSTPROCESS_FN
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError:
        _POSTPROCESS_FN = lambda text: text
    else:
        _POSTPROCESS_FN = rich_transcription_postprocess
    return _POSTPROCESS_FN


def postprocess_text(text: str) -> str:
    """执行 FunASR 富文本后处理；不可用时安全退回原文。"""

    return _postprocess_fn()(text)


def split_lines_by_punct(text: str) -> str:
    """按主要标点拆行。

    该函数只做格式化，不承担语义级断句判断。
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

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PromptAsset:
    label: str
    path: Path
    default: bool = False


def _read_text_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_prompt_assets(root: Path | None = None) -> list[PromptAsset]:
    """Return selectable prompt files for the UI.

    Priority:
    1. repo-level .codex/agents.md, if present
    2. prompts/*.md sorted by filename
    """

    root = root or ROOT
    assets: list[PromptAsset] = []
    default_agent = root / ".codex" / "agents.md"
    if default_agent.exists():
        assets.append(PromptAsset(label=".codex/agents.md", path=default_agent, default=True))

    prompt_dir = root / "prompts"
    if prompt_dir.exists():
        for path in sorted(prompt_dir.glob("*.md"), key=lambda p: p.name.lower()):
            assets.append(PromptAsset(label=f"prompts/{path.name}", path=path, default=False))

    return assets


def prompt_labels(root: Path | None = None) -> list[str]:
    return [asset.label for asset in discover_prompt_assets(root)]


def default_prompt_labels(root: Path | None = None) -> list[str]:
    assets = discover_prompt_assets(root)
    defaults = [asset.label for asset in assets if asset.default]
    if defaults:
        return defaults
    for asset in assets:
        if asset.path.name == "daily_chat.md":
            return [asset.label]
    return assets[:1] and [assets[0].label] or []


def profile_prompt_label(prompt_name: str, root: Path | None = None) -> str:
    if not prompt_name:
        return ""
    target = f"prompts/{prompt_name}"
    labels = set(prompt_labels(root))
    if target in labels:
        return target
    normalized = prompt_name.strip().lower()
    for label in labels:
        name = label.split("/", 1)[-1].lower()
        stem = name[:-3] if name.endswith(".md") else name
        if normalized == name or normalized == stem:
            return label
    return ""


def load_selected_prompts(
    selected_labels: Sequence[str] | None,
    *,
    root: Path | None = None,
) -> list[tuple[str, str]]:
    root = root or ROOT
    by_label = {asset.label: asset.path for asset in discover_prompt_assets(root)}
    prompt_blocks: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label in selected_labels or []:
        if not label or label in seen:
            continue
        path = by_label.get(label)
        if not path or not path.exists():
            continue
        prompt_blocks.append((label, _read_text_safe(path).strip()))
        seen.add(label)
    return prompt_blocks


def load_uploaded_prompt_files(files: object) -> list[tuple[str, str]]:
    paths = _as_path_list(files)
    prompt_blocks: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        prompt_blocks.append((f"uploaded/{path.name}", _read_text_safe(path).strip()))
    return prompt_blocks


def _as_path_list(files: object) -> list[Path]:
    if not files:
        return []
    if isinstance(files, (str, Path)):
        return [Path(str(files))]
    paths: list[Path] = []
    try:
        iterator = iter(files)  # type: ignore[arg-type]
    except TypeError:
        iterator = iter([files])
    for item in iterator:
        if item is None:
            continue
        if isinstance(item, Path):
            paths.append(item)
        elif isinstance(item, str):
            paths.append(Path(item))
        else:
            path_like = getattr(item, "path", None) or getattr(item, "name", None)
            if path_like:
                paths.append(Path(str(path_like)))
            elif isinstance(item, dict):
                for key in ("path", "name"):
                    value = item.get(key)
                    if value:
                        paths.append(Path(str(value)))
                        break
    return paths


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


def compose_summary_prompt(
    *,
    transcript_text: str,
    source_value: str,
    prompt_blocks: Sequence[tuple[str, str]],
    inline_prompt: str = "",
    profile_name: str = "",
    tag: str = "",
    media_path: str = "",
    output_dir: str = "",
    suggested_note_path: str = "",
    speaker_transcript_text: str = "",
) -> str:
    """Build the single prompt handed to an external LLM/agent."""

    parts: list[str] = []
    if prompt_blocks:
        parts.append("# 已选择的 Agent / Prompt 指令")
        for label, body in prompt_blocks:
            if not body.strip():
                continue
            parts.extend([f"\n## {label}", body.strip()])
        parts.append("")

    inline = (inline_prompt or "").strip()
    if inline:
        parts.extend(["# 本次追加指令", inline, ""])

    parts.extend(
        [
            "# 本次音视频处理上下文",
            f"- 来源: {source_value or '未指定'}",
            f"- 工作流配置: {profile_name or '通用'}",
            f"- 标签: {tag or '未指定'}",
            f"- 媒体路径: {media_path or '未指定'}",
            f"- 转写目录: {output_dir or '未指定'}",
            f"- 建议落盘路径: {suggested_note_path or '未指定'}",
            "",
            "# 任务边界",
            "- 严格依据转写内容总结，不要编造原文没有的信息。",
            "- 如果转写存在错字或语音识别不确定，请以“可能/疑似”表达。",
            "- 输出 Markdown，可包含标题、要点、分节总结、行动项、待核对问题。",
            "",
        ]
    )

    if speaker_transcript_text.strip():
        parts.extend(
            [
                "# 带发言人转写（如可用，优先参考）",
                speaker_transcript_text.strip(),
                "",
            ]
        )

    parts.extend(
        [
            "# 原始转写",
            transcript_text.strip(),
            "",
        ]
    )

    return "\n".join(parts).strip() + "\n"


def render_final_markdown(
    *,
    llm_output: str,
    prompt_text: str,
    transcript_text: str,
    title: str = "音视频总结",
    source_value: str = "",
    include_prompt: bool = False,
    include_transcript: bool = False,
) -> str:
    parts = [f"# {title.strip() or '音视频总结'}", ""]
    if source_value:
        parts.extend([f"- 来源: {source_value}", ""])
    if llm_output.strip():
        parts.extend([llm_output.strip(), ""])
    else:
        parts.extend(
            [
                "> 未调用大模型或大模型未返回内容。下面保留可复制给 agent 的汇总提示词。",
                "",
            ]
        )
        include_prompt = True
    if include_prompt:
        parts.extend(["---", "", "## 本次完整提示词", "", prompt_text.strip(), ""])
    if include_transcript:
        parts.extend(["---", "", "## 转写原文", "", transcript_text.strip(), ""])
    return "\n".join(parts).rstrip() + "\n"


def sanitize_filename(name: str, default: str = "untitled") -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    sanitized = sanitized.rstrip(". ")
    return sanitized or default


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(2, 1000):
        candidate = parent / f"{stem} ({idx}){suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem} ({os.getpid()}){suffix}"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PromptSource:
    """表示一个可被 Audio Workbench 选中的 Prompt 来源。

    功能：
        把 `.codex/agents.md`、`prompts/*.md` 等离散文件统一抽象成可选项。

    关键参数：
        key：写入 manifest 和 WebUI 选择器的稳定标识。
        label：给用户展示的名称。
        path：磁盘上的真实文件路径。
        group：来源分组，便于后续区分“默认规则”和“项目模板”。

    返回值：
        数据类本身就是 Prompt 条目的承载对象。

    状态影响：
        无副作用。

    调用时机：
        WebUI 初始化 Prompt 列表，或工作流准备生成 `summary_prompt.md` 时。
    """

    key: str
    label: str
    path: Path
    group: str
    default_selected: bool = False


def project_root() -> Path:
    """返回当前仓库根目录。"""

    return Path(__file__).resolve().parents[2]


def discover_prompt_sources() -> list[PromptSource]:
    """扫描当前项目里可供组合的 Prompt 文件。

    功能：
        自动收集默认代理规则和项目级 Prompt 模板，避免把文件列表写死在代码里。

    关键参数：
        无。

    返回值：
        `PromptSource` 列表，已按展示顺序组织好。

    状态影响：
        会读取仓库目录结构，但不会修改文件。

    调用时机：
        WebUI 启动时，或正式组装 `summary_prompt.md` 之前。
    """

    root = project_root()
    sources: list[PromptSource] = []

    default_prompt = root / ".codex" / "agents.md"
    if default_prompt.exists():
        sources.append(
            PromptSource(
                key=".codex/agents.md",
                label=".codex/agents.md (默认)",
                path=default_prompt,
                group="default",
                default_selected=True,
            )
        )

    prompt_dir = root / "prompts"
    if prompt_dir.exists():
        for path in sorted(prompt_dir.glob("*.md"), key=lambda item: item.name.lower()):
            relative = path.relative_to(root).as_posix()
            sources.append(
                PromptSource(
                    key=relative,
                    label=relative,
                    path=path,
                    group="project",
                )
            )

    return sources


def prompt_source_index() -> dict[str, PromptSource]:
    """把 Prompt 列表整理成按 key 查询的映射。"""

    return {source.key: source for source in discover_prompt_sources()}


def default_prompt_keys() -> list[str]:
    """返回首次进入 Audio Workbench 时建议默认勾选的 Prompt。"""

    return [source.key for source in discover_prompt_sources() if source.default_selected]


def prompt_name_to_key(prompt_name: str) -> str | None:
    """把旧式 `--prompt xxx` 选择转换成新的 Prompt key。

    功能：
        兼容原先 pipeline 的“按 prompt 文件名或 stem 选择模板”习惯，
        让 CLI 和新工作台都能走同一套 Prompt 组装逻辑。

    关键参数：
        prompt_name：CLI 或 profile 里传入的 prompt 名称。

    返回值：
        能命中时返回稳定 key，不能命中时返回 `None`。

    状态影响：
        无副作用。

    调用时机：
        旧 pipeline 迁移到新工作流编排层时。
    """

    raw = (prompt_name or "").strip()
    if not raw:
        return None
    candidate_names = {raw, raw if raw.endswith(".md") else f"{raw}.md"}
    available = prompt_source_index()
    for key, source in available.items():
        if key in candidate_names:
            return key
        if source.path.name in candidate_names or source.path.stem == raw:
            return key
    return None


def resolve_selected_prompt_sources(selected_keys: list[str]) -> list[PromptSource]:
    """根据用户选择的 key 解析出实际 Prompt 文件列表。"""

    available = prompt_source_index()
    resolved: list[PromptSource] = []
    for key in selected_keys:
        source = available.get(key)
        if source is None:
            continue
        resolved.append(source)
    return resolved


def _read_text(path: Path) -> str:
    """统一按 UTF-8 读取 Prompt 文件内容。"""

    return path.read_text(encoding="utf-8").strip()


def compose_summary_prompt_markdown(
    *,
    selected_sources: list[PromptSource],
    uploaded_prompt_path: Path | None,
    extra_instruction: str,
    tag: str,
    profile_name: str,
    source_value: str,
    output_dir: Path,
    media_library_path: Path | None,
    note_target_path: Path | None,
    transcript_text: str,
    speaker_transcript_text: str,
) -> str:
    """把默认 Prompt、项目模板和本次补充说明组装成最终 `summary_prompt.md`。

    功能：
        为后续 LLM 总结提供一个可追踪、可复核的单一输入包。

    关键参数：
        selected_sources：用户从列表里勾选的 Prompt 文件。
        uploaded_prompt_path：本次额外上传的 Prompt 文件。
        extra_instruction：用户在文本框补充的即时指令。
        transcript_text：清洗后的正文转写。
        speaker_transcript_text：带发言人分段的转写文本，可选。

    返回值：
        一段完整 Markdown，可直接写入 `summary_prompt.md`。

    状态影响：
        只做字符串拼装，不会修改磁盘。

    调用时机：
        ASR 完成后、LLM 调用前。
    """

    lines = [
        "# Audio Workbench Summary Package",
        "",
        "## 使用要求",
        "- 严格依据转写内容整理，不要补造原始音视频里没有的信息。",
        "- 若转写存在歧义、缺字或说话人不确定，请直接标注“不确定”。",
        "- 优先参考“带发言人转写”段落，再参考原始转写正文。",
        "- 输出应为结构清晰、便于继续沉淀到 Obsidian 的 Markdown。",
        "",
        "## 运行上下文",
        f"- 标签: {tag or '未指定'}",
        f"- 工作流配置: {profile_name or '通用'}",
        f"- 原始输入: {source_value or '未提供'}",
        f"- 运行目录: {output_dir}",
        f"- 媒体存档路径: {media_library_path or '未指定'}",
        f"- 建议总结落盘路径: {note_target_path or '未指定'}",
        "",
    ]

    if selected_sources:
        lines.extend(["## 已选 Prompt 来源", ""])
        for source in selected_sources:
            lines.extend(
                [
                    f"### {source.label}",
                    "",
                    _read_text(source.path),
                    "",
                ]
            )

    if uploaded_prompt_path and uploaded_prompt_path.exists():
        lines.extend(
            [
                "## 本次上传 Prompt",
                "",
                _read_text(uploaded_prompt_path),
                "",
            ]
        )

    if (extra_instruction or "").strip():
        lines.extend(
            [
                "## 本次补充说明",
                "",
                extra_instruction.strip(),
                "",
            ]
        )

    if speaker_transcript_text.strip():
        lines.extend(
            [
                "## 带发言人转写（优先参考）",
                "",
                speaker_transcript_text.strip(),
                "",
            ]
        )

    lines.extend(
        [
            "## 原始转写",
            "",
            transcript_text.strip(),
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"

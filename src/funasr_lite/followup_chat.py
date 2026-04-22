from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .brave_search import BraveSearchClient, BraveSearchSettings, format_web_results
from .llm_client import LLMSettings, OpenAICompatibleClient
from .prompting import sanitize_filename


@dataclass(frozen=True)
class FollowupChatResult:
    """表示一次追问会话执行后的关键产物。"""

    run_dir: Path
    session_id: str
    chat_history_pairs: list[tuple[str, str]]
    answer_markdown: str
    current_summary_text: str
    current_summary_path: Path
    transcript_path: Path
    web_context_text: str
    meta: dict[str, Any]


def new_session_id() -> str:
    """生成新的追问会话标识。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]


def resolve_run_dir(run_reference: str) -> Path:
    """把 run 目录或 manifest 路径统一解析成真实 run 目录。

    功能：
        允许 UI 既传目录，也传 `run_manifest.json` 文件。

    关键参数：
        run_reference：run 目录路径或 manifest 文件路径。

    返回值：
        对应的 run 目录。

    状态影响：
        无外部副作用。

    调用时机：
        追问前加载上下文，或刷新会话状态时。
    """

    raw = (run_reference or "").strip()
    if not raw:
        raise RuntimeError("请先提供一个有效的 run 目录。")
    path = Path(raw).expanduser()
    if path.is_file():
        if path.name.lower() == "run_manifest.json":
            path = path.parent
        else:
            path = path.parent
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"Run 目录不存在：{path}")
    return path.resolve()


def followup_cache_dir(run_dir: Path) -> Path:
    """返回追问会话专用缓存目录。"""

    cache_dir = run_dir / "_followup_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def current_summary_path(run_dir: Path) -> Path:
    """返回当前“工作摘要”的文件路径。

    功能：
        优先使用追问缓存里最新的 `summary_current.md`，否则回落到第一轮生成的
        `final_summary.md`。

    关键参数：
        run_dir：当前工作流输出目录。

    返回值：
        当前应继续编辑/展示的摘要路径。

    状态影响：
        无外部副作用。

    调用时机：
        追问开始前和每轮追问结束后。
    """

    cache_summary = followup_cache_dir(run_dir) / "summary_current.md"
    if cache_summary.exists():
        return cache_summary
    final_summary = run_dir / "final_summary.md"
    if final_summary.exists():
        return final_summary
    raise RuntimeError(f"未找到可用于追问的总结文件：{final_summary}")


def transcript_path_for_run(run_dir: Path) -> Path:
    """返回 run 目录中最适合追问引用的转写文件。"""

    transcript_txt = run_dir / "transcript.txt"
    if transcript_txt.exists():
        return transcript_txt
    transcript_md = run_dir / "transcript.md"
    if transcript_md.exists():
        return transcript_md
    raise RuntimeError(f"未找到转写文件：{run_dir}")


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """读取 run_manifest.json。"""

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _history_file(run_dir: Path, session_id: str) -> Path:
    """返回当前会话的历史文件路径。"""

    return followup_cache_dir(run_dir) / f"{sanitize_filename(session_id, 'session')}.history.json"


def load_followup_history(run_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """读取当前追问会话的历史记录。"""

    history_path = _history_file(run_dir, session_id)
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def save_followup_history(run_dir: Path, session_id: str, history: list[dict[str, Any]]) -> None:
    """写回追问会话历史。"""

    history_path = _history_file(run_dir, session_id)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def history_to_pairs(history: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """把历史记录转换成 Gradio Chatbot 可直接展示的问答对。"""

    pairs: list[tuple[str, str]] = []
    for item in history:
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer_markdown") or "").strip()
        if question or answer:
            pairs.append((question, answer))
    return pairs


def load_followup_session(run_reference: str, session_id: str) -> dict[str, Any]:
    """加载追问会话上下文。

    功能：
        读取当前 summary、transcript、manifest 和历史记录，供 UI 刷新展示。

    关键参数：
        run_reference：run 目录或 manifest 路径。
        session_id：会话标识。

    返回值：
        一个字典，包含当前会话 UI 需要的所有主要字段。

    状态影响：
        无外部副作用。

    调用时机：
        用户切换 run 目录、刷新追问面板或重置会话时。
    """

    run_dir = resolve_run_dir(run_reference)
    current_summary = current_summary_path(run_dir)
    transcript_path = transcript_path_for_run(run_dir)
    history = load_followup_history(run_dir, session_id)
    manifest = load_manifest(run_dir)
    return {
        "run_dir": str(run_dir),
        "session_id": session_id,
        "current_summary_path": str(current_summary),
        "current_summary_text": current_summary.read_text(encoding="utf-8", errors="replace"),
        "transcript_path": str(transcript_path),
        "transcript_text": transcript_path.read_text(encoding="utf-8", errors="replace"),
        "chat_history_pairs": history_to_pairs(history),
        "meta": {
            "manifest": manifest,
            "history_count": len(history),
        },
    }


def _query_seed_from_context(run_dir: Path, summary_text: str, manifest: dict[str, Any]) -> str:
    """从 run 上下文中提取适合作为 Brave 检索种子的文本。"""

    heading_match = re.search(r"^#\s+(.+)$", summary_text, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    source_media = str(manifest.get("source_media_path") or "")
    if source_media:
        return Path(source_media).stem
    source_value = str(manifest.get("source_value") or "")
    if source_value:
        return source_value
    return run_dir.name


def _split_followup_response(response_text: str) -> tuple[str, str]:
    """解析模型返回的“回答区”和“更新后的摘要区”。"""

    answer_marker = "<<<ANSWER>>>"
    summary_marker = "<<<UPDATED_SUMMARY>>>"
    text = response_text.strip()
    if answer_marker not in text:
        return text, ""
    answer_part = text.split(answer_marker, 1)[1]
    if summary_marker not in answer_part:
        return answer_part.strip(), ""
    answer_text, summary_text = answer_part.split(summary_marker, 1)
    return answer_text.strip(), summary_text.strip()


def _write_summary_version(run_dir: Path, updated_summary: str) -> Path:
    """把更新后的摘要写入缓存，并刷新 `summary_current.md`。"""

    cache_dir = followup_cache_dir(run_dir)
    current_path = cache_dir / "summary_current.md"
    existing_versions = sorted(cache_dir.glob("summary_v*.md"))
    next_index = len(existing_versions) + 1
    version_path = cache_dir / f"summary_v{next_index:02d}.md"
    version_path.write_text(updated_summary.rstrip() + "\n", encoding="utf-8")
    current_path.write_text(updated_summary.rstrip() + "\n", encoding="utf-8")
    return version_path


def _build_followup_messages(
    *,
    question: str,
    summary_text: str,
    transcript_text: str,
    history: list[dict[str, Any]],
    web_context_text: str,
    update_summary: bool,
) -> list[dict[str, str]]:
    """构造追问会话的 LLM 消息。"""

    history_lines: list[str] = []
    for item in history[-8:]:
        history_lines.append(f"Q: {item.get('question', '')}")
        history_lines.append(f"A: {item.get('answer_markdown', '')}")

    instructions = [
        "你是一个严谨的中文音视频总结助手。",
        "你必须优先依据当前工作摘要和原始转写回答。",
        "如果提供了网页补充，只能把它当作外部背景，不能覆盖转写证据。",
        "请严格按如下标记返回：",
        "<<<ANSWER>>>",
        "这里写对用户追问的 Markdown 回答",
    ]
    if update_summary:
        instructions.extend(
            [
                "<<<UPDATED_SUMMARY>>>",
                "这里写更新后的完整总结 Markdown；如果你认为无需改写，请返回当前摘要的改进版。",
            ]
        )

    user_sections = [
        "# 当前工作摘要",
        summary_text.strip(),
        "",
        "# 原始转写",
        transcript_text.strip(),
        "",
    ]
    if history_lines:
        user_sections.extend(["# 历史追问", "\n".join(history_lines).strip(), ""])
    if web_context_text.strip():
        user_sections.extend(["# 外部补充背景（不可覆盖转写证据）", web_context_text.strip(), ""])
    user_sections.extend(["# 本轮用户问题", question.strip()])

    return [
        {"role": "system", "content": "\n".join(instructions)},
        {"role": "user", "content": "\n".join(user_sections).strip()},
    ]


def run_followup_turn(
    *,
    run_reference: str,
    session_id: str,
    question: str,
    llm_settings: LLMSettings,
    update_summary: bool,
    enable_web_search: bool,
    brave_settings: BraveSearchSettings,
) -> FollowupChatResult:
    """围绕既有 run 目录执行一轮追问。

    功能：
        基于已经生成好的 `final_summary.md / transcript / manifest` 继续多轮对话，
        并在需要时把新的 summary 版本写到 run 目录缓存中。

    关键参数：
        run_reference：run 目录或 manifest 路径。
        session_id：当前追问会话标识。
        question：本轮追问内容。
        llm_settings：追问阶段使用的 LLM 配置。
        update_summary：是否让模型同时生成更新后的完整摘要。
        enable_web_search：是否启用 Brave Search。
        brave_settings：Brave Search 配置。

    返回值：
        `FollowupChatResult`，供 WebUI 刷新聊天记录、当前摘要和下载入口。

    状态影响：
        会在 run 目录下的 `_followup_cache` 中写入历史、可选的网页上下文和摘要版本。

    调用时机：
        用户在追问 Tab 中点击发送时。
    """

    raw_question = (question or "").strip()
    if not raw_question:
        raise RuntimeError("请输入追问内容。")

    run_dir = resolve_run_dir(run_reference)
    manifest = load_manifest(run_dir)
    summary_path = current_summary_path(run_dir)
    transcript_path = transcript_path_for_run(run_dir)
    summary_text = summary_path.read_text(encoding="utf-8", errors="replace")
    transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
    history = load_followup_history(run_dir, session_id)

    brave_results_text = ""
    brave_meta: dict[str, Any] = {"enabled": bool(enable_web_search)}
    if enable_web_search:
        client = BraveSearchClient(BraveSearchSettings(
            enabled=True,
            api_key=brave_settings.api_key,
            api_key_env=brave_settings.api_key_env,
            base_url=brave_settings.base_url,
            count=brave_settings.count,
            timeout_seconds=brave_settings.timeout_seconds,
        ))
        if client.enabled:
            query_seed = _query_seed_from_context(run_dir, summary_text, manifest)
            query = f"\"{query_seed}\" {raw_question}"
            try:
                results = client.search(query, count=brave_settings.count)
                brave_results_text = format_web_results(results)
                brave_meta.update({"query": query, "result_count": len(results)})
            except Exception as exc:  # noqa: BLE001
                brave_meta["error"] = str(exc)
        else:
            brave_meta["error"] = "Brave Search 未启用或缺少 API key。"

    messages = _build_followup_messages(
        question=raw_question,
        summary_text=summary_text,
        transcript_text=transcript_text,
        history=history,
        web_context_text=brave_results_text,
        update_summary=update_summary,
    )

    answer_markdown = ""
    updated_summary_text = ""
    llm_meta: dict[str, Any] = {
        "model": llm_settings.model,
        "base_url": llm_settings.base_url,
        "enabled": bool(llm_settings.enabled),
    }
    try:
        client = OpenAICompatibleClient(llm_settings)
        response_text = client.chat_text(messages, temperature=llm_settings.temperature)
        answer_markdown, updated_summary_text = _split_followup_response(response_text)
        if not answer_markdown:
            answer_markdown = response_text.strip()
    except Exception as exc:  # noqa: BLE001
        answer_markdown = (
            "### 追问失败\n\n"
            f"- 错误：{exc}\n"
            "- 已保留当前摘要与历史，不会覆盖已有总结。"
        )
        llm_meta["error"] = str(exc)

    latest_summary_path = summary_path
    if update_summary and updated_summary_text.strip():
        latest_summary_path = _write_summary_version(run_dir, updated_summary_text.strip())
    current_summary_text = latest_summary_path.read_text(encoding="utf-8", errors="replace")

    web_context_path = ""
    if brave_results_text.strip():
        cache_dir = followup_cache_dir(run_dir)
        web_file = cache_dir / f"web_context_{len(history) + 1:02d}.md"
        web_file.write_text(brave_results_text.rstrip() + "\n", encoding="utf-8")
        web_context_path = str(web_file)

    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": raw_question,
            "answer_markdown": answer_markdown,
            "updated_summary_path": str(latest_summary_path) if latest_summary_path != summary_path or update_summary else "",
            "web_context_path": web_context_path,
            "update_summary": bool(update_summary),
            "web_search": brave_meta,
            "llm": llm_meta,
        }
    )
    save_followup_history(run_dir, session_id, history)

    return FollowupChatResult(
        run_dir=run_dir,
        session_id=session_id,
        chat_history_pairs=history_to_pairs(history),
        answer_markdown=answer_markdown,
        current_summary_text=current_summary_text,
        current_summary_path=latest_summary_path,
        transcript_path=transcript_path,
        web_context_text=brave_results_text,
        meta={
            "run_dir": str(run_dir),
            "session_id": session_id,
            "history_count": len(history),
            "llm": llm_meta,
            "web_search": brave_meta,
            "current_summary_path": str(latest_summary_path),
            "transcript_path": str(transcript_path),
        },
    )

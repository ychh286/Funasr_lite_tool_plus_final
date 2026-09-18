from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .brave_search import BraveSearchSettings
from .config import DEFAULT_SCENARIO, MODEL_PRESETS, PUNC_PRESETS, SCENARIO_PRESETS, VAD_PRESETS
from .followup_chat import load_followup_session, new_session_id, run_followup_turn
from .llm_client import DEFAULT_LLM_TIMEOUT_SECONDS, LLM_MODEL_CHOICES, LLMSettings, OpenAICompatibleClient, default_llm_api_key_env, default_llm_base_url, default_llm_model
from .local_dialogs import choose_directory, format_runtime_status_markdown
from .pipeline_workflow import MEDIA_EXTENSIONS, inspect_audio_workflow_environment, run_audio_workflow
from .prompting import default_prompt_labels, profile_prompt_label, prompt_labels
from .webui import (
    TECH_CSS,
    _apply_scenario_defaults,
    _as_path_list,
    _metric_card_html,
    _run_batch,
    _section_banner_html,
    _topbar_html,
)
from .workflow_profiles import get_workflow_profile, get_workflow_profiles


SINGLE_INPUT_FILE_TYPES = ["audio", "video", ".txt", ".md"]


WORKBENCH_CSS = TECH_CSS + r"""
.workflow-grid { display: grid; grid-template-columns: minmax(320px, 440px) minmax(0, 1fr); gap: 24px; align-items: start; }
.workflow-sidebar, .workflow-main { min-width: 0; }
.workflow-sidebar .glass-card, .workflow-main .glass-card { margin-bottom: 18px; }
.workflow-hero { padding: 38px 40px; }
.workflow-hero-title { margin: 12px 0; font-size: clamp(2rem, 3.5vw, 3rem); font-weight: 700; letter-spacing: -0.04em; line-height: 1.1; background: var(--grad-text); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.workflow-note { color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
.small-code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 0.82rem; }
.path-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: end; }
.status-box { padding: 14px 16px; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
.chat-panel { min-height: 280px; }
@media (max-width: 1100px) { .workflow-grid { grid-template-columns: 1fr; } }
"""


def _workflow_hero_html() -> str:
    return """
    <div class="hero-panel workflow-hero">
      <div class="eyebrow">Audio / Video Workbench V2</div>
      <div class="workflow-hero-title">完整工作流、批量转写、总结追问，分三条线做。</div>
      <div class="hero-copy">
        默认入口继续负责“链接/本地媒体 → 下载 → 预处理 → 转写 → Prompt 组合 → LLM 总结”，也支持已有转写稿直接总结。
        同时保留批量转写能力，并新增总结后的多轮追问与可选 Brave 联网补充。
      </div>
      <div class="hero-chips">
        <div class="chip"><strong>完整工作流</strong>&nbsp; 单素材端到端总结</div>
        <div class="chip"><strong>批量转写</strong>&nbsp; 多文件/目录批处理，不强绑 LLM</div>
        <div class="chip"><strong>追问会话</strong>&nbsp; 基于既有 run 目录继续改写总结</div>
      </div>
    </div>
    """


def _profiles_help_html() -> str:
    profiles = get_workflow_profiles()
    if not profiles:
        return "<div class='workflow-note'>未检测到 workflow_profiles.local.json；将使用通用输出目录。</div>"
    lines = ["<div class='workflow-note'><b>可用本地路由配置</b><br/>"]
    for key, profile in profiles.items():
        lines.append(
            f"<span class='small-code'>{key}</span> — {profile.display_name}"
            f" / prompt: <span class='small-code'>{profile.prompt_name or '未指定'}</span><br/>"
        )
    lines.append("</div>")
    return "".join(lines)


def _scenario_outputs(name: str):
    bs, ms, ml, lang, itn, mv = _apply_scenario_defaults(name)
    return (
        gr.update(value=bs),
        gr.update(value=ms),
        gr.update(value=ml),
        gr.update(value=lang),
        gr.update(value=itn),
        gr.update(value=mv),
    )


def _speaker_toggle(enabled: bool, current_model: str, current_punc: str):
    if enabled:
        target_model = (
            current_model
            if current_model == "paraformer-zh (speaker diarization)"
            else "paraformer-zh (speaker diarization)"
        )
        target_punc = current_punc if current_punc != "none" else "ct-punc-c"
        return gr.update(value=target_model), gr.update(value=target_punc)
    if current_model == "paraformer-zh (speaker diarization)":
        return gr.update(value="paraformer-zh (offline)"), gr.update()
    return gr.update(), gr.update()


def _on_profile_change(profile_key: str):
    profile = get_workflow_profile(profile_key) if profile_key and profile_key != "auto" else None
    defaults = default_prompt_labels()
    if profile:
        label = profile_prompt_label(profile.prompt_name)
        selected = defaults + ([label] if label and label not in defaults else [])
        return (
            gr.update(value=profile.tag or ""),
            gr.update(value=selected),
            gr.update(value=profile.download_root or ""),
            gr.update(value=profile.note_root or ""),
            gr.update(value=profile.output_root or ""),
        )
    return (
        gr.update(value=""),
        gr.update(value=defaults),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )


def _check_llm(base_url: str, model: str, api_key: str, api_key_env: str, timeout_seconds: int, temperature: float):
    try:
        client = OpenAICompatibleClient(
            LLMSettings(
                enabled=True,
                base_url=base_url,
                model=model,
                api_key=api_key,
                api_key_env=api_key_env,
                timeout_seconds=int(timeout_seconds or 60),
                temperature=float(temperature or 0),
            )
        )
        health = client.health_check()
        if health.ok:
            return f"✅ LLM 可用：{health.message}"
        return f"❌ LLM 检查失败：{health.message}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ LLM 检查异常：{exc}"


def _browse_directory(current_value: str, title: str):
    chosen = choose_directory(current_value, title)
    return chosen or current_value or ""


def _inspect_runtime(
    source_url: str,
    uploaded_media: Any,
    profile_key: str,
    selected_prompts: list[str],
    output_root: str,
    download_root: str,
    note_root: str,
    cookie_mode: str,
    cookie_path: str,
    uploaded_cookie: Any,
    ytdlp_path: str,
    ffmpeg_path: str,
):
    info = inspect_audio_workflow_environment(
        source_url=source_url,
        uploaded_media=uploaded_media,
        profile_key=profile_key,
        selected_prompt_labels=selected_prompts or [],
        output_root=output_root,
        download_root=download_root,
        note_root=note_root,
        cookie_mode=cookie_mode,
        cookie_path=cookie_path,
        uploaded_cookie=uploaded_cookie,
        ytdlp_path=ytdlp_path,
        ffmpeg_path=ffmpeg_path,
    )
    markdown = format_runtime_status_markdown(
        profile_label=info.get("profile_label", ""),
        profile_prompt=info.get("profile_prompt", ""),
        source_value=info.get("source_value", ""),
        source_kind=info.get("source_kind", "empty"),
        resolved_output_root=info.get("resolved_output_root", ""),
        resolved_download_root=info.get("resolved_download_root", ""),
        resolved_note_root=info.get("resolved_note_root", ""),
        ytdlp_status=info.get("ytdlp_status", ""),
        ffmpeg_status=info.get("ffmpeg_status", ""),
        cookie_status=info.get("cookie_status", ""),
        prompt_defaults=info.get("default_prompts", []),
    )
    return markdown, info


def _build_run_status_markdown(meta: dict[str, Any]) -> str:
    files = meta.get("files") or {}
    llm_meta = meta.get("llm") or {}
    return "\n".join(
        [
            "### 本次完整工作流已完成",
            "",
            f"- 输出目录：`{meta.get('output_dir', '')}`",
            f"- 来源：`{meta.get('source_value', '')}`",
            f"- 当前摘要：`{files.get('final_markdown', '')}`",
            f"- 转写文件：`{files.get('transcript_md', '') or files.get('transcript_txt', '')}`",
            f"- LLM：`{'启用' if llm_meta.get('enabled') else '未启用'}`",
            f"- 追问入口：可直接切到“总结追问与改写”Tab，run 目录会自动带过去。",
        ]
    )


def _run_end_to_end(
    source_url: str,
    uploaded_media: Any,
    profile_key: str,
    tag: str,
    selected_prompts: list[str],
    uploaded_prompt_files: Any,
    inline_prompt: str,
    output_root: str,
    download_root: str,
    note_root: str,
    download_mode: str,
    keep_download: bool,
    cookie_mode: str,
    uploaded_cookie: Any,
    cookie_path: str,
    ytdlp_path: str,
    ffmpeg_path: str,
    scenario_name: str,
    model_preset: str,
    model_custom: str,
    vad_preset: str,
    vad_custom: str,
    punc_preset: str,
    punc_custom: str,
    language: str,
    device: str,
    use_itn: bool,
    merge_vad: bool,
    speaker_diarization: bool,
    speaker_count: int,
    verbose: bool,
    batch_size_s: int,
    max_single_segment_ms: int,
    merge_length_s: int,
    split_lines: bool,
    run_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    llm_api_key_env: str,
    llm_temperature: float,
    llm_timeout_seconds: int,
    include_prompt_in_final: bool,
    include_transcript_in_final: bool,
    write_note: bool,
    overwrite_note: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    def _progress(pct: float, message: str) -> None:
        progress(pct, desc=message)

    try:
        result = run_audio_workflow(
            source_url=source_url,
            uploaded_media=uploaded_media,
            profile_key=profile_key,
            tag=tag,
            selected_prompt_labels=selected_prompts or [],
            uploaded_prompt_files=uploaded_prompt_files,
            inline_prompt=inline_prompt,
            output_root=output_root,
            download_root=download_root,
            note_root=note_root,
            download_mode=download_mode,
            keep_download=bool(keep_download),
            cookie_mode=cookie_mode,
            cookie_path=cookie_path,
            uploaded_cookie=uploaded_cookie,
            ytdlp_path=ytdlp_path,
            ffmpeg_path=ffmpeg_path,
            scenario_name=scenario_name,
            model_preset=model_preset,
            model_custom=model_custom,
            vad_preset=vad_preset,
            vad_custom=vad_custom,
            punc_preset=punc_preset,
            punc_custom=punc_custom,
            language=language,
            device=device,
            use_itn=bool(use_itn),
            merge_vad=bool(merge_vad),
            speaker_diarization=bool(speaker_diarization),
            speaker_count=int(speaker_count or 0),
            verbose=bool(verbose),
            batch_size_s=int(batch_size_s),
            max_single_segment_ms=int(max_single_segment_ms),
            merge_length_s=int(merge_length_s),
            split_lines=bool(split_lines),
            run_llm=bool(run_llm),
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_api_key_env=llm_api_key_env,
            llm_temperature=float(llm_temperature or 0),
            llm_timeout_seconds=int(llm_timeout_seconds or 180),
            include_prompt_in_final=bool(include_prompt_in_final),
            include_transcript_in_final=bool(include_transcript_in_final),
            write_note=bool(write_note),
            overwrite_note=bool(overwrite_note),
            progress=_progress,
        )
        files = result.meta.get("files") or {}
        transcript_download = files.get("transcript_md") or files.get("transcript_txt")
        run_dir = result.meta.get("output_dir", "")
        return (
            result.preview,
            result.summary_preview,
            result.summary_preview,
            result.final_markdown_path,
            transcript_download,
            result.bundle_path,
            result.meta,
            _build_run_status_markdown(result.meta),
            run_dir,
            run_dir,
            f"### 追问说明\n\n已自动带入最近一次 run：`{run_dir}`\n\n现在可以切到“总结追问与改写”Tab 继续对话。",
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        error_markdown = f"❌ 工作流失败：{exc}\n\n```text\n{tb}\n```"
        return (
            "",
            error_markdown,
            error_markdown,
            None,
            None,
            None,
            {"error": str(exc), "traceback": tb},
            "### 工作流失败\n\n请查看下方元数据里的完整异常。",
            "",
            "",
            "### 追问说明\n\n当前没有可用 run，请先完成一次完整工作流。",
        )


def _merge_batch_media(uploaded_files: Any, uploaded_directory: Any) -> list[Path]:
    merged: list[Path] = []
    seen: set[str] = set()
    for path in _as_path_list(uploaded_files) + _as_path_list(uploaded_directory):
        resolved = Path(path).expanduser()
        if resolved.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        merged.append(resolved)
    return merged


def _run_batch_tab(
    uploaded_files: Any,
    uploaded_directory: Any,
    scenario_name: str,
    model_preset: str,
    model_custom: str,
    vad_preset: str,
    vad_custom: str,
    punc_preset: str,
    punc_custom: str,
    language: str,
    device: str,
    use_itn: bool,
    merge_vad: bool,
    speaker_diarization: bool,
    speaker_count: int,
    verbose: bool,
    batch_size_s: int,
    max_single_segment_ms: int,
    merge_length_s: int,
    split_lines: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    merged = _merge_batch_media(uploaded_files, uploaded_directory)
    if not merged:
        return "", None, {"error": "请先选择多个媒体文件，或选择一个媒体目录。"}
    return _run_batch(
        merged,
        scenario_name,
        model_preset,
        model_custom,
        vad_preset,
        vad_custom,
        punc_preset,
        punc_custom,
        language,
        device,
        use_itn,
        merge_vad,
        speaker_diarization,
        speaker_count,
        verbose,
        batch_size_s,
        max_single_segment_ms,
        merge_length_s,
        "md",
        split_lines,
        progress,
    )


def _followup_status_markdown(run_dir: str, session_id: str, history_count: int) -> str:
    return "\n".join(
        [
            "### 当前追问会话",
            "",
            f"- Run 目录：`{run_dir or '未指定'}`",
            f"- Session：`{session_id or '未创建'}`",
            f"- 已有轮次：`{history_count}`",
            "- 追问产生的新摘要版本只会写入 run 目录下的 `_followup_cache`，不会污染目标 note 目录。",
        ]
    )


def _load_followup_run(run_dir_value: str, latest_run_dir: str, session_id: str):
    target = (run_dir_value or latest_run_dir or "").strip()
    if not target:
        return (
            run_dir_value,
            session_id,
            "### 尚未选择 run 目录",
            [],
            "",
            "",
            None,
            None,
            "",
            {},
        )
    try:
        active_session = (session_id or "").strip() or new_session_id()
        session = load_followup_session(target, active_session)
        status = _followup_status_markdown(
            session["run_dir"],
            active_session,
            int(session["meta"].get("history_count", 0)),
        )
        return (
            session["run_dir"],
            active_session,
            status,
            session["chat_history_pairs"],
            session["current_summary_text"],
            session["current_summary_text"],
            session["current_summary_path"],
            session["transcript_path"],
            "",
            session["meta"],
        )
    except Exception as exc:  # noqa: BLE001
        return (
            target,
            session_id or "",
            f"### 追问面板加载失败\n\n- 错误：{exc}",
            [],
            "",
            "",
            None,
            None,
            "",
            {"error": str(exc)},
        )


def _reset_followup_session(run_dir_value: str, latest_run_dir: str):
    new_id = new_session_id()
    return _load_followup_run(run_dir_value, latest_run_dir, new_id)


def _run_followup_round(
    run_dir_value: str,
    latest_run_dir: str,
    session_id: str,
    question: str,
    update_summary: bool,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    llm_api_key_env: str,
    llm_temperature: float,
    llm_timeout_seconds: int,
    enable_web_search: bool,
    brave_api_key: str,
    brave_api_key_env: str,
    brave_count: int,
    brave_timeout_seconds: int,
):
    target = (run_dir_value or latest_run_dir or "").strip()
    active_session = (session_id or "").strip() or new_session_id()
    try:
        result = run_followup_turn(
            run_reference=target,
            session_id=active_session,
            question=question,
            llm_settings=LLMSettings(
                enabled=True,
                base_url=llm_base_url,
                model=llm_model,
                api_key=llm_api_key,
                api_key_env=llm_api_key_env,
                timeout_seconds=int(llm_timeout_seconds or 180),
                temperature=float(llm_temperature or 0.1),
            ),
            update_summary=bool(update_summary),
            enable_web_search=bool(enable_web_search),
            brave_settings=BraveSearchSettings(
                enabled=bool(enable_web_search),
                api_key=brave_api_key,
                api_key_env=brave_api_key_env,
                count=int(brave_count or 5),
                timeout_seconds=int(brave_timeout_seconds or 25),
            ),
        )
        status = _followup_status_markdown(
            str(result.run_dir),
            result.session_id,
            len(result.chat_history_pairs),
        )
        return (
            str(result.run_dir),
            result.session_id,
            result.chat_history_pairs,
            result.current_summary_text,
            result.current_summary_text,
            str(result.current_summary_path),
            str(result.transcript_path),
            result.web_context_text or "",
            result.meta,
            status,
            "",
        )
    except Exception as exc:  # noqa: BLE001
        return (
            target,
            active_session,
            [],
            "",
            "",
            None,
            None,
            "",
            {"error": str(exc)},
            f"### 追问失败\n\n- 错误：{exc}",
            question,
        )


def main() -> None:
    scenario_names = list(SCENARIO_PRESETS.keys())
    model_preset_keys = list(MODEL_PRESETS.keys())
    vad_preset_keys = list(VAD_PRESETS.keys())
    punc_preset_keys = list(PUNC_PRESETS.keys())
    profile_keys = ["auto"] + sorted(get_workflow_profiles().keys())
    prompt_choices = prompt_labels()
    prompt_defaults = default_prompt_labels()

    theme = gr.themes.Base(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate", radius_size="lg", spacing_size="md")
    with gr.Blocks(title="FunASR Audio Workbench V2", css=WORKBENCH_CSS, theme=theme) as demo:
        latest_run_dir = gr.State("")
        followup_session_id = gr.State("")

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(_topbar_html())
            with gr.Row(elem_classes=["hero-grid"]):
                gr.HTML(_workflow_hero_html())
                with gr.Column(elem_classes=["hero-aside"]):
                    gr.HTML(_metric_card_html("完整工作流", "Single input", "单素材端到端处理，也支持已有转写稿直接总结。"))
                    gr.HTML(_metric_card_html("批量转写", "Legacy batch", "保留多文件转写入口，但不强绑单一 final_summary。"))
                    gr.HTML(_metric_card_html("追问改写", "Cached follow-up", "围绕既有 run 目录继续追问和改写，不污染目标导出路径。"))

            with gr.Tabs():
                with gr.TabItem("完整工作流 V2"):
                    with gr.Row(equal_height=False, elem_classes=["workflow-grid"]):
                        with gr.Column(elem_classes=["workflow-sidebar"]):
                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 1. 输入源与路由配置\n`Profile / 本地路由配置` 负责默认 tag、prompt 与目录；本地 `.txt/.md` 会跳过 ffmpeg/FunASR，直接进入总结。")
                                source_url = gr.Textbox(label="视频/音频链接", placeholder="Bilibili / YouTube / 其他 yt-dlp 支持链接")
                                uploaded_media = gr.File(label="或上传本地媒体/转写稿", file_types=SINGLE_INPUT_FILE_TYPES, type="filepath")
                                profile_key = gr.Dropdown(choices=profile_keys, value="auto", label="Profile / 本地路由配置")
                                gr.HTML(_profiles_help_html())
                                tag = gr.Textbox(label="标签", placeholder="#组会 / #项目会 / #人文 / #技术 ...")

                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 2. Prompt 规则\n建议保留 `.codex/agents.md`，再额外选择一个场景 Prompt；`inline prompt` 只放本轮附加要求。")
                                selected_prompts = gr.CheckboxGroup(choices=prompt_choices, value=prompt_defaults, label="当前要组合的 prompts")
                                uploaded_prompt_files = gr.Files(label="本次追加 prompt 文件", file_types=[".md", ".txt"], type="filepath")
                                inline_prompt = gr.Textbox(label="本次追加对话指令", lines=5, placeholder="例如：请输出适合 Obsidian 的层级标题；保留关键原话；列出待核对问题。")

                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 3. 下载、目录与工具\n路径一般可留空走自动探测；只有工具没装好或 cookies 资源受限时才需要覆盖。")
                                download_mode = gr.Radio(["audio", "video"], value="audio", label="URL 下载模式")
                                keep_download = gr.Checkbox(label="保留下载的源媒体", value=True)
                                with gr.Row():
                                    output_root = gr.Textbox(label="运行输出根目录", placeholder="默认 outputs/workbench_runs")
                                    output_root_pick = gr.Button("选择", variant="secondary")
                                with gr.Row():
                                    download_root = gr.Textbox(label="媒体下载/归档目录", placeholder="留空则使用 profile 或当前 run 缓存")
                                    download_root_pick = gr.Button("选择", variant="secondary")
                                with gr.Row():
                                    note_root = gr.Textbox(label="最终总结 md 落盘目录", placeholder="留空则只生成下载文件")
                                    note_root_pick = gr.Button("选择", variant="secondary")
                                with gr.Accordion("Cookies 与工具覆盖项", open=False):
                                    cookie_mode = gr.Radio(["auto", "upload/path", "none"], value="auto", label="Cookies 模式")
                                    uploaded_cookie = gr.File(label="上传 cookies.txt", file_types=[".txt"], type="filepath")
                                    cookie_path = gr.Textbox(label="Cookies 文件路径", placeholder="例如 E:\\Software_tools\\yt-dlp\\cookies\\youtube_cookies.txt")
                                    ytdlp_path = gr.Textbox(label="yt-dlp 路径", placeholder="留空则自动探测")
                                    ffmpeg_path = gr.Textbox(label="ffmpeg 路径", placeholder="留空则自动探测")
                                inspect_btn = gr.Button("检查环境与路径", variant="secondary")
                                runtime_status_md = gr.Markdown("### 当前环境探测\n\n点击上面的按钮查看本次实际会使用的路径与工具。")
                                runtime_status_meta = gr.JSON(label="environment details")

                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 4. FunASR 参数\n这些参数只影响当前 run，不会改变你的本地 profile 配置。")
                                scenario_name = gr.Dropdown(choices=scenario_names, value=DEFAULT_SCENARIO, label="Scenario preset")
                                model_preset = gr.Dropdown(choices=model_preset_keys, value="paraformer-zh (offline)", label="Model preset")
                                model_custom = gr.Textbox(label="Custom model id", placeholder="可选；填写后覆盖 preset")
                                vad_preset = gr.Dropdown(choices=vad_preset_keys, value="fsmn-vad", label="VAD preset")
                                vad_custom = gr.Textbox(label="Custom VAD model id", placeholder="可选")
                                punc_preset = gr.Dropdown(choices=punc_preset_keys, value="none", label="Punctuation preset")
                                punc_custom = gr.Textbox(label="Custom punctuation model id", placeholder="可选")
                                language = gr.Dropdown(["auto", "zh", "en", "ja", "ko"], value="auto", label="Language")
                                device = gr.Dropdown(["auto", "cuda:0", "cuda", "cpu"], value="auto", label="Device")
                                with gr.Row():
                                    use_itn = gr.Checkbox(label="Use ITN", value=True)
                                    merge_vad = gr.Checkbox(label="Merge VAD segments", value=True)
                                with gr.Row():
                                    speaker_diarization = gr.Checkbox(label="Speaker diarization V1", value=False)
                                    split_lines = gr.Checkbox(label="按标点拆行", value=True)
                                speaker_count = gr.Slider(0, 10, value=0, step=1, label="Known speaker count (0 = auto)")
                                batch_size_s = gr.Slider(1, 60, value=10, step=1, label="batch_size_s")
                                max_single_segment_ms = gr.Slider(5000, 120000, value=30000, step=1000, label="max_single_segment_time (ms)")
                                merge_length_s = gr.Slider(1, 60, value=15, step=1, label="merge_length_s")
                                verbose = gr.Checkbox(label="Verbose meta", value=False)

                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 5. LLM 总结与导出\n第一轮总结默认严格基于 transcript；联网补充会放到追问阶段。")
                                run_llm = gr.Checkbox(label="调用大模型生成 final_summary.md", value=False)
                                llm_base_url = gr.Textbox(label="Base URL", value=default_llm_base_url(), info="LiteLLM 默认 http://127.0.0.1:4000/v1")
                                llm_model = gr.Dropdown(choices=list(LLM_MODEL_CHOICES), value=default_llm_model(), label="Model", allow_custom_value=True)
                                llm_api_key = gr.Textbox(label="API Key（可留空走环境变量）", type="password")
                                llm_api_key_env = gr.Textbox(label="API Key 环境变量名", value=default_llm_api_key_env())
                                llm_temperature = gr.Slider(0, 2, value=0.1, step=0.05, label="Temperature")
                                llm_timeout_seconds = gr.Slider(30, 1800, value=DEFAULT_LLM_TIMEOUT_SECONDS, step=30, label="Timeout seconds")
                                include_prompt_in_final = gr.Checkbox(label="在 final_summary.md 末尾附完整 prompt", value=False)
                                include_transcript_in_final = gr.Checkbox(label="在 final_summary.md 末尾附转写全文", value=False)
                                write_note = gr.Checkbox(label="设置 note_root / profile 建议路径时同步写入 Markdown", value=True)
                                overwrite_note = gr.Checkbox(label="允许覆盖已有同名笔记", value=False)
                                with gr.Row():
                                    check_llm_btn = gr.Button("检查 LLM 连接", variant="secondary")
                                    start_btn = gr.Button("开始完整工作流", variant="primary", size="lg")
                                llm_status = gr.Markdown("")

                        with gr.Column(elem_classes=["workflow-main"]):
                            gr.HTML(_section_banner_html("Results", "预览、下载与运行证据", "完整工作流会生成 final_summary.md、转写文件和 workflow_outputs.zip。"))
                            with gr.Column(elem_classes=["glass-card", "tech-panel", "results-panel"]):
                                run_status_md = gr.Markdown("### 等待执行\n\n完成一次 run 后，这里会显示产物路径和追问入口提示。")
                                with gr.Tabs():
                                    with gr.TabItem("转写预览"):
                                        transcript_preview = gr.Textbox(lines=24, interactive=False, label="Transcript / Speaker Preview")
                                    with gr.TabItem("总结渲染预览"):
                                        summary_rendered = gr.Markdown()
                                    with gr.TabItem("总结原文"):
                                        summary_raw = gr.Textbox(lines=24, interactive=False, label="Final Markdown Preview")
                                    with gr.TabItem("下载"):
                                        final_download = gr.File(label="final_summary.md")
                                        transcript_download = gr.File(label="transcript.md / transcript.txt")
                                        bundle_download = gr.File(label="workflow_outputs.zip")
                                    with gr.TabItem("运行元数据"):
                                        meta = gr.JSON(label="run_manifest")

                with gr.TabItem("批量转写"):
                    with gr.Row(equal_height=False, elem_classes=["workflow-grid"]):
                        with gr.Column(elem_classes=["workflow-sidebar"]):
                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 批量媒体输入\n这里专门保留旧版多文件批量转写能力，不会尝试为所有文件合成一个统一总结。")
                                batch_files = gr.Files(label="批量选择媒体文件", file_types=["audio", "video"], type="filepath")
                                batch_directory = gr.File(label="或选择一个媒体目录", file_count="directory", type="filepath")
                                gr.Markdown("<div class='workflow-note'>如果目录里包含非媒体文件，会自动跳过。</div>")
                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 批量转写复用的 FunASR 参数\n这里复用完整工作流的同一套模型参数，便于切换场景时保持一致。")
                                batch_status = gr.Markdown("### 批量入口说明\n\n输出为转写预览 + 单文件或 zip 下载，不接 LLM。")
                                batch_start_btn = gr.Button("开始批量转写", variant="primary", size="lg")
                        with gr.Column(elem_classes=["workflow-main"]):
                            with gr.Column(elem_classes=["glass-card", "tech-panel", "results-panel"]):
                                with gr.Tabs():
                                    with gr.TabItem("批量预览"):
                                        batch_preview = gr.Textbox(lines=28, interactive=False, label="Batch Transcript Preview")
                                    with gr.TabItem("批量下载"):
                                        batch_download = gr.File(label="Generated file or zip bundle")
                                    with gr.TabItem("批量元数据"):
                                        batch_meta = gr.JSON(label="batch meta")

                with gr.TabItem("总结追问与改写"):
                    with gr.Row(equal_height=False, elem_classes=["workflow-grid"]):
                        with gr.Column(elem_classes=["workflow-sidebar"]):
                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 1. 选择要继续追问的 run\n可以使用刚跑完的 run，也可以手动选择历史 run 目录。")
                                followup_run_dir = gr.Textbox(label="Run 目录", placeholder="可直接粘贴 run 目录，或点击选择。")
                                with gr.Row():
                                    followup_run_pick = gr.Button("选择 run 目录", variant="secondary")
                                    use_latest_run_btn = gr.Button("使用最近一次 run", variant="secondary")
                                with gr.Row():
                                    followup_refresh_btn = gr.Button("刷新会话", variant="secondary")
                                    followup_reset_btn = gr.Button("新建会话", variant="secondary")
                                followup_status = gr.Markdown("### 追问说明\n\n先选择一个 run 目录，再刷新会话。")

                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 2. 追问参数\n这里的追问只会修改 run 目录缓存中的“当前工作摘要”，不会改动目标导出目录。")
                                followup_question = gr.Textbox(label="追问内容", lines=6, placeholder="例如：请把这个总结改成更适合发给导师的版本；补充行动项；列出不确定点。")
                                followup_update_summary = gr.Checkbox(label="允许根据追问更新当前工作摘要", value=True)
                                followup_llm_base_url = gr.Textbox(label="追问 LLM Base URL", value=default_llm_base_url(), info="LiteLLM 默认 http://127.0.0.1:4000/v1")
                                followup_llm_model = gr.Dropdown(choices=list(LLM_MODEL_CHOICES), value=default_llm_model(), label="追问 LLM Model", allow_custom_value=True)
                                followup_llm_api_key = gr.Textbox(label="追问 LLM API Key（可留空走环境变量）", type="password")
                                followup_llm_api_key_env = gr.Textbox(label="追问 LLM API Key 环境变量名", value=default_llm_api_key_env())
                                followup_llm_temperature = gr.Slider(0, 2, value=0.1, step=0.05, label="追问 Temperature")
                                followup_llm_timeout = gr.Slider(30, 1800, value=DEFAULT_LLM_TIMEOUT_SECONDS, step=30, label="追问 Timeout seconds")

                            with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                                gr.Markdown("### 3. Brave 联网补充（可选）\n参考 paper-workbench-v2 的做法，只在追问阶段作为外部背景使用，不覆盖 transcript 证据。")
                                followup_enable_web = gr.Checkbox(label="本轮追问启用 Brave Search", value=False)
                                followup_brave_api_key = gr.Textbox(label="Brave API Key（可留空走环境变量）", type="password")
                                followup_brave_api_key_env = gr.Textbox(label="Brave API Key 环境变量名", value="BRAVE_SEARCH_API_KEY")
                                followup_brave_count = gr.Slider(1, 10, value=5, step=1, label="Brave 返回条数")
                                followup_brave_timeout = gr.Slider(5, 60, value=25, step=5, label="Brave Timeout seconds")
                                followup_start_btn = gr.Button("发送追问", variant="primary", size="lg")

                        with gr.Column(elem_classes=["workflow-main"]):
                            with gr.Column(elem_classes=["glass-card", "tech-panel", "results-panel"]):
                                followup_chatbot = gr.Chatbot(label="追问会话", height=360)
                                with gr.Tabs():
                                    with gr.TabItem("当前摘要渲染"):
                                        followup_summary_rendered = gr.Markdown()
                                    with gr.TabItem("当前摘要原文"):
                                        followup_summary_raw = gr.Textbox(lines=24, interactive=False, label="Current Working Summary")
                                    with gr.TabItem("原始 transcript"):
                                        followup_transcript_file = gr.File(label="transcript.md / transcript.txt")
                                    with gr.TabItem("外部补充背景"):
                                        followup_web_context = gr.Markdown()
                                    with gr.TabItem("追问元数据"):
                                        followup_meta = gr.JSON(label="followup meta")
                                with gr.Row():
                                    followup_summary_file = gr.File(label="当前工作摘要下载")

        output_root_pick.click(lambda current: _browse_directory(current, "选择运行输出根目录"), inputs=[output_root], outputs=[output_root])
        download_root_pick.click(lambda current: _browse_directory(current, "选择媒体下载/归档目录"), inputs=[download_root], outputs=[download_root])
        note_root_pick.click(lambda current: _browse_directory(current, "选择总结落盘目录"), inputs=[note_root], outputs=[note_root])
        followup_run_pick.click(lambda current: _browse_directory(current, "选择一个 run 目录"), inputs=[followup_run_dir], outputs=[followup_run_dir])
        use_latest_run_btn.click(lambda latest: latest or "", inputs=[latest_run_dir], outputs=[followup_run_dir])

        scenario_name.change(_scenario_outputs, inputs=[scenario_name], outputs=[batch_size_s, max_single_segment_ms, merge_length_s, language, use_itn, merge_vad])
        speaker_diarization.change(_speaker_toggle, inputs=[speaker_diarization, model_preset, punc_preset], outputs=[model_preset, punc_preset])
        profile_key.change(_on_profile_change, inputs=[profile_key], outputs=[tag, selected_prompts, download_root, note_root, output_root])
        inspect_btn.click(
            _inspect_runtime,
            inputs=[source_url, uploaded_media, profile_key, selected_prompts, output_root, download_root, note_root, cookie_mode, cookie_path, uploaded_cookie, ytdlp_path, ffmpeg_path],
            outputs=[runtime_status_md, runtime_status_meta],
        )
        check_llm_btn.click(
            _check_llm,
            inputs=[llm_base_url, llm_model, llm_api_key, llm_api_key_env, llm_timeout_seconds, llm_temperature],
            outputs=[llm_status],
        )
        start_btn.click(
            _run_end_to_end,
            inputs=[
                source_url,
                uploaded_media,
                profile_key,
                tag,
                selected_prompts,
                uploaded_prompt_files,
                inline_prompt,
                output_root,
                download_root,
                note_root,
                download_mode,
                keep_download,
                cookie_mode,
                uploaded_cookie,
                cookie_path,
                ytdlp_path,
                ffmpeg_path,
                scenario_name,
                model_preset,
                model_custom,
                vad_preset,
                vad_custom,
                punc_preset,
                punc_custom,
                language,
                device,
                use_itn,
                merge_vad,
                speaker_diarization,
                speaker_count,
                verbose,
                batch_size_s,
                max_single_segment_ms,
                merge_length_s,
                split_lines,
                run_llm,
                llm_base_url,
                llm_model,
                llm_api_key,
                llm_api_key_env,
                llm_temperature,
                llm_timeout_seconds,
                include_prompt_in_final,
                include_transcript_in_final,
                write_note,
                overwrite_note,
            ],
            outputs=[
                transcript_preview,
                summary_rendered,
                summary_raw,
                final_download,
                transcript_download,
                bundle_download,
                meta,
                run_status_md,
                latest_run_dir,
                followup_run_dir,
                followup_status,
            ],
        )
        batch_start_btn.click(
            _run_batch_tab,
            inputs=[
                batch_files,
                batch_directory,
                scenario_name,
                model_preset,
                model_custom,
                vad_preset,
                vad_custom,
                punc_preset,
                punc_custom,
                language,
                device,
                use_itn,
                merge_vad,
                speaker_diarization,
                speaker_count,
                verbose,
                batch_size_s,
                max_single_segment_ms,
                merge_length_s,
                split_lines,
            ],
            outputs=[batch_preview, batch_download, batch_meta],
        )
        followup_refresh_btn.click(
            _load_followup_run,
            inputs=[followup_run_dir, latest_run_dir, followup_session_id],
            outputs=[
                followup_run_dir,
                followup_session_id,
                followup_status,
                followup_chatbot,
                followup_summary_rendered,
                followup_summary_raw,
                followup_summary_file,
                followup_transcript_file,
                followup_web_context,
                followup_meta,
            ],
        )
        followup_reset_btn.click(
            _reset_followup_session,
            inputs=[followup_run_dir, latest_run_dir],
            outputs=[
                followup_run_dir,
                followup_session_id,
                followup_status,
                followup_chatbot,
                followup_summary_rendered,
                followup_summary_raw,
                followup_summary_file,
                followup_transcript_file,
                followup_web_context,
                followup_meta,
            ],
        )
        followup_start_btn.click(
            _run_followup_round,
            inputs=[
                followup_run_dir,
                latest_run_dir,
                followup_session_id,
                followup_question,
                followup_update_summary,
                followup_llm_base_url,
                followup_llm_model,
                followup_llm_api_key,
                followup_llm_api_key_env,
                followup_llm_temperature,
                followup_llm_timeout,
                followup_enable_web,
                followup_brave_api_key,
                followup_brave_api_key_env,
                followup_brave_count,
                followup_brave_timeout,
            ],
            outputs=[
                followup_run_dir,
                followup_session_id,
                followup_chatbot,
                followup_summary_rendered,
                followup_summary_raw,
                followup_summary_file,
                followup_transcript_file,
                followup_web_context,
                followup_meta,
                followup_status,
                followup_question,
            ],
        )

    import argparse

    ap = argparse.ArgumentParser(description="FunASR Audio Workbench V2")
    ap.add_argument("--host", default=os.getenv("FUNASR_LITE_HOST", "127.0.0.1"), help="Bind host")
    ap.add_argument("--port", type=int, default=int(os.getenv("FUNASR_LITE_PORT", "7860")), help="Bind port")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = ap.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

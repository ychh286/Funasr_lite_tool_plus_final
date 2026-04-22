from __future__ import annotations

import os
import sys
import traceback
from typing import Any

import gradio as gr

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .config import DEFAULT_SCENARIO, MODEL_PRESETS, PUNC_PRESETS, SCENARIO_PRESETS, VAD_PRESETS
from .llm_client import LLMSettings, OpenAICompatibleClient
from .pipeline_workflow import run_audio_workflow
from .prompting import default_prompt_labels, profile_prompt_label, prompt_labels
from .webui import TECH_CSS, _apply_scenario_defaults, _metric_card_html, _section_banner_html, _topbar_html
from .workflow_profiles import get_workflow_profile, get_workflow_profiles


WORKBENCH_CSS = TECH_CSS + r"""
.workflow-grid { display: grid; grid-template-columns: minmax(310px, 430px) minmax(0, 1fr); gap: 24px; align-items: start; }
.workflow-sidebar, .workflow-main { min-width: 0; }
.workflow-sidebar .glass-card { margin-bottom: 18px; }
.workflow-hero { padding: 38px 40px; }
.workflow-hero-title { margin: 12px 0; font-size: clamp(2rem, 3.5vw, 3rem); font-weight: 700; letter-spacing: -0.04em; line-height: 1.1; background: var(--grad-text); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.workflow-note { color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
.small-code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 0.82rem; }
@media (max-width: 1100px) { .workflow-grid { grid-template-columns: 1fr; } }
"""


def _workflow_hero_html() -> str:
    return """
    <div class="hero-panel workflow-hero">
      <div class="eyebrow">Audio / Video Workbench V2</div>
      <div class="workflow-hero-title">下载 → 预处理 → 转写 → Agent 总结 → 落盘。</div>
      <div class="hero-copy">
        这个入口把 B 站、YouTube 等链接、本地媒体、cookies、yt-dlp、ffmpeg、FunASR、
        Prompt 选择和 OpenAI-compatible 大模型调用串成一个可复核的本地工作流。
      </div>
      <div class="hero-chips">
        <div class="chip"><strong>输入</strong>&nbsp; URL / 本地上传 / Cookies</div>
        <div class="chip"><strong>中间层</strong>&nbsp; yt-dlp + ffmpeg + FunASR</div>
        <div class="chip"><strong>输出</strong>&nbsp; final_summary.md + prompt + transcript + manifest</div>
      </div>
    </div>
    """


def _profiles_help_html() -> str:
    profiles = get_workflow_profiles()
    if not profiles:
        return "<div class='workflow-note'>未检测到 workflow_profiles.local.json；将使用通用输出目录。</div>"
    lines = ["<div class='workflow-note'><b>可用本地 Profile</b><br/>"]
    for key, profile in profiles.items():
        lines.append(
            f"<span class='small-code'>{key}</span> — {profile.display_name}"
            f" / prompt: <span class='small-code'>{profile.prompt_name or '未指定'}</span><br/>"
        )
    lines.append("</div>")
    return "".join(lines)


def _scenario_outputs(name: str):
    bs, ms, ml, lang, itn, mv = _apply_scenario_defaults(name)
    return (gr.update(value=bs), gr.update(value=ms), gr.update(value=ml), gr.update(value=lang), gr.update(value=itn), gr.update(value=mv))


def _speaker_toggle(enabled: bool, current_model: str, current_punc: str):
    if enabled:
        target_model = current_model if current_model == "paraformer-zh (speaker diarization)" else "paraformer-zh (speaker diarization)"
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
        return (gr.update(value=profile.tag or ""), gr.update(value=selected), gr.update(value=profile.download_root or ""), gr.update(value=profile.note_root or ""), gr.update(value=profile.output_root or ""))
    return (gr.update(value=""), gr.update(value=defaults), gr.update(value=""), gr.update(value=""), gr.update(value=""))


def _check_llm(base_url: str, model: str, api_key: str, api_key_env: str, timeout_seconds: int, temperature: float):
    try:
        client = OpenAICompatibleClient(LLMSettings(enabled=True, base_url=base_url, model=model, api_key=api_key, api_key_env=api_key_env, timeout_seconds=int(timeout_seconds or 60), temperature=float(temperature or 0)))
        health = client.health_check()
        if health.ok:
            return f"✅ LLM 可用：{health.message}"
        return f"❌ LLM 检查失败：{health.message}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ LLM 检查异常：{exc}"


def _run_end_to_end(
    source_url: str, uploaded_media: Any, profile_key: str, tag: str, selected_prompts: list[str], uploaded_prompt_files: Any,
    inline_prompt: str, output_root: str, download_root: str, note_root: str, download_mode: str, keep_download: bool,
    cookie_mode: str, uploaded_cookie: Any, cookie_path: str, ytdlp_path: str, ffmpeg_path: str,
    scenario_name: str, model_preset: str, model_custom: str, vad_preset: str, vad_custom: str, punc_preset: str,
    punc_custom: str, language: str, device: str, use_itn: bool, merge_vad: bool, speaker_diarization: bool,
    speaker_count: int, verbose: bool, batch_size_s: int, max_single_segment_ms: int, merge_length_s: int, split_lines: bool,
    run_llm: bool, llm_base_url: str, llm_model: str, llm_api_key: str, llm_api_key_env: str, llm_temperature: float,
    llm_timeout_seconds: int, include_prompt_in_final: bool, include_transcript_in_final: bool, write_note: bool, overwrite_note: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    def _progress(pct: float, message: str) -> None:
        progress(pct, desc=message)

    try:
        result = run_audio_workflow(
            source_url=source_url, uploaded_media=uploaded_media, profile_key=profile_key, tag=tag,
            selected_prompt_labels=selected_prompts or [], uploaded_prompt_files=uploaded_prompt_files, inline_prompt=inline_prompt,
            output_root=output_root, download_root=download_root, note_root=note_root, download_mode=download_mode,
            keep_download=bool(keep_download), cookie_mode=cookie_mode, cookie_path=cookie_path, uploaded_cookie=uploaded_cookie,
            ytdlp_path=ytdlp_path, ffmpeg_path=ffmpeg_path, scenario_name=scenario_name, model_preset=model_preset,
            model_custom=model_custom, vad_preset=vad_preset, vad_custom=vad_custom, punc_preset=punc_preset, punc_custom=punc_custom,
            language=language, device=device, use_itn=bool(use_itn), merge_vad=bool(merge_vad), speaker_diarization=bool(speaker_diarization),
            speaker_count=int(speaker_count or 0), verbose=bool(verbose), batch_size_s=int(batch_size_s), max_single_segment_ms=int(max_single_segment_ms),
            merge_length_s=int(merge_length_s), split_lines=bool(split_lines), run_llm=bool(run_llm), llm_base_url=llm_base_url,
            llm_model=llm_model, llm_api_key=llm_api_key, llm_api_key_env=llm_api_key_env, llm_temperature=float(llm_temperature or 0),
            llm_timeout_seconds=int(llm_timeout_seconds or 180), include_prompt_in_final=bool(include_prompt_in_final),
            include_transcript_in_final=bool(include_transcript_in_final), write_note=bool(write_note), overwrite_note=bool(overwrite_note), progress=_progress,
        )
        return (result.preview, result.summary_preview, result.final_markdown_path, result.bundle_path, result.meta)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return ("", f"❌ 工作流失败：{exc}\n\n```text\n{tb}\n```", None, None, {"error": str(exc), "traceback": tb})


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
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(_topbar_html())
            with gr.Row(elem_classes=["hero-grid"]):
                gr.HTML(_workflow_hero_html())
                with gr.Column(elem_classes=["hero-aside"]):
                    gr.HTML(_metric_card_html("Mode", "Full workflow", "从链接/本地媒体到最终 Markdown，一次运行保留可复核 manifest。"))
                    gr.HTML(_metric_card_html("LLM", "OpenAI-compatible", "可接 LiteLLM / OpenAI 兼容网关 / 本地模型服务。"))
                    gr.HTML(_metric_card_html("Legacy", "Still available", "旧版快速转写入口仍可通过 python -m funasr_lite.webui 启动。"))

            with gr.Row(equal_height=False, elem_classes=["workflow-grid"]):
                with gr.Column(elem_classes=["workflow-sidebar"]):
                    with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                        gr.Markdown("### 1. 输入源与路由\n链接优先；未填链接时使用上传的本地媒体。Profile 用于自动选择下载库、总结目录和默认 prompt。")
                        source_url = gr.Textbox(label="视频/音频链接", placeholder="Bilibili / YouTube / 其他 yt-dlp 支持链接")
                        uploaded_media = gr.File(label="或上传本地媒体", file_types=["audio", "video"], type="filepath")
                        profile_key = gr.Dropdown(choices=profile_keys, value="auto", label="Workflow profile")
                        gr.HTML(_profiles_help_html())
                        tag = gr.Textbox(label="标签", placeholder="#组会 / #项目会 / #人文 / #技术 ...")
                        with gr.Accordion("下载与落盘路径", open=False):
                            download_mode = gr.Radio(["audio", "video"], value="audio", label="URL 下载模式")
                            keep_download = gr.Checkbox(label="保留下载的源媒体", value=True)
                            output_root = gr.Textbox(label="运行输出根目录", placeholder="默认 outputs/workbench_runs")
                            download_root = gr.Textbox(label="媒体下载/归档目录", placeholder="留空则使用 profile 或临时目录")
                            note_root = gr.Textbox(label="最终总结 md 落盘目录", placeholder="留空则只生成下载文件")
                        with gr.Accordion("Cookies 与工具路径", open=False):
                            cookie_mode = gr.Radio(["auto", "upload/path", "none"], value="auto", label="Cookies")
                            uploaded_cookie = gr.File(label="上传 cookies.txt", file_types=[".txt"], type="filepath")
                            cookie_path = gr.Textbox(label="Cookies 文件路径", placeholder="例如 E:\\Software_tools\\yt-dlp\\cookies\\youtube_cookies.txt")
                            ytdlp_path = gr.Textbox(label="yt-dlp 路径", placeholder="留空则 PATH / .local_tools / E:\\Software_tools 自动查找")
                            ffmpeg_path = gr.Textbox(label="ffmpeg 路径", placeholder="留空则 PATH / .local_tools\\ffmpeg\\bin 自动查找")

                    with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                        gr.Markdown("### 2. Prompt 与大模型\n默认会纳入 `.codex/agents.md`（若存在）和你勾选的项目 prompt；还可上传本次专用 prompt 文件。")
                        selected_prompts = gr.CheckboxGroup(choices=prompt_choices, value=prompt_defaults, label="可勾选的 prompts")
                        uploaded_prompt_files = gr.Files(label="本次追加 prompt 文件", file_types=[".md", ".txt"], type="filepath")
                        inline_prompt = gr.Textbox(label="本次追加对话指令", lines=5, placeholder="例如：请输出适合 Obsidian 的层级标题；保留关键原话；列出待核对问题。")
                        run_llm = gr.Checkbox(label="调用大模型生成 final_summary.md", value=False)
                        with gr.Accordion("LLM API 设置", open=False):
                            llm_base_url = gr.Textbox(label="Base URL", value=os.getenv("FUNASR_LITE_LLM_BASE_URL", "http://127.0.0.1:4000/v1"))
                            llm_model = gr.Textbox(label="Model", value=os.getenv("FUNASR_LITE_LLM_MODEL", "gpt-5.4"))
                            llm_api_key = gr.Textbox(label="API Key（可留空走环境变量）", type="password")
                            llm_api_key_env = gr.Textbox(label="API Key 环境变量名", value=os.getenv("FUNASR_LITE_LLM_API_KEY_ENV", "LITELLM_API_KEY"))
                            llm_temperature = gr.Slider(0, 2, value=0.1, step=0.05, label="Temperature")
                            llm_timeout_seconds = gr.Slider(30, 900, value=180, step=30, label="Timeout seconds")
                            check_llm_btn = gr.Button("检查 LLM 连接", variant="secondary")
                            llm_status = gr.Markdown("")
                        with gr.Accordion("最终 Markdown 选项", open=False):
                            include_prompt_in_final = gr.Checkbox(label="在 final_summary.md 末尾附完整 prompt", value=False)
                            include_transcript_in_final = gr.Checkbox(label="在 final_summary.md 末尾附转写全文", value=False)
                            write_note = gr.Checkbox(label="同时写入 note_root / profile 建议路径", value=False)
                            overwrite_note = gr.Checkbox(label="允许覆盖已有同名笔记", value=False)

                    with gr.Column(elem_classes=["glass-card", "tech-panel"]):
                        gr.Markdown("### 3. FunASR 参数\n这些参数与旧版 WebUI 保持一致；开启说话人分离时会自动切到兼容模型。")
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

                    with gr.Column(elem_classes=["glass-card", "action-bar", "execute-panel"]):
                        gr.Markdown("### 触发工作流")
                        gr.HTML('<div class="action-copy">将依次执行下载/上传解析、ffmpeg 预处理、FunASR 转写、prompt 组装、大模型处理与产物打包。</div>')
                        start_btn = gr.Button("开始完整工作流", variant="primary", size="lg")

                with gr.Column(elem_classes=["workflow-main"]):
                    gr.HTML(_section_banner_html("Results", "预览、下载与运行证据", "最终会生成 final_summary.md、workflow_outputs.zip 和 run_manifest.json。"))
                    with gr.Column(elem_classes=["glass-card", "tech-panel", "results-panel"]):
                        with gr.Tabs():
                            with gr.TabItem("转写预览"):
                                transcript_preview = gr.Textbox(lines=24, interactive=False, label="Transcript / Speaker Preview")
                            with gr.TabItem("总结预览"):
                                summary_preview = gr.Textbox(lines=28, interactive=False, label="Final Markdown Preview")
                            with gr.TabItem("下载"):
                                final_download = gr.File(label="final_summary.md")
                                bundle_download = gr.File(label="workflow_outputs.zip")
                            with gr.TabItem("运行元数据"):
                                meta = gr.JSON(label="run_manifest")

            scenario_name.change(_scenario_outputs, inputs=[scenario_name], outputs=[batch_size_s, max_single_segment_ms, merge_length_s, language, use_itn, merge_vad])
            speaker_diarization.change(_speaker_toggle, inputs=[speaker_diarization, model_preset, punc_preset], outputs=[model_preset, punc_preset])
            profile_key.change(_on_profile_change, inputs=[profile_key], outputs=[tag, selected_prompts, download_root, note_root, output_root])
            check_llm_btn.click(_check_llm, inputs=[llm_base_url, llm_model, llm_api_key, llm_api_key_env, llm_timeout_seconds, llm_temperature], outputs=[llm_status])
            start_btn.click(
                _run_end_to_end,
                inputs=[source_url, uploaded_media, profile_key, tag, selected_prompts, uploaded_prompt_files, inline_prompt, output_root, download_root, note_root, download_mode, keep_download, cookie_mode, uploaded_cookie, cookie_path, ytdlp_path, ffmpeg_path, scenario_name, model_preset, model_custom, vad_preset, vad_custom, punc_preset, punc_custom, language, device, use_itn, merge_vad, speaker_diarization, speaker_count, verbose, batch_size_s, max_single_segment_ms, merge_length_s, split_lines, run_llm, llm_base_url, llm_model, llm_api_key, llm_api_key_env, llm_temperature, llm_timeout_seconds, include_prompt_in_final, include_transcript_in_final, write_note, overwrite_note],
                outputs=[transcript_preview, summary_preview, final_download, bundle_download, meta],
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

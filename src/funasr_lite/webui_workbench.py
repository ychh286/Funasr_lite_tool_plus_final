from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .config import DEFAULT_SCENARIO, SCENARIO_PRESETS
from .llm_client import LLMSettings, check_health
from .pipeline_workflow import (
    DEFAULT_MODEL,
    DEFAULT_PUNC,
    WorkflowRunRequest,
    run_audio_workbench,
)
from .prompting import default_prompt_keys, discover_prompt_sources
from .workflow_profiles import get_workflow_profiles


WORKBENCH_CSS = """
:root {
  --bg: #070b13;
  --panel: rgba(18, 26, 42, 0.78);
  --panel-strong: rgba(12, 18, 30, 0.88);
  --border: rgba(151, 180, 255, 0.18);
  --text: #eef4ff;
  --muted: #9fb0ce;
  --accent: #77d7f6;
  --accent-2: #b3ffb6;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 34%),
    radial-gradient(circle at top right, rgba(134, 239, 172, 0.14), transparent 28%),
    linear-gradient(180deg, #060913 0%, #0b1120 100%) !important;
  color: var(--text) !important;
  font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif !important;
}

.gradio-container {
  max-width: 1520px !important;
  margin: 0 auto !important;
  padding: 28px 24px 56px !important;
}

.hero,
.panel,
.result-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  backdrop-filter: blur(18px);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
}

.hero {
  padding: 28px 32px;
  margin-bottom: 22px;
}

.hero h1 {
  margin: 0 0 12px 0;
  font-size: 2.4rem;
  letter-spacing: -0.04em;
}

.hero p,
.hero li,
.panel p,
.hint {
  color: var(--muted);
}

.panel {
  padding: 20px 22px;
}

.result-panel {
  padding: 22px;
}

.action-panel {
  background: var(--panel-strong);
}

.status-card {
  padding: 18px 20px;
  border-radius: 16px;
  background: rgba(7, 12, 22, 0.58);
  border: 1px solid rgba(119, 215, 246, 0.18);
}

.gr-button-primary,
button.primary {
  background: linear-gradient(135deg, #77d7f6 0%, #68f0c6 100%) !important;
  color: #04101b !important;
  border: none !important;
  font-weight: 700 !important;
}

.gr-button-secondary,
button:not(.primary) {
  background: rgba(255, 255, 255, 0.06) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

textarea,
input,
.wrap,
.dropdown {
  background: rgba(255, 255, 255, 0.04) !important;
  color: var(--text) !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
}
"""


def _prompt_choices() -> list[tuple[str, str]]:
    """构造 Workbench 的 Prompt 多选项。"""

    return [(source.label, source.key) for source in discover_prompt_sources()]


def _profile_choices() -> list[tuple[str, str]]:
    """构造 profile 下拉选项。"""

    choices = [("自动识别 / 不使用", "")]
    for key, profile in sorted(get_workflow_profiles().items()):
        label = f"{profile.display_name} ({key})" if profile.display_name != key else key
        choices.append((label, key))
    return choices


def _build_status_markdown(result) -> str:
    """把一次运行结果转换成顶部状态摘要。"""

    manifest = result.manifest
    warnings = manifest.get("download_warnings") or []
    llm_error = (manifest.get("llm_error") or "").strip()

    lines = [
        "### 本次运行完成",
        "",
        f"- 输出目录：`{manifest.get('output_dir', '')}`",
        f"- 原始输入：`{manifest.get('source', '')}`",
        f"- Prompt 组合：`{len(manifest.get('selected_prompt_keys') or [])}` 个来源",
        f"- 最终总结：`{manifest.get('final_summary', '')}`",
    ]
    if warnings:
        for warning in warnings:
            lines.append(f"- 下载提醒：{warning}")
    if llm_error:
        lines.append(f"- LLM 回退：{llm_error}")
    elif manifest.get("llm", {}).get("enabled"):
        lines.append("- LLM 状态：已调用成功。")
    else:
        lines.append("- LLM 状态：未启用，已生成可继续处理的总结草稿。")
    return "\n".join(lines)


def _check_llm_connection(
    enabled: bool,
    base_url: str,
    model: str,
    api_key_env: str,
    api_key: str,
    temperature: float,
    timeout_seconds: float,
) -> str:
    """在 WebUI 中执行一次轻量的 LLM 健康检查。"""

    settings = LLMSettings(
        enabled=bool(enabled),
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        api_key=api_key,
        temperature=float(temperature),
        timeout_seconds=int(timeout_seconds),
    )
    health = check_health(settings)
    if health.ok:
        return f"连接成功：`{health.message}`"
    return f"连接失败：`{health.message}`"


def _run_workbench(
    source_url: str,
    local_media_path: str,
    profile_key: str,
    download_mode: str,
    cookies_mode: str,
    cookies_path: str,
    uploaded_cookies_path: str,
    ytdlp_path: str,
    ffmpeg_path: str,
    scenario: str,
    device: str,
    speaker_diarization: bool,
    speaker_count: int,
    tag: str,
    selected_prompt_keys: list[str],
    uploaded_prompt_path: str,
    extra_instruction: str,
    llm_enabled: bool,
    llm_base_url: str,
    llm_model: str,
    llm_api_key_env: str,
    llm_api_key: str,
    llm_temperature: float,
    llm_timeout_seconds: float,
):
    """把 WebUI 表单值转换成 `WorkflowRunRequest` 并执行。"""

    request = WorkflowRunRequest(
        source_url=source_url,
        local_media_path=local_media_path,
        profile_key=profile_key,
        download_mode=download_mode,
        cookies_mode=cookies_mode,
        cookies_path=cookies_path,
        uploaded_cookies_path=uploaded_cookies_path,
        ytdlp_path=ytdlp_path,
        ffmpeg_path=ffmpeg_path,
        scenario=scenario,
        device=device,
        model=DEFAULT_MODEL,
        punc=DEFAULT_PUNC,
        speaker_diarization=bool(speaker_diarization),
        speaker_count=int(speaker_count),
        tag=tag,
        selected_prompt_keys=tuple(selected_prompt_keys or []),
        uploaded_prompt_path=uploaded_prompt_path,
        extra_instruction=extra_instruction,
        default_output_root=str(Path(__file__).resolve().parents[2] / "outputs" / "workbench_runs"),
        prepared_audio_name="audio_16k_mono.wav",
        llm=LLMSettings(
            enabled=bool(llm_enabled),
            base_url=llm_base_url,
            model=llm_model,
            api_key_env=llm_api_key_env,
            api_key=llm_api_key,
            temperature=float(llm_temperature),
            timeout_seconds=int(llm_timeout_seconds),
        ),
    )
    result = run_audio_workbench(request)
    return (
        _build_status_markdown(result),
        result.transcript_preview,
        result.summary_preview,
        str(result.final_summary_path),
        str(result.workflow_zip_path),
        result.manifest,
    )


def main() -> None:
    """启动 Audio Workbench V2 的 Gradio WebUI。"""

    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="emerald",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
    )
    prompt_choices = _prompt_choices()
    scenario_choices = list(SCENARIO_PRESETS.keys())

    with gr.Blocks(title="FunASR Audio Workbench V2", css=WORKBENCH_CSS, theme=theme) as demo:
        gr.HTML(
            """
            <div class="hero">
              <p style="text-transform: uppercase; letter-spacing: 0.14em; margin: 0 0 10px 0;">Audio Workbench V2</p>
              <h1>把下载、转写、Prompt 组合和总结，放进同一个入口。</h1>
              <p>
                当前页面默认面向“链接或本地媒体 -> yt-dlp / ffmpeg -> FunASR -> Prompt 组合 -> 可选 LLM -> Markdown / zip”这条完整工作流。
                如果你只想保留旧版本地批量转写界面，可以继续使用 <code>Start-FunASRWebUI.ps1 -Legacy</code>。
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### 输入源")
                    source_url = gr.Textbox(
                        label="视频链接",
                        placeholder="支持 Bilibili、YouTube 或其他 yt-dlp 支持的链接。若填写了 URL，将优先使用 URL。",
                    )
                    local_media = gr.File(
                        label="本地媒体文件",
                        type="filepath",
                        file_types=["audio", "video"],
                    )
                    profile_key = gr.Dropdown(
                        choices=_profile_choices(),
                        value="",
                        label="工作流配置（profile）",
                    )
                    download_mode = gr.Radio(
                        choices=[("保留音频", "audio"), ("保留视频", "video")],
                        value="audio",
                        label="URL 下载模式",
                    )
                    gr.Markdown("<div class='hint'>profile 会自动带入默认 tag、prompt、下载目录和建议的笔记落盘路径。</div>")

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### 下载与 cookies")
                    cookies_mode = gr.Radio(
                        choices=[
                            ("自动发现", "auto"),
                            ("手动路径", "path"),
                            ("上传本次 cookies", "upload"),
                            ("禁用 cookies", "none"),
                        ],
                        value="auto",
                        label="cookies 模式",
                    )
                    cookies_path = gr.Textbox(
                        label="cookies 文件路径",
                        placeholder="当模式为“手动路径”时填写。",
                    )
                    uploaded_cookies = gr.File(
                        label="上传 cookies 文件",
                        type="filepath",
                        file_types=[".txt", ".cookies"],
                    )
                    ytdlp_path = gr.Textbox(
                        label="yt-dlp 路径（可选）",
                        placeholder="留空时自动从环境变量、.local_tools 或 PATH 解析。",
                    )
                    ffmpeg_path = gr.Textbox(
                        label="ffmpeg 路径（可选）",
                        placeholder="留空时自动从环境变量、.local_tools 或 PATH 解析。",
                    )

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### 转写参数")
                    scenario = gr.Dropdown(
                        choices=scenario_choices,
                        value=DEFAULT_SCENARIO,
                        label="场景预设",
                    )
                    device = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="设备",
                    )
                    speaker_diarization = gr.Checkbox(
                        label="启用说话人分离（Speaker Diarization V1）",
                        value=False,
                    )
                    speaker_count = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        label="已知说话人数（0 表示自动聚类）",
                    )
                    tag = gr.Textbox(
                        label="快捷标签（可选）",
                        placeholder="例如 #技术、#组会、#人文，用于补充 prompt 选择。",
                    )

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Prompt 组合")
                    selected_prompt_keys = gr.Dropdown(
                        choices=prompt_choices,
                        value=default_prompt_keys(),
                        multiselect=True,
                        label="已选 Prompt",
                    )
                    uploaded_prompt = gr.File(
                        label="本次附加 Prompt 文件",
                        type="filepath",
                        file_types=[".md", ".txt"],
                    )
                    extra_instruction = gr.Textbox(
                        label="本次补充说明",
                        lines=8,
                        placeholder="补充你希望 agent 或模型本次额外关注的点，例如输出结构、口吻、写作限制等。",
                    )

                with gr.Column(elem_classes=["panel", "action-panel"]):
                    gr.Markdown("### LLM 总结（可选）")
                    llm_enabled = gr.Checkbox(label="启用 OpenAI-compatible LLM 总结", value=False)
                    llm_base_url = gr.Textbox(label="base_url", value="http://127.0.0.1:4000/v1")
                    llm_model = gr.Textbox(label="model", value="gpt-5.4")
                    llm_api_key_env = gr.Textbox(label="api_key_env", value="OPENAI_API_KEY")
                    llm_api_key = gr.Textbox(label="api_key", type="password", placeholder="可留空，优先从环境变量读取。")
                    llm_temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.2, label="temperature")
                    llm_timeout_seconds = gr.Number(value=180, label="timeout_seconds")
                    with gr.Row():
                        llm_check_btn = gr.Button("检查连接", variant="secondary")
                        start_btn = gr.Button("开始执行 Audio Workbench", variant="primary")
                    llm_status = gr.Markdown("未检查连接。")

            with gr.Column(scale=8):
                with gr.Column(elem_classes=["result-panel"]):
                    status_md = gr.Markdown("### 等待执行\n\n填好左侧参数后点击开始。")
                with gr.Tabs():
                    with gr.TabItem("转写预览"):
                        transcript_preview = gr.Textbox(label="Transcript Preview", lines=22, interactive=False)
                    with gr.TabItem("总结预览"):
                        summary_preview = gr.Textbox(label="Final Summary Preview", lines=22, interactive=False)
                    with gr.TabItem("运行清单"):
                        manifest_json = gr.JSON(label="run_manifest.json")
                with gr.Row():
                    final_summary_file = gr.File(label="final_summary.md")
                    workflow_zip_file = gr.File(label="workflow_outputs.zip")

        llm_check_btn.click(
            _check_llm_connection,
            inputs=[
                llm_enabled,
                llm_base_url,
                llm_model,
                llm_api_key_env,
                llm_api_key,
                llm_temperature,
                llm_timeout_seconds,
            ],
            outputs=[llm_status],
        )

        start_btn.click(
            _run_workbench,
            inputs=[
                source_url,
                local_media,
                profile_key,
                download_mode,
                cookies_mode,
                cookies_path,
                uploaded_cookies,
                ytdlp_path,
                ffmpeg_path,
                scenario,
                device,
                speaker_diarization,
                speaker_count,
                tag,
                selected_prompt_keys,
                uploaded_prompt,
                extra_instruction,
                llm_enabled,
                llm_base_url,
                llm_model,
                llm_api_key_env,
                llm_api_key,
                llm_temperature,
                llm_timeout_seconds,
            ],
            outputs=[
                status_md,
                transcript_preview,
                summary_preview,
                final_summary_file,
                workflow_zip_file,
                manifest_json,
            ],
        )

    import argparse

    parser = argparse.ArgumentParser(description="FunASR Audio Workbench V2")
    parser.add_argument("--host", default=os.getenv("FUNASR_LITE_HOST", "127.0.0.1"), help="Bind host")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FUNASR_LITE_PORT", "7860")),
        help="Bind port",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (default OFF for safety).",
    )
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

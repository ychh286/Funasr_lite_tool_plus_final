from __future__ import annotations

import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .config import (
    DEFAULT_SCENARIO,
    MODEL_PRESETS,
    PUNC_PRESETS,
    SCENARIO_PRESETS,
    VAD_PRESETS,
    DecodeConfig,
    ModelConfig,
    ObsConfig,
    OutputConfig,
    apply_speaker_diarization_defaults,
    validate_speaker_diarization_config,
)
from .core import TranscribeResult, transcribe_file
from .formatting import postprocess_text, split_lines_by_punct
from .save import render_speaker_content, save_json, save_speaker_text, save_text


TECH_CSS = '''
:root {
  --background: #000000;
  --foreground: #ffffff;
  --muted: #a1a1aa;
  --accent: #fff;
  --panel: rgba(18, 18, 20, 0.75);
  --panel-border: rgba(255, 255, 255, 0.08);
  --shadow-sm: 0 4px 14px 0 rgba(0, 0, 0, 0.39);
  --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.5);
  --brand-glow: radial-gradient(circle at 50% -20%, rgba(255, 255, 255, 0.08), transparent 60%);
  --grad-text: linear-gradient(180deg, #fff 0%, #a1a1aa 100%);
}

body,
.gradio-container {
  background: var(--background) !important;
  color: var(--foreground) !important;
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Microsoft YaHei UI", sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

.gradio-container {
  max-width: 1560px !important;
  margin: 0 auto !important;
  padding: 40px 32px 80px !important;
}

/* Base shell & nice top background glow */
.app-shell::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 600px;
  background: var(--brand-glow);
  pointer-events: none;
  z-index: 0;
}

.topbar,
.hero-panel,
.glass-card,
.result-card,
.metric-card,
.download-card {
  position: relative;
  z-index: 1;
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  box-shadow: var(--shadow-sm);
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 28px;
  margin-bottom: 32px;
  border-radius: 16px;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 16px;
}

.brand-badge {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: #ffffff;
  color: #000000;
  font-weight: 700;
  font-size: 1.1rem;
  letter-spacing: -0.02em;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.1), 0 8px 16px rgba(255,255,255,0.15);
}

.brand-title {
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.brand-subtitle,
.eyebrow,
.metric-label,
.section-kicker {
  color: var(--muted);
  font-size: 0.8rem;
  font-weight: 500;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.brand-subtitle {
  margin-top: 4px;
}

.status-pills { display: flex; gap: 8px; flex-wrap: wrap; }

.status-pill, .chip {
  padding: 6px 14px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
  color: #d4d4d8;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(8px);
}

.status-pill strong, .chip strong { color: #fff; font-weight: 600; }

.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(300px, 0.8fr);
  gap: 24px;
  margin-bottom: 40px;
}

.hero-panel {
  padding: 48px 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.hero-title {
  margin: 16px 0;
  font-size: clamp(2.5rem, 4vw, 3.5rem);
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.1;
  background: var(--grad-text);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-copy {
  color: #d4d4d8;
  font-size: 1.05rem;
  line-height: 1.6;
  max-width: 600px;
}

.hero-chips {
  margin-top: 32px;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.hero-aside {
  display: grid;
  gap: 16px;
}

.metric-card {
  padding: 24px;
}
.metric-card:hover { border-color: rgba(255,255,255,0.2); }

.metric-value {
  margin-top: 8px;
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.metric-note {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.85rem;
  line-height: 1.5;
}

.section-banner {
  margin: 48px 0 24px 8px;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.03em;
  margin-top: 4px;
}

.section-copy {
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.95rem;
}

.glass-card { padding: 28px 32px; }

.glass-card > .gradio-markdown { margin-bottom: 24px; }
.glass-card > .gradio-markdown h3 { font-size: 1.25rem; font-weight: 600; margin: 0 0 8px 0; letter-spacing: -0.01em;}
.glass-card > .gradio-markdown p { color: var(--muted); font-size: 0.9rem; margin: 0;}

.form-row { gap: 24px !important; }

/* Overriding Gradio Core Elements */
.gradio-container * {
  border-radius: 8px !important;
}

.tech-panel label, 
.tech-panel .block-label {
  color: #e4e4e7 !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  margin-bottom: 6px !important;
}

.tech-panel input,
.tech-panel textarea,
.tech-panel .wrap,
.tech-panel .dropdown,
.tech-panel .gr-box,
.tech-panel .cm-editor {
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #fff !important;
  transition: border-color 0.2s, background 0.2s;
  box-shadow: none !important;
}

.tech-panel input:focus,
.tech-panel textarea:focus,
.tech-panel .wrap.focus {
  border-color: rgba(255, 255, 255, 0.3) !important;
  background: rgba(255, 255, 255, 0.07) !important;
}

.gradio-container .gr-button-primary,
.gradio-container button.primary {
  background: #ffffff !important;
  color: #000000 !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  border: none !important;
  box-shadow: 0 4px 14px rgba(255,255,255,0.2) !important;
  transition: all 0.2s ease !important;
}

.gradio-container .gr-button-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(255,255,255,0.3) !important;
}

.gradio-container .gr-button-secondary,
.gradio-container button:not(.primary) {
  background: rgba(255, 255, 255, 0.08) !important;
  color: #fff !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  font-weight: 500 !important;
}
.gradio-container .gr-button-secondary:hover {
  background: rgba(255, 255, 255, 0.12) !important;
}

.action-bar {
  margin: 32px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.action-bar .action-copy {
  color: var(--muted);
  font-size: 0.95rem;
}

.result-card, .download-card { padding: 24px; }
.results-grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(300px, 0.8fr); gap: 24px; align-items: start; }
.result-stack { display: grid; gap: 24px; }

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }

@media (max-width: 1024px) {
  .hero-grid, .results-grid { grid-template-columns: 1fr; }
  .gradio-container { padding: 20px 16px 40px !important; }
}
'''


def _topbar_html() -> str:
    return """
    <div class="topbar">
      <div class="brand-lockup">
        <div class="brand-badge">FA</div>
        <div>
          <div class="brand-title">FunASR Pro Console</div>
          <div class="brand-subtitle">高性能本地化智能语音转写与说话人区分工作流引擎</div>
        </div>
      </div>
      <div class="status-pills">
        <div class="status-pill"><strong>Local & Privacy</strong>&nbsp; 纯本地推理架构</div>
        <div class="status-pill"><strong>Speaker Engine</strong>&nbsp; Paraformer \+ CAM\+\+</div>
        <div class="status-pill"><strong>Batch Process</strong>&nbsp; 多端媒体流并发处理</div>
      </div>
    </div>
    """


def _hero_html() -> str:
    return """
    <div class="hero-panel">
      <div class="eyebrow">Workspace</div>
      <div class="hero-title">高精转写，<br/>智能分离与导出。</div>
      <div class="hero-copy">
        专为离线生产力打造的音视频媒体处理中心。支持多文件极速批处理、高精说话人聚类过滤（Speaker Diarization V1），并自动生成清晰的 Markdown 面包屑数据，无缝对接大语言模型下游摘要工作流。
      </div>
      <div class="hero-chips">
        <div class="chip"><strong>输入源支持</strong>&nbsp; wav / mp3 / mp4 等音视频容器</div>
        <div class="chip"><strong>交付产物</strong>&nbsp; 逐字稿 \+ 扬声器数据边车 (Sidecars)</div>
        <div class="chip"><strong>计算效能</strong>&nbsp; CPU \+ CUDA 异构加速计算</div>
      </div>
    </div>
    """


def _metric_card_html(title: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def _section_banner_html(kicker: str, title: str, copy: str) -> str:
    return f"""
    <div class="section-banner">
      <div class="section-kicker">{kicker}</div>
      <div class="section-title">{title}</div>
      <div class="section-copy">{copy}</div>
    </div>
    """

def _as_path_list(files: Any) -> List[Path]:
    if not files:
        return []
    if isinstance(files, (str, Path)):
        return [Path(str(files))]

    paths: List[Path] = []
    for f in files:
        if f is None:
            continue
        if isinstance(f, Path):
            paths.append(f)
            continue
        if isinstance(f, str):
            paths.append(Path(f))
            continue
        path_like = getattr(f, "path", None) or getattr(f, "name", None)
        if path_like:
            paths.append(Path(str(path_like)))
            continue
        if isinstance(f, dict):
            for k in ("path", "name"):
                if k in f and f[k]:
                    paths.append(Path(str(f[k])))
                    break
    return paths


def _resolve_model_id(preset_key: str, custom: str) -> str:
    custom = (custom or "").strip()
    if custom:
        return custom
    return MODEL_PRESETS.get(preset_key, preset_key)


def _resolve_vad_id(preset_key: str, custom: str) -> str:
    custom = (custom or "").strip()
    if custom:
        return custom
    return VAD_PRESETS.get(preset_key, preset_key)


def _resolve_punc_id(preset_key: str, custom: str) -> Optional[str]:
    custom = (custom or "").strip()
    if custom:
        return custom
    return PUNC_PRESETS.get(preset_key)


def _apply_scenario_defaults(scenario_name: str) -> Tuple[int, int, int, str, bool, bool]:
    preset = SCENARIO_PRESETS.get(scenario_name) or SCENARIO_PRESETS[DEFAULT_SCENARIO]
    d = preset.decode
    return (
        int(d.batch_size_s),
        int(d.max_single_segment_ms),
        int(d.merge_length_s),
        d.language,
        bool(d.use_itn),
        bool(d.merge_vad),
    )


def _speaker_sidecar_paths(base_dir: Path, stem: str, suffix: str) -> tuple[Path, Path]:
    speaker_text = base_dir / f"{stem}_speakers{suffix}"
    speaker_json = base_dir / f"{stem}_speakers.json"
    return speaker_text, speaker_json


def _run_batch(
    files: Any,
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
    out_fmt: str,
    split_lines: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    paths = _as_path_list(files)
    if not paths:
        return "", None, {"error": "No input files."}

    model_cfg = ModelConfig(
        model_id=_resolve_model_id(model_preset, model_custom),
        vad_model_id=_resolve_vad_id(vad_preset, vad_custom),
        punc_model_id=_resolve_punc_id(punc_preset, punc_custom),
        device=None if device == "auto" else device,
        disable_update=True,
    )
    if speaker_diarization:
        model_cfg = apply_speaker_diarization_defaults(
            model_cfg,
            allow_auto_model_swap=not bool(model_custom),
        )
        validate_speaker_diarization_config(model_cfg)

    decode_cfg = DecodeConfig(
        language=language,
        use_itn=bool(use_itn),
        batch_size_s=int(batch_size_s),
        merge_vad=bool(merge_vad),
        merge_length_s=int(merge_length_s),
        max_single_segment_ms=int(max_single_segment_ms),
        preset_spk_num=int(speaker_count) if int(speaker_count) > 0 else None,
    )
    output_cfg = OutputConfig(fmt=out_fmt, split_lines=bool(split_lines))
    obs_cfg = ObsConfig(verbose=bool(verbose))

    meta: Dict[str, Any] = {
        "scenario": scenario_name,
        "model_cfg": asdict(model_cfg),
        "decode_cfg": asdict(decode_cfg),
        "output_cfg": asdict(output_cfg),
        "files": [p.name for p in paths],
        "results": [],
    }

    preview_parts: List[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="funasr_lite_batch_"))
    out_files: List[Path] = []
    suffix = ".md" if out_fmt == "md" else ".txt"

    n = len(paths)
    for i, p in enumerate(paths, start=1):
        progress((i - 1) / max(n, 1), desc=f"Transcribing {p.name} ({i}/{n})")
        try:
            r: TranscribeResult = transcribe_file(
                wav_path=str(p),
                model_cfg=model_cfg,
                decode_cfg=decode_cfg,
                output_cfg=output_cfg,
                obs_cfg=obs_cfg,
            )
        except Exception as e:
            err_type = type(e).__name__
            err_msg = (str(e) or "").strip() or repr(e)
            err = f"{err_type}: {err_msg}"
            if obs_cfg.verbose:
                import traceback

                err += "\n" + traceback.format_exc()
            meta["results"].append({"file": p.name, "ok": False, "error": err})
            preview_parts.append(f"## {p.name}\n\n[ERROR] {err}\n")
            continue

        text = postprocess_text(r.text)
        if split_lines:
            text = split_lines_by_punct(text)

        plain_out_path = tmpdir / f"{p.stem}{suffix}"
        save_text(text, plain_out_path, fmt=out_fmt, split_lines=split_lines, source_name=p.stem)
        out_files.append(plain_out_path)

        if speaker_diarization and r.segments:
            preview_parts.append(render_speaker_content(r.segments, fmt=out_fmt, source_name=p.name).strip())
            speaker_text_path, speaker_json_path = _speaker_sidecar_paths(tmpdir, p.stem, suffix)
            save_speaker_text(r.segments, speaker_text_path, fmt=out_fmt, source_name=p.stem)
            save_json(r.segments, speaker_json_path)
            out_files.extend([speaker_text_path, speaker_json_path])
        else:
            if out_fmt == "md":
                preview_parts.append(f"## {p.name}\n\n{text}\n")
            else:
                preview_parts.append(f"[{p.name}]\n{text}\n")

        meta["results"].append(
            {
                "file": p.name,
                "ok": True,
                "speaker_segments": len(r.segments),
                "meta": r.meta,
            }
        )

    progress(1.0, desc="Packaging outputs")

    if len(out_files) == 1:
        download_path = str(out_files[0])
    else:
        zip_path = tmpdir / "outputs.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in out_files:
                z.write(f, arcname=f.name)
        download_path = str(zip_path)

    preview = "\n\n".join(part for part in preview_parts if part).strip()
    return preview, download_path, meta


def main() -> None:
    scenario_names = list(SCENARIO_PRESETS.keys())
    model_preset_keys = list(MODEL_PRESETS.keys())
    vad_preset_keys = list(VAD_PRESETS.keys())
    punc_preset_keys = list(PUNC_PRESETS.keys())

    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
    )

    with gr.Blocks(title="funasr-lite - Local transcription", css=TECH_CSS, theme=theme) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(_topbar_html())

            with gr.Row(elem_classes=["hero-grid"]):
                gr.HTML(_hero_html())
                with gr.Column(elem_classes=["hero-aside"]):
                    gr.HTML(
                        _metric_card_html(
                            "Default mode",
                            DEFAULT_SCENARIO,
                            "场景预设会联动 batch_size、语言、ITN 和分段合并等默认参数。",
                        )
                    )
                    gr.HTML(
                        _metric_card_html(
                            "Speaker V1",
                            "Optional",
                            "开启后会自动切到支持时间戳的 Paraformer 配置，并输出 speaker sidecars。",
                        )
                    )
                    gr.HTML(
                        _metric_card_html(
                            "Export",
                            "Markdown / TXT",
                            "单文件直接下载，多文件自动打包 zip，适合继续喂给总结工作流。",
                        )
                    )

            gr.HTML(
                _section_banner_html(
                    "Mission Setup",
                    "Configure the run surface",
                    "上传文件、选择推理模式和输出策略。界面分为输入输出、模型后端、运行时控制与结果面板，尽量把常用设置收敛在一屏内。",
                )
            )

            with gr.Row(equal_height=False, elem_classes=["form-row"]):
                with gr.Column(scale=6, elem_classes=["glass-card", "tech-panel"]):
                    gr.Markdown(
                        "### 输入源与产物格式\n"
                        "选择输入媒体与导出格式。默认保留最适合构建下游大语言模型摘要工作流的 Markdown 格式。"
                    )
                    files = gr.Files(
                        label="Audio / Video files",
                        file_types=["audio", "video"],
                    )
                    with gr.Row():
                        scenario = gr.Dropdown(
                            choices=scenario_names,
                            value=DEFAULT_SCENARIO,
                            label="Scenario preset",
                        )
                        out_fmt = gr.Radio(["md", "txt"], value="md", label="Output format")
                    split_lines = gr.Checkbox(
                        label="Split transcript lines by punctuation",
                        value=False,
                    )

                with gr.Column(scale=6, elem_classes=["glass-card", "tech-panel"]):
                    gr.Markdown(
                        "### 模型基座与后端引擎\n"
                        "配置主转写模型、VAD 断句引擎及标点预测后端。输入自定义模型 ID 将自动覆盖默认参数。"
                    )
                    model_preset = gr.Dropdown(
                        choices=model_preset_keys,
                        value="iic/SenseVoiceSmall",
                        label="Model preset",
                    )
                    model_custom = gr.Textbox(
                        label="Custom model id",
                        placeholder="Optional. Overrides preset if provided.",
                    )
                    with gr.Row():
                        vad_preset = gr.Dropdown(
                            choices=vad_preset_keys,
                            value="fsmn-vad",
                            label="VAD preset",
                        )
                        punc_preset = gr.Dropdown(
                            choices=punc_preset_keys,
                            value="none",
                            label="Punctuation preset",
                        )
                    vad_custom = gr.Textbox(
                        label="Custom VAD model id",
                        placeholder="Optional. Overrides preset if provided.",
                    )
                    punc_custom = gr.Textbox(
                        label="Custom punctuation model id",
                        placeholder="Optional. Overrides preset if provided.",
                    )

            with gr.Row(equal_height=False, elem_classes=["form-row"]):
                with gr.Column(scale=6, elem_classes=["glass-card", "tech-panel"]):
                    gr.Markdown(
                        "### 运行时与解码单元\n"
                        "指定语言、计算设备及流式分段参数。为确保逻辑闭环，说话人声纹特征提取（Speaker Diarization）选项亦配置于此。"
                    )
                    with gr.Row():
                        language = gr.Dropdown(
                            ["auto", "zh", "en", "ja", "ko"],
                            value="auto",
                            label="Language",
                        )
                        device = gr.Dropdown(
                            ["auto", "cuda:0", "cpu"],
                            value="auto",
                            label="Device",
                        )
                    with gr.Row():
                        use_itn = gr.Checkbox(label="Use ITN", value=True)
                        merge_vad = gr.Checkbox(label="Merge VAD segments", value=True)
                        verbose = gr.Checkbox(label="Verbose meta", value=False)
                    speaker_diarization = gr.Checkbox(
                        label="Enable speaker diarization (V1)",
                        value=False,
                    )
                    speaker_count = gr.Slider(
                        0,
                        10,
                        value=0,
                        step=1,
                        label="Known speaker count (0 = auto cluster)",
                    )
                    batch_size_s = gr.Slider(1, 60, value=10, step=1, label="batch_size_s (seconds)")
                    max_single_segment_ms = gr.Slider(
                        5000,
                        120000,
                        value=30000,
                        step=1000,
                        label="max_single_segment_time (ms)",
                    )
                    merge_length_s = gr.Slider(
                        1,
                        60,
                        value=15,
                        step=1,
                        label="merge_length_s (seconds)",
                    )

                with gr.Column(scale=6, elem_classes=["glass-card"]):
                    gr.Markdown(
                        "### 最佳实践指南\n"
                        "声纹分离（Speaker Diarization V1）专为会议、对谈场景设计。若所处声学环境存在严重多人同重叠堆叠或复杂混响，可能导致局部话语边界解析下降。"
                    )
                    gr.HTML(
                        _metric_card_html(
                            "Best for",
                            "Meetings",
                            "轮流发言、录音相对干净的素材，最容易稳定得到可总结的 speaker transcript。",
                        )
                    )
                    gr.HTML(
                        _metric_card_html(
                            "Fallback",
                            "Plain ASR",
                            "不开启说话人模式时，仍走普通转写链路，适合快速粗转或单人音频。",
                        )
                    )

            with gr.Row(elem_classes=["glass-card", "action-bar"]):
                with gr.Column(scale=7):
                    gr.Markdown("### Execute")
                    gr.HTML(
                        '<div class="action-copy">点击执行批处理工作流。系统将在完成转写与声纹聚类后，自动打包产物结构至下载区，供后续业务长链条调用。</div>'
                    )
                with gr.Column(scale=3, min_width=220):
                    start_btn = gr.Button("Start transcription", variant="primary", size="lg")

            gr.HTML(
                _section_banner_html(
                    "Results",
                    "检视实时转写日志与交付产物",
                    "左侧用于快速查看转写结果，右侧保留下载与元数据。这样既方便直接复制内容，也方便排查推理配置和资源占用。",
                )
            )

            with gr.Row(equal_height=False, elem_classes=["results-grid"]):
                with gr.Column(elem_classes=["result-stack"]):
                    with gr.Column(elem_classes=["result-card", "tech-panel"]):
                        gr.Markdown("### Transcript Preview")
                        preview = gr.Textbox(
                            lines=22,
                            interactive=False,
                            label="Preview",
                        )
                    with gr.Column(elem_classes=["download-card", "tech-panel"]):
                        gr.Markdown("### Download Artifacts")
                        download = gr.File(label="Generated file or zip bundle")

                with gr.Column(elem_classes=["result-stack"]):
                    with gr.Column(elem_classes=["result-card", "tech-panel"]):
                        gr.Markdown("### Runtime Meta")
                        meta = gr.JSON(label="timings / memory / model selection")

        def _on_scenario_change(name: str):
            bs, ms, ml, lang, itn, mv = _apply_scenario_defaults(name)
            return (
                gr.update(value=bs),
                gr.update(value=ms),
                gr.update(value=ml),
                gr.update(value=lang),
                gr.update(value=itn),
                gr.update(value=mv),
            )

        scenario.change(
            _on_scenario_change,
            inputs=[scenario],
            outputs=[batch_size_s, max_single_segment_ms, merge_length_s, language, use_itn, merge_vad],
        )

        start_btn.click(
            _run_batch,
            inputs=[
                files,
                scenario,
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
                out_fmt,
                split_lines,
            ],
            outputs=[preview, download, meta],
        )

    import argparse

    ap = argparse.ArgumentParser(description="funasr-lite web UI")
    ap.add_argument("--host", default=os.getenv("FUNASR_LITE_HOST", "127.0.0.1"), help="Bind host")
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FUNASR_LITE_PORT", "7860")),
        help="Bind port",
    )
    ap.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (default OFF for safety).",
    )
    args = ap.parse_args()

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

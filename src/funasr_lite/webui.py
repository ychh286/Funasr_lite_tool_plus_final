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


TECH_CSS = """
:root {
  --bg: #07111f;
  --bg-soft: #0b1b31;
  --panel: rgba(8, 22, 40, 0.78);
  --panel-strong: rgba(10, 28, 49, 0.92);
  --panel-border: rgba(112, 201, 255, 0.18);
  --text: #e9f5ff;
  --muted: #8aa5bf;
  --accent: #51e3ff;
  --accent-2: #4dd6a8;
  --accent-3: #86a8ff;
  --danger: #ff7f98;
  --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(77, 214, 168, 0.12), transparent 30%),
    radial-gradient(circle at top right, rgba(81, 227, 255, 0.16), transparent 32%),
    linear-gradient(180deg, #06111e 0%, #081624 45%, #07111f 100%);
  color: var(--text);
  font-family: "Segoe UI Variable Display", "Bahnschrift", "Microsoft YaHei UI", "Noto Sans SC", sans-serif;
}

.gradio-container {
  max-width: 1480px !important;
  padding: 28px 24px 48px !important;
}

.app-shell {
  position: relative;
}

.app-shell::before,
.app-shell::after {
  content: "";
  position: fixed;
  inset: auto;
  width: 440px;
  height: 440px;
  border-radius: 999px;
  filter: blur(80px);
  opacity: 0.22;
  pointer-events: none;
  z-index: 0;
}

.app-shell::before {
  top: -140px;
  left: -120px;
  background: rgba(81, 227, 255, 0.45);
}

.app-shell::after {
  right: -100px;
  bottom: -140px;
  background: rgba(77, 214, 168, 0.30);
}

.topbar,
.hero-panel,
.glass-card,
.result-card,
.metric-card,
.download-card {
  position: relative;
  z-index: 1;
  backdrop-filter: blur(18px);
  background: linear-gradient(180deg, rgba(9, 25, 44, 0.9), rgba(6, 17, 31, 0.82));
  border: 1px solid var(--panel-border);
  border-radius: 26px;
  box-shadow: var(--shadow);
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 24px;
  padding: 18px 24px;
  margin-bottom: 20px;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 16px;
}

.brand-badge {
  width: 56px;
  height: 56px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  color: #02111d;
  font-weight: 800;
  letter-spacing: 0.08em;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08), 0 18px 40px rgba(77, 214, 168, 0.24);
}

.brand-title {
  font-size: 1.55rem;
  font-weight: 760;
  line-height: 1.1;
}

.brand-subtitle,
.eyebrow,
.metric-label,
.section-kicker {
  color: var(--muted);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.brand-subtitle {
  margin-top: 4px;
  font-size: 0.83rem;
}

.status-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.status-pill,
.chip {
  padding: 10px 16px;
  border-radius: 999px;
  font-size: 0.92rem;
  color: #d7ecff;
  border: 1px solid rgba(127, 203, 255, 0.15);
  background: rgba(12, 33, 57, 0.7);
}

.status-pill strong,
.chip strong {
  color: #ffffff;
}

.hero-grid,
.results-grid {
  display: grid;
  gap: 20px;
}

.hero-grid {
  grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.85fr);
  margin-bottom: 20px;
}

.hero-panel {
  padding: 34px 34px 28px;
  overflow: hidden;
}

.hero-panel::before {
  content: "";
  position: absolute;
  inset: -10% auto auto 48%;
  width: 420px;
  height: 420px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(81, 227, 255, 0.18), transparent 60%);
  pointer-events: none;
}

.hero-title {
  margin: 10px 0 14px;
  max-width: 9.2em;
  font-size: clamp(2.7rem, 5vw, 4.6rem);
  line-height: 0.96;
  font-weight: 820;
  letter-spacing: -0.04em;
}

.hero-copy {
  max-width: 780px;
  color: #c7dff6;
  font-size: 1.06rem;
  line-height: 1.8;
}

.hero-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 24px;
}

.hero-aside {
  display: grid;
  gap: 18px;
}

.metric-card {
  padding: 22px 24px;
}

.metric-value {
  margin-top: 10px;
  font-size: 1.85rem;
  font-weight: 780;
  letter-spacing: -0.03em;
}

.metric-note {
  margin-top: 8px;
  color: #c1d5e8;
  font-size: 0.95rem;
  line-height: 1.6;
}

.section-banner {
  margin: 0 0 16px;
  padding: 0 4px;
}

.section-kicker {
  font-size: 0.8rem;
  margin-bottom: 8px;
}

.section-title {
  font-size: 1.65rem;
  font-weight: 760;
  letter-spacing: -0.03em;
}

.section-copy {
  margin-top: 8px;
  color: #9fb9d1;
  line-height: 1.75;
}

.glass-card {
  padding: 20px 20px 18px;
}

.glass-card > .gradio-markdown,
.result-card > .gradio-markdown {
  margin-bottom: 10px;
}

.card-title h3,
.card-title h2,
.card-title p {
  margin: 0;
}

.card-title h3 {
  font-size: 1.2rem;
  font-weight: 730;
}

.card-title p {
  margin-top: 8px;
  color: var(--muted);
  line-height: 1.65;
}

.form-row {
  gap: 18px !important;
}

.action-bar {
  margin: 18px 0 20px;
  padding: 18px 22px;
}

.action-copy {
  color: #9fb7cf;
  line-height: 1.7;
}

.result-card,
.download-card {
  padding: 22px 22px 18px;
}

.results-grid {
  grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.85fr);
  align-items: start;
}

.result-stack {
  display: grid;
  gap: 20px;
}

.tech-panel textarea,
.tech-panel pre,
.tech-panel .cm-editor,
.tech-panel .cm-scroller,
.tech-panel .wrap,
.tech-panel .message-wrap,
.tech-panel .scroll-hide {
  background: rgba(6, 17, 31, 0.6) !important;
  color: #ecf8ff !important;
}

.tech-panel textarea,
.tech-panel .cm-editor,
.tech-panel .json-container {
  border-radius: 18px !important;
  border: 1px solid rgba(107, 190, 255, 0.18) !important;
}

.tech-panel label,
.tech-panel .block-label,
.tech-panel .wrap label {
  color: #d7ecff !important;
  font-weight: 630 !important;
}

.gradio-container .gr-button-primary,
.gradio-container button.primary {
  background: linear-gradient(135deg, #56e0ff, #54ddb3) !important;
  color: #03111c !important;
  border: none !important;
  box-shadow: 0 16px 40px rgba(81, 227, 255, 0.28);
}

.gradio-container button:not(.primary),
.gradio-container .gr-button-secondary {
  background: rgba(12, 33, 57, 0.74) !important;
  color: #dcefff !important;
  border: 1px solid rgba(111, 189, 255, 0.16) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .message,
.gradio-container .container,
.gradio-container .form,
.gradio-container .dropdown,
.gradio-container .gr-box,
.gradio-container .gradio-dropdown,
.gradio-container .gradio-slider {
  border-color: rgba(111, 189, 255, 0.16) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .gr-box,
.gradio-container .gradio-dropdown {
  background: rgba(10, 25, 44, 0.82) !important;
  color: #eff8ff !important;
}

.gradio-container .gr-file,
.gradio-container .gr-files {
  border-radius: 20px !important;
}

.gradio-container .gr-accordion,
.gradio-container .gr-panel,
.gradio-container .gr-box {
  box-shadow: none !important;
}

@media (max-width: 1120px) {
  .hero-grid,
  .results-grid {
    grid-template-columns: 1fr;
  }

  .hero-title {
    max-width: none;
  }
}

@media (max-width: 760px) {
  .gradio-container {
    padding: 18px 14px 32px !important;
  }

  .topbar,
  .hero-panel,
  .glass-card,
  .result-card,
  .metric-card,
  .download-card {
    border-radius: 22px;
  }

  .topbar {
    flex-direction: column;
    align-items: flex-start;
  }

  .hero-panel {
    padding: 26px 22px 22px;
  }

  .hero-title {
    font-size: 2.4rem;
  }
}
"""


def _topbar_html() -> str:
    return """
    <div class="topbar">
      <div class="brand-lockup">
        <div class="brand-badge">FX</div>
        <div>
          <div class="brand-title">FunASR Lite Console</div>
          <div class="brand-subtitle">Local transcription, diarization, and export workflow</div>
        </div>
      </div>
      <div class="status-pills">
        <div class="status-pill"><strong>Local-first</strong>&nbsp; No cloud dependency</div>
        <div class="status-pill"><strong>Speaker V1</strong>&nbsp; Paraformer + CAM++</div>
        <div class="status-pill"><strong>Batch</strong>&nbsp; Audio / video in one run</div>
      </div>
    </div>
    """


def _hero_html() -> str:
    return """
    <div class="hero-panel">
      <div class="eyebrow">Control Surface</div>
      <div class="hero-title">Transcribe, separate, export.</div>
      <div class="hero-copy">
        为本地音频与视频转写设计的控制台。当前版本支持多文件转写、说话人区分 V1、
        Markdown / TXT 导出，以及适合后续大模型总结的结构化产物输出。
      </div>
      <div class="hero-chips">
        <div class="chip"><strong>Input</strong>&nbsp; wav / mp3 / mp4 / more</div>
        <div class="chip"><strong>Output</strong>&nbsp; transcript + speaker sidecars</div>
        <div class="chip"><strong>Runtime</strong>&nbsp; CPU or CUDA</div>
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
                        "### Ingest & Output\n"
                        "选择输入文件和导出格式。默认保留最适合后续总结的 Markdown 输出。"
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
                        "### Model Backbone\n"
                        "主模型、VAD 和标点后端。自定义 ID 一旦填写，就会覆盖预设值。"
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
                        "### Runtime & Decode\n"
                        "语言、设备和分段参数。说话人区分开关也放在这里，避免和模型设置割裂。"
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
                        "### Run Notes\n"
                        "Speaker diarization V1 适合会议、访谈、讨论录音。若是多人重叠说话严重或混响很重，发言人边界仍可能不稳。"
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
                        '<div class="action-copy">点击后会按当前配置执行批处理。若存在 speaker sidecar，会一并写入下载结果并在预览区展示结构化内容。</div>'
                    )
                with gr.Column(scale=3, min_width=220):
                    start_btn = gr.Button("Start transcription", variant="primary", size="lg")

            gr.HTML(
                _section_banner_html(
                    "Results",
                    "Inspect transcript and artifacts",
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

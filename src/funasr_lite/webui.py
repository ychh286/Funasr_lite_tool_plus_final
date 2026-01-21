from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import asdict

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
)
from .core import TranscribeResult, transcribe_file
from .formatting import postprocess_text, split_lines_by_punct


def _as_path_list(files: Any) -> List[Path]:
    """Normalize Gradio file payloads into a list of Paths."""
    if not files:
        return []

    # gr.Files typically returns a list
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
        # Gradio FileData-like
        # In newer Gradio versions, Files/UploadButton returns FileData objects.
        # The actual temporary file path is typically in `.path`, while `.name`
        # may be the original filename.
        path_like = getattr(f, "path", None) or getattr(f, "name", None)
        if path_like:
            paths.append(Path(str(path_like)))
            continue
        # Dict payload
        if isinstance(f, dict):
            for k in ("path", "name"):
                if k in f and f[k]:
                    paths.append(Path(str(f[k])))
                    break
            continue
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

    # Scenario is an opinionated default set; user inputs take precedence.
    # (We still keep it in meta for reproducibility.)

    resolved_model_id = _resolve_model_id(model_preset, model_custom)
    resolved_vad_id = _resolve_vad_id(vad_preset, vad_custom)
    resolved_punc_id = _resolve_punc_id(punc_preset, punc_custom)

    model_cfg = ModelConfig(
        model_id=resolved_model_id,
        vad_model_id=resolved_vad_id,
        punc_model_id=resolved_punc_id,
        device=None if device == "auto" else device,
        disable_update=True,
    )

    decode_cfg = DecodeConfig(
        language=language,
        use_itn=bool(use_itn),
        batch_size_s=int(batch_size_s),
        merge_vad=bool(merge_vad),
        merge_length_s=int(merge_length_s),
        max_single_segment_ms=int(max_single_segment_ms),
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

    # Prepare output payload
    tmpdir = Path(tempfile.mkdtemp(prefix="funasr_lite_batch_"))
    out_files: List[Path] = []

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
            err_msg = (str(e) or "").strip()
            if not err_msg:
                # Some exceptions have an empty string representation on Windows
                err_msg = repr(e)
            err = f"{err_type}: {err_msg}"
            # Older revisions used `verbose_meta`. Current config is `verbose`.
            if getattr(obs_cfg, "verbose", False):
                import traceback
                err += "\n" + traceback.format_exc()

            meta["results"].append({"file": p.name, "ok": False, "error": err})
            preview_parts.append(f"## {p.name}\n\n[ERROR] {err}\n")
            continue

        text = postprocess_text(r.text)
        if split_lines:
            text = split_lines_by_punct(text)

        # Preview
        if out_fmt == "md":
            preview_parts.append(f"## {p.name}\n\n{text}\n")
        else:
            preview_parts.append(f"[{p.name}]\n{text}\n")

        # Write per-file output
        safe_stem = p.stem
        suffix = ".md" if out_fmt == "md" else ".txt"
        out_path = tmpdir / f"{safe_stem}{suffix}"
        out_path.write_text(text + "\n", encoding="utf-8")
        out_files.append(out_path)

        meta["results"].append({
            "file": p.name,
            "ok": True,
            "meta": r.meta,
        })

    progress(1.0, desc="Packaging outputs")

    # Single file: return it directly; multi: zip
    download_path: Optional[str] = None
    if len(out_files) == 1:
        download_path = str(out_files[0])
    else:
        zip_path = tmpdir / "outputs.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in out_files:
                z.write(f, arcname=f.name)
        download_path = str(zip_path)

    preview = "\n\n".join(preview_parts).strip()
    return preview, download_path, meta


def main() -> None:
    scenario_names = list(SCENARIO_PRESETS.keys())
    model_preset_keys = list(MODEL_PRESETS.keys())
    vad_preset_keys = list(VAD_PRESETS.keys())
    punc_preset_keys = list(PUNC_PRESETS.keys())

    with gr.Blocks(title="funasr-lite - Local transcription") as demo:
        gr.Markdown(
            "# funasr-lite - Local transcription (FunASR)\n"
            "Upload/select one or multiple files, transcribe, then download .md/.txt."
        )

        with gr.Row():
            files = gr.Files(
                label="Audio/Video file(s)",
                file_types=["audio", "video"],
            )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("## Presets")
                scenario = gr.Dropdown(
                    choices=scenario_names,
                    value=DEFAULT_SCENARIO,
                    label="Scenario preset",
                )

            with gr.Column(scale=5):
                gr.Markdown("## Output")
                out_fmt = gr.Radio(["md", "txt"], value="md", label="Output format")
                split_lines = gr.Checkbox(
                    label="Split lines by punctuation (experimental)",
                    value=False,
                )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("## Model")
                model_preset = gr.Dropdown(
                    choices=model_preset_keys,
                    value="iic/SenseVoiceSmall",
                    label="Model preset",
                )
                model_custom = gr.Textbox(
                    label="Custom model id (optional)",
                    placeholder="If set, overrides preset",
                )

                vad_preset = gr.Dropdown(
                    choices=vad_preset_keys,
                    value="fsmn-vad",
                    label="VAD preset",
                )
                vad_custom = gr.Textbox(
                    label="Custom VAD model id (optional)",
                    placeholder="If set, overrides preset",
                )

                punc_preset = gr.Dropdown(
                    choices=punc_preset_keys,
                    value="none",
                    label="Punctuation preset (optional)",
                )
                punc_custom = gr.Textbox(
                    label="Custom punctuation model id (optional)",
                    placeholder="If set, overrides preset",
                )

            with gr.Column(scale=5):
                gr.Markdown("## Decode / Runtime")
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

                batch_size_s = gr.Slider(
                    1,
                    60,
                    value=10,
                    step=1,
                    label="batch_size_s (seconds)",
                )
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

        start_btn = gr.Button("Start transcription", variant="primary")

        gr.Markdown("## Text preview")
        # `show_copy_button` is not supported in some Gradio versions (e.g., 6.3.0).
        preview = gr.Textbox(lines=18, interactive=False)

        with gr.Row():
            download = gr.File(label="Download")

        gr.Markdown("## Meta (timings / memory)")
        meta = gr.JSON()

        # Scenario -> set opinionated defaults
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
    import os

    ap = argparse.ArgumentParser(description="funasr-lite web UI")
    ap.add_argument("--host", default=os.getenv("FUNASR_LITE_HOST", "127.0.0.1"), help="Bind host")
    ap.add_argument("--port", type=int, default=int(os.getenv("FUNASR_LITE_PORT", "7860")), help="Bind port")
    ap.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (default OFF for safety).",
    )
    args = ap.parse_args()

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

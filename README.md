# funasr-lite (CLI + WebGUI)

A small local transcription utility built on FunASR.

## Prerequisites
- Python >= 3.11
- `ffmpeg` available in PATH
- A working PyTorch installation (CPU or CUDA).  
  Note: this project does **not** pin `torch` because GPU vs CPU wheels differ. Install the torch build you want first.

## Install (uv recommended)
```bash
# inside this project folder
uv sync
```

If you do not have `uv`, you can use `pip`, but `uv` is recommended for reproducibility.

## Quick start: WebGUI
```bash
uv run funasr-lite-web
```
Open the local URL shown in terminal, upload/select an audio/video file, then download `.md` or `.txt`.

Optional (advanced):
```bash
# Bind to all network interfaces (LAN access)
uv run funasr-lite-web -- --host 0.0.0.0 --port 7860

# Create a Gradio public link (share=True)
uv run funasr-lite-web -- --share
```

## Quick start: CLI
```bash
uv run funasr-lite "path/to/audio.mp3" --format md --out-dir outputs
```

Examples:
```bash
# Choose a scenario preset (affects defaults for language / VAD merge / chunk sizes)
uv run funasr-lite "audio.mp3" --scenario cn_meeting

# Enable punctuation model
uv run funasr-lite "audio.mp3" --enable-punc --punc-preset punc-zh
```

## Diagnostics script (optional)
If you want detailed timing / GPU memory / cache hints:
```bash
uv run python tools/asr_test2_print.py
```

## Notes
- First run may download model weights into user cache (ModelScope / HuggingFace caches).
- If you hit OOM, try lowering:
  - `--batch-size-s` (e.g., 10 -> 5)
  - `--max-single-segment-ms` (e.g., 30000 -> 15000)


## torch模型
damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

## punc模型
damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
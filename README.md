# funasr-lite (CLI + WebGUI)

A local transcription utility built on FunASR, plus a project-local `yt-dlp -> ffmpeg -> transcript -> summary prompt` workflow layer.

## Prerequisites
- Python >= 3.11
- `ffmpeg` available in PATH
- A working PyTorch installation (CPU or CUDA)  
  Note: this project does **not** pin `torch` because GPU vs CPU wheels differ. Install the torch build you want first.

## Install (Windows, recommended)
Assume you are already in this project root and can see `pyproject.toml`.

### 1. Create the virtual environment (Python 3.11)
```powershell
uv venv .venv -p 3.11
```

### 2. Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Torch (GPU, cu128)
```powershell
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install FunASR and common dependencies
Without mirror:
```powershell
uv pip install -U funasr modelscope huggingface_hub soundfile gradio
```

With Tsinghua mirror:
```powershell
uv pip install -U funasr modelscope huggingface_hub soundfile gradio --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. Install this project into the environment
```powershell
uv pip install -e . --no-deps
```

## Quick start: WebGUI
```powershell
uv run --active funasr-lite-web
```
Open the local URL shown in terminal, upload/select an audio/video file, then download `.md` or `.txt`.

Windows one-click launch from project root:
```bat
Start-FunASRWebUI.cmd
```

When speaker diarization is enabled, the download bundle also includes:
- `*_speakers.md`
- `*_speakers.json`

### Windows: one-click launch + Start Menu shortcut (recommended)
After the environment is ready, run once:

```bat
scripts\install_startmenu_shortcut.bat
```

Then open **Start Menu** and run:

- `funasr-lite` -> `funasr-lite WebUI`

Notes:
- The WebUI binds to `127.0.0.1:7860` by default and auto-opens the browser.
- To stop the service, close the console window (or press Ctrl+C).
- Optional overrides (no file edits): set environment variables `FUNASR_LITE_HOST` and/or `FUNASR_LITE_PORT`.

Optional (advanced):
```powershell
# Bind to all network interfaces (LAN access)
uv run --active funasr-lite-web -- --host 0.0.0.0 --port 7860

# Create a Gradio public link (share=True)
uv run --active funasr-lite-web -- --share
```

## Quick start: CLI
```powershell
.\Run-FunASR.ps1 "D:\path\to\audio.mp3" --format md --out .\outputs\sample.md
```

Examples:
```powershell
# Choose a scenario preset (affects defaults for language / VAD merge / chunk sizes)
.\Run-FunASR.ps1 "D:\audio.mp3" --scenario "中文会议/普通录音"

# Enable punctuation model
.\Run-FunASR.ps1 "D:\audio.mp3" --punc "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

# Enable speaker diarization (V1)
# `--device auto` is the default and is usually the right choice.
.\Run-FunASR.ps1 "D:\audio.mp3" --speaker-diarization --device auto --out .\outputs\sample.md
```

## Quick start: Pipeline
The project-local pipeline wraps `yt-dlp`, `ffmpeg`, `transcribe_local.py`, transcript cleaning, and optional prompt selection.

```powershell
.\Run-FunASRPipeline.ps1 "D:\recordings\meeting.mp3" --tag "#组会"
```

```powershell
.\Run-FunASRPipeline.ps1 "https://www.youtube.com/watch?v=..." --tag "#技术"
```

```powershell
# Enable speaker diarization in the project-local pipeline
.\Run-FunASRPipeline.ps1 "D:\recordings\meeting.mp3" --speaker-diarization --speaker-count 0
```

Managed channel/library profiles can also be enabled via a local config file:

```powershell
# Use a locally configured profile name from workflow_profiles.local.json
.\Run-FunASRPipeline.ps1 "https://www.bilibili.com/video/BV..." --profile my_channel
```

Configuration notes:
- Copy `workflow_profiles.example.json` to `workflow_profiles.local.json`
- Fill in your local `download_root`, `note_root`, `output_root`, and `prompt_name`
- `workflow_profiles.local.json` is ignored by git and intended for owner-local routing rules

When a profile is enabled, the pipeline will:
- keep the downloaded media in the profile's library folder instead of discarding it
- prefer the profile-specific prompt template
- write a recommended final summary note path into both `summary_prompt.md` and `run_manifest.json`
- store transcript artifacts under the profile's `_workflow_outputs\` directory

Outputs go to `.\outputs\transcripts\<timestamp>\`:
- `audio.wav`
- `transcript.md`
- `transcript.txt`
- `transcript_speakers.md` when speaker diarization is enabled
- `transcript_speakers.json` when speaker diarization is enabled
- `summary_prompt.md` when a matching prompt template is resolved

With a managed profile, outputs go to the profile-specific workflow directory configured in `workflow_profiles.local.json`.

The preferred cookies location for `yt-dlp` is `.\secrets\cookies\all_cookies.txt`.
If it is missing, the pipeline falls back to `%USERPROFILE%\.openclaw\secrets\cookies\all_cookies.txt` and `~/.openclaw/secrets/cookies/all_cookies.txt`.

## Diagnostics script (optional)
If you want detailed timing / GPU memory / cache hints:
```powershell
uv run python tools/asr_test2_print.py
```

## Notes
- First run may download model weights into user cache (ModelScope / HuggingFace caches).
- Speaker diarization V1 automatically switches to a timestamp-capable Paraformer model plus `ct-punc-c` and `cam++`.
- V1 is aimed at single-channel recordings where speakers mostly take turns. Heavy overlap is still a known weak spot.
- If you hit OOM, try lowering:
  - `--batch-size-s` (e.g., 10 -> 5)
  - `--max-single-segment-ms` (e.g., 30000 -> 15000)

## Torch model
damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

## Punctuation model
damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch

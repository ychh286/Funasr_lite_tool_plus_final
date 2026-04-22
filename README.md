# funasr-lite (CLI + Audio Workbench V2)

A local transcription utility built on FunASR, plus a project-local `yt-dlp -> ffmpeg -> transcript -> prompt -> LLM summary -> Markdown` workflow layer.

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

## Quick start: Audio Workbench V2

The default WebUI is now **Audio Workbench V2**. It can run the complete local workflow:

```text
URL / local media
  -> yt-dlp download when URL is provided
  -> ffmpeg 16k mono WAV preprocessing
  -> FunASR transcription
  -> selectable project prompts + uploaded one-off prompts + inline instruction
  -> optional OpenAI-compatible LLM call
  -> final_summary.md + transcript artifacts + run_manifest.json + workflow_outputs.zip
```

Launch from the activated project environment:

```powershell
uv run --active funasr-lite-web
```

Windows one-click launch from project root:

```bat
Start-FunASRWebUI.cmd
```

The old quick transcription WebUI is still available:

```powershell
.\Start-FunASRWebUI.ps1 -Legacy
# or
uv run --active funasr-lite-web-legacy
```

Typical V2 outputs go to `outputs/workbench_runs/<timestamp>/` unless a profile or UI path overrides them:

- `audio_16k_mono.wav`
- `transcript.md`
- `transcript.txt`
- `transcript_speakers.md` / `transcript_speakers.json` when speaker diarization is enabled
- `summary_prompt.md`
- `final_summary.md`
- `run_manifest.json`
- `workflow_outputs.zip`

Cookies are used only at runtime for `yt-dlp`; generated zip bundles intentionally exclude runtime cookie copies.

See `docs/audio_workbench_v2_design.md` for the design and implementation notes.

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
- store transcript artifacts under the profile's configured cache/output directory, separate from the media library when desired

If the input is a URL, the pipeline can also auto-match a locally configured profile when the URL contains one of that profile's configured `url_patterns`.

For managed libraries that already use sequence naming, the pipeline also tries to rename freshly downloaded media into:
- `NN - 标题 [BV号].ext`

before generating the suggested final summary note path.

Outputs go to `.\outputs\transcripts\<timestamp>\`:
- `audio.wav`
- `transcript.md`
- `transcript.txt`
- `transcript_speakers.md` when speaker diarization is enabled
- `transcript_speakers.json` when speaker diarization is enabled
- `summary_prompt.md` when a matching prompt template is resolved

With a managed profile, outputs go to the profile-specific workflow directory configured in `workflow_profiles.local.json`.

The preferred Windows cookies location for `yt-dlp` is:
- `E:\Software_tools\yt-dlp\cookies\bili_cookies.txt` for Bilibili
- `E:\Software_tools\yt-dlp\cookies\youtube_cookies.txt` for YouTube
- `E:\Software_tools\yt-dlp\cookies\patreon_cookies.txt` for Patreon
- `E:\Software_tools\yt-dlp\cookies\cookies.txt` as a generic fallback

Optional override:
- `FUNASR_LITE_COOKIE_FILE`
- `YTDLP_COOKIE_FILE`

Project-local `.\secrets\cookies\all_cookies.txt` is only kept as a final fallback for compatibility.

## Audio Workbench V2 configuration notes

Prompt discovery order:

- `.codex/agents.md` is selected by default when present.
- `prompts/*.md` are listed in the UI and can be multi-selected.
- Uploaded `.md/.txt` prompt files are included only for that run.
- The final inline textbox is appended as the run-specific instruction.

OpenAI-compatible LLM settings can be provided in the UI or via environment variables:

```powershell
$env:FUNASR_LITE_LLM_BASE_URL = "http://127.0.0.1:4000/v1"
$env:FUNASR_LITE_LLM_MODEL = "gpt-5.4"
$env:FUNASR_LITE_LLM_API_KEY_ENV = "LITELLM_API_KEY"
$env:LITELLM_API_KEY = "..."
```

External tool lookup order for V2:

- explicit UI path
- `FUNASR_LITE_YTDLP` / `YTDLP_BIN` and `FUNASR_LITE_FFMPEG` / `FFMPEG_BINARY` / `IMAGEIO_FFMPEG_EXE`
- project-local `.local_tools`
- common Windows paths such as `E:\Software_tools\yt-dlp\yt-dlp.exe` and `E:\Software_tools\ffmpeg\bin\ffmpeg.exe`
- PATH

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

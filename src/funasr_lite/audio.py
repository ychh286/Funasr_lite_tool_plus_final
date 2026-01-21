\
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile


class FfmpegNotFoundError(RuntimeError):
    pass


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found in PATH. Please install ffmpeg and ensure `ffmpeg` is available in your terminal."
        )


def ensure_wav_16k_mono(src: Path) -> Path:
    """
    Convert any audio/video file to 16kHz mono WAV (PCM).
    Returns a temp wav path.
    """
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    ensure_ffmpeg_available()

    tmp_wav = Path(tempfile.gettempdir()) / f"{src.stem}_16k_mono.wav"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-vn", str(tmp_wav)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        err = err[-1500:] if len(err) > 1500 else err
        raise RuntimeError(f"ffmpeg conversion failed:\n{err}")
    return tmp_wav

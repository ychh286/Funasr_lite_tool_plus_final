\
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def sync_cuda_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_mem_stats(device: str) -> Dict[str, Any]:
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return {"gpu": None}
    idx = torch.cuda.current_device()
    return {
        "gpu": torch.cuda.get_device_name(idx),
        "allocated": torch.cuda.memory_allocated(idx),
        "reserved": torch.cuda.memory_reserved(idx),
        "max_allocated": torch.cuda.max_memory_allocated(idx),
        "max_reserved": torch.cuda.max_memory_reserved(idx),
    }


def process_rss_bytes() -> Optional[int]:
    """
    Return process RSS if psutil is installed, else None.
    This is optional and should not break the tool.
    """
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss)
    except Exception:
        return None


def candidate_cache_dirs() -> Iterable[Path]:
    home = Path.home()

    env_candidates = []
    for k in ["MODELSCOPE_CACHE", "HF_HOME", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME"]:
        v = os.environ.get(k)
        if v:
            env_candidates.append(Path(v))

    defaults = [
        home / ".cache" / "modelscope",
        home / ".cache" / "huggingface",
        home / ".cache" / "funasr",
    ]

    if platform.system().lower().startswith("win"):
        defaults += [
            home / "AppData" / "Local" / "modelscope",
            home / "AppData" / "Local" / "huggingface",
        ]

    seen = set()
    for d in env_candidates + defaults:
        d = d.expanduser()
        if str(d) not in seen:
            seen.add(str(d))
            yield d


def dir_size_bytes(path: Path, max_files: int = 200_000) -> Optional[int]:
    if not path.exists():
        return None
    total = 0
    cnt = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
                cnt += 1
                if cnt >= max_files:
                    break
        return total
    except Exception:
        return None

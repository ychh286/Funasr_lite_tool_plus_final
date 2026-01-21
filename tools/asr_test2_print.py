from __future__ import annotations

from pathlib import Path
import os
import platform
import subprocess
import tempfile
import time
from typing import Optional, Dict, Any, Iterable, Tuple

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


# ---------------------------
# 观测工具：时间、显存、内存、缓存目录
# ---------------------------

def _human_bytes(n: int) -> str:
    """把字节数转成人类可读格式。"""
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def _sync_cuda_if_needed(device: str) -> None:
    """为了让计时更准确：GPU 异步执行需要 synchronize。"""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem_stats(device: str) -> Dict[str, Any]:
    """返回当前 GPU 显存统计。"""
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


def _print_gpu_mem(title: str, device: str) -> None:
    s = _gpu_mem_stats(device)
    print(f"\n[{title}] GPU memory:")
    if s["gpu"] is None:
        print("  (no cuda) running on CPU")
        return
    print(f"  gpu          : {s['gpu']}")
    print(f"  allocated    : {_human_bytes(int(s['allocated']))}")
    print(f"  reserved     : {_human_bytes(int(s['reserved']))}")
    print(f"  max_allocated: {_human_bytes(int(s['max_allocated']))}")
    print(f"  max_reserved : {_human_bytes(int(s['max_reserved']))}")


def _try_process_mem() -> Optional[int]:
    """
    进程内存（RSS）观测：可选依赖 psutil。
    没有 psutil 也不影响运行，只是少一个指标。
    """
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss)
    except Exception:
        return None


def _print_proc_mem(title: str) -> None:
    rss = _try_process_mem()
    print(f"\n[{title}] Process memory:")
    if rss is None:
        print("  psutil not installed -> skip RSS (optional: `uv add psutil`)")
    else:
        print(f"  RSS: {_human_bytes(rss)}")


def _candidate_cache_dirs() -> Iterable[Path]:
    """
    模型权重通常不在 site-packages，而在用户缓存目录。
    FunASR 常与 ModelScope/HF 缓存机制联动，所以这里给一组候选目录。
    """
    home = Path.home()
    # 环境变量线索（如果你自己设置过）
    env_candidates = []
    for k in ["MODELSCOPE_CACHE", "HF_HOME", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME"]:
        v = os.environ.get(k)
        if v:
            env_candidates.append(Path(v))

    # 常见默认目录
    defaults = [
        home / ".cache" / "modelscope",
        home / ".cache" / "huggingface",
        home / ".cache" / "funasr",
    ]

    # Windows 常见补充
    if platform.system().lower().startswith("win"):
        defaults += [
            home / "AppData" / "Local" / "modelscope",
            home / "AppData" / "Local" / "huggingface",
        ]

    # 去重
    seen = set()
    for d in env_candidates + defaults:
        d = d.expanduser()
        if str(d) not in seen:
            seen.add(str(d))
            yield d


def _dir_size_bytes(path: Path, max_files: int = 200_000) -> Optional[int]:
    """
    粗略统计目录大小。大缓存目录可能很大，统计会有开销，所以设置 max_files 上限。
    """
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
                    # 超过上限就不继续扫，避免卡太久
                    break
        return total
    except Exception:
        return None


def _print_cache_status(title: str, model_hint: str = "SenseVoice") -> None:
    print(f"\n[{title}] Cache directories (candidates):")
    for d in _candidate_cache_dirs():
        if d.exists():
            size = _dir_size_bytes(d)
            size_str = _human_bytes(size) if size is not None else "(size unknown)"
            print(f"  EXISTS: {d}  | approx size: {size_str}")
        else:
            print(f"  MISSING: {d}")
    print("  Hint: after first run, the directory that grows is typically where weights are cached.")
    print(f"  Hint: you can search in the above dirs for keywords like '{model_hint}' / 'SenseVoice' / 'fsmn-vad'.")


# ---------------------------
# 音频预处理
# ---------------------------

def ensure_wav_16k_mono(src: Path) -> Path:
    """把任意音频转成 16kHz 单声道 wav（PCM），返回临时 wav 路径。"""
    if not src.exists():
        raise FileNotFoundError(f"找不到音频文件：{src}")

    tmp_wav = Path(tempfile.gettempdir()) / f"{src.stem}_16k_mono.wav"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-vn", str(tmp_wav)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败：\n{p.stderr}")
    return tmp_wav


# ---------------------------
# 主流程（加观测）
# ---------------------------

def main():
    here = Path(__file__).resolve().parent
    audio_path = here / "audio" / "Gaoning20251212-2.mp3"
    print("Input audio:", audio_path)

    # 观测：运行前缓存状态
    _print_cache_status("PRE-RUN cache snapshot", model_hint="SenseVoiceSmall")

    t0 = time.perf_counter()
    wav_path = ensure_wav_16k_mono(audio_path)
    t1 = time.perf_counter()
    print("WAV (tmp):", wav_path)
    print(f"[TIMING] ffmpeg preprocess: {(t1 - t0):.3f}s")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 观测：模型初始化前资源
    _print_proc_mem("BEFORE model init")
    _print_gpu_mem("BEFORE model init", device)

    # 为了让“峰值显存统计”更干净
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------
    # Step 3：权重加载（AutoModel 初始化）
    # ---------------------------
    t2 = time.perf_counter()
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 120000},  # 30s 一段；若仍 OOM 改 15000
        device=device,
        disable_update=True,
    )
    _sync_cuda_if_needed(device)
    t3 = time.perf_counter()
    print(f"[TIMING] AutoModel init (load+prepare): {(t3 - t2):.3f}s")

    # 观测：模型初始化后资源（这一步最能反映“权重加载占了多少显存/内存”）
    _print_proc_mem("AFTER model init")
    _print_gpu_mem("AFTER model init", device)

    # ---------------------------
    # Step 4：识别推理（generate）
    # ---------------------------
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t4 = time.perf_counter()
    with torch.inference_mode():
        res = model.generate(
            input=[str(wav_path)],
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,      # 先用 10 更稳（60 不是必然安全）
            merge_vad=True,
            merge_length_s=30,
        )
    _sync_cuda_if_needed(device)
    t5 = time.perf_counter()
    print(f"[TIMING] generate() inference: {(t5 - t4):.3f}s")

    # 观测：推理后资源（峰值显存尤其关键）
    _print_proc_mem("AFTER generate")
    _print_gpu_mem("AFTER generate (peak included)", device)

    # 后处理
    t6 = time.perf_counter()
    text = rich_transcription_postprocess(res[0]["text"])
    t7 = time.perf_counter()
    print(f"[TIMING] postprocess: {(t7 - t6):.3f}s")

    print("\n=== ASR Result ===")
    # print(text)

    # 写 Markdown
    out_dir = here / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{wav_path.stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ASR 转录：{wav_path.name}\n\n")
        f.write(text.strip() + "\n")

    print("Saved markdown:", md_path)

    # 观测：运行后缓存状态（第一次跑通常会看到某个 cache dir 变大）
    _print_cache_status("POST-RUN cache snapshot", model_hint="SenseVoiceSmall")


if __name__ == "__main__":
    main()

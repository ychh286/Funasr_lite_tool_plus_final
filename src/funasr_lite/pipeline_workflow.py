from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (
    DecodeConfig,
    MODEL_PRESETS,
    PUNC_PRESETS,
    SCENARIO_PRESETS,
    VAD_PRESETS,
    ModelConfig,
    ObsConfig,
    OutputConfig,
    apply_speaker_diarization_defaults,
    validate_speaker_diarization_config,
)
from .core import TranscribeResult, transcribe_file
from .formatting import postprocess_text, split_lines_by_punct
from .llm_client import LLMSettings, OpenAICompatibleClient
from .prompting import (
    clean_transcript_text,
    compose_summary_prompt,
    load_selected_prompts,
    load_uploaded_prompt_files,
    render_final_markdown,
    sanitize_filename,
    unique_path,
)
from .save import render_speaker_content, save_json, save_speaker_text, save_text
from .workflow_profiles import (
    WorkflowProfile,
    build_managed_media_path,
    build_suggested_note_path,
    detect_workflow_profile_from_input,
    get_workflow_profile,
)


ROOT = Path(__file__).resolve().parents[2]
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v",
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus",
}
ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class WorkflowPaths:
    output_dir: Path
    source_media_path: Path
    wav_path: Path
    transcript_md: Path
    transcript_txt: Path
    transcript_speakers_md: Path | None = None
    transcript_speakers_json: Path | None = None
    summary_prompt: Path | None = None
    final_markdown: Path | None = None
    note_path: Path | None = None
    bundle_zip: Path | None = None


@dataclass
class AudioWorkflowResult:
    preview: str
    summary_preview: str
    final_markdown_path: str
    bundle_path: str
    manifest_path: str
    meta: dict[str, Any] = field(default_factory=dict)


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", (value or "").strip(), re.IGNORECASE))


def _report(progress: ProgressFn | None, pct: float, message: str) -> None:
    if progress:
        progress(float(pct), message)


def run_command(cmd: list[str], *, cwd: Path | None = None, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        detail = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
        raise RuntimeError(f"Command failed ({exc.returncode}): {' '.join(cmd)}\n{detail}") from exc


def _path_candidates_for_exe(name: str) -> list[Path]:
    exe_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    return [
        ROOT / ".local_tools" / name / exe_name,
        ROOT / ".local_tools" / name / "bin" / exe_name,
        ROOT / ".local_tools" / exe_name,
    ]


def resolve_executable(
    name: str,
    *,
    explicit_path: str = "",
    env_names: tuple[str, ...] = (),
    extra_candidates: list[str | Path] | None = None,
) -> str:
    raw = (explicit_path or "").strip().strip('"')
    if raw:
        p = Path(raw).expanduser()
        if p.exists():
            return str(p)
        found = shutil.which(raw)
        if found:
            return found
        raise RuntimeError(f"{name} path does not exist: {raw}")

    for env_name in env_names:
        env_value = (os.getenv(env_name, "") or "").strip().strip('"')
        if not env_value:
            continue
        p = Path(env_value).expanduser()
        if p.exists():
            return str(p)
        found = shutil.which(env_value)
        if found:
            return found

    candidates: list[Path] = []
    for item in extra_candidates or []:
        candidates.append(Path(item).expanduser())
    candidates.extend(_path_candidates_for_exe(name))
    if name == "ffmpeg":
        candidates.extend([
            ROOT / ".local_tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
            Path(r"E:\Software_tools\ffmpeg\bin\ffmpeg.exe"),
        ])
    if name == "yt-dlp":
        candidates.extend([
            ROOT / ".local_tools" / "yt-dlp" / "yt-dlp.exe",
            ROOT / ".local_tools" / "yt-dlp.exe",
            Path(r"E:\Software_tools\yt-dlp\yt-dlp.exe"),
        ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        found = shutil.which(f"{name}.exe")
        if found:
            return found
    raise RuntimeError(f"Cannot find {name}. Put it on PATH, set the explicit path in UI, or place it under .local_tools.")


def ffmpeg_location_for_ytdlp(ffmpeg_bin: str) -> str:
    p = Path(ffmpeg_bin)
    if p.name.lower() in {"ffmpeg.exe", "ffmpeg"}:
        return str(p.parent)
    return str(p)


def site_cookie_candidates(input_value: str) -> list[Path]:
    raw = (input_value or "").lower()
    cookies_roots = [ROOT / "secrets" / "cookies", Path(r"E:\Software_tools\yt-dlp\cookies")]
    if "bilibili.com" in raw or "b23.tv" in raw:
        names = ["bili_cookies.txt", "cookies.txt", "all_cookies.txt"]
    elif "youtube.com" in raw or "youtu.be" in raw:
        names = ["youtube_cookies.txt", "cookies.txt", "all_cookies.txt"]
    elif "patreon.com" in raw:
        names = ["patreon_cookies.txt", "cookies.txt", "all_cookies.txt"]
    else:
        names = ["cookies.txt", "all_cookies.txt"]
    candidates: list[Path] = []
    for root in cookies_roots:
        candidates.extend(root / name for name in names)
    return candidates


def _as_path_list(files: object) -> list[Path]:
    if not files:
        return []
    if isinstance(files, (str, Path)):
        return [Path(str(files))]
    paths: list[Path] = []
    try:
        iterator = iter(files)  # type: ignore[arg-type]
    except TypeError:
        iterator = iter([files])
    for item in iterator:
        if item is None:
            continue
        if isinstance(item, Path):
            paths.append(item)
        elif isinstance(item, str):
            paths.append(Path(item))
        else:
            path_like = getattr(item, "path", None) or getattr(item, "name", None)
            if path_like:
                paths.append(Path(str(path_like)))
            elif isinstance(item, dict):
                for key in ("path", "name"):
                    if item.get(key):
                        paths.append(Path(str(item[key])))
                        break
    return paths


def resolve_cookie_file(input_value: str, *, cookie_mode: str = "auto", cookie_path: str = "", uploaded_cookie: object = None) -> Path | None:
    mode = (cookie_mode or "auto").strip().lower()
    if mode in {"none", "不使用"}:
        return None

    explicit_values: list[str] = []
    if cookie_path:
        explicit_values.append(cookie_path)
    explicit_values.extend(str(path) for path in _as_path_list(uploaded_cookie))
    for raw in explicit_values:
        p = Path(raw).expanduser()
        if p.exists() and p.is_file():
            return p

    if mode in {"upload/path", "上传或路径", "manual"}:
        raise RuntimeError("Cookie mode is upload/path, but no readable cookie file was provided.")

    for env_name in ("FUNASR_LITE_COOKIE_FILE", "YTDLP_COOKIE_FILE"):
        raw = (os.getenv(env_name, "") or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.exists() and p.is_file():
                return p
    for candidate in site_cookie_candidates(input_value):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def snapshot_cookie_file(cookie_file: Path, runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    suffix = cookie_file.suffix or ".txt"
    runtime_path = runtime_dir / f"{sanitize_filename(cookie_file.stem, 'cookies')}.runtime{suffix}"
    shutil.copy2(cookie_file, runtime_path)
    return runtime_path


def _latest_media_candidate(target_dir: Path, before_ts: float = 0.0) -> Path | None:
    candidates = [
        p for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS and p.stat().st_mtime >= before_ts and "_runtime_cookies" not in p.parts
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def predict_ytdlp_path(cmd: list[str]) -> Path | None:
    try:
        cp = run_command(cmd, capture_output=True)
    except Exception:
        return None
    lines = [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    return Path(lines[-1])


def download_url_media(
    url: str,
    out_dir: Path,
    *,
    download_root: Path | None = None,
    download_mode: str = "audio",
    cookie_mode: str = "auto",
    cookie_path: str = "",
    uploaded_cookie: object = None,
    ytdlp_path: str = "",
    ffmpeg_path: str = "",
) -> tuple[Path, dict[str, Any]]:
    target_dir = download_root or out_dir / "downloaded_media"
    target_dir.mkdir(parents=True, exist_ok=True)
    ytdlp_bin = resolve_executable("yt-dlp", explicit_path=ytdlp_path, env_names=("FUNASR_LITE_YTDLP", "YTDLP_BIN"))
    ffmpeg_bin = resolve_executable("ffmpeg", explicit_path=ffmpeg_path, env_names=("FUNASR_LITE_FFMPEG", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"))
    cookie_file = resolve_cookie_file(url, cookie_mode=cookie_mode, cookie_path=cookie_path, uploaded_cookie=uploaded_cookie)
    cookie_runtime: Path | None = None
    cookie_runtime_root: tempfile.TemporaryDirectory[str] | None = None
    if cookie_file:
        cookie_runtime_root = tempfile.TemporaryDirectory(prefix="funasr_lite_cookie_runtime_")
        cookie_runtime = snapshot_cookie_file(cookie_file, Path(cookie_runtime_root.name))

    source_tpl = target_dir / "%(title)s [%(id)s].%(ext)s"
    archive_path = target_dir / ".yt-dlp-archive.txt"
    common = [
        ytdlp_bin, "--newline", "--no-playlist", "--windows-filenames", "--trim-filenames", "180",
        "--download-archive", str(archive_path), "-o", str(source_tpl),
        "--ffmpeg-location", ffmpeg_location_for_ytdlp(ffmpeg_bin),
    ]
    predict = [
        ytdlp_bin, "--no-playlist", "--windows-filenames", "--trim-filenames", "180",
        "-o", str(source_tpl), "--get-filename", "--ffmpeg-location", ffmpeg_location_for_ytdlp(ffmpeg_bin),
    ]
    mode = (download_mode or "audio").strip().lower()
    if mode == "audio":
        common.extend(["-x", "--audio-format", "wav", "--audio-quality", "0"])
        predict.extend(["-x", "--audio-format", "wav", "--audio-quality", "0"])
    else:
        common.extend(["--merge-output-format", "mp4"])
        predict.extend(["--merge-output-format", "mp4"])
    if cookie_runtime:
        common.extend(["--cookies", str(cookie_runtime)])
        predict.extend(["--cookies", str(cookie_runtime)])
    predict.append(url)
    common.append(url)

    before_ts = datetime.now().timestamp() - 2
    predicted = predict_ytdlp_path(predict)
    try:
        cp = run_command(common, capture_output=True)
        output = "\n".join(part for part in [cp.stdout or "", cp.stderr or ""] if part)
        if predicted and predicted.exists():
            media = predicted
        else:
            media = _latest_media_candidate(target_dir, before_ts=before_ts) or _latest_media_candidate(target_dir, before_ts=0.0)
            if not media:
                raise RuntimeError("yt-dlp completed, but no media file was found in the target directory.")
    finally:
        if cookie_runtime_root is not None:
            cookie_runtime_root.cleanup()
    meta = {
        "tool": "yt-dlp", "yt_dlp": ytdlp_bin, "ffmpeg_for_ytdlp": ffmpeg_bin,
        "download_mode": mode, "download_root": str(target_dir), "archive_path": str(archive_path),
        "cookie_used": bool(cookie_runtime), "cookie_source": str(cookie_file) if cookie_file else "",
        "cookie_runtime": str(cookie_runtime) if cookie_runtime else "", "log_tail": output[-6000:],
    }
    return media, meta


def resolve_input_media(
    *, source_url: str, uploaded_media: object, out_dir: Path, download_root: Path | None = None,
    download_mode: str = "audio", cookie_mode: str = "auto", cookie_path: str = "",
    uploaded_cookie: object = None, ytdlp_path: str = "", ffmpeg_path: str = "",
) -> tuple[Path, dict[str, Any], str]:
    source_url = (source_url or "").strip()
    if source_url:
        media, meta = download_url_media(
            source_url, out_dir, download_root=download_root, download_mode=download_mode,
            cookie_mode=cookie_mode, cookie_path=cookie_path, uploaded_cookie=uploaded_cookie,
            ytdlp_path=ytdlp_path, ffmpeg_path=ffmpeg_path,
        )
        return media, meta, source_url
    media_paths = _as_path_list(uploaded_media)
    if not media_paths:
        raise RuntimeError("请提供一个视频/音频链接，或上传一个本地媒体文件。")
    media = media_paths[0].expanduser().resolve()
    if not media.exists():
        raise RuntimeError(f"Uploaded/local media not found: {media}")
    return media, {"tool": "local_upload"}, str(media)


def preprocess_to_wav(media: Path, out_dir: Path, *, ffmpeg_path: str = "") -> tuple[Path, dict[str, Any]]:
    ffmpeg_bin = resolve_executable("ffmpeg", explicit_path=ffmpeg_path, env_names=("FUNASR_LITE_FFMPEG", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"))
    wav_path = out_dir / "audio_16k_mono.wav"
    run_command([ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(media), "-ac", "1", "-ar", "16000", str(wav_path)], capture_output=True)
    return wav_path, {"tool": "ffmpeg", "ffmpeg": ffmpeg_bin, "wav_path": str(wav_path)}


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


def transcribe_prepared_wav(
    wav_path: Path, out_dir: Path, *, scenario_name: str, model_preset: str, model_custom: str,
    vad_preset: str, vad_custom: str, punc_preset: str, punc_custom: str, language: str, device: str,
    use_itn: bool, merge_vad: bool, speaker_diarization: bool, speaker_count: int, verbose: bool,
    batch_size_s: int, max_single_segment_ms: int, merge_length_s: int, split_lines: bool,
) -> tuple[str, str, Path, Path, Path | None, Path | None, dict[str, Any]]:
    model_cfg = ModelConfig(
        model_id=_resolve_model_id(model_preset, model_custom),
        vad_model_id=_resolve_vad_id(vad_preset, vad_custom),
        punc_model_id=_resolve_punc_id(punc_preset, punc_custom),
        device=None if device == "auto" else device,
        disable_update=True,
    )
    if speaker_diarization:
        model_cfg = apply_speaker_diarization_defaults(model_cfg, allow_auto_model_swap=not bool(model_custom))
        validate_speaker_diarization_config(model_cfg)
    decode_cfg = DecodeConfig(
        language=language, use_itn=bool(use_itn), batch_size_s=int(batch_size_s), merge_vad=bool(merge_vad),
        merge_length_s=int(merge_length_s), max_single_segment_ms=int(max_single_segment_ms),
        preset_spk_num=int(speaker_count) if int(speaker_count) > 0 else None,
    )
    output_cfg = OutputConfig(fmt="md", split_lines=bool(split_lines))
    obs_cfg = ObsConfig(verbose=bool(verbose))
    result: TranscribeResult = transcribe_file(str(wav_path), model_cfg=model_cfg, decode_cfg=decode_cfg, output_cfg=output_cfg, obs_cfg=obs_cfg)
    text = postprocess_text(result.text)
    if split_lines:
        text = split_lines_by_punct(text)
    transcript_md = out_dir / "transcript.md"
    transcript_txt = out_dir / "transcript.txt"
    save_text(text, transcript_md, fmt="md", split_lines=split_lines, source_name=wav_path.stem)
    transcript_txt.write_text(clean_transcript_text(transcript_md.read_text(encoding="utf-8")), encoding="utf-8")
    speaker_text = ""
    speaker_md: Path | None = None
    speaker_json: Path | None = None
    if speaker_diarization and result.segments:
        speaker_md = out_dir / "transcript_speakers.md"
        speaker_json = out_dir / "transcript_speakers.json"
        speaker_text = render_speaker_content(result.segments, fmt="md", source_name=wav_path.stem).strip()
        save_speaker_text(result.segments, speaker_md, fmt="md", source_name=wav_path.stem)
        save_json(result.segments, speaker_json)
    meta = {"scenario": scenario_name, "model_cfg": asdict(model_cfg), "decode_cfg": asdict(decode_cfg), "output_cfg": asdict(output_cfg), "transcribe_meta": result.meta}
    return text, speaker_text, transcript_md, transcript_txt, speaker_md, speaker_json, meta


def resolve_profile(profile_key: str, source_value: str) -> WorkflowProfile | None:
    key = (profile_key or "").strip()
    if key and key.lower() not in {"auto", "自动匹配"}:
        return get_workflow_profile(key)
    return detect_workflow_profile_from_input(source_value)


def _path_or_none(raw: str) -> Path | None:
    raw = (raw or "").strip().strip('"')
    if not raw:
        return None
    return Path(raw).expanduser()


def build_bundle(out_dir: Path, files: list[Path]) -> Path:
    zip_path = out_dir / "workflow_outputs.zip"
    seen: set[Path] = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if not path or not path.exists() or not path.is_file():
                continue
            if "_runtime_cookies" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                arcname = path.relative_to(out_dir)
            except ValueError:
                arcname = Path(path.name)
            z.write(path, arcname=str(arcname))
    return zip_path


def run_audio_workflow(
    *,
    source_url: str = "", uploaded_media: object = None, profile_key: str = "auto", tag: str = "",
    selected_prompt_labels: list[str] | None = None, uploaded_prompt_files: object = None, inline_prompt: str = "",
    output_root: str = "", download_root: str = "", note_root: str = "", download_mode: str = "audio",
    keep_download: bool = True, cookie_mode: str = "auto", cookie_path: str = "", uploaded_cookie: object = None,
    ytdlp_path: str = "", ffmpeg_path: str = "", scenario_name: str = "中文会议/普通录音",
    model_preset: str = "paraformer-zh (offline)", model_custom: str = "", vad_preset: str = "fsmn-vad", vad_custom: str = "",
    punc_preset: str = "none", punc_custom: str = "", language: str = "auto", device: str = "auto",
    use_itn: bool = True, merge_vad: bool = True, speaker_diarization: bool = False, speaker_count: int = 0,
    verbose: bool = False, batch_size_s: int = 10, max_single_segment_ms: int = 30000, merge_length_s: int = 15,
    split_lines: bool = True, run_llm: bool = False, llm_base_url: str = "http://127.0.0.1:4000/v1",
    llm_model: str = "gpt-5.4", llm_api_key: str = "", llm_api_key_env: str = "LITELLM_API_KEY",
    llm_temperature: float = 0.1, llm_timeout_seconds: int = 180, include_prompt_in_final: bool = False,
    include_transcript_in_final: bool = False, write_note: bool = False, overwrite_note: bool = False,
    progress: ProgressFn | None = None,
) -> AudioWorkflowResult:
    _report(progress, 0.01, "初始化运行目录")
    source_probe = (source_url or "").strip()
    if not source_probe:
        uploaded_paths = _as_path_list(uploaded_media)
        source_probe = str(uploaded_paths[0]) if uploaded_paths else ""
    detected_profile = resolve_profile(profile_key, source_probe)
    effective_tag = tag or (detected_profile.tag if detected_profile else "")
    default_output_root = ROOT / "outputs" / "workbench_runs"
    out_root = _path_or_none(output_root) or (_path_or_none(detected_profile.output_root) if detected_profile and detected_profile.output_root else None) or default_output_root
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_download_root = _path_or_none(download_root) or (_path_or_none(detected_profile.download_root) if detected_profile and detected_profile.download_root else None)
    effective_note_root = _path_or_none(note_root) or (_path_or_none(detected_profile.note_root) if detected_profile and detected_profile.note_root else None)

    _report(progress, 0.08, "解析输入媒体 / 下载链接")
    source_media, input_meta, source_value = resolve_input_media(
        source_url=source_url, uploaded_media=uploaded_media, out_dir=out_dir, download_root=effective_download_root,
        download_mode=download_mode, cookie_mode=cookie_mode, cookie_path=cookie_path, uploaded_cookie=uploaded_cookie,
        ytdlp_path=ytdlp_path, ffmpeg_path=ffmpeg_path,
    )
    managed_media_path = source_media
    if detected_profile and effective_download_root and source_media.exists():
        try:
            candidate = build_managed_media_path(detected_profile, source_media, download_root=effective_download_root)
            if candidate != source_media:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                if not candidate.exists():
                    shutil.move(str(source_media), str(candidate))
                managed_media_path = candidate
        except Exception:
            managed_media_path = source_media

    _report(progress, 0.22, "ffmpeg 预处理为 16k 单声道 WAV")
    wav_path, ffmpeg_meta = preprocess_to_wav(managed_media_path, out_dir, ffmpeg_path=ffmpeg_path)
    if is_url(source_value) and not keep_download and not effective_download_root:
        try:
            managed_media_path.unlink(missing_ok=True)
        except OSError:
            pass

    _report(progress, 0.38, "FunASR 转写")
    transcript_text, speaker_text, transcript_md, transcript_txt, speaker_md, speaker_json, asr_meta = transcribe_prepared_wav(
        wav_path, out_dir, scenario_name=scenario_name, model_preset=model_preset, model_custom=model_custom,
        vad_preset=vad_preset, vad_custom=vad_custom, punc_preset=punc_preset, punc_custom=punc_custom,
        language=language, device=device, use_itn=use_itn, merge_vad=merge_vad, speaker_diarization=speaker_diarization,
        speaker_count=speaker_count, verbose=verbose, batch_size_s=batch_size_s, max_single_segment_ms=max_single_segment_ms,
        merge_length_s=merge_length_s, split_lines=split_lines,
    )

    _report(progress, 0.70, "组装 prompts")
    prompt_blocks = load_selected_prompts(selected_prompt_labels)
    prompt_blocks.extend(load_uploaded_prompt_files(uploaded_prompt_files))
    note_target_path: Path | None = None
    if detected_profile and effective_note_root:
        note_target_path = build_suggested_note_path(detected_profile, managed_media_path, note_root=effective_note_root)
    elif effective_note_root:
        note_target_path = effective_note_root / f"{sanitize_filename(managed_media_path.stem, 'summary')}.md"
    prompt_text = compose_summary_prompt(
        transcript_text=transcript_txt.read_text(encoding="utf-8", errors="replace"), source_value=source_value,
        prompt_blocks=prompt_blocks, inline_prompt=inline_prompt, profile_name=detected_profile.display_name if detected_profile else "",
        tag=effective_tag, media_path=str(managed_media_path), output_dir=str(out_dir),
        suggested_note_path=str(note_target_path) if note_target_path else "", speaker_transcript_text=speaker_text,
    )
    prompt_path = out_dir / "summary_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    llm_output = ""
    llm_meta: dict[str, Any] = {"enabled": bool(run_llm)}
    if run_llm:
        _report(progress, 0.78, "调用 OpenAI-compatible 大模型")
        settings = LLMSettings(enabled=True, base_url=llm_base_url, model=llm_model, api_key=llm_api_key, api_key_env=llm_api_key_env, timeout_seconds=int(llm_timeout_seconds), temperature=float(llm_temperature))
        llm_meta.update({"base_url": llm_base_url, "model": llm_model, "temperature": float(llm_temperature), "api_key_env": llm_api_key_env, "api_key_supplied": bool((llm_api_key or "").strip())})
        try:
            client = OpenAICompatibleClient(settings)
            messages = [
                {"role": "system", "content": "你是一个严谨的中文音视频资料整理助手。你必须依据用户提供的转写文本输出 Markdown。"},
                {"role": "user", "content": prompt_text},
            ]
            llm_output = client.chat_text(messages, temperature=settings.temperature)
        except Exception as exc:  # noqa: BLE001
            llm_output = ""
            llm_meta["error"] = str(exc)
            llm_meta["fallback_to_prompt_bundle"] = True

    _report(progress, 0.90, "生成 Markdown 与 manifest")
    title_prefix = detected_profile.note_title_prefix if detected_profile else ""
    title = f"{title_prefix}{managed_media_path.stem}" if title_prefix else managed_media_path.stem
    final_md = out_dir / "final_summary.md"
    final_text = render_final_markdown(
        llm_output=llm_output, prompt_text=prompt_text, transcript_text=transcript_txt.read_text(encoding="utf-8", errors="replace"),
        title=title or "音视频总结", source_value=source_value, include_prompt=include_prompt_in_final, include_transcript=include_transcript_in_final,
    )
    final_md.write_text(final_text, encoding="utf-8")
    written_note_path: Path | None = None
    if write_note and note_target_path:
        note_target_path.parent.mkdir(parents=True, exist_ok=True)
        target = note_target_path if overwrite_note else unique_path(note_target_path)
        target.write_text(final_text, encoding="utf-8")
        written_note_path = target
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "profile": detected_profile.key if detected_profile else "",
        "profile_name": detected_profile.display_name if detected_profile else "", "tag": effective_tag, "source_value": source_value,
        "source_media_path": str(managed_media_path), "output_dir": str(out_dir), "download_root": str(effective_download_root) if effective_download_root else "",
        "note_root": str(effective_note_root) if effective_note_root else "", "suggested_note_path": str(note_target_path) if note_target_path else "",
        "written_note_path": str(written_note_path) if written_note_path else "", "input": input_meta, "ffmpeg": ffmpeg_meta,
        "asr": asr_meta, "prompts": [label for label, _ in prompt_blocks], "llm": llm_meta,
        "files": {"wav": str(wav_path), "transcript_md": str(transcript_md), "transcript_txt": str(transcript_txt), "transcript_speakers_md": str(speaker_md) if speaker_md else "", "transcript_speakers_json": str(speaker_json) if speaker_json else "", "summary_prompt": str(prompt_path), "final_markdown": str(final_md)},
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle_inputs = [transcript_md, transcript_txt, prompt_path, final_md, manifest_path]
    if speaker_md:
        bundle_inputs.append(speaker_md)
    if speaker_json:
        bundle_inputs.append(speaker_json)
    bundle_zip = build_bundle(out_dir, bundle_inputs)
    preview = speaker_text.strip() if speaker_text.strip() else transcript_text.strip()
    _report(progress, 1.0, "完成")
    return AudioWorkflowResult(preview=preview, summary_preview=final_text.strip(), final_markdown_path=str(final_md), bundle_path=str(bundle_zip), manifest_path=str(manifest_path), meta=manifest)

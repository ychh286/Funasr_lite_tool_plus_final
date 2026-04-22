from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm_client import LLMSettings, chat_text
from .prompting import (
    compose_summary_prompt_markdown,
    default_prompt_keys,
    prompt_name_to_key,
    resolve_selected_prompt_sources,
)
from .workflow_profiles import (
    build_managed_media_path,
    build_suggested_note_path,
    detect_workflow_profile_from_input,
    get_workflow_profile,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
DEFAULT_PUNC = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
DEFAULT_SCENARIO = "中文会议/普通录音"
TAG_PROMPTS = {
    "#组会": "group_meeting.md",
    "#项目会": "meeting_project.md",
    "#讲座": "seminar_audio.md",
    "#研讨会": "seminar_audio.md",
    "#技术": "tech_video.md",
    "#日常": "daily_chat.md",
    "#人文": "humanities_video.md",
}
SUPPORTED_MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".m4a", ".wav", ".mp3"}


@dataclass(slots=True)
class WorkflowRunRequest:
    """描述一次 Audio Workbench 运行所需的全部输入参数。

    功能：
        统一承载 URL/本地媒体、Prompt、下载配置、转写配置和 LLM 配置，
        让 CLI 和 WebUI 可以复用同一条执行主线。

    关键参数：
        source_url / local_media_path：二选一，描述输入源。
        selected_prompt_keys：新的多 Prompt 选择结果。
        profile_key：本地工作流配置名，可为空。
        llm：最终总结所用的 LLM 配置。

    返回值：
        数据类本身就是请求对象，不直接返回其它结果。

    状态影响：
        无外部副作用，真正的文件读写与命令执行发生在 `run_audio_workbench()`。

    调用时机：
        WebUI 点击开始执行时，或命令行入口准备运行 pipeline 时。
    """

    source_url: str = ""
    local_media_path: str = ""
    tag: str = ""
    prompt_name: str = ""
    selected_prompt_keys: tuple[str, ...] = ()
    uploaded_prompt_path: str = ""
    extra_instruction: str = ""
    profile_key: str = ""
    scenario: str = DEFAULT_SCENARIO
    device: str = "auto"
    model: str = DEFAULT_MODEL
    punc: str = DEFAULT_PUNC
    speaker_diarization: bool = False
    speaker_count: int = 0
    output_root: str = ""
    default_output_root: str = ""
    download_root: str = ""
    note_root: str = ""
    download_mode: str = "audio"
    retention_days: int = 7
    keep_download: bool = False
    cookies_mode: str = "auto"
    cookies_path: str = ""
    uploaded_cookies_path: str = ""
    ytdlp_path: str = ""
    ffmpeg_path: str = ""
    timestamp: str = ""
    prepared_audio_name: str = "audio.wav"
    workflow_zip_name: str = "workflow_outputs.zip"
    passthrough_args: tuple[str, ...] = ()
    llm: LLMSettings = field(default_factory=LLMSettings)


@dataclass(slots=True)
class WorkflowRunResult:
    """承载一次 Audio Workbench 执行后的关键产物。"""

    output_dir: Path
    transcript_preview: str
    summary_preview: str
    final_summary_path: Path
    workflow_zip_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def clean_transcript_text(raw_text: str) -> str:
    """清理 FunASR 输出里的控制标记和元信息行。"""

    lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"<\|[^>]*\>", "", line).strip()
        if not line:
            continue
        if line.startswith("# ASR Transcript:"):
            continue
        if line.startswith("# Speaker Transcript:"):
            continue
        if line.startswith("- Generated:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """统一执行子进程命令，并强制使用 UTF-8 文本模式。"""

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _echo_completed_output(cp: subprocess.CompletedProcess[str]) -> None:
    """把捕获到的 stdout/stderr 原样转发回当前终端。"""

    if cp.stdout:
        sys.stdout.buffer.write(cp.stdout.encode("utf-8", errors="replace"))
        if not cp.stdout.endswith("\n"):
            sys.stdout.buffer.write(b"\n")
    if cp.stderr:
        sys.stderr.buffer.write(cp.stderr.encode("utf-8", errors="replace"))
        if not cp.stderr.endswith("\n"):
            sys.stderr.buffer.write(b"\n")


def _resolve_source_value(request: WorkflowRunRequest) -> str:
    """统一解析“本次运行到底使用哪个输入源”。"""

    source_url = (request.source_url or "").strip()
    if source_url:
        return source_url
    local_media_path = (request.local_media_path or "").strip()
    if local_media_path:
        return local_media_path
    raise SystemExit("请提供 URL 或本地媒体文件。")


def _is_url(input_value: str) -> bool:
    """判断输入值是否为 HTTP/HTTPS 链接。"""

    return bool(re.match(r"^https?://", input_value or ""))


def _resolve_profile(request: WorkflowRunRequest, source_value: str):
    """按“显式选择优先，自动识别兜底”的顺序解析工作流 profile。"""

    if (request.profile_key or "").strip():
        return get_workflow_profile(request.profile_key)
    return detect_workflow_profile_from_input(source_value)


def _resolve_output_root(request: WorkflowRunRequest, profile) -> Path:
    """确定本次运行的产物根目录。"""

    candidate = (request.output_root or "").strip()
    if candidate:
        return Path(candidate).expanduser()
    if profile and profile.output_root:
        return Path(profile.output_root).expanduser()
    default_root = (request.default_output_root or "").strip()
    if default_root:
        return Path(default_root).expanduser()
    return ROOT / "outputs" / "workbench_runs"


def _resolve_optional_path(raw_path: str, fallback_value: str = "") -> Path | None:
    """把可选路径字符串解析成 `Path`。"""

    value = (raw_path or fallback_value or "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def cleanup_old_run_dirs(out_root: Path, retention_days: int) -> None:
    """按保留天数清理旧的运行目录。

    功能：
        避免 `outputs/workbench_runs` 或旧的 transcript 输出目录无限膨胀。

    关键参数：
        out_root：需要清理的运行根目录。
        retention_days：保留天数，`<= 0` 时不清理。

    返回值：
        无。

    状态影响：
        会删除超期的运行子目录，因此必须只对明确的输出根目录执行。

    调用时机：
        每次创建新的运行目录之前。
    """

    if retention_days <= 0 or not out_root.exists():
        return
    cutoff = datetime.now().timestamp() - retention_days * 24 * 3600
    for child in out_root.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def resolve_tool_path(
    *,
    explicit_path: str,
    env_names: tuple[str, ...],
    local_candidates: tuple[Path, ...],
    program_name: str,
) -> Path | None:
    """按“显式参数 -> 环境变量 -> 本地工具目录 -> PATH”顺序查找外部工具。"""

    raw_explicit = (explicit_path or "").strip()
    if raw_explicit:
        candidate = Path(raw_explicit).expanduser()
        if candidate.exists():
            return candidate
        raise SystemExit(f"工具路径不存在：{candidate}")

    for env_name in env_names:
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            return candidate

    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    resolved = shutil.which(program_name)
    if resolved:
        return Path(resolved)
    return None


def _site_cookie_names(input_value: str) -> list[str]:
    """根据输入站点返回优先尝试的 cookies 文件名列表。"""

    lowered = (input_value or "").lower()
    if "bilibili.com" in lowered or "b23.tv" in lowered:
        return ["bili_cookies.txt", "cookies.txt", "all_cookies.txt"]
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return ["youtube_cookies.txt", "cookies.txt", "all_cookies.txt"]
    if "patreon.com" in lowered:
        return ["patreon_cookies.txt", "cookies.txt", "all_cookies.txt"]
    return ["cookies.txt", "all_cookies.txt"]


def resolve_cookie_file(
    input_value: str,
    request: WorkflowRunRequest,
    runtime_dir: Path,
) -> tuple[Path | None, str]:
    """按配置解析本次 `yt-dlp` 要使用的 cookies 文件。

    功能：
        支持 auto / path / upload / none 四种模式，并避免把上传的 cookies
        直接打进最终产物目录。

    关键参数：
        input_value：当前 URL，用于做站点级 cookies 优先级判断。
        request：本次运行请求。
        runtime_dir：本次运行的临时目录，可安全放入上传 cookies 的副本。

    返回值：
        `(cookie_path, cookie_source)`，其中 `cookie_source` 仅用于 manifest 记录。

    状态影响：
        在 `upload` 模式下会把上传文件复制到临时目录。

    调用时机：
        URL 下载前。
    """

    mode = (request.cookies_mode or "auto").strip().lower()
    if mode == "none":
        return None, "disabled"

    if mode == "path":
        candidate = _resolve_optional_path(request.cookies_path)
        if candidate is None or not candidate.exists():
            raise SystemExit("cookies 模式为 path，但未提供可用文件。")
        return candidate, "manual_path"

    if mode == "upload":
        upload_path = _resolve_optional_path(request.uploaded_cookies_path)
        if upload_path is None or not upload_path.exists():
            raise SystemExit("cookies 模式为 upload，但未提供上传文件。")
        copied = runtime_dir / upload_path.name
        shutil.copy2(upload_path, copied)
        return copied, "uploaded"

    env_file_candidates = [
        os.getenv("FUNASR_LITE_COOKIE_FILE", "").strip(),
        os.getenv("YTDLP_COOKIE_FILE", "").strip(),
    ]
    for raw in env_file_candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return candidate, f"env_file:{candidate.name}"

    root_candidates: list[Path] = []
    for env_name in ("FUNASR_LITE_COOKIE_ROOT", "YTDLP_COOKIE_ROOT"):
        raw_root = os.getenv(env_name, "").strip()
        if raw_root:
            root_candidates.append(Path(raw_root).expanduser())

    root_candidates.extend(
        [
            ROOT / "secrets" / "cookies",
            Path.home() / ".openclaw" / "secrets" / "cookies",
        ]
    )

    for root in root_candidates:
        for cookie_name in _site_cookie_names(input_value):
            candidate = root / cookie_name
            if candidate.exists():
                return candidate, f"auto:{candidate.name}"

    return None, "not_found"


def _extract_download_warnings(text: str) -> list[str]:
    """从 `yt-dlp` 输出里提取对用户有意义的下载提醒。"""

    warnings: list[str] = []
    lowered = text.lower()
    if "premium member to download" in lowered or "1080p 高清 are missing" in lowered or "1080p 高码率" in text:
        warnings.append("当前 cookies 可能不具备目标站点的高权限下载能力，如需更高清晰度请更新 cookies 配置。")
    return warnings


def _predict_ytdlp_path(cmd: list[str]) -> Path | None:
    """通过 `yt-dlp --get-filename` 预估下载后文件路径。"""

    try:
        cp = run_command(cmd, capture_output=True)
    except subprocess.CalledProcessError:
        return None

    lines = [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    return Path(lines[-1])


def resolve_media(
    input_value: str,
    out_dir: Path,
    *,
    ytdlp_path: Path | None,
    cookie_file: Path | None,
    download_root: Path | None,
    download_mode: str,
) -> tuple[Path, list[str]]:
    """把 URL 或本地媒体统一解析成后续可处理的真实文件路径。"""

    if _is_url(input_value):
        if ytdlp_path is None:
            raise SystemExit("当前输入是 URL，但未找到 yt-dlp。")
        target_dir = download_root or out_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        source_tpl = target_dir / "%(title)s [%(id)s].%(ext)s"
        archive_path = target_dir / ".yt-dlp-archive.txt"

        predict_cmd = [str(ytdlp_path), "--get-filename", "-o", str(source_tpl)]
        download_cmd = [
            str(ytdlp_path),
            "--download-archive",
            str(archive_path),
            "-o",
            str(source_tpl),
        ]
        if download_mode == "video":
            predict_cmd.extend(["--merge-output-format", "mp4"])
            download_cmd.extend(["--merge-output-format", "mp4"])
        else:
            predict_cmd.extend(["-x", "--audio-format", "wav", "--audio-quality", "0"])
            download_cmd.extend(["-x", "--audio-format", "wav", "--audio-quality", "0"])

        if cookie_file:
            predict_cmd.extend(["--cookies", str(cookie_file)])
            download_cmd.extend(["--cookies", str(cookie_file)])

        predict_cmd.append(input_value)
        download_cmd.append(input_value)

        predicted = _predict_ytdlp_path(predict_cmd)
        cp = run_command(download_cmd, capture_output=True)
        _echo_completed_output(cp)
        warnings = _extract_download_warnings((cp.stdout or "") + "\n" + (cp.stderr or ""))
        if predicted and predicted.exists():
            return predicted, warnings

        candidates = sorted(
            [
                path
                for path in target_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit("yt-dlp 已执行，但没有找到可用下载文件。")
        return candidates[0], warnings

    media_path = Path(input_value).expanduser().resolve()
    if not media_path.exists():
        raise SystemExit(f"输入文件不存在：{media_path}")
    return media_path, []


def resolve_prompt_name(tag: str, prompt_name: str) -> str:
    """兼容旧 pipeline 的 tag/prompt 解析规则。"""

    raw_prompt = (prompt_name or "").strip()
    if raw_prompt:
        return raw_prompt
    raw_tag = (tag or "").strip()
    if not raw_tag:
        return ""
    return TAG_PROMPTS.get(raw_tag, "")


def _build_fallback_final_summary(summary_prompt_markdown: str, *, llm_enabled: bool, llm_error: str) -> str:
    """在未启用 LLM 或 LLM 失败时生成一个可继续复核的总结草稿。"""

    if llm_enabled and llm_error:
        status_line = f"LLM 调用失败，已回退为总结草稿。错误：{llm_error}"
    else:
        status_line = "当前未启用 LLM，以下内容为可直接继续交给 Agent 或模型处理的总结草稿。"

    return (
        "# Final Summary Draft\n\n"
        f"{status_line}\n\n"
        "## Prompt Bundle\n\n"
        f"{summary_prompt_markdown.rstrip()}\n"
    )


def package_workflow_outputs(output_dir: Path, artifact_paths: list[Path], *, zip_name: str) -> Path:
    """把本次运行的关键产物打包成一个可下载 zip。"""

    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in artifact_paths:
            if not path.exists():
                continue
            if path == zip_path:
                continue
            archive.write(path, arcname=path.name)
    return zip_path


def run_audio_workbench(request: WorkflowRunRequest) -> WorkflowRunResult:
    """执行完整的 Audio Workbench 工作流。

    功能：
        把“URL/本地媒体 -> 下载/预处理 -> FunASR 转写 -> Prompt 组装 ->
        可选 LLM 总结 -> 打包输出”串成一条可复用的服务层主线。

    关键参数：
        request：本次运行的全部输入配置。

    返回值：
        `WorkflowRunResult`，包含预览文本、关键文件路径和完整 manifest。

    状态影响：
        会创建输出目录、执行外部命令、生成多个中间产物，并在需要时调用外部 LLM。

    调用时机：
        WebUI 或 CLI 真正开始执行长任务时。
    """

    source_value = _resolve_source_value(request)
    profile = _resolve_profile(request, source_value)
    effective_tag = (request.tag or "").strip() or (profile.tag if profile else "")
    effective_prompt_name = resolve_prompt_name(
        effective_tag,
        (request.prompt_name or "").strip() or (profile.prompt_name if profile else ""),
    )

    output_root = _resolve_output_root(request, profile)
    cleanup_old_run_dirs(output_root, int(request.retention_days))
    timestamp = (request.timestamp or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    download_root = _resolve_optional_path(request.download_root, getattr(profile, "download_root", ""))
    note_root = _resolve_optional_path(request.note_root, getattr(profile, "note_root", ""))

    ffmpeg_path = resolve_tool_path(
        explicit_path=request.ffmpeg_path,
        env_names=("FUNASR_LITE_FFMPEG", "FFMPEG_BINARY"),
        local_candidates=(ROOT / ".local_tools" / "ffmpeg" / "bin" / "ffmpeg.exe",),
        program_name="ffmpeg",
    )
    if ffmpeg_path is None:
        raise SystemExit("未找到 ffmpeg，请将其加入 PATH，或在 WebUI 中手动填写路径。")

    ytdlp_path = None
    if _is_url(source_value):
        ytdlp_path = resolve_tool_path(
            explicit_path=request.ytdlp_path,
            env_names=("FUNASR_LITE_YTDLP", "YTDLP_BINARY"),
            local_candidates=(
                ROOT / ".local_tools" / "yt-dlp" / "yt-dlp.exe",
                ROOT / ".local_tools" / "yt-dlp.exe",
            ),
            program_name="yt-dlp",
        )
        if ytdlp_path is None:
            raise SystemExit("当前输入是 URL，但未找到 yt-dlp，请先安装或手动填写路径。")

    with tempfile.TemporaryDirectory(prefix="funasr_lite_runtime_") as runtime_dir_str:
        runtime_dir = Path(runtime_dir_str)
        cookie_file, cookie_source = resolve_cookie_file(source_value, request, runtime_dir)
        media_path, download_warnings = resolve_media(
            source_value,
            output_dir,
            ytdlp_path=ytdlp_path,
            cookie_file=cookie_file,
            download_root=download_root,
            download_mode=(request.download_mode or "audio").strip() or "audio",
        )

        if profile and download_root and media_path.exists():
            managed_media_path = build_managed_media_path(profile, media_path, download_root=download_root)
            if managed_media_path != media_path:
                managed_media_path.parent.mkdir(parents=True, exist_ok=True)
                if not managed_media_path.exists():
                    media_path.rename(managed_media_path)
                media_path = managed_media_path

        prepared_audio_path = output_dir / (request.prepared_audio_name or "audio.wav")
        run_command(
            [
                str(ffmpeg_path),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(media_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(prepared_audio_path),
            ]
        )

        transcript_md = output_dir / "transcript.md"
        transcript_txt = output_dir / "transcript.txt"
        transcript_speakers_md = output_dir / "transcript_speakers.md"
        transcript_speakers_json = output_dir / "transcript_speakers.json"

        transcribe_cmd = [
            sys.executable,
            str(ROOT / "transcribe_local.py"),
            str(prepared_audio_path),
            "--scenario",
            request.scenario,
            "--device",
            request.device,
            "--out",
            str(transcript_md),
            "--split-lines",
        ]
        model_env_override = os.getenv("FUNASR_MODEL_ID")
        punc_env_override = os.getenv("FUNASR_PUNC_MODEL_ID")
        if not (request.speaker_diarization and request.model == DEFAULT_MODEL and not model_env_override):
            transcribe_cmd.extend(["--model", request.model])
        if not (request.speaker_diarization and request.punc == DEFAULT_PUNC and not punc_env_override):
            transcribe_cmd.extend(["--punc", request.punc])
        if request.speaker_diarization:
            transcribe_cmd.append("--speaker-diarization")
        if int(request.speaker_count) > 0:
            transcribe_cmd.extend(["--speaker-count", str(int(request.speaker_count))])
        transcribe_cmd.extend(list(request.passthrough_args))
        run_command(transcribe_cmd, cwd=ROOT)

        transcript_body = clean_transcript_text(transcript_md.read_text(encoding="utf-8"))
        transcript_txt.write_text(transcript_body, encoding="utf-8")

        speaker_transcript_text = ""
        if transcript_speakers_md.exists():
            speaker_transcript_text = transcript_speakers_md.read_text(encoding="utf-8")

        selected_prompt_keys = list(request.selected_prompt_keys)
        if not selected_prompt_keys:
            selected_prompt_keys.extend(default_prompt_keys())
        prompt_key = prompt_name_to_key(effective_prompt_name)
        if prompt_key and prompt_key not in selected_prompt_keys:
            selected_prompt_keys.append(prompt_key)
        selected_sources = resolve_selected_prompt_sources(selected_prompt_keys)

        note_target_path = None
        if profile and note_root:
            note_target_path = build_suggested_note_path(profile, media_path, note_root=note_root)

        uploaded_prompt_path = _resolve_optional_path(request.uploaded_prompt_path)
        summary_prompt_markdown = compose_summary_prompt_markdown(
            selected_sources=selected_sources,
            uploaded_prompt_path=uploaded_prompt_path,
            extra_instruction=request.extra_instruction,
            tag=effective_tag,
            profile_name=profile.display_name if profile else "",
            source_value=source_value,
            output_dir=output_dir,
            media_library_path=media_path,
            note_target_path=note_target_path,
            transcript_text=transcript_body,
            speaker_transcript_text=speaker_transcript_text,
        )
        summary_prompt_path = output_dir / "summary_prompt.md"
        summary_prompt_path.write_text(summary_prompt_markdown, encoding="utf-8")

        llm_error = ""
        llm_output = ""
        if request.llm.enabled:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个严谨的中文音视频内容整理助手。请严格依据用户提供的 Prompt 包和转写内容输出 Markdown，总结中不要虚构事实。",
                },
                {
                    "role": "user",
                    "content": summary_prompt_markdown,
                },
            ]
            try:
                llm_output = chat_text(request.llm, messages)
            except Exception as exc:  # noqa: BLE001
                llm_error = str(exc)

        final_summary_path = output_dir / "final_summary.md"
        if llm_output.strip():
            final_summary_path.write_text(llm_output.strip() + "\n", encoding="utf-8")
        else:
            final_summary_path.write_text(
                _build_fallback_final_summary(
                    summary_prompt_markdown,
                    llm_enabled=bool(request.llm.enabled),
                    llm_error=llm_error,
                ),
                encoding="utf-8",
            )

        if media_path.parent == output_dir and not request.keep_download and media_path != prepared_audio_path:
            try:
                media_path.unlink()
            except OSError:
                pass

        artifact_paths = [
            prepared_audio_path,
            transcript_md,
            transcript_txt,
            summary_prompt_path,
            final_summary_path,
        ]
        if transcript_speakers_md.exists():
            artifact_paths.append(transcript_speakers_md)
        if transcript_speakers_json.exists():
            artifact_paths.append(transcript_speakers_json)

        manifest: dict[str, Any] = {
            "source": source_value,
            "profile": profile.key if profile else "",
            "profile_name": profile.display_name if profile else "",
            "tag": effective_tag,
            "prompt_template": effective_prompt_name,
            "selected_prompt_keys": selected_prompt_keys,
            "selected_prompt_files": [str(source.path) for source in selected_sources],
            "uploaded_prompt_path": str(uploaded_prompt_path) if uploaded_prompt_path else "",
            "extra_instruction": (request.extra_instruction or "").strip(),
            "speaker_diarization": bool(request.speaker_diarization),
            "speaker_count": int(request.speaker_count),
            "download_mode": (request.download_mode or "audio").strip() or "audio",
            "download_root": str(download_root) if download_root else "",
            "note_root": str(note_root) if note_root else "",
            "suggested_note_path": str(note_target_path) if note_target_path else "",
            "cookies_mode": (request.cookies_mode or "auto").strip().lower(),
            "cookies_source": cookie_source,
            "download_warnings": download_warnings,
            "tool_paths": {
                "ffmpeg": str(ffmpeg_path),
                "yt_dlp": str(ytdlp_path) if ytdlp_path else "",
            },
            "llm": request.llm.safe_summary(),
            "llm_error": llm_error,
            "output_dir": str(output_dir),
            "source_media_path": str(media_path),
            "prepared_audio_path": str(prepared_audio_path),
            "transcript_md": str(transcript_md),
            "transcript_txt": str(transcript_txt),
            "transcript_speakers_md": str(transcript_speakers_md) if transcript_speakers_md.exists() else "",
            "transcript_speakers_json": str(transcript_speakers_json) if transcript_speakers_json.exists() else "",
            "summary_prompt": str(summary_prompt_path),
            "final_summary": str(final_summary_path),
            "workflow_zip": str(output_dir / request.workflow_zip_name),
        }
        manifest_path = output_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact_paths.append(manifest_path)
        workflow_zip_path = package_workflow_outputs(output_dir, artifact_paths, zip_name=request.workflow_zip_name)

    transcript_preview_path = transcript_speakers_md if transcript_speakers_md.exists() else transcript_md
    return WorkflowRunResult(
        output_dir=output_dir,
        transcript_preview=transcript_preview_path.read_text(encoding="utf-8"),
        summary_preview=final_summary_path.read_text(encoding="utf-8"),
        final_summary_path=final_summary_path,
        workflow_zip_path=workflow_zip_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )

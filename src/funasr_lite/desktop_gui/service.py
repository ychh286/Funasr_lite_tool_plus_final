from __future__ import annotations

"""桌面 GUI 的服务/用例层。

这里是 Qt 页面与既有业务模块之间的边界：页面只提交 `SingleRunRequest`、
`BatchRunRequest` 或 `FollowupRunRequest`，服务层负责调用下载、ffmpeg、FunASR、
LLM、Brave Search 和追问缓存逻辑。
"""

import json
import shutil
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..brave_search import BraveSearchSettings
from ..config import DEFAULT_SCENARIO
from ..followup_chat import (
    current_summary_path,
    load_followup_session,
    new_session_id,
    resolve_run_dir,
    run_followup_turn,
)
from ..llm_client import LLMSettings, OpenAICompatibleClient
from ..pipeline_workflow import (
    MEDIA_EXTENSIONS,
    build_bundle,
    inspect_audio_workflow_environment,
    preprocess_to_wav,
    run_audio_workflow,
    transcribe_prepared_wav,
)
from ..prompting import default_prompt_labels, sanitize_filename
from ..workflow_profiles import get_workflow_profiles, reload_workflow_profiles
from .models import (
    BatchRunRequest,
    EnvironmentCard,
    EnvironmentSnapshot,
    FollowupRunRequest,
    GuiRunResult,
    LLMGuiSettings,
    PersistedGuiState,
    SingleRunRequest,
)

ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = ROOT / ".local_state" / "desktop_gui_state.json"
ProgressFn = Callable[[float, str], None]


def _report(progress: ProgressFn | None, pct: float, message: str) -> None:
    if progress:
        progress(float(pct), message)


def _tone_from_status(text: str) -> str:
    value = text or ""
    if "❌" in value or "失败" in value or "不存在" in value:
        return "error"
    if "⚠️" in value or "未命中" in value or "未指定" in value or "不需要" in value:
        return "warning"
    return "ok"


def _safe_read(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _llm_settings(settings: LLMGuiSettings) -> LLMSettings:
    return LLMSettings(
        enabled=settings.run_llm,
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        api_key_env=settings.api_key_env,
        timeout_seconds=int(settings.timeout_seconds),
        temperature=float(settings.temperature),
    )


class DesktopWorkflowService:
    """桌面 GUI 的稳定业务入口。

    生命周期：主窗口创建一个实例，后台 worker 复用这个实例调用长任务。
    该类不持有 QWidget，也不从工作线程直接更新 UI。
    """

    def inspect_environment(self, request: SingleRunRequest) -> EnvironmentSnapshot:
        """根据当前表单生成环境探测快照。"""

        info = inspect_audio_workflow_environment(
            source_url=request.source_url,
            uploaded_media=request.local_media,
            profile_key=request.profile_key,
            selected_prompt_labels=request.selected_prompt_labels,
            output_root=request.output_root,
            download_root=request.download_root,
            note_root=request.note_root,
            cookie_mode=request.cookie_mode,
            cookie_path=request.cookie_path,
            uploaded_cookie=None,
            ytdlp_path=request.ytdlp_path,
            ffmpeg_path=request.ffmpeg_path,
        )
        cards = [
            EnvironmentCard(
                "输入来源",
                info.get("source_value") or "尚未选择",
                info.get("input_status") or f"类型：{info.get('source_kind', 'empty')}",
                _tone_from_status(info.get("input_status", "")),
            ),
            EnvironmentCard(
                "Workflow 路由",
                info.get("profile_label") or "通用模式",
                f"默认 Prompt：{info.get('profile_prompt') or '未指定'}",
                "info",
            ),
            EnvironmentCard("Python / FunASR 运行时", info.get("python_runtime_status") or "未检查", "检查 funasr、modelscope、torch 是否可导入。", _tone_from_status(info.get("python_runtime_status", ""))),
            EnvironmentCard("运行输出目录", info.get("resolved_output_root") or "未解析", info.get("output_root_status") or "转写、manifest、zip 都会先写到这里。", _tone_from_status(info.get("output_root_status", ""))),
            EnvironmentCard("媒体归档目录", info.get("resolved_download_root") or "未指定；URL 媒体保存在单次 run 内", info.get("download_root_status") or "只影响 yt-dlp 下载后的原始媒体保存位置。", _tone_from_status(info.get("download_root_status", ""))),
            EnvironmentCard("总结落盘目录", info.get("resolved_note_root") or "未指定；只生成 run 内 final_summary.md", info.get("note_root_status") or "设置后默认会额外同步一份 Markdown。", _tone_from_status(info.get("note_root_status", ""))),
            EnvironmentCard("ffmpeg", info.get("ffmpeg_status") or "未检查", "负责抽音频、转 16k 单声道 WAV。", _tone_from_status(info.get("ffmpeg_status", ""))),
            EnvironmentCard("yt-dlp", info.get("ytdlp_status") or "未检查", "仅 URL 输入时需要。", _tone_from_status(info.get("ytdlp_status", ""))),
            EnvironmentCard("Cookies", info.get("cookie_status") or "未检查", "需要登录或会员内容时提供；不会打进导出 zip。", _tone_from_status(info.get("cookie_status", ""))),
        ]
        has_error = any(card.tone == "error" for card in cards)
        headline = "环境探测发现阻塞项，请先修正红色卡片。" if has_error else "环境探测完成，可以开始运行。"
        return EnvironmentSnapshot(state="failed" if has_error else "ready", headline=headline, cards=cards, raw=info)

    def run_single(self, request: SingleRunRequest, progress: ProgressFn | None = None) -> GuiRunResult:
        """执行 URL/本地单素材端到端总结工作流。"""

        result = run_audio_workflow(
            source_url=request.source_url,
            uploaded_media=request.local_media,
            profile_key=request.profile_key,
            tag=request.tag,
            selected_prompt_labels=request.selected_prompt_labels or default_prompt_labels(),
            uploaded_prompt_files=request.uploaded_prompt_files,
            inline_prompt=request.inline_prompt,
            output_root=request.output_root,
            download_root=request.download_root,
            note_root=request.note_root,
            download_mode=request.download_mode,
            keep_download=request.keep_download,
            cookie_mode=request.cookie_mode,
            cookie_path=request.cookie_path,
            uploaded_cookie=None,
            ytdlp_path=request.ytdlp_path,
            ffmpeg_path=request.ffmpeg_path,
            scenario_name=request.model.scenario_name,
            model_preset=request.model.model_preset,
            model_custom=request.model.model_custom,
            vad_preset=request.model.vad_preset,
            vad_custom=request.model.vad_custom,
            punc_preset=request.model.punc_preset,
            punc_custom=request.model.punc_custom,
            language=request.model.language,
            device=request.model.device,
            use_itn=request.model.use_itn,
            merge_vad=request.model.merge_vad,
            speaker_diarization=request.model.speaker_diarization,
            speaker_count=request.model.speaker_count,
            verbose=request.model.verbose,
            batch_size_s=request.model.batch_size_s,
            max_single_segment_ms=request.model.max_single_segment_ms,
            merge_length_s=request.model.merge_length_s,
            split_lines=request.model.split_lines,
            run_llm=request.llm.run_llm,
            llm_base_url=request.llm.base_url,
            llm_model=request.llm.model,
            llm_api_key=request.llm.api_key,
            llm_api_key_env=request.llm.api_key_env,
            llm_temperature=request.llm.temperature,
            llm_timeout_seconds=request.llm.timeout_seconds,
            include_prompt_in_final=request.include_prompt_in_final,
            include_transcript_in_final=request.include_transcript_in_final,
            write_note=request.write_note,
            overwrite_note=request.overwrite_note,
            enable_web_search=request.llm.enable_web_search,
            brave_api_key=request.llm.brave_api_key,
            brave_api_key_env=request.llm.brave_api_key_env,
            brave_count=request.llm.brave_count,
            brave_timeout_seconds=request.llm.brave_timeout_seconds,
            progress=progress,
        )
        files = result.meta.get("files") or {}
        return GuiRunResult(
            headline="完整工作流完成",
            preview_text=result.preview,
            summary_text=result.summary_preview,
            output_dir=str(result.meta.get("output_dir") or ""),
            final_markdown_path=result.final_markdown_path,
            transcript_path=str(files.get("transcript_speakers_md") or files.get("transcript_md") or files.get("transcript_txt") or ""),
            bundle_path=result.bundle_path,
            manifest_path=result.manifest_path,
            meta=result.meta,
        )

    def run_batch(self, request: BatchRunRequest, progress: ProgressFn | None = None) -> GuiRunResult:
        """批量转写多个本地媒体文件。

        这个入口不做 LLM 总结，定位是“批处理得到可复用文字稿”。视频文件会先
        经 ffmpeg 转成 16k 单声道 WAV，再交给 FunASR。
        """

        media_paths = self._collect_batch_media(request)
        if not media_paths:
            raise RuntimeError("请先选择至少一个音频/视频文件，或选择一个包含媒体的目录。")

        out_root = Path(request.output_root).expanduser() if request.output_root.strip() else ROOT / "outputs" / "batch_runs"
        batch_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        batch_dir.mkdir(parents=True, exist_ok=True)
        result_files: list[Path] = []
        preview_parts: list[str] = []
        results: list[dict[str, object]] = []

        normalized_format = (request.out_format or "md").lower().strip(".")
        if normalized_format not in {"md", "txt"}:
            normalized_format = "md"

        for index, media_path in enumerate(media_paths, start=1):
            pct_base = (index - 1) / max(1, len(media_paths))
            _report(progress, pct_base, f"准备 {media_path.name} ({index}/{len(media_paths)})")
            item_dir = batch_dir / f"{index:02d}_{sanitize_filename(media_path.stem, 'media')}"
            item_dir.mkdir(parents=True, exist_ok=True)
            try:
                transcribe_input = media_path
                if request.preprocess_to_wav:
                    _report(progress, pct_base + 0.05 / len(media_paths), f"ffmpeg 预处理：{media_path.name}")
                    transcribe_input, _ = preprocess_to_wav(media_path, item_dir, ffmpeg_path=request.ffmpeg_path)
                _report(progress, pct_base + 0.25 / len(media_paths), f"FunASR 转写：{media_path.name}")
                transcript_text, speaker_text, transcript_md, transcript_txt, speaker_md, speaker_json, asr_meta = transcribe_prepared_wav(
                    transcribe_input,
                    item_dir,
                    scenario_name=request.model.scenario_name,
                    model_preset=request.model.model_preset,
                    model_custom=request.model.model_custom,
                    vad_preset=request.model.vad_preset,
                    vad_custom=request.model.vad_custom,
                    punc_preset=request.model.punc_preset,
                    punc_custom=request.model.punc_custom,
                    language=request.model.language,
                    device=request.model.device,
                    use_itn=request.model.use_itn,
                    merge_vad=request.model.merge_vad,
                    speaker_diarization=request.model.speaker_diarization,
                    speaker_count=request.model.speaker_count,
                    verbose=request.model.verbose,
                    batch_size_s=request.model.batch_size_s,
                    max_single_segment_ms=request.model.max_single_segment_ms,
                    merge_length_s=request.model.merge_length_s,
                    split_lines=request.model.split_lines,
                    output_fmt=normalized_format,
                )
                primary_transcript = transcript_txt if normalized_format == "txt" else transcript_md
                item_files = [primary_transcript]
                if speaker_md:
                    item_files.append(speaker_md)
                if speaker_json:
                    item_files.append(speaker_json)
                result_files.extend(item_files)
                preview_parts.append(f"## {media_path.name}\n\n{(speaker_text or transcript_text).strip()}\n")
                results.append(
                    {
                        "file": str(media_path),
                        "ok": True,
                        "output_dir": str(item_dir),
                        "output_format": normalized_format,
                        "transcript_file": str(primary_transcript),
                        "speaker_file": str(speaker_md) if speaker_md else "",
                        "asr": asr_meta,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error_path = item_dir / "ERROR.txt"
                error_path.write_text(str(exc), encoding="utf-8")
                result_files.append(error_path)
                preview_parts.append(f"## {media_path.name}\n\n[ERROR] {exc}\n")
                results.append({"file": str(media_path), "ok": False, "error": str(exc), "output_dir": str(item_dir)})

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(batch_dir),
            "output_format": normalized_format,
            "files": [str(path) for path in media_paths],
            "model": asdict(request.model),
            "results": results,
        }
        manifest_path = batch_dir / "batch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result_files.append(manifest_path)
        bundle_path = build_bundle(batch_dir, result_files)
        _report(progress, 1.0, "批量转写完成")
        return GuiRunResult(
            headline="批量转写完成",
            preview_text="\n\n".join(preview_parts).strip(),
            output_dir=str(batch_dir),
            bundle_path=str(bundle_path),
            manifest_path=str(manifest_path),
            meta=manifest,
        )

    def load_followup(self, run_reference: str, session_id: str = "") -> dict[str, object]:
        active_session = session_id.strip() or new_session_id()
        return load_followup_session(run_reference, active_session)

    def run_followup(self, request: FollowupRunRequest, progress: ProgressFn | None = None) -> GuiRunResult:
        _report(progress, 0.05, "加载 run 目录与追问上下文")
        result = run_followup_turn(
            run_reference=request.run_reference,
            session_id=request.session_id or new_session_id(),
            question=request.question,
            llm_settings=_llm_settings(request.llm),
            update_summary=request.update_summary,
            enable_web_search=request.llm.enable_web_search,
            brave_settings=BraveSearchSettings(
                enabled=request.llm.enable_web_search,
                api_key=request.llm.brave_api_key,
                api_key_env=request.llm.brave_api_key_env,
                count=int(request.llm.brave_count),
                timeout_seconds=int(request.llm.brave_timeout_seconds),
            ),
        )
        _report(progress, 1.0, "追问完成")
        return GuiRunResult(
            headline="追问完成",
            preview_text=result.answer_markdown,
            summary_text=result.current_summary_text,
            output_dir=str(result.run_dir),
            final_markdown_path=str(result.current_summary_path),
            transcript_path=str(result.transcript_path),
            meta=result.meta,
        )

    def promote_current_summary(self, run_reference: str) -> GuiRunResult:
        """把追问缓存中的当前摘要安全写回 final_summary.md。"""

        run_dir = resolve_run_dir(run_reference)
        current_path = current_summary_path(run_dir)
        final_path = run_dir / "final_summary.md"
        if not current_path.exists():
            raise RuntimeError("当前摘要文件不存在，无法写回。")
        backup_path = ""
        if final_path.exists():
            backup = run_dir / f"final_summary.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            shutil.copy2(final_path, backup)
            backup_path = str(backup)
        shutil.copy2(current_path, final_path)
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            manifest.setdefault("followup", {})
            manifest["followup"].update(
                {
                    "promoted_at": datetime.now().isoformat(timespec="seconds"),
                    "promoted_from": str(current_path),
                    "backup_path": backup_path,
                }
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return GuiRunResult(
            headline="已写回 final_summary.md",
            summary_text=_safe_read(final_path),
            output_dir=str(run_dir),
            final_markdown_path=str(final_path),
            manifest_path=str(manifest_path) if manifest_path.exists() else "",
            meta={"backup_path": backup_path, "promoted_from": str(current_path)},
        )

    def check_llm(self, settings: LLMGuiSettings) -> str:
        health = OpenAICompatibleClient(_llm_settings(settings)).health_check()
        if health.ok:
            return f"✅ LLM 可用：{health.message}"
        return f"❌ LLM 检查失败：{health.message}"

    def profile_options(self) -> list[tuple[str, str]]:
        reload_workflow_profiles()
        options = [("auto", "auto — 自动匹配 / 通用模式")]
        for key, profile in get_workflow_profiles().items():
            options.append((key, f"{key} — {profile.display_name}"))
        return options

    def _collect_batch_media(self, request: BatchRunRequest) -> list[Path]:
        paths: list[Path] = []
        seen: set[str] = set()
        for raw in request.media_files:
            p = Path(raw).expanduser()
            if p.exists() and p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
                key = str(p.resolve())
                if key not in seen:
                    paths.append(p.resolve())
                    seen.add(key)
        if request.media_directory.strip():
            root = Path(request.media_directory).expanduser()
            iterator = root.rglob("*") if request.recursive else root.glob("*")
            for p in iterator:
                if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
                    key = str(p.resolve())
                    if key not in seen:
                        paths.append(p.resolve())
                        seen.add(key)
        paths.sort(key=lambda item: item.name.lower())
        return paths


class GuiStateStore:
    """读写桌面 GUI 的低敏偏好。"""

    def load(self) -> PersistedGuiState:
        if not STATE_PATH.exists():
            return PersistedGuiState()
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return PersistedGuiState()
        allowed = set(PersistedGuiState.__dataclass_fields__.keys())
        return PersistedGuiState(**{key: value for key, value in data.items() if key in allowed})

    def save(self, state: PersistedGuiState) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_SCENARIO",
    "DesktopWorkflowService",
    "GuiStateStore",
    "MEDIA_EXTENSIONS",
    "ROOT",
]

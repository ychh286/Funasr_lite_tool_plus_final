from __future__ import annotations

"""桌面 GUI 与服务层之间共享的数据合同。

本模块只定义不可依赖 Qt 的请求、快照和结果对象。这样 GUI 页面可以专注
于展示与收集用户意图，后台服务也能在命令行测试或单元测试里独立运行。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..llm_client import DEFAULT_LLM_API_KEY_ENV, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_TIMEOUT_SECONDS


RunState = Literal["ready", "busy", "done", "failed", "degraded"]
StatusTone = Literal["info", "ok", "warning", "error"]


@dataclass(slots=True)
class EnvironmentCard:
    """环境探测卡片。

    功能：把 profile、目录、工具和 cookies 的来源拆开展示，避免 GUI 用一个
    模糊的“可运行”状态掩盖真实前置条件。
    """

    title: str
    value: str
    detail: str = ""
    tone: StatusTone = "info"


@dataclass(slots=True)
class EnvironmentSnapshot:
    """一次“检查路径和环境”的只读快照。"""

    state: RunState
    headline: str
    cards: list[EnvironmentCard] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelSettings:
    """FunASR 解码与分段配置。

    这些字段基本对应 `pipeline_workflow.run_audio_workflow` 的参数。GUI 不在
    页面中直接拼模型对象，避免把模型决策散落在控件代码里。
    """

    scenario_name: str = "中文会议/普通录音"
    model_preset: str = "paraformer-zh (offline)"
    model_custom: str = ""
    vad_preset: str = "fsmn-vad"
    vad_custom: str = ""
    punc_preset: str = "none"
    punc_custom: str = ""
    language: str = "auto"
    device: str = "auto"
    use_itn: bool = True
    merge_vad: bool = True
    speaker_diarization: bool = False
    speaker_count: int = 0
    verbose: bool = False
    batch_size_s: int = 10
    max_single_segment_ms: int = 30000
    merge_length_s: int = 15
    split_lines: bool = True


@dataclass(slots=True)
class LLMGuiSettings:
    """OpenAI-compatible 与 Brave Search 配置。"""

    run_llm: bool = False
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    api_key: str = ""
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV
    temperature: float = 0.1
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    enable_web_search: bool = False
    brave_api_key: str = ""
    brave_api_key_env: str = "BRAVE_SEARCH_API_KEY"
    brave_count: int = 5
    brave_timeout_seconds: int = 25


@dataclass(slots=True)
class SingleRunRequest:
    """单素材端到端工作流请求。"""

    source_url: str = ""
    local_media: str = ""
    profile_key: str = "auto"
    tag: str = ""
    selected_prompt_labels: list[str] = field(default_factory=list)
    uploaded_prompt_files: list[str] = field(default_factory=list)
    inline_prompt: str = ""
    output_root: str = ""
    download_root: str = ""
    note_root: str = ""
    download_mode: str = "audio"
    keep_download: bool = True
    cookie_mode: str = "auto"
    cookie_path: str = ""
    ytdlp_path: str = ""
    ffmpeg_path: str = ""
    include_prompt_in_final: bool = False
    include_transcript_in_final: bool = False
    write_note: bool = False
    overwrite_note: bool = False
    model: ModelSettings = field(default_factory=ModelSettings)
    llm: LLMGuiSettings = field(default_factory=LLMGuiSettings)


@dataclass(slots=True)
class BatchRunRequest:
    """批量转写请求。"""

    media_files: list[str] = field(default_factory=list)
    media_directory: str = ""
    recursive: bool = True
    output_root: str = ""
    ffmpeg_path: str = ""
    out_format: str = "md"
    preprocess_to_wav: bool = True
    model: ModelSettings = field(default_factory=ModelSettings)


@dataclass(slots=True)
class FollowupRunRequest:
    """基于既有 run 目录执行追问或改写的请求。"""

    run_reference: str = ""
    session_id: str = ""
    question: str = ""
    update_summary: bool = True
    llm: LLMGuiSettings = field(default_factory=LLMGuiSettings)


@dataclass(slots=True)
class GuiRunResult:
    """服务层交回 GUI 的统一结果。"""

    headline: str
    preview_text: str = ""
    summary_text: str = ""
    output_dir: str = ""
    final_markdown_path: str = ""
    transcript_path: str = ""
    bundle_path: str = ""
    manifest_path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PersistedGuiState:
    """需要跨启动保存的轻量 GUI 配置。

    不保存 API key 或 cookies 正文，只保存路径、模型名、环境变量名等低敏信息。
    """

    profile_key: str = "auto"
    output_root: str = ""
    download_root: str = ""
    note_root: str = ""
    ffmpeg_path: str = ""
    ytdlp_path: str = ""
    batch_output_root: str = ""
    batch_ffmpeg_path: str = ""
    followup_run_dir: str = ""
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key_env: str = DEFAULT_LLM_API_KEY_ENV
    brave_api_key_env: str = "BRAVE_SEARCH_API_KEY"
    selected_prompt_labels: list[str] = field(default_factory=list)


MEDIA_FILE_FILTER = "音视频文件 (*.mp4 *.mkv *.webm *.mov *.avi *.flv *.m4v *.m4a *.mp3 *.wav *.aac *.flac *.ogg *.opus);;所有文件 (*.*)"
SINGLE_INPUT_FILE_FILTER = "音视频或转写稿 (*.mp4 *.mkv *.webm *.mov *.avi *.flv *.m4v *.m4a *.mp3 *.wav *.aac *.flac *.ogg *.opus *.txt *.md);;音视频文件 (*.mp4 *.mkv *.webm *.mov *.avi *.flv *.m4v *.m4a *.mp3 *.wav *.aac *.flac *.ogg *.opus);;转写稿 (*.txt *.md);;所有文件 (*.*)"
PROMPT_FILE_FILTER = "Prompt 文件 (*.md *.txt);;所有文件 (*.*)"
COOKIE_FILE_FILTER = "Cookies 文件 (*.txt *.cookies);;所有文件 (*.*)"
EXECUTABLE_FILE_FILTER = "可执行文件 (*.exe);;所有文件 (*.*)"


def existing_path_or_empty(raw: str) -> str:
    """把用户输入路径规整为字符串；不存在也不报错，只供 GUI 显示。"""

    value = (raw or "").strip().strip('"')
    if not value:
        return ""
    try:
        return str(Path(value).expanduser())
    except OSError:
        return value

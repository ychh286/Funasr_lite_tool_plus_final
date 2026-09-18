from __future__ import annotations

"""FunASR Audio Workbench 的本地 PySide6 GUI。

本模块只承担桌面窗口、页面布局、用户意图信号和后台线程调度。真实的下载、
转码、转写、总结、追问和文件写回都交给 `DesktopWorkflowService`，避免 GUI
控件直接拥有业务真相。
"""

import json
import faulthandler
import logging
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import DEFAULT_SCENARIO, MODEL_PRESETS, PUNC_PRESETS, SCENARIO_PRESETS, VAD_PRESETS
from ..llm_client import DEFAULT_LLM_MODEL, DEFAULT_LLM_TIMEOUT_SECONDS, LLM_MODEL_CHOICES, default_llm_api_key_env, default_llm_base_url, default_llm_model
from ..prompting import default_prompt_labels, profile_prompt_label, prompt_labels
from ..workflow_profiles import get_workflow_profile
from .models import (
    BatchRunRequest,
    COOKIE_FILE_FILTER,
    EXECUTABLE_FILE_FILTER,
    MEDIA_FILE_FILTER,
    PROMPT_FILE_FILTER,
    SINGLE_INPUT_FILE_FILTER,
    FollowupRunRequest,
    GuiRunResult,
    LLMGuiSettings,
    ModelSettings,
    PersistedGuiState,
    SingleRunRequest,
)
from .service import DesktopWorkflowService, GuiStateStore, ROOT
from .style import app_qss
from .widgets import (
    BrowseRow,
    Card,
    CheckList,
    CollapsibleCard,
    FormRow,
    PathList,
    StatusText,
    combo_value,
    label,
    setup_combo,
)


_GUI_LOG_PATH: Path | None = None
_FAULTHANDLER_FILE = None


def configure_splitter(splitter: QSplitter, *, left: int, right: int, handle_width: int = 6) -> None:
    """统一初始化桌面页左右分栏的默认比例与手柄宽度。"""

    splitter.setHandleWidth(handle_width)
    splitter.setSizes([left, right])


def current_gui_log_path() -> Path | None:
    """返回当前 GUI 会话日志文件路径，供错误面板展示。"""

    return _GUI_LOG_PATH


def install_gui_logging() -> Path:
    """安装 GUI 会话日志和 faulthandler，保留闪退时的最小证据。"""

    global _FAULTHANDLER_FILE, _GUI_LOG_PATH
    if _GUI_LOG_PATH is not None:
        return _GUI_LOG_PATH

    log_dir = ROOT / "outputs" / "gui_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _GUI_LOG_PATH = log_dir / f"desktop_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=str(_GUI_LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        encoding="utf-8",
        force=True,
    )

    _FAULTHANDLER_FILE = _GUI_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_FAULTHANDLER_FILE, all_threads=True)

    original_excepthook = sys.excepthook

    def log_uncaught_exception(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001 - sys.excepthook contract
        logging.critical("Uncaught GUI exception", exc_info=(exc_type, exc_value, exc_tb))
        original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = log_uncaught_exception
    return _GUI_LOG_PATH


class TaskWorker(QObject):
    """把长任务放进 QThread，并只通过 signal 回到 UI 线程。"""

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, task: Callable[[Callable[[float, str], None]], object]) -> None:
        super().__init__()
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            logging.info("GUI worker task started")
            result = self._task(lambda pct, message: self.progress.emit(float(pct), str(message)))
            logging.info("GUI worker task finished")
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            logging.exception("GUI worker task failed")
            self.failed.emit(str(exc), traceback.format_exc())


class ModelSettingsPanel(CollapsibleCard):
    """FunASR 模型与分段参数面板。"""

    def __init__(self, *, collapsed: bool = True) -> None:
        super().__init__(
            "FunASR 转写参数",
            "普通场景只需要选择“场景预设”和设备；模型 ID、VAD、分段阈值放在高级设置里。",
            collapsed=collapsed,
        )
        self.scenario = QComboBox()
        setup_combo(self.scenario, SCENARIO_PRESETS.keys(), DEFAULT_SCENARIO)
        self.device = QComboBox()
        setup_combo(self.device, ["auto", "cuda", "cpu"], "auto")
        self.language = QComboBox()
        setup_combo(self.language, ["auto", "zh", "en", "yue", "ja", "ko"], "auto")
        self.speaker = QCheckBox("启用说话人区分 V1（会议/访谈可用，重叠发言仍可能不准）")
        self.use_itn = QCheckBox("数字、日期等文本归一化 ITN")
        self.use_itn.setChecked(True)
        self.merge_vad = QCheckBox("合并 VAD 分段")
        self.merge_vad.setChecked(True)
        self.split_lines = QCheckBox("按标点拆行，方便后续阅读")
        self.split_lines.setChecked(True)

        compact = QWidget()
        compact_layout = QGridLayout(compact)
        compact_layout.setContentsMargins(0, 0, 0, 0)
        compact_layout.setHorizontalSpacing(10)
        compact_layout.setVerticalSpacing(10)
        compact_layout.addWidget(FormRow("场景预设", self.scenario, "会联动推荐的 batch、语言、ITN 和分段参数。"), 0, 0)
        compact_layout.addWidget(FormRow("运行设备", self.device, "auto 会让 FunASR 自行选择；显存紧张时可手动选 cpu。"), 0, 1)
        compact_layout.addWidget(FormRow("语言", self.language), 1, 0)
        compact_layout.addWidget(self.speaker, 1, 1)
        compact_layout.addWidget(self.use_itn, 2, 0)
        compact_layout.addWidget(self.merge_vad, 2, 1)
        compact_layout.addWidget(self.split_lines, 3, 0, 1, 2)
        self.layout.insertWidget(2, compact)

        self.model_preset = QComboBox()
        setup_combo(self.model_preset, MODEL_PRESETS.keys(), "paraformer-zh (offline)")
        self.model_custom = QLineEdit()
        self.model_custom.setPlaceholderText("可选：自定义 ASR 模型 ID，填写后覆盖 preset")
        self.vad_preset = QComboBox()
        setup_combo(self.vad_preset, VAD_PRESETS.keys(), "fsmn-vad")
        self.vad_custom = QLineEdit()
        self.vad_custom.setPlaceholderText("可选：自定义 VAD 模型 ID")
        self.punc_preset = QComboBox()
        setup_combo(self.punc_preset, PUNC_PRESETS.keys(), "none")
        self.punc_custom = QLineEdit()
        self.punc_custom.setPlaceholderText("可选：自定义标点模型 ID")
        self.speaker_count = QSpinBox()
        self.speaker_count.setRange(0, 20)
        self.speaker_count.setSpecialValueText("自动")
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 120)
        self.batch_size.setValue(10)
        self.max_segment = QSpinBox()
        self.max_segment.setRange(5000, 180000)
        self.max_segment.setSingleStep(5000)
        self.max_segment.setValue(30000)
        self.merge_length = QSpinBox()
        self.merge_length.setRange(1, 120)
        self.merge_length.setValue(15)
        self.verbose = QCheckBox("输出详细调试信息")

        advanced = QGridLayout()
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setHorizontalSpacing(10)
        advanced.setVerticalSpacing(10)
        advanced.addWidget(FormRow("ASR preset", self.model_preset), 0, 0)
        advanced.addWidget(FormRow("Custom ASR model", self.model_custom), 0, 1)
        advanced.addWidget(FormRow("VAD preset", self.vad_preset), 1, 0)
        advanced.addWidget(FormRow("Custom VAD model", self.vad_custom), 1, 1)
        advanced.addWidget(FormRow("Punctuation preset", self.punc_preset), 2, 0)
        advanced.addWidget(FormRow("Custom punctuation model", self.punc_custom), 2, 1)
        advanced.addWidget(FormRow("已知说话人数", self.speaker_count, "0 表示自动聚类。"), 3, 0)
        advanced.addWidget(FormRow("batch_size_s", self.batch_size), 3, 1)
        advanced.addWidget(FormRow("max_single_segment_ms", self.max_segment), 4, 0)
        advanced.addWidget(FormRow("merge_length_s", self.merge_length), 4, 1)
        advanced.addWidget(self.verbose, 5, 0, 1, 2)
        self.content_layout.addLayout(advanced)
        self.speaker.toggled.connect(self._speaker_toggled)

    def _speaker_toggled(self, enabled: bool) -> None:
        if enabled:
            idx = self.model_preset.findText("paraformer-zh (speaker diarization)")
            if idx >= 0:
                self.model_preset.setCurrentIndex(idx)
            if combo_value(self.punc_preset) == "none":
                idx = self.punc_preset.findText("ct-punc-c")
                if idx >= 0:
                    self.punc_preset.setCurrentIndex(idx)

    def settings(self) -> ModelSettings:
        return ModelSettings(
            scenario_name=combo_value(self.scenario),
            model_preset=combo_value(self.model_preset),
            model_custom=self.model_custom.text().strip(),
            vad_preset=combo_value(self.vad_preset),
            vad_custom=self.vad_custom.text().strip(),
            punc_preset=combo_value(self.punc_preset),
            punc_custom=self.punc_custom.text().strip(),
            language=combo_value(self.language),
            device=combo_value(self.device),
            use_itn=self.use_itn.isChecked(),
            merge_vad=self.merge_vad.isChecked(),
            speaker_diarization=self.speaker.isChecked(),
            speaker_count=self.speaker_count.value(),
            verbose=self.verbose.isChecked(),
            batch_size_s=self.batch_size.value(),
            max_single_segment_ms=self.max_segment.value(),
            merge_length_s=self.merge_length.value(),
            split_lines=self.split_lines.isChecked(),
        )


class LLMSettingsPanel(Card):
    """LLM 与 Brave Search 配置面板。"""

    def __init__(self, *, purpose: str = "summary") -> None:
        help_text = (
            "OpenAI-compatible 接口用于生成 Markdown 总结。Brave Search 是可选外部背景，"
            "会明确写入 prompt，不能覆盖转写证据。"
            if purpose == "summary"
            else "追问/改写必须调用 LLM；Brave Search 只作为本轮问题的外部背景。"
        )
        super().__init__("LLM 与联网补充", help_text)
        self.run_llm = QCheckBox("调用 LLM 生成总结" if purpose == "summary" else "调用 LLM 进行追问/改写")
        self.run_llm.setChecked(purpose != "summary")
        self.base_url = QLineEdit(default_llm_base_url())
        self.model = QComboBox()
        self.model.setEditable(True)
        setup_combo(self.model, LLM_MODEL_CHOICES, default_llm_model() or DEFAULT_LLM_MODEL)
        self.api_key_env = QLineEdit(default_llm_api_key_env())
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("可留空，优先使用环境变量")
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 2)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(0.1)
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 1800)
        self.timeout.setValue(DEFAULT_LLM_TIMEOUT_SECONDS)
        self.enable_web = QCheckBox("本轮启用 Brave Search 联网补充")
        self.brave_key_env = QLineEdit("BRAVE_SEARCH_API_KEY")
        self.brave_key = QLineEdit()
        self.brave_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.brave_count = QSpinBox()
        self.brave_count.setRange(1, 10)
        self.brave_count.setValue(5)
        self.brave_timeout = QSpinBox()
        self.brave_timeout.setRange(5, 120)
        self.brave_timeout.setValue(25)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(self.run_llm, 0, 0, 1, 2)
        grid.addWidget(FormRow("Base URL", self.base_url, "默认 http://127.0.0.1:4000/v1；也可填其他 OpenAI-compatible 地址。"), 1, 0)
        grid.addWidget(FormRow("模型名", self.model, "默认 gpt-5.6-sol；已内置其他 GPT-5.6、GPT-5.5 与 DeepSeek 备用模型。"), 1, 1)
        grid.addWidget(FormRow("API Key 环境变量", self.api_key_env), 2, 0)
        grid.addWidget(FormRow("API Key（可选）", self.api_key), 2, 1)
        grid.addWidget(FormRow("temperature", self.temperature), 3, 0)
        grid.addWidget(FormRow("timeout 秒", self.timeout), 3, 1)
        grid.addWidget(self.enable_web, 4, 0, 1, 2)
        grid.addWidget(FormRow("Brave API Key 环境变量", self.brave_key_env), 5, 0)
        grid.addWidget(FormRow("Brave API Key（可选）", self.brave_key), 5, 1)
        grid.addWidget(FormRow("Brave 返回条数", self.brave_count), 6, 0)
        grid.addWidget(FormRow("Brave timeout 秒", self.brave_timeout), 6, 1)
        self.layout.addLayout(grid)

    def apply_state(self, state: PersistedGuiState) -> None:
        self.base_url.setText(state.llm_base_url)
        model = (state.llm_model or DEFAULT_LLM_MODEL).strip()
        idx = self.model.findData(model)
        if idx < 0:
            idx = self.model.findText(model)
        if idx >= 0:
            self.model.setCurrentIndex(idx)
        else:
            self.model.setCurrentText(model)
        self.api_key_env.setText(state.llm_api_key_env)
        self.brave_key_env.setText(state.brave_api_key_env)

    def settings(self) -> LLMGuiSettings:
        return LLMGuiSettings(
            run_llm=self.run_llm.isChecked(),
            base_url=self.base_url.text().strip(),
            model=self.model.currentText().strip(),
            api_key=self.api_key.text().strip(),
            api_key_env=self.api_key_env.text().strip(),
            temperature=float(self.temperature.value()),
            timeout_seconds=int(self.timeout.value()),
            enable_web_search=self.enable_web.isChecked(),
            brave_api_key=self.brave_key.text().strip(),
            brave_api_key_env=self.brave_key_env.text().strip(),
            brave_count=int(self.brave_count.value()),
            brave_timeout_seconds=int(self.brave_timeout.value()),
        )


class ResultsPanel(Card):
    """运行状态、转写预览、总结预览和元数据。"""

    def __init__(self) -> None:
        super().__init__("运行结果", "这里显示进度、日志、预览和文件路径；本地 GUI 不需要浏览器下载链接，直接打开文件/目录。")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("等待任务。")
        self.status_label.setObjectName("toolbarBadge")
        self.tabs = QTabWidget()
        self.log = StatusText("等待运行。")
        self.transcript = StatusText("转写预览将在任务完成后显示。")
        self.summary = StatusText("总结预览将在任务完成后显示。")
        self.manifest = StatusText("Manifest / 元数据将在任务完成后显示。")
        self.tabs.addTab(self.log, "运行日志")
        self.tabs.addTab(self.transcript, "转写预览")
        self.tabs.addTab(self.summary, "总结预览")
        self.tabs.addTab(self.manifest, "元数据")
        self.open_dir_button = QPushButton("打开输出目录")
        self.open_dir_button.setObjectName("secondaryButton")
        self.open_final_button = QPushButton("打开 Markdown")
        self.open_final_button.setObjectName("secondaryButton")
        self.open_bundle_button = QPushButton("打开 zip 所在目录")
        self.open_bundle_button.setObjectName("secondaryButton")
        row = QHBoxLayout()
        row.addWidget(self.open_dir_button)
        row.addWidget(self.open_final_button)
        row.addWidget(self.open_bundle_button)
        row.addStretch(1)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.tabs, 1)
        self.layout.addLayout(row)
        self._output_dir = ""
        self._final_path = ""
        self._bundle_path = ""
        self.open_dir_button.clicked.connect(lambda: self._open_path(self._output_dir))
        self.open_final_button.clicked.connect(lambda: self._open_path(self._final_path))
        self.open_bundle_button.clicked.connect(lambda: self._open_path(str(Path(self._bundle_path).parent) if self._bundle_path else ""))

    def reset(self) -> None:
        self.progress.setValue(0)
        self.status_label.setText("等待任务。")
        self.log.setPlainText("等待运行。")
        self.transcript.setPlainText("")
        self.summary.setPlainText("")
        self.manifest.setPlainText("")
        self._output_dir = ""
        self._final_path = ""
        self._bundle_path = ""

    def append_log(self, message: str) -> None:
        old = self.log.toPlainText().strip()
        self.log.setPlainText((old + "\n" if old else "") + message)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def set_progress(self, pct: float, message: str) -> None:
        self.progress.setValue(max(0, min(100, int(pct * 100))))
        self.status_label.setText(message or "运行中")
        self.append_log(f"[{self.progress.value():03d}%] {message}")

    def show_result(self, result: GuiRunResult) -> None:
        self.progress.setValue(100)
        self.status_label.setText(result.headline)
        self._output_dir = result.output_dir
        self._final_path = result.final_markdown_path
        self._bundle_path = result.bundle_path
        self.transcript.setPlainText(result.preview_text or "")
        self.summary.setPlainText(result.summary_text or "")
        self.manifest.setPlainText(json.dumps(result.meta, ensure_ascii=False, indent=2) if result.meta else "")
        self.append_log(f"完成：{result.headline}")
        if result.output_dir:
            self.append_log(f"输出目录：{result.output_dir}")
        if result.final_markdown_path:
            self.append_log(f"Markdown：{result.final_markdown_path}")
        if result.bundle_path:
            self.append_log(f"Zip：{result.bundle_path}")

    def show_error(self, message: str, tb: str) -> None:
        self.status_label.setText("任务失败")
        self.append_log(f"❌ {message}\n{tb}")
        log_path = current_gui_log_path()
        if log_path:
            self.append_log(f"GUI 日志：{log_path}")
        self.manifest.setPlainText(tb)

    @staticmethod
    def _open_path(path: str) -> None:
        if not path:
            return
        target = Path(path).expanduser()
        if not target.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


class SingleWorkflowPage(QWidget):
    inspect_requested = Signal(object)
    run_requested = Signal(object)
    check_llm_requested = Signal(object)
    state_changed = Signal()

    def __init__(self, service: DesktopWorkflowService, state: PersistedGuiState) -> None:
        super().__init__()
        self.service = service
        self.state = state
        self.setObjectName("pageSurface")
        self._build_ui()
        self._apply_state(state)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        title = QLabel("单条音视频总结")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(label("适合 B 站 / YouTube / 本地单个录音录像，也支持已有转写稿：下载或读取媒体 → ffmpeg/FunASR 或直接读取 transcript → prompt 组合 → 可选 LLM 总结。", "bodyText"))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        form_holder = QWidget()
        form_holder.setObjectName("pageScrollContent")
        form_layout = QVBoxLayout(form_holder)
        form_layout.setContentsMargins(0, 0, 10, 0)
        form_layout.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_holder)
        splitter.addWidget(scroll)

        source_card = Card("1. 输入来源与本地路由", "URL 和本地文件二选一；如果同时填写 URL，优先使用 URL。多文件请使用“批量转写”。")
        self.source_url = QLineEdit()
        self.source_url.setPlaceholderText("Bilibili / YouTube / 其他 yt-dlp 支持链接")
        self.media_list = PathList()
        self.media_list.add_button.clicked.connect(self._choose_media)
        self.profile = QComboBox()
        setup_combo(self.profile, self.service.profile_options(), "auto")
        self.tag = QLineEdit()
        self.output_root = BrowseRow("默认 outputs/workbench_runs")
        self.download_root = BrowseRow("可选：媒体下载/归档目录")
        self.note_root = BrowseRow("可选：最终 Markdown 笔记目录")
        source_card.layout.addWidget(FormRow("链接输入", self.source_url, "会员/充电/私有内容通常需要 cookies。"))
        source_card.layout.addWidget(FormRow("本地媒体/转写稿", self.media_list, "单条模式只会取列表第一个文件；.txt/.md 会跳过转写，直接总结。"))
        source_card.layout.addWidget(FormRow("Workflow / 本地路由配置", self.profile, "只决定默认标签、目录和默认 prompt，不等于真正的 prompt 内容。"))
        source_card.layout.addWidget(FormRow("标签", self.tag))
        source_card.layout.addWidget(FormRow("运行输出根目录", self.output_root, "中间文件、转写、manifest、zip 都会保存在这里。"))
        source_card.layout.addWidget(FormRow("媒体下载/归档目录", self.download_root))
        source_card.layout.addWidget(FormRow("总结笔记目录", self.note_root))
        form_layout.addWidget(source_card)

        prompt_card = Card("2. Prompt 组合", "真正发送给 LLM/agent 的指令来自这里：默认 agents.md、项目 prompts、本次上传 prompt 和最后的追加说明会按顺序合并。")
        self.prompt_checks = CheckList()
        prompt_card.layout.addWidget(self.prompt_checks)
        self.prompt_files = PathList("尚未添加本次 prompt 文件")
        self.prompt_files.add_button.clicked.connect(self._choose_prompts)
        self.inline_prompt = QTextEdit()
        self.inline_prompt.setPlaceholderText("例如：请输出适合 Obsidian 的层级标题；保留关键论点和可引用原句；最后列出待人工核对点。")
        self.inline_prompt.setMinimumHeight(92)
        prompt_card.layout.addWidget(FormRow("本次 prompt 文件", self.prompt_files))
        prompt_card.layout.addWidget(FormRow("本次追加指令", self.inline_prompt))
        form_layout.addWidget(prompt_card)

        download_card = CollapsibleCard("3. 下载、Cookies 与外部工具", "普通公开视频可保持 auto；登录态或会员内容请提供 cookies。", collapsed=True)
        self.download_mode = QComboBox()
        setup_combo(self.download_mode, [("audio", "只抽音频，最快"), ("video", "下载视频并合并 mp4")], "audio")
        self.keep_download = QCheckBox("保留 URL 下载得到的原始媒体")
        self.keep_download.setChecked(True)
        self.cookie_mode = QComboBox()
        setup_combo(self.cookie_mode, [("auto", "auto：自动找常用 cookies 路径"), ("upload/path", "手动：使用下面 cookies 路径"), ("none", "不使用 cookies")], "auto")
        self.cookie_path = BrowseRow("cookies.txt / bili_cookies.txt / youtube_cookies.txt")
        self.ytdlp_path = BrowseRow("可选：yt-dlp.exe 路径")
        self.ffmpeg_path = BrowseRow("可选：ffmpeg.exe 路径")
        download_card.content_layout.addWidget(FormRow("URL 下载模式", self.download_mode))
        download_card.content_layout.addWidget(self.keep_download)
        download_card.content_layout.addWidget(FormRow("Cookies 模式", self.cookie_mode))
        download_card.content_layout.addWidget(FormRow("Cookies 文件", self.cookie_path))
        download_card.content_layout.addWidget(FormRow("yt-dlp 路径", self.ytdlp_path))
        download_card.content_layout.addWidget(FormRow("ffmpeg 路径", self.ffmpeg_path))
        form_layout.addWidget(download_card)

        self.model_panel = ModelSettingsPanel(collapsed=True)
        form_layout.addWidget(self.model_panel)
        self.llm_panel = LLMSettingsPanel(purpose="summary")
        form_layout.addWidget(self.llm_panel)

        output_card = Card("5. 输出策略", "final_summary.md 永远保存在 run 目录；如果设置了总结笔记目录，默认会额外同步一份 Markdown。")
        self.include_prompt = QCheckBox("在 final_summary.md 中附上完整 prompt（便于交给 agent 复核）")
        self.include_transcript = QCheckBox("在 final_summary.md 中附上转写全文")
        self.write_note = QCheckBox("设置总结笔记目录时，同步写入一份 Markdown")
        self.write_note.setChecked(True)
        self.overwrite_note = QCheckBox("允许覆盖同名笔记（默认不覆盖，会自动加序号）")
        output_card.layout.addWidget(self.include_prompt)
        output_card.layout.addWidget(self.include_transcript)
        output_card.layout.addWidget(self.write_note)
        output_card.layout.addWidget(self.overwrite_note)
        form_layout.addWidget(output_card)

        action_card = Card("开始运行", "建议先检查路径，再启动长任务。运行中窗口会保持响应。")
        self.inspect_button = QPushButton("检查路径和环境")
        self.inspect_button.setObjectName("secondaryButton")
        self.check_llm_button = QPushButton("检查 LLM")
        self.check_llm_button.setObjectName("secondaryButton")
        self.run_button = QPushButton("开始完整工作流")
        self.run_button.setObjectName("primaryButton")
        action_row = QHBoxLayout()
        action_row.addWidget(self.inspect_button)
        action_row.addWidget(self.check_llm_button)
        action_row.addWidget(self.run_button)
        action_card.layout.addLayout(action_row)
        form_layout.addWidget(action_card)
        form_layout.addStretch(1)

        self.results = ResultsPanel()
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 7)
        configure_splitter(splitter, left=760, right=980)

        for row in [self.output_root, self.download_root, self.note_root]:
            row.button.clicked.connect(lambda checked=False, target=row: self._choose_directory(target))
        self.cookie_path.button.clicked.connect(lambda: self._choose_file_for_row(self.cookie_path, COOKIE_FILE_FILTER))
        self.ytdlp_path.button.clicked.connect(lambda: self._choose_file_for_row(self.ytdlp_path, EXECUTABLE_FILE_FILTER))
        self.ffmpeg_path.button.clicked.connect(lambda: self._choose_file_for_row(self.ffmpeg_path, EXECUTABLE_FILE_FILTER))
        self.profile.currentIndexChanged.connect(self._on_profile_changed)
        self.inspect_button.clicked.connect(lambda: self.inspect_requested.emit(self.request()))
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.request()))
        self.check_llm_button.clicked.connect(lambda: self.check_llm_requested.emit(self.llm_panel.settings()))

    def _apply_state(self, state: PersistedGuiState) -> None:
        setup_combo(self.profile, self.service.profile_options(), state.profile_key or "auto")
        defaults = state.selected_prompt_labels or default_prompt_labels()
        self.prompt_checks.set_options(prompt_labels(), defaults)
        self.output_root.set_text(state.output_root)
        self.download_root.set_text(state.download_root)
        self.note_root.set_text(state.note_root)
        self.ffmpeg_path.set_text(state.ffmpeg_path)
        self.ytdlp_path.set_text(state.ytdlp_path)
        self.llm_panel.apply_state(state)

    def _on_profile_changed(self) -> None:
        key = combo_value(self.profile)
        profile = get_workflow_profile(key) if key and key != "auto" else None
        selected = default_prompt_labels()
        if profile:
            self.tag.setText(profile.tag or "")
            self.download_root.set_text(profile.download_root or "")
            self.note_root.set_text(profile.note_root or "")
            self.output_root.set_text(profile.output_root or "")
            label_value = profile_prompt_label(profile.prompt_name)
            if label_value and label_value not in selected:
                selected.append(label_value)
        self.prompt_checks.set_selected(selected)
        self.state_changed.emit()

    def _choose_media(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择本地媒体或转写稿", "", SINGLE_INPUT_FILE_FILTER)
        self.media_list.add_paths(files)

    def _choose_prompts(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择本次 prompt 文件", "", PROMPT_FILE_FILTER)
        self.prompt_files.add_paths(files)

    def _choose_directory(self, row: BrowseRow) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", row.text() or str(ROOT))
        if path:
            row.set_text(path)

    def _choose_file_for_row(self, row: BrowseRow, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", row.text() or str(ROOT), file_filter)
        if path:
            row.set_text(path)

    def request(self) -> SingleRunRequest:
        return SingleRunRequest(
            source_url=self.source_url.text().strip(),
            local_media=self.media_list.first_path(),
            profile_key=combo_value(self.profile),
            tag=self.tag.text().strip(),
            selected_prompt_labels=self.prompt_checks.selected(),
            uploaded_prompt_files=self.prompt_files.paths(),
            inline_prompt=self.inline_prompt.toPlainText().strip(),
            output_root=self.output_root.text(),
            download_root=self.download_root.text(),
            note_root=self.note_root.text(),
            download_mode=combo_value(self.download_mode),
            keep_download=self.keep_download.isChecked(),
            cookie_mode=combo_value(self.cookie_mode),
            cookie_path=self.cookie_path.text(),
            ytdlp_path=self.ytdlp_path.text(),
            ffmpeg_path=self.ffmpeg_path.text(),
            include_prompt_in_final=self.include_prompt.isChecked(),
            include_transcript_in_final=self.include_transcript.isChecked(),
            write_note=self.write_note.isChecked(),
            overwrite_note=self.overwrite_note.isChecked(),
            model=self.model_panel.settings(),
            llm=self.llm_panel.settings(),
        )

    def persisted_state(self) -> PersistedGuiState:
        req = self.request()
        return PersistedGuiState(
            profile_key=req.profile_key,
            output_root=req.output_root,
            download_root=req.download_root,
            note_root=req.note_root,
            ffmpeg_path=req.ffmpeg_path,
            ytdlp_path=req.ytdlp_path,
            llm_base_url=req.llm.base_url,
            llm_model=req.llm.model,
            llm_api_key_env=req.llm.api_key_env,
            brave_api_key_env=req.llm.brave_api_key_env,
            selected_prompt_labels=req.selected_prompt_labels,
        )

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.inspect_button.setEnabled(not busy)
        self.check_llm_button.setEnabled(not busy)


class BatchPage(QWidget):
    run_requested = Signal(object)

    def __init__(self, state: PersistedGuiState) -> None:
        super().__init__()
        self.state = state
        self.setObjectName("pageSurface")
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        title = QLabel("批量转写")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(label("适合一次处理多个本地音视频。该页只做转写和打包，不强制调用 LLM，也不会生成单个 final_summary.md。", "bodyText"))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        form_holder = QWidget()
        form_holder.setObjectName("pageScrollContent")
        form_layout = QVBoxLayout(form_holder)
        form_layout.setContentsMargins(0, 0, 10, 0)
        form_layout.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_holder)
        splitter.addWidget(scroll)

        media_card = Card("1. 批量媒体", "可选择多个文件，也可以选择目录；目录默认递归查找常见音视频后缀。")
        self.files = PathList()
        self.files.add_button.clicked.connect(self._choose_files)
        self.directory = BrowseRow("可选：包含音视频的目录")
        self.directory.button.clicked.connect(self._choose_directory)
        self.recursive = QCheckBox("递归扫描子目录")
        self.recursive.setChecked(True)
        media_card.layout.addWidget(FormRow("媒体文件", self.files))
        media_card.layout.addWidget(FormRow("媒体目录", self.directory))
        media_card.layout.addWidget(self.recursive)
        form_layout.addWidget(media_card)

        output_card = Card("2. 输出与预处理", "每个文件会生成独立子目录，最后统一打包 workflow_outputs.zip。")
        self.output_root = BrowseRow("默认 outputs/batch_runs")
        self.output_root.set_text(self.state.batch_output_root or self.state.output_root)
        self.output_root.button.clicked.connect(lambda: self._choose_directory_for_row(self.output_root))
        self.ffmpeg_path = BrowseRow("可选：ffmpeg.exe 路径")
        self.ffmpeg_path.set_text(self.state.batch_ffmpeg_path or self.state.ffmpeg_path)
        self.ffmpeg_path.button.clicked.connect(lambda: self._choose_file_for_row(self.ffmpeg_path, EXECUTABLE_FILE_FILTER))
        self.preprocess = QCheckBox("每个文件先转成 16k 单声道 WAV（推荐，视频必须开启）")
        self.preprocess.setChecked(True)
        self.out_format = QComboBox()
        setup_combo(self.out_format, ["md", "txt"], "md")
        output_card.layout.addWidget(FormRow("批量输出根目录", self.output_root))
        output_card.layout.addWidget(FormRow("ffmpeg 路径", self.ffmpeg_path))
        output_card.layout.addWidget(self.preprocess)
        output_card.layout.addWidget(FormRow("输出格式", self.out_format))
        form_layout.addWidget(output_card)
        self.model_panel = ModelSettingsPanel(collapsed=True)
        form_layout.addWidget(self.model_panel)
        action = Card("开始批量转写")
        self.run_button = QPushButton("开始批量转写")
        self.run_button.setObjectName("primaryButton")
        action.layout.addWidget(self.run_button)
        form_layout.addWidget(action)
        form_layout.addStretch(1)
        self.results = ResultsPanel()
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 7)
        configure_splitter(splitter, left=700, right=960)
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.request()))

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择多个媒体文件", "", MEDIA_FILE_FILTER)
        self.files.add_paths(files)

    def _choose_directory(self) -> None:
        self._choose_directory_for_row(self.directory)

    def _choose_directory_for_row(self, row: BrowseRow) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", row.text() or str(ROOT))
        if path:
            row.set_text(path)

    def _choose_file_for_row(self, row: BrowseRow, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", row.text() or str(ROOT), file_filter)
        if path:
            row.set_text(path)

    def request(self) -> BatchRunRequest:
        return BatchRunRequest(
            media_files=self.files.paths(),
            media_directory=self.directory.text(),
            recursive=self.recursive.isChecked(),
            output_root=self.output_root.text(),
            ffmpeg_path=self.ffmpeg_path.text(),
            out_format=combo_value(self.out_format),
            preprocess_to_wav=self.preprocess.isChecked(),
            model=self.model_panel.settings(),
        )

    def persisted_state_patch(self) -> dict[str, str]:
        return {
            "batch_output_root": self.output_root.text(),
            "batch_ffmpeg_path": self.ffmpeg_path.text(),
        }

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)


class FollowupPage(QWidget):
    load_requested = Signal(str, str)
    run_requested = Signal(object)
    promote_requested = Signal(str)

    def __init__(self, state: PersistedGuiState) -> None:
        super().__init__()
        self.state = state
        self.setObjectName("pageSurface")
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        title = QLabel("总结追问与改写")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(label("基于已有 run 目录继续问答、改写 summary。默认写入 _followup_cache；确认后可安全写回 final_summary.md。", "bodyText"))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        form_holder = QWidget()
        form_holder.setObjectName("pageScrollContent")
        form_layout = QVBoxLayout(form_holder)
        form_layout.setContentsMargins(0, 0, 10, 0)
        form_layout.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_holder)
        splitter.addWidget(scroll)

        context = Card("1. 选择 run 目录", "可以选择 run 目录，也可以选择 run_manifest.json 所在目录。")
        self.run_dir = BrowseRow("outputs/workbench_runs/<timestamp>")
        self.run_dir.set_text(self.state.followup_run_dir)
        self.run_dir.button.clicked.connect(self._choose_run_dir)
        self.session_id = QLineEdit()
        self.session_id.setPlaceholderText("留空会自动创建新会话")
        self.load_button = QPushButton("加载上下文")
        self.load_button.setObjectName("secondaryButton")
        context.layout.addWidget(FormRow("Run 目录", self.run_dir))
        context.layout.addWidget(FormRow("Session ID", self.session_id))
        context.layout.addWidget(self.load_button)
        form_layout.addWidget(context)

        question_card = Card("2. 本轮追问 / 改写要求", "可以问问题，也可以要求重写结构、补充行动项、提炼适合 Obsidian 的版本。")
        self.question = QTextEdit()
        self.question.setMinimumHeight(120)
        self.question.setPlaceholderText("例如：请把当前总结改写成“背景-主要观点-证据-争议-待核对问题”的结构，并保留原文依据。")
        self.update_summary = QCheckBox("同时生成更新后的完整总结版本")
        self.update_summary.setChecked(True)
        question_card.layout.addWidget(FormRow("问题或改写要求", self.question))
        question_card.layout.addWidget(self.update_summary)
        form_layout.addWidget(question_card)

        self.llm_panel = LLMSettingsPanel(purpose="followup")
        self.llm_panel.apply_state(self.state)
        form_layout.addWidget(self.llm_panel)
        action = Card("3. 执行与写回", "追问产生的新摘要先进入缓存；确认满意后再写回 final_summary.md，并自动备份旧文件。")
        self.run_button = QPushButton("发送追问 / 改写")
        self.run_button.setObjectName("primaryButton")
        self.promote_button = QPushButton("将当前摘要写回 final_summary.md")
        self.promote_button.setObjectName("secondaryButton")
        row = QHBoxLayout()
        row.addWidget(self.run_button)
        row.addWidget(self.promote_button)
        action.layout.addLayout(row)
        form_layout.addWidget(action)
        form_layout.addStretch(1)
        self.results = ResultsPanel()
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 7)
        configure_splitter(splitter, left=860, right=840)
        self.load_button.clicked.connect(lambda: self.load_requested.emit(self.run_dir.text(), self.session_id.text().strip()))
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.request()))
        self.promote_button.clicked.connect(lambda: self.promote_requested.emit(self.run_dir.text()))

    def _choose_run_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 run 目录", self.run_dir.text() or str(ROOT / "outputs"))
        if path:
            self.run_dir.set_text(path)

    def set_run_dir(self, run_dir: str) -> None:
        if run_dir:
            self.run_dir.set_text(run_dir)

    def show_session(self, session: dict[str, object]) -> None:
        self.run_dir.set_text(str(session.get("run_dir") or ""))
        self.session_id.setText(str(session.get("session_id") or self.session_id.text()))
        self.results.status_label.setText("已加载追问上下文")
        self.results.summary.setPlainText(str(session.get("current_summary_text") or ""))
        self.results.transcript.setPlainText(str(session.get("transcript_text") or ""))
        self.results.manifest.setPlainText(json.dumps(session.get("meta") or {}, ensure_ascii=False, indent=2))
        self.results.append_log(f"加载 run：{session.get('run_dir')}")

    def request(self) -> FollowupRunRequest:
        settings = self.llm_panel.settings()
        settings.run_llm = True
        return FollowupRunRequest(
            run_reference=self.run_dir.text(),
            session_id=self.session_id.text().strip(),
            question=self.question.toPlainText().strip(),
            update_summary=self.update_summary.isChecked(),
            llm=settings,
        )

    def persisted_state_patch(self) -> dict[str, str]:
        return {
            "followup_run_dir": self.run_dir.text(),
        }

    def set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.load_button.setEnabled(not busy)
        self.promote_button.setEnabled(not busy)


class MainWindow(QMainWindow):
    """Audio Workbench 本地桌面主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.service = DesktopWorkflowService()
        self.state_store = GuiStateStore()
        self.state = self.state_store.load()
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None
        self.setWindowTitle("FunASR Audio Workbench Desktop")
        self.resize(1480, 920)
        self.setMinimumSize(1180, 760)
        self._build_ui()
        self._build_menu()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(18)

        header = QFrame()
        header.setObjectName("topHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 20)
        header_layout.setSpacing(18)
        hero = QLabel("FunASR Audio Workbench")
        hero.setObjectName("heroTitle")
        subtitle = label("本地音视频下载、预处理、转写、Prompt 组合、LLM 总结、追问改写的一体化桌面入口。", "bodyText")
        title_col = QVBoxLayout()
        title_col.setSpacing(6)
        title_col.addWidget(hero)
        title_col.addWidget(subtitle)
        header_layout.addLayout(title_col, 1)
        for text in ["Local first", "yt-dlp + ffmpeg", "FunASR + LLM", "Brave 可选"]:
            badge = QLabel(text)
            badge.setObjectName("toolbarBadge")
            header_layout.addWidget(badge)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(18)
        root.addLayout(body, 1)
        nav = QFrame()
        nav.setObjectName("navPanel")
        nav.setFixedWidth(230)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(14, 14, 14, 14)
        nav_layout.setSpacing(10)
        self.single_nav = self._nav_button("单条总结")
        self.batch_nav = self._nav_button("批量转写")
        self.follow_nav = self._nav_button("追问改写")
        nav_layout.addWidget(label("工作流", "sectionTitle"))
        nav_layout.addWidget(self.single_nav)
        nav_layout.addWidget(self.batch_nav)
        nav_layout.addWidget(self.follow_nav)
        nav_layout.addStretch(1)
        nav_layout.addWidget(label("状态说明\nready：可运行\nbusy：后台任务中\nfailed：需要处理错误", "helperText"))
        body.addWidget(nav)

        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        self.single_page = SingleWorkflowPage(self.service, self.state)
        self.batch_page = BatchPage(self.state)
        self.follow_page = FollowupPage(self.state)
        self.stack.addWidget(self.single_page)
        self.stack.addWidget(self.batch_page)
        self.stack.addWidget(self.follow_page)
        self.single_nav.clicked.connect(lambda: self._select_page(0))
        self.batch_nav.clicked.connect(lambda: self._select_page(1))
        self.follow_nav.clicked.connect(lambda: self._select_page(2))
        self.single_nav.setChecked(True)

        self.single_page.inspect_requested.connect(self._inspect_environment)
        self.single_page.run_requested.connect(self._run_single)
        self.single_page.check_llm_requested.connect(self._check_llm)
        self.single_page.state_changed.connect(self._save_state)
        self.batch_page.run_requested.connect(self._run_batch)
        self.follow_page.load_requested.connect(self._load_followup)
        self.follow_page.run_requested.connect(self._run_followup)
        self.follow_page.promote_requested.connect(self._promote_summary)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_root = QAction("打开项目目录", self)
        open_root.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT))))
        save_state = QAction("保存当前 GUI 设置", self)
        save_state.triggered.connect(self._save_state)
        file_menu.addAction(open_root)
        file_menu.addAction(save_state)
        help_menu = self.menuBar().addMenu("帮助")
        about = QAction("关于", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _nav_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return button

    def _select_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, button in enumerate([self.single_nav, self.batch_nav, self.follow_nav]):
            button.setChecked(i == index)

    def _set_busy(self, busy: bool) -> None:
        self.single_page.set_busy(busy)
        self.batch_page.set_busy(busy)
        self.follow_page.set_busy(busy)

    def _run_worker(
        self,
        task: Callable[[Callable[[float, str], None]], object],
        progress_target: ResultsPanel,
        done: Callable[[object], None],
    ) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "任务进行中", "已有后台任务正在运行，请等待完成。")
            return
        progress_target.reset()
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = TaskWorker(task)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(progress_target.set_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(done, Qt.ConnectionType.QueuedConnection)
        self._worker.failed.connect(progress_target.show_error, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._finish_worker, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    @Slot()
    def _finish_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)
        self._save_state()

    def _inspect_environment(self, request: SingleRunRequest) -> None:
        try:
            snapshot = self.service.inspect_environment(request)
        except Exception as exc:  # noqa: BLE001
            self.single_page.results.show_error(str(exc), traceback.format_exc())
            return
        lines = [snapshot.headline, ""]
        for card in snapshot.cards:
            lines.append(f"## {card.title}")
            lines.append(f"- 状态：{card.tone}")
            lines.append(f"- 值：{card.value}")
            if card.detail:
                lines.append(f"- 说明：{card.detail}")
            lines.append("")
        self.single_page.results.log.setPlainText("\n".join(lines).strip())
        self.single_page.results.status_label.setText(snapshot.headline)
        self.single_page.results.manifest.setPlainText(json.dumps(snapshot.raw, ensure_ascii=False, indent=2))

    def _run_single(self, request: SingleRunRequest) -> None:
        self._save_state()
        self._run_worker(
            lambda progress: self.service.run_single(request, progress=progress),
            self.single_page.results,
            self._single_done,
        )

    def _single_done(self, result: object) -> None:
        if isinstance(result, GuiRunResult):
            self.single_page.results.show_result(result)
            self.follow_page.set_run_dir(result.output_dir)
        else:
            self.single_page.results.append_log(str(result))

    def _run_batch(self, request: BatchRunRequest) -> None:
        self._run_worker(
            lambda progress: self.service.run_batch(request, progress=progress),
            self.batch_page.results,
            self._batch_done,
        )

    def _batch_done(self, result: object) -> None:
        if isinstance(result, GuiRunResult):
            self.batch_page.results.show_result(result)

    def _load_followup(self, run_dir: str, session_id: str) -> None:
        try:
            session = self.service.load_followup(run_dir, session_id)
            self.follow_page.show_session(session)
        except Exception as exc:  # noqa: BLE001
            self.follow_page.results.show_error(str(exc), traceback.format_exc())

    def _run_followup(self, request: FollowupRunRequest) -> None:
        self._run_worker(
            lambda progress: self.service.run_followup(request, progress=progress),
            self.follow_page.results,
            self._followup_done,
        )

    def _followup_done(self, result: object) -> None:
        if isinstance(result, GuiRunResult):
            self.follow_page.results.show_result(result)

    def _promote_summary(self, run_dir: str) -> None:
        if not run_dir.strip():
            QMessageBox.warning(self, "缺少 run 目录", "请先选择或加载一个 run 目录。")
            return
        answer = QMessageBox.question(
            self,
            "确认写回",
            "这会把当前追问缓存摘要写回 final_summary.md，并自动备份旧文件。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self.service.promote_current_summary(run_dir)
            self.follow_page.results.show_result(result)
        except Exception as exc:  # noqa: BLE001
            self.follow_page.results.show_error(str(exc), traceback.format_exc())

    def _check_llm(self, settings: LLMGuiSettings) -> None:
        self._run_worker(
            lambda progress: self._check_llm_task(settings, progress),
            self.single_page.results,
            self._check_llm_done,
        )

    def _check_llm_task(self, settings: LLMGuiSettings, progress: Callable[[float, str], None]) -> GuiRunResult:
        progress(0.2, "发送 LLM 健康检查请求")
        message = self.service.check_llm(settings)
        progress(1.0, "LLM 检查完成")
        return GuiRunResult(headline="LLM 检查完成", preview_text=message, summary_text=message, meta={"message": message})

    def _check_llm_done(self, result: object) -> None:
        if isinstance(result, GuiRunResult):
            self.single_page.results.show_result(result)

    def _save_state(self) -> None:
        try:
            state = self.single_page.persisted_state()
            for key, value in self.batch_page.persisted_state_patch().items():
                setattr(state, key, value)
            for key, value in self.follow_page.persisted_state_patch().items():
                setattr(state, key, value)
            self.state_store.save(state)
        except Exception:
            pass

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "关于 FunASR Audio Workbench Desktop",
            "本地桌面版把页面、服务、运行时分层：\n"
            "Page/Dialog → Service → yt-dlp/ffmpeg/FunASR/LLM → 本地文件。\n\n"
            "Cookies 和 API Key 不会保存在 GUI 状态文件中。",
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt override
        self._save_state()
        if self._thread is not None:
            QMessageBox.warning(self, "后台任务进行中", "任务仍在运行。为了避免线程在退出时被销毁，请先等待任务完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)


def run_gui() -> int:
    """启动 PySide6 GUI。"""

    log_path = install_gui_logging()
    logging.info("Starting FunASR desktop GUI, log file: %s", log_path)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("FunASR Audio Workbench")
    app.setStyleSheet(app_qss())
    window = MainWindow()
    window.show()
    return int(app.exec())

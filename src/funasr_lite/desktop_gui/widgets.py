from __future__ import annotations

"""桌面 GUI 的可复用控件。

控件只负责展示和收集用户意图，不直接调用 FunASR、ffmpeg 或 LLM。这样可以
保证页面层不会越过服务边界直接控制运行时。
"""

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class Card(QFrame):
    """卡片容器，统一标题、说明和内容布局。"""

    def __init__(self, title: str, help_text: str = "", *, soft: bool = False) -> None:
        super().__init__()
        self.setObjectName("softCard" if soft else "card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 16)
        self.layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        self.layout.addWidget(title_label)
        if help_text:
            helper = QLabel(help_text)
            helper.setObjectName("helperText")
            helper.setWordWrap(True)
            self.layout.addWidget(helper)


class FormRow(QWidget):
    """一行标签 + 控件 + 可选说明。"""

    def __init__(self, label: str, widget: QWidget, help_text: str = "") -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(5)
        label_widget = QLabel(label)
        label_widget.setObjectName("helperText")
        root.addWidget(label_widget)
        root.addWidget(widget)
        if help_text:
            helper = QLabel(help_text)
            helper.setObjectName("helperText")
            helper.setWordWrap(True)
            root.addWidget(helper)


class BrowseRow(QWidget):
    """路径输入 + 浏览按钮。"""

    def __init__(self, placeholder: str = "", button_text: str = "选择") -> None:
        super().__init__()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton(button_text)
        self.button.setObjectName("secondaryButton")
        self.button.setFixedWidth(92)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self.edit, 1)
        root.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_text(self, value: str) -> None:
        self.edit.setText(value or "")


class CheckList(QWidget):
    """Prompt 多选列表。"""

    def __init__(self) -> None:
        super().__init__()
        self._checks: dict[str, QCheckBox] = {}
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(8)

    def set_options(self, labels: Iterable[str], selected: Iterable[str] = ()) -> None:
        while self.root.count():
            item = self.root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checks.clear()
        selected_set = set(selected or [])
        for label in labels:
            check = QCheckBox(label)
            check.setChecked(label in selected_set)
            self._checks[label] = check
            self.root.addWidget(check)
        self.root.addStretch(1)

    def selected(self) -> list[str]:
        return [label for label, check in self._checks.items() if check.isChecked()]

    def set_selected(self, labels: Iterable[str]) -> None:
        selected_set = set(labels or [])
        for label, check in self._checks.items():
            check.setChecked(label in selected_set)


class PathList(QWidget):
    """文件路径列表，支持批量追加、清空和取值。"""

    def __init__(self, empty_text: str = "尚未选择文件") -> None:
        super().__init__()
        self.empty_text = empty_text
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setMinimumHeight(86)
        self.add_button = QPushButton("添加")
        self.add_button.setObjectName("secondaryButton")
        self.remove_button = QPushButton("移除选中")
        self.remove_button.setObjectName("secondaryButton")
        self.clear_button = QPushButton("清空")
        self.clear_button.setObjectName("secondaryButton")
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.remove_button)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self.list_widget)
        root.addLayout(button_row)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear)

    def add_paths(self, paths: Iterable[str]) -> None:
        existing = set(self.paths())
        for raw in paths:
            path = str(Path(raw).expanduser()) if raw else ""
            if not path or path in existing:
                continue
            self.list_widget.addItem(path)
            existing.add(path)

    def paths(self) -> list[str]:
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

    def first_path(self) -> str:
        return self.paths()[0] if self.list_widget.count() else ""

    def remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def clear(self) -> None:
        self.list_widget.clear()


class CollapsibleCard(Card):
    """可折叠高级参数卡片。

    初始折叠可以减少主工作流噪音；仍然允许用户展开修改模型、VAD、工具路径等
    细节。
    """

    def __init__(self, title: str, help_text: str = "", *, collapsed: bool = True) -> None:
        super().__init__(title, help_text)
        self.toggle_button = QToolButton()
        self.toggle_button.setText("展开高级设置" if collapsed else "收起高级设置")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not collapsed)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow if collapsed else Qt.ArrowType.DownArrow)
        self.toggle_button.setObjectName("secondaryButton")
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.content.setVisible(not collapsed)
        self.layout.addWidget(self.toggle_button)
        self.layout.addWidget(self.content)
        self.toggle_button.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle_button.setText("收起高级设置" if expanded else "展开高级设置")
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


class StatusText(QTextBrowser):
    """只读状态/预览文本框。"""

    def __init__(self, placeholder: str = "") -> None:
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setMinimumHeight(180)
        if placeholder:
            self.setPlainText(placeholder)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


def setup_combo(combo: QComboBox, items: Iterable[str] | Iterable[tuple[str, str]], current: str = "") -> None:
    """统一 combo 下拉样式并填充选项。

    选项可以是字符串，也可以是 `(data, label)`。data 会作为 itemData 保存，方便
    profile 这类“显示名 != 实际 key”的字段。
    """

    view = QListView(combo)
    view.setUniformItemSizes(True)
    view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    combo.setView(view)
    combo.clear()
    for item in items:
        if isinstance(item, tuple):
            data, label = item
            combo.addItem(label, data)
        else:
            combo.addItem(str(item), str(item))
    if current:
        idx = combo.findData(current)
        if idx < 0:
            idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)


def combo_value(combo: QComboBox) -> str:
    data = combo.currentData()
    return str(data if data is not None else combo.currentText()).strip()


def label(text: str, object_name: str = "helperText", *, wrap: bool = True) -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(object_name)
    widget.setWordWrap(wrap)
    return widget

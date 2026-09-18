from __future__ import annotations

"""桌面 GUI 的 QSS 设计令牌。

视觉目标是“本地工作台”：浅色、卡片化、信息密度适中，重点突出当前运行状态
和主要操作，避免 WebUI 截图中所有内容挤在左侧的问题。
"""


def app_qss() -> str:
    """返回全局 QSS。

    调用时机：`QApplication` 创建后、主窗口展示前。
    """

    return """
    QMainWindow, QWidget#centralRoot, QWidget#pageSurface, QWidget#pageScrollContent {
        background: #f4f7fb;
        color: #17233b;
        font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
        font-size: 15px;
    }
    QWidget {
        color: #17233b;
    }
    QDialog, QMessageBox {
        background: #ffffff;
        color: #17233b;
    }
    QMessageBox QLabel {
        background: transparent;
        color: #17233b;
        min-width: 320px;
    }
    QMessageBox QPushButton {
        min-width: 96px;
        min-height: 38px;
        padding: 0 14px;
        border-radius: 12px;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 15px;
        font-weight: 800;
    }
    QMessageBox QPushButton:hover {
        background: #dbeafe;
    }
    QMessageBox QPushButton:default {
        background: #2563eb;
        color: #ffffff;
        border: 1px solid #2563eb;
    }
    QMenuBar {
        background: #f4f7fb;
        color: #516780;
        border: none;
    }
    QMenuBar::item {
        background: transparent;
        color: #516780;
        padding: 6px 10px;
        border-radius: 8px;
    }
    QMenuBar::item:selected {
        background: #eaf2ff;
        color: #1d4ed8;
    }
    QMenu {
        background: #ffffff;
        color: #1d2940;
        border: 1px solid #d8e6fb;
        padding: 6px;
    }
    QMenu::item {
        padding: 7px 18px;
        border-radius: 8px;
    }
    QMenu::item:selected {
        background: #eaf2ff;
        color: #1d4ed8;
    }
    QFrame#topHeader {
        background: #ffffff;
        border: 1px solid #d8e6fb;
        border-radius: 24px;
    }
    QFrame#card {
        background: #ffffff;
        border: 1px solid #d8e6fb;
        border-radius: 22px;
    }
    QFrame#softCard {
        background: #f7fbff;
        border: 1px solid #d5e6fb;
        border-radius: 18px;
    }
    QFrame#navPanel {
        background: #ffffff;
        border: 1px solid #d8e6fb;
        border-radius: 24px;
    }
    QFrame#statusOk {
        background: #ecfeff;
        border: 1px solid #67e8f9;
        border-radius: 18px;
    }
    QFrame#statusWarning {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 18px;
    }
    QFrame#statusError {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 18px;
    }
    QLabel#heroTitle {
        font-size: 34px;
        font-weight: 900;
        color: #17233b;
    }
    QLabel#pageTitle {
        font-size: 28px;
        font-weight: 900;
        color: #17233b;
    }
    QLabel#sectionTitle {
        font-size: 18px;
        font-weight: 800;
        color: #17233b;
    }
    QLabel#bodyText, QLabel#helperText {
        color: #5f7392;
        line-height: 22px;
    }
    QLabel#helperText {
        font-size: 13px;
    }
    QLabel#toolbarBadge {
        background: #f3f8ff;
        border: 1px solid #d8e6fb;
        border-radius: 14px;
        padding: 7px 11px;
        font-size: 13px;
        font-weight: 700;
        color: #46617f;
    }
    QPushButton#primaryButton {
        min-height: 46px;
        padding: 0 18px;
        border-radius: 16px;
        border: none;
        background: #2563eb;
        color: white;
        font-size: 16px;
        font-weight: 900;
    }
    QPushButton#primaryButton:hover { background: #1d4ed8; }
    QPushButton#secondaryButton, QPushButton#navButton, QToolButton#secondaryButton {
        min-height: 42px;
        padding: 0 15px;
        border-radius: 15px;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 15px;
        font-weight: 800;
    }
    QPushButton#secondaryButton:hover, QPushButton#navButton:hover, QToolButton#secondaryButton:hover { background: #dbeafe; }
    QPushButton#navButton:checked {
        background: #2563eb;
        color: white;
        border: 1px solid #2563eb;
    }
    QPushButton#dangerButton {
        min-height: 42px;
        padding: 0 15px;
        border-radius: 15px;
        border: 1px solid #fecaca;
        background: #fef2f2;
        color: #b91c1c;
        font-weight: 800;
    }
    QPushButton:disabled {
        color: #94a3b8;
        background: #e2e8f0;
        border-color: #cbd5e1;
    }
    QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser, QListWidget, QListView,
    QSpinBox, QDoubleSpinBox, QComboBox, QAbstractSpinBox {
        background: #ffffff;
        color: #1d2940;
        border: 1px solid #cfe0f7;
        border-radius: 13px;
        min-height: 36px;
        padding: 6px 9px;
        selection-background-color: #bfdbfe;
        selection-color: #17233b;
    }
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus,
    QListWidget:focus, QListView:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QComboBox:focus, QAbstractSpinBox:focus {
        border: 1px solid #60a5fa;
    }
    QLineEdit[echoMode="2"] {
        letter-spacing: 1px;
    }
    QTextEdit, QPlainTextEdit {
        line-height: 22px;
    }
    QLineEdit, QTextEdit, QPlainTextEdit {
        placeholder-text-color: #8aa0bf;
    }
    QTextBrowser {
        background: #ffffff;
        color: #1d2940;
        line-height: 22px;
    }
    QListWidget {
        background: #ffffff;
        color: #1d2940;
        outline: none;
    }
    QListWidget::item {
        min-height: 28px;
        padding: 4px 6px;
        border-radius: 8px;
    }
    QListWidget::item:selected {
        background: #eaf2ff;
        color: #1d4ed8;
    }
    QAbstractItemView {
        background: #ffffff;
        color: #1d2940;
        selection-background-color: #eaf2ff;
        selection-color: #1d4ed8;
        alternate-background-color: #f8fbff;
        outline: none;
    }
    QComboBox::drop-down {
        width: 28px;
        border: none;
    }
    QComboBox::down-arrow {
        image: none;
        width: 0px;
        height: 0px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 7px solid #5f7392;
        margin-right: 10px;
    }
    QComboBox QAbstractItemView {
        background: #ffffff;
        color: #1d2940;
        border: 1px solid #cfe0f7;
        selection-background-color: #eaf2ff;
        selection-color: #1d4ed8;
        outline: none;
    }
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
        width: 18px;
        border: none;
        background: transparent;
    }
    QCheckBox {
        spacing: 8px;
        color: #17233b;
    }
    QCheckBox::indicator {
        width: 17px;
        height: 17px;
        border-radius: 5px;
        border: 1px solid #9db8da;
        background: #ffffff;
    }
    QCheckBox::indicator:checked {
        background: #2563eb;
        border: 1px solid #2563eb;
    }
    QTabWidget::pane {
        border: 1px solid #d8e6fb;
        border-radius: 18px;
        background: #ffffff;
        top: -1px;
    }
    QTabBar::tab {
        min-height: 34px;
        padding: 8px 16px;
        border-top-left-radius: 13px;
        border-top-right-radius: 13px;
        background: #eaf2ff;
        color: #46617f;
        font-weight: 700;
    }
    QTabBar::tab:selected {
        background: #ffffff;
        color: #1d4ed8;
    }
    QProgressBar {
        min-height: 18px;
        border-radius: 9px;
        border: 1px solid #cfe0f7;
        background: #eef5ff;
        text-align: center;
        color: #17233b;
    }
    QProgressBar::chunk {
        border-radius: 8px;
        background: #2563eb;
    }
    QSplitter::handle {
        background: #d7e4f7;
        border-radius: 3px;
    }
    QSplitter::handle:horizontal {
        width: 6px;
        margin: 6px 0;
    }
    QSplitter::handle:vertical {
        height: 6px;
        margin: 0 6px;
    }
    QSplitter::handle:hover {
        background: #93c5fd;
    }
    QScrollArea {
        border: none;
        background: transparent;
    }
    QScrollArea > QWidget > QWidget {
        background: #f4f7fb;
    }
    QScrollBar:vertical {
        background: #edf4ff;
        width: 12px;
        margin: 2px 0 2px 0;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background: #9db8da;
        min-height: 30px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical:hover {
        background: #6f8fb8;
    }
    QScrollBar:horizontal {
        background: #edf4ff;
        height: 12px;
        margin: 0 2px 0 2px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal {
        background: #9db8da;
        min-width: 30px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #6f8fb8;
    }
    QScrollBar::add-line, QScrollBar::sub-line,
    QScrollBar::add-page, QScrollBar::sub-page {
        background: transparent;
        border: none;
        width: 0px;
        height: 0px;
    }
    """

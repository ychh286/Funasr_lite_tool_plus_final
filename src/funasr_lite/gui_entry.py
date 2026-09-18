from __future__ import annotations

"""本地桌面 GUI 启动入口。

模块保持很薄，只在真正启动时导入 PySide6 相关窗口代码。这样 CLI、WebUI 和测试
环境即使没有安装 PySide6，也不会因为普通导入 `funasr_lite` 而失败。
"""


def main() -> int:
    """启动 FunASR Audio Workbench Desktop。"""

    try:
        from .desktop_gui.app import run_gui
    except ImportError as exc:  # pragma: no cover - 只在缺少 PySide6 的本地环境触发
        raise SystemExit(
            "未安装 PySide6，无法启动本地 GUI。请先运行：\n"
            "  uv pip install -e .[gui]\n"
            "或：\n"
            "  uv pip install PySide6"
        ) from exc
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())

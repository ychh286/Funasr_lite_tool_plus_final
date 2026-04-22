from __future__ import annotations

from pathlib import Path


def _run_tk_directory_dialog(initial_dir: str, title: str) -> str:
    """调用本地 Tk 文件夹选择对话框。

    功能：
        在本地桌面环境弹出文件夹选择窗口，并返回用户选中的目录。

    关键参数：
        initial_dir：初始打开目录，可为空。
        title：对话框标题。

    返回值：
        用户选中的目录路径；若取消则返回空字符串。

    状态影响：
        会弹出本地 GUI 对话框。

    调用时机：
        WebUI 中用户点击“选择文件夹”按钮时。
    """

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        if initial_dir and Path(initial_dir).expanduser().exists():
            chosen = filedialog.askdirectory(initialdir=str(Path(initial_dir).expanduser()), title=title, mustexist=False)
        else:
            chosen = filedialog.askdirectory(title=title, mustexist=False)
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return str(chosen or "")


def choose_directory(initial_dir: str = "", title: str = "选择文件夹") -> str:
    """安全地弹出文件夹选择器。

    功能：
        对 GUI 缺失、远程环境或用户取消等情况做统一兜底，避免把异常抛回 WebUI。

    关键参数：
        initial_dir：初始目录。
        title：对话框标题。

    返回值：
        成功时返回目录路径；失败或取消时返回空字符串。

    状态影响：
        在可用的本地桌面环境中会尝试打开目录选择窗口。

    调用时机：
        WebUI 中的路径浏览按钮。
    """

    try:
        return _run_tk_directory_dialog(initial_dir, title).strip()
    except Exception:
        return ""


def format_runtime_status_markdown(
    *,
    profile_label: str,
    profile_prompt: str,
    source_value: str,
    source_kind: str,
    resolved_output_root: str,
    resolved_download_root: str,
    resolved_note_root: str,
    ytdlp_status: str,
    ffmpeg_status: str,
    cookie_status: str,
    prompt_defaults: list[str],
) -> str:
    """把运行时探测结果整理成适合 Gradio Markdown 展示的文本。

    功能：
        用简洁的人话把 profile、Prompt、工具路径、cookies 与目录解析结果汇总给用户。

    关键参数：
        profile_label：当前命中的 profile 显示名。
        profile_prompt：profile 默认 prompt。
        source_value：当前输入源文本。
        source_kind：`url` / `local` / `empty`。
        resolved_output_root：最终输出根目录。
        resolved_download_root：媒体下载目录。
        resolved_note_root：总结写入目录。
        ytdlp_status：yt-dlp 实际解析结果。
        ffmpeg_status：ffmpeg 实际解析结果。
        cookie_status：cookies 实际解析结果。
        prompt_defaults：当前默认会纳入的 Prompt 列表。

    返回值：
        一段 Markdown 文本。

    状态影响：
        无外部副作用。

    调用时机：
        用户点击“检查环境与路径”或切换 profile 后刷新状态时。
    """

    lines = [
        "### 当前环境探测",
        "",
        f"- 输入类型：`{source_kind}`",
        f"- 当前输入：`{source_value or '未指定'}`",
        f"- 路由配置：`{profile_label or '通用模式'}`",
        f"- Profile 默认 Prompt：`{profile_prompt or '未指定'}`",
        f"- 当前默认 Prompt 组合：`{', '.join(prompt_defaults) if prompt_defaults else '无'}`",
        "",
        "### 目录解析",
        "",
        f"- 输出目录：`{resolved_output_root or '未解析'}`",
        f"- 下载目录：`{resolved_download_root or '未解析 / 当前 run 临时目录'}`",
        f"- 总结目录：`{resolved_note_root or '未指定'}`",
        "",
        "### 工具与 Cookies",
        "",
        f"- yt-dlp：{ytdlp_status}",
        f"- ffmpeg：{ffmpeg_status}",
        f"- Cookies：{cookie_status}",
    ]
    return "\n".join(lines)

# FunASR Audio Workbench Desktop GUI 设计说明

## 设计 Intake

| Area | Decision |
|---|---|
| User / environment | 本地 Windows 工作站操作者，通常拥有本地模型缓存、ffmpeg、yt-dlp、cookies、Obsidian/本地笔记目录和一个 OpenAI-compatible API 入口。运行时可能是长任务，用户需要知道当前卡在哪一步。 |
| Primary flow | 单条：URL/本地媒体 → 环境检查 → yt-dlp/ffmpeg → FunASR → prompt 组合 → 可选 Brave → 可选 LLM → final_summary.md。批量：多个本地媒体 → 预处理 → 转写 → 打包。追问：已有 run → 加载 summary/transcript → LLM 追问/改写 → 可选写回。 |
| Truth sources | 用户输入：URL、路径、prompt 勾选、API key。保存配置：workflow_profiles.local.json、.local_state/desktop_gui_state.json。计算快照：inspect_audio_workflow_environment。运行产物：run_manifest.json、final_summary.md、transcript 文件。外部工具：yt-dlp、ffmpeg、FunASR、LLM、Brave Search。 |
| State model | ready：环境检查无红色阻塞项；busy：QThread 后台任务运行；done：任务返回 GuiRunResult；failed：worker 捕获异常；degraded：未指定可选目录、未命中 cookies、未启用 LLM 或 Brave。 |
| Boundaries | Page 只负责布局和收集用户意图；Model dataclass 定义请求；DesktopWorkflowService 调用现有 pipeline/followup/LLM；pipeline_workflow 继续拥有下载、转码、转写和导出事实；followup_chat 拥有追问缓存事实。 |
| Failure handling | UI 线程不跑长任务；worker 捕获异常并显示 traceback；缺 ffmpeg/yt-dlp/cookies 会在检查卡片中显式标红或警告；追问写回 final_summary.md 前自动备份；GUI 关闭时若有后台任务会二次确认。 |
| Evidence target | 当前达到 T2/T3：页面与服务合同存在，mock/patch 条件下单测通过；未在本环境进行真实 FunASR、真实 yt-dlp、真实 PySide6 窗口、真实 LLM/Brave 硬件/网络验证。 |

## 为什么从 WebUI 改为本地 GUI

上一版 WebUI 功能已能串起流程，但主要问题是：

1. 控件集中在左侧，解释和结果区比例失衡。
2. Gradio 的文件/路径选择和长期运行状态不够像“本地工作台”。
3. 多文件转写、单条总结、追问改写是三种不同任务，放在一个页面会让概念变混。
4. cookies、下载目录、run 输出目录、note 目录、prompt 之间需要更明确的中文解释。

本地 GUI 的改造原则是：**服务层复用，不重写核心 pipeline；UI 层重做。**

## 新增文件

```text
src/funasr_lite/gui_entry.py                  桌面 GUI 启动入口
src/funasr_lite/desktop_gui/models.py         Qt 无关的数据合同
src/funasr_lite/desktop_gui/service.py        GUI 服务/用例层
src/funasr_lite/desktop_gui/style.py          QSS 视觉设计令牌
src/funasr_lite/desktop_gui/widgets.py        复用控件
src/funasr_lite/desktop_gui/app.py            PySide6 主窗口与页面
Start-FunASRGUI.ps1 / .cmd                    Windows 启动脚本
tests/test_desktop_gui_service.py             服务层回归测试
```

## 页面结构

桌面版采用左侧导航 + 右侧工作区，避免全部内容挤在屏幕左侧：

```text
顶部品牌/状态条
左侧导航：单条总结 / 批量转写 / 追问改写
右侧工作区：
  左半区：当前任务表单卡片，可滚动
  右半区：进度、日志、转写预览、总结预览、元数据
```

### 单条总结

负责端到端主链路：

```text
URL 或本地单文件
→ workflow profile 路由
→ prompt 多选/上传/追加说明
→ cookies/yt-dlp/ffmpeg 高级设置
→ FunASR 参数
→ LLM/Brave 可选
→ 输出策略
→ final_summary.md / workflow_outputs.zip
```

关键解释：

- `Workflow / 本地路由配置` 只决定默认标签、目录和默认 prompt，不等于真正 prompt 内容。
- `Prompt 组合` 决定真正交给 LLM/agent 的指令文本。
- `运行输出根目录` 保存中间文件和 run_manifest。
- `媒体下载/归档目录` 只影响 URL 下载后的原始媒体保存位置。
- `总结笔记目录` 只在勾选“同时写入总结笔记目录”时额外写入。

### 批量转写

负责多个本地媒体的转写，不强行做 LLM 总结：

```text
多个文件 / 一个目录
→ 可递归扫描
→ 每个文件独立子目录
→ 可 ffmpeg 预处理
→ FunASR 转写
→ batch_manifest.json + workflow_outputs.zip
```

### 追问改写

负责基于已有 run 目录继续处理：

```text
run 目录
→ 读取 final_summary.md / _followup_cache/summary_current.md
→ 读取 transcript.txt / transcript.md
→ LLM 追问或改写
→ 可选 Brave Search
→ 写入 _followup_cache
→ 用户确认后写回 final_summary.md，并备份旧版本
```

## API 与 Brave Search

首轮总结现在也支持可选 Brave Search。实现方式是在 FunASR 转写完成后，基于 URL 或媒体文件名生成检索词，把 Brave 返回结果作为 `brave_search/首轮外部背景` prompt block 合入总结提示词。

约束：

- Brave 结果只作为外部背景。
- prompt 中仍明确要求“不能覆盖转写证据”。
- manifest 中记录 `web_search` 元数据，包括启用状态、查询词、结果条数或错误信息。

## 安全与隐私

- GUI 状态文件 `.local_state/desktop_gui_state.json` 不保存 API key 或 cookies 内容。
- cookies 路径只作为运行时输入；pipeline 仍会避免把 cookies 打入 zip。
- 追问改写默认只写 `_followup_cache`，不会自动覆盖最终 Markdown。
- 写回 `final_summary.md` 前会弹确认框，并自动生成 `final_summary.backup.<timestamp>.md`。

## 启动

安装 GUI 依赖：

```powershell
uv pip install -e .[gui]
```

启动：

```powershell
.\Start-FunASRGUI.ps1
```

如果只想用命令入口：

```powershell
uv run --active funasr-lite-gui
```

## 验证记录

已在当前交付环境完成：

```text
python -S -m compileall -q src/funasr_lite tests
PYTHONPATH=src python -S -m unittest tests.test_audio_workbench tests.test_followup_chat tests.test_local_dialogs tests.test_desktop_gui_service
```

通过 12 个测试。

未在当前环境验证：

- 真实 PySide6 窗口渲染，因为当前容器没有桌面显示服务。
- 真实 FunASR 模型推理，因为容器没有本地模型/GPU 环境。
- 真实 B 站/YouTube 下载，因为容器没有你的 cookies 和外网下载环境。
- 真实 LLM/Brave 调用，因为容器没有你的 API key 和服务端点。

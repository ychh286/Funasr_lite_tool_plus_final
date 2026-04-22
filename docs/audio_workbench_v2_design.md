# Audio Workbench V2 设计与接入说明

> 目标：把“视频/音频链接或本地文件 → yt-dlp 下载 → ffmpeg 预处理 → FunASR 转写 → prompt 组装 → OpenAI-compatible 大模型处理 → Markdown 预览/下载/落盘”串成一个本地可复核工作流。

## 1. 结论：先扩展 WebUI，不急于重构桌面 GUI

当前需求的核心不是复杂多窗口交互，而是端到端任务编排和产物追踪。保留 Gradio WebUI 有几个直接收益：

- 复用现有 FunASR 参数面板、暗色视觉风格、Web 启动脚本与虚拟环境。
- 运行路径短：无需引入 PySide/Tauri/Electron、打包器、IPC、线程模型和额外状态同步。
- Gradio 对文件上传、文件下载、进度条、长任务执行和本地浏览器入口足够好。
- 真正需要沉淀的是 workflow service 层；只要服务层独立，未来再换成本地 GUI 也不需要重写业务逻辑。

因此 V2 采用：

```text
WebUI 页面
  -> pipeline_workflow.run_audio_workflow
  -> yt-dlp / ffmpeg / FunASR / LLM client
  -> prompt 与 manifest / markdown / zip 输出
```

这与 operator GUI skill 包里的分层建议一致：UI 只发出用户意图，服务层负责编排动作，外部工具被封装为运行时适配器，最终产物和 manifest 作为事实来源。

## 2. 用户流程

### 输入区

- URL 输入：Bilibili、YouTube 或其他 yt-dlp 支持的视频/音频链接。
- 本地媒体上传：未填写 URL 时使用上传文件。
- Profile：可以自动或手动选择 `workflow_profiles.local.json` 中的媒体库/笔记目录规则。
- Cookie：支持 `auto`、`upload/path`、`none` 三种模式。
- 工具路径：yt-dlp 与 ffmpeg 可手动指定；留空则从环境变量、`.local_tools`、常见 Windows 路径、PATH 中自动查找。

### Prompt 与 LLM 区

- 默认纳入 `.codex/agents.md`。
- 自动列出 `prompts/*.md`，可多选。
- 可上传本次专用 `.md/.txt` prompt 文件。
- 可填写本次追加指令。
- 可调用 OpenAI-compatible `/chat/completions` 接口，参数包括 `base_url`、`model`、`api_key`、`api_key_env`、`temperature`、`timeout`。

### ASR 区

- 保留旧版 WebUI 的模型、VAD、标点、语言、设备、ITN、VAD 合并、说话人区分等参数。
- 开启说话人区分时自动切换到兼容模型和标点模型。

### 输出区

- 转写预览。
- 最终 Markdown 预览。
- 下载 `final_summary.md`。
- 下载 `workflow_outputs.zip`。
- 查看 `run_manifest.json`。

## 3. 事实模型与状态来源

| 对象 | 来源 | 是否最终事实 | 说明 |
|---|---|---:|---|
| 输入 URL / 上传文件 | UI | 否 | 用户意图，需要解析成真实媒体文件 |
| cookies 文件 | UI / 环境变量 / 本地路径 | 否 | 只用于 yt-dlp，运行 zip 不包含 cookie 内容 |
| 源媒体文件 | yt-dlp 或上传路径 | 是 | 进入 ffmpeg 前的媒体源 |
| `audio_16k_mono.wav` | ffmpeg | 是 | FunASR 的标准输入 |
| `transcript.md/.txt` | FunASR | 是 | 后续 prompt 与总结的文本基础 |
| `summary_prompt.md` | prompt service | 是 | 可复制给任意 agent 的完整上下文 |
| `final_summary.md` | LLM 或 fallback 渲染器 | 是 | 用户主要落盘对象 |
| `run_manifest.json` | workflow service | 是 | 单次运行的可复核证据 |
| `workflow_outputs.zip` | bundle builder | 是 | 便携导出包，排除 cookies |

## 4. 失败边界

V2 没有把失败吞掉，而是在 UI 的“总结预览/运行元数据”里暴露错误信息和 traceback，便于继续调试。

常见失败点：

- yt-dlp 不存在、路径错误或站点需要 cookies。
- cookies 过期或权限不足。
- ffmpeg 路径错误或媒体格式异常。
- FunASR 模型未下载、GPU/CPU 环境不匹配、显存不足。
- speaker diarization 模型组合不兼容。
- LLM base URL、模型名、API key 或超时设置错误。
- note_root 不存在或文件名冲突；默认不覆盖，使用 `xxx (2).md` 形式避让。

## 5. 文件变更概览

新增核心模块：

- `src/funasr_lite/pipeline_workflow.py`：端到端工作流服务。
- `src/funasr_lite/prompting.py`：prompt 发现、组合、fallback markdown 渲染。
- `src/funasr_lite/llm_client.py`：轻量 OpenAI-compatible 客户端。
- `src/funasr_lite/webui_workbench.py`：新 WebUI 入口。
- `.codex/agents.md`：默认 Agent 指令。

更新入口：

- `main.py` 默认启动 `webui_workbench`。
- `pyproject.toml` 中 `funasr-lite-web` 指向 V2，另保留 `funasr-lite-web-legacy`。
- `Start-FunASRWebUI.ps1` 默认启动 V2，并增加 `-Legacy`。

## 6. 推荐运行方式

```powershell
Set-Location D:\AI_projects\Workflows_local\funasr-lite-workflow
.\.venv\Scripts\Activate.ps1
.\Start-FunASRWebUI.ps1
```

旧版快速转写入口仍可用：

```powershell
.\Start-FunASRWebUI.ps1 -Legacy
```

也可以用 console script：

```powershell
uv run --active funasr-lite-web
uv run --active funasr-lite-web-legacy
```

## 7. 后续可选增强

- 批量 URL 队列：一行一个链接，逐条生成独立 manifest。
- 断点续跑：如果已有 `audio_16k_mono.wav` 或 `transcript.txt`，可选择从后续阶段继续。
- Prompt 配方保存：把勾选组合保存为“项目会/组会/峰video/渤海小吏”等一键 presets。
- LLM 多轮后处理：先粗总结，再结构化改写，再输出 Obsidian frontmatter。
- 独立桌面 GUI：当需要后台队列、系统托盘、跨会话任务列表、长期任务管理时，再复用当前 service 层接入 PySide/Tauri。

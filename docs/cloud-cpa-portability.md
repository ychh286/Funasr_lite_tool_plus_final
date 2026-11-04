---
tags:
  - FunASR
  - 部署迁移
  - CPA
created: 2026-11-04 11:36
updated: 2026-11-04 12:05
author: codex-app-GPT-6
---

# FunASR 换主机运行与云端 CPA 配置

## 核心概览

FunASR工作流可以迁移到其他主机。Git克隆提供程序代码、通用模板和启动入口；目标机还需要Python虚拟环境、匹配硬件的PyTorch、FFmpeg，以及首次下载或另行复制的语音模型权重。语音识别、说话人区分和媒体预处理在目标主机执行；云端CPA承担转写后的文字总结、追问和改写。2026年11月4日已验证当前程序既能通过本机LiteLLM转发到云端，也能直接使用公网HTTPS入口，后者无需目标机再部署LiteLLM、CPA或Clash。API密钥通过用户环境变量单独配置，不随Git传输。已有录音、历史输出、个人路径和下载网站Cookie也需要按需迁移。当前仅验证了本机代码的云端调用和环境信息，尚未在另一台全新主机完成完整音频转写验收。

## 新旧主机各自负责什么

| 组件 | 运行位置/迁移要求 |
|---|---|
| CPA与上游代理 | 阿里云，已运行；目标机只需能访问HTTPS入口 |
| FunASR、PyTorch | 目标机；CPU或适合目标GPU的版本 |
| FFmpeg | 目标机；加入PATH |
| 语音模型 | 目标机首次下载或从已有缓存复制；Git仓库不包含模型权重 |
| Python虚拟环境 | 在目标机重新创建，不直接搬运旧 `.venv` |
| API密钥 | 单独安全配置到用户环境；不写入仓库或提交记录 |
| 录音与结果 | 按需要复制，当前 `outputs/`、音视频文件等不随Git提交 |
| 个人路径与Cookie | 按目标机调整；网站下载场景才需要相应授权/Cookie |

现有部署可保留 `http://127.0.0.1:4000/v1` 的LiteLLM入口，只有其中的GPT/图像CPA上游切换到云端，DeepSeek等其他选项继续使用原路由。新主机若只需要CPA中的模型，可直接使用下面的云端配置；需要其他供应商时分别添加相应服务，或部署自己的聚合入口。

## Windows安装顺序

1. 从自己的仓库克隆包含本次修改的版本；本次本地提交需要同步到远端后，其他电脑才能通过远端Git获得。
2. 创建项目虚拟环境。项目要求Python3.11以上，已验证环境的版本可作为兼容性参考。
3. 根据目标机CPU/GPU及驱动选择PyTorch安装包，参照[官方安装选择器](https://pytorch.org/get-started/locally/)；CPU版本可运行，速度需用实际录音评估。
4. 安装FFmpeg，确认命令行可以执行 `ffmpeg -version`。
5. 安装项目及桌面GUI依赖。

```powershell
# 在已经克隆的FunASR项目根目录运行
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 此处先执行与目标硬件匹配的PyTorch安装命令，使用上述虚拟环境的python -m pip。
# 然后安装项目：
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
```

迁移测试时已观察到的版本是torch2.7.1+cu128、torchaudio2.7.1+cu128、FunASR1.3.0、Gradio6.3.0、ModelScope1.33.0、PySide6 6.11.0，CUDA可用。此列表是当前工作环境快照，目标GPU驱动及操作系统仍需单独匹配。仓库中的 `pyproject.toml`、`uv.lock` 和README应一并保留；当前依赖声明中PyTorch需要根据硬件另行处理。

## 新主机直接连接云端

在PowerShell7中运行以下配置。密钥从私有连接文件取得，通过掩码提示输入；命令历史中只保留变量名。

```powershell
$cloudKey = Read-Host '输入你的云端CPA工作流API密钥' -MaskInput
[Environment]::SetEnvironmentVariable('CPA_CLOUD_API_KEY', $cloudKey, 'User')
[Environment]::SetEnvironmentVariable('CPA_CLOUD_BASE_URL', 'https://<你的CPA地址>/v1', 'User')
[Environment]::SetEnvironmentVariable('FUNASR_LITE_LLM_BASE_URL', 'https://<你的CPA地址>/v1', 'User')
[Environment]::SetEnvironmentVariable('FUNASR_LITE_LLM_API_KEY_ENV', 'CPA_CLOUD_API_KEY', 'User')
[Environment]::SetEnvironmentVariable('FUNASR_LITE_LLM_MODEL', 'gpt-5.6-sol', 'User')
Remove-Variable cloudKey
```

随后重新启动FunASR。若复制了旧 `.local_state/desktop_gui_state.json`，界面可能继续显示以前保存的4000地址；在界面中将地址与密钥变量名一起改为上面的云端配置。密钥变量名是 `CPA_CLOUD_API_KEY`，字段中填写变量名即可，真实密钥留在用户环境中。

程序在这个云端凭据配置下会直接连接HTTPS，不依赖系统代理。现有LiteLLM及其他供应商配置保留原网络行为。此方式只提供当前云端CPA账户可用的模型；已验证 `gpt-5.6-sol`。历史GPT-5.4/GPT-5.5选项可留作旧配置记录，当前云端未列出这些模型，使用时应选择现有模型列表中的名称。

Linux/macOS使用相同的五个环境变量，在自己的启动环境中配置即可。启动时使用平台对应的 `.venv/bin` 命令；本项目Windows `.cmd`/`.ps1` 启动器适用于Windows，跨平台优先使用安装后的 `funasr-lite`、`funasr-lite-web` 命令。其他平台完整GUI与转写流程本轮尚未验证。

## 换机验收顺序

1. 检查Python、PyTorch和FFmpeg能正常加载。
2. 在FunASR中执行一次模型连接检查，确认使用云端地址。
3. 用一段短录音检查语音模型下载、转写、标点与输出目录。
4. 再做一次“转写→文字总结→追问”的完整流程。
5. 最后迁入个人录音、历史输出和 `workflow_profiles.local.json`，逐项修正旧的D盘/E盘路径。

本轮已完成：在故意配置不可用HTTP代理的测试进程中，直接云端配置返回OK；当前电脑的LiteLLM链路也在本机CPA停止时返回OK。后者仍依赖当前电脑上的LiteLLM进程，适合保留原有多供应商入口；新主机的直接云端配置可省去这一层。

## 维护边界

- 云端API密钥和服务器SSH私钥分别管理；新主机调用API只需API密钥。
- 上游账号、代理订阅和云服务器期限继续有效，云端API才能持续工作。
- 19.5GB月流量保护关机会中断总结请求；已经生成的本地转写文件仍在目标机。
- 云端API主机继续承担CPA服务，音频模型计算先留在运行FunASR的主机。
- FunASR官方项目：[modelscope/FunASR](https://github.com/modelscope/FunASR)。本工作流的启动和配置以当前仓库代码为准。

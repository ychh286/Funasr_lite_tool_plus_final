from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class LLMSettings:
    """描述一次 OpenAI-compatible 调用所需的最小配置。

    功能：
        统一承载 Audio Workbench 的模型地址、模型名、鉴权和超时参数。

    关键参数：
        enabled：是否启用 LLM 总结。
        base_url：模型服务基础地址，可以是 `/v1` 或完整 `/chat/completions`。
        model：请求体里的模型名。
        api_key：直接输入的 API key。
        api_key_env：当 `api_key` 为空时，用来补全密钥的环境变量名。

    返回值：
        数据类本身就是配置承载对象，不直接返回其它结果。

    状态影响：
        不会主动访问网络；只负责把后续请求所需参数组织起来。

    调用时机：
        在 WebUI 点击“检查连接”或正式运行总结流程之前构造。
    """

    enabled: bool = False
    base_url: str = "http://127.0.0.1:4000/v1"
    model: str = "gpt-5.4"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    timeout_seconds: int = 180

    def resolved_api_key(self) -> str:
        """解析本次调用实际使用的 API key。"""

        direct_value = (self.api_key or "").strip()
        if direct_value:
            return direct_value
        env_name = (self.api_key_env or "").strip()
        if not env_name:
            return ""
        return os.getenv(env_name, "").strip()

    def safe_summary(self) -> dict[str, Any]:
        """返回适合写入 manifest 的脱敏配置摘要。"""

        return {
            "enabled": bool(self.enabled),
            "base_url": (self.base_url or "").strip(),
            "model": (self.model or "").strip(),
            "api_key_env": (self.api_key_env or "").strip(),
            "has_api_key": bool(self.resolved_api_key()),
            "temperature": float(self.temperature),
            "timeout_seconds": int(self.timeout_seconds),
        }


@dataclass(slots=True)
class LLMHealth:
    """表示一次健康检查的结果。"""

    ok: bool
    message: str


def normalize_chat_completions_url(base_url: str) -> str:
    """把任意 OpenAI-compatible 基础地址规整成 `chat/completions` 端点。

    功能：
        兼容用户输入 `http://host:port/v1`、`.../v1/` 或完整
        `.../chat/completions` 这几种常见写法。

    关键参数：
        base_url：用户输入的服务地址。

    返回值：
        可直接用于 POST 的完整 `chat/completions` URL。

    状态影响：
        不访问网络，只做字符串规整。

    调用时机：
        每次正式发请求之前。
    """

    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("LLM base_url 不能为空。")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def extract_message_text(response_json: dict[str, Any]) -> str:
    """从 OpenAI-compatible 响应中提取第一条消息正文。

    功能：
        兼容 `message.content` 为纯字符串或分段列表两种常见返回形态。

    关键参数：
        response_json：模型返回的完整 JSON 对象。

    返回值：
        第一条候选消息中的文本正文。

    状态影响：
        无外部副作用。

    调用时机：
        成功收到模型 JSON 响应之后。
    """

    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("LLM 响应缺少 choices。")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                text_parts.append(str(item["text"]))
        text = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
        if text:
            return text
    raise ValueError("LLM 响应缺少可用文本内容。")


def _headers(settings: LLMSettings) -> dict[str, str]:
    """构造统一请求头，并在可用时追加 Bearer 鉴权。"""

    headers = {"Content-Type": "application/json; charset=utf-8"}
    api_key = settings.resolved_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    """发送 JSON POST 请求并把响应解析成字典。

    功能：
        把网络异常、HTTP 错误和 JSON 解析失败都转换成更直观的运行时错误，
        方便 WebUI 把真实失败原因反馈给用户。

    关键参数：
        url：目标接口地址。
        headers：请求头。
        payload：JSON 请求体。
        timeout_seconds：网络超时。

    返回值：
        解析后的 JSON 字典。

    状态影响：
        会发起一次真实的网络请求。

    调用时机：
        健康检查和正式聊天请求都会走这里。
    """

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM 请求失败：HTTP {exc.code}，响应片段：{raw_body[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM 请求失败：{exc.reason}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM 返回的内容不是合法 JSON：{raw_body[:400]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM 返回的 JSON 不是对象结构。")
    return parsed


def chat_text(
    settings: LLMSettings,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
) -> str:
    """向 OpenAI-compatible 接口发送对话并返回纯文本结果。

    功能：
        这是 Audio Workbench 的统一 LLM 调用出口。

    关键参数：
        settings：本次调用的模型配置。
        messages：符合 chat/completions 约定的消息列表。
        temperature：可选覆盖默认温度。

    返回值：
        模型生成的正文文本。

    状态影响：
        会向外部模型接口发送一次真实请求。

    调用时机：
        当工作流需要根据转写结果生成最终总结时。
    """

    if not settings.enabled:
        raise ValueError("LLM 未启用，不能执行聊天请求。")
    model_name = (settings.model or "").strip()
    if not model_name:
        raise ValueError("LLM model 不能为空。")

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": float(settings.temperature if temperature is None else temperature),
    }
    response_json = _post_json(
        normalize_chat_completions_url(settings.base_url),
        headers=_headers(settings),
        payload=payload,
        timeout_seconds=int(settings.timeout_seconds),
    )
    return extract_message_text(response_json)


def check_health(settings: LLMSettings) -> LLMHealth:
    """用一个最小请求检查 LLM 接口是否可达。

    功能：
        帮助用户在正式跑长任务前，先确认 base URL、模型名和鉴权是否大体可用。

    关键参数：
        settings：健康检查时使用的接口配置。

    返回值：
        `LLMHealth`，包含是否成功以及简洁的提示信息。

    状态影响：
        会发出一次轻量的网络请求。

    调用时机：
        WebUI 中点击“检查连接”按钮时。
    """

    if not settings.enabled:
        return LLMHealth(ok=False, message="LLM 未启用。")

    probe_messages = [
        {"role": "system", "content": "You are a connectivity probe."},
        {"role": "user", "content": "Reply with the single word OK."},
    ]
    try:
        reply = chat_text(settings, probe_messages, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        return LLMHealth(ok=False, message=str(exc))
    return LLMHealth(ok=True, message=reply.strip() or "接口连通，但返回为空。")

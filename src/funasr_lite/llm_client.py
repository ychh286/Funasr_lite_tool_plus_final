from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


LITELLM_LLM_BASE_URL = "http://127.0.0.1:4000/v1"
LITELLM_LLM_API_KEY_ENV = "LITELLM_API_KEY"
DEEPSEEK_V4_PRO_MODEL = "deepseek-v4-pro"
CPA_GPT56_SOL_MODEL = "gpt-5.6-sol"
CPA_GPT56_LUNA_MODEL = "gpt-5.6-luna"
CPA_GPT56_TERRA_MODEL = "gpt-5.6-terra"
CPA_GPT55_MODEL = "gpt-5.5"
CPA_GPT54_MODEL = "gpt-5.4"
LLM_MODEL_CHOICES = (
    CPA_GPT56_SOL_MODEL,
    CPA_GPT56_LUNA_MODEL,
    CPA_GPT56_TERRA_MODEL,
    CPA_GPT55_MODEL,
    DEEPSEEK_V4_PRO_MODEL,
    CPA_GPT54_MODEL,
)
DEFAULT_LLM_BASE_URL = LITELLM_LLM_BASE_URL
DEFAULT_LLM_MODEL = CPA_GPT56_SOL_MODEL
DEFAULT_LLM_API_KEY_ENV = LITELLM_LLM_API_KEY_ENV
DEFAULT_LLM_TIMEOUT_SECONDS = 1200
DEFAULT_LLM_REASONING_EFFORT = "high"


def _read_windows_registry_env(name: str) -> str:
    """读取 Windows 持久环境变量，避免新开进程没有继承用户级 API key。"""

    if os.name != "nt":
        return ""
    try:
        import winreg
    except Exception:  # noqa: BLE001
        return ""

    locations = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for hive, subkey in locations:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _value_type = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _get_env(name: str) -> str:
    """读取当前进程或 Windows 持久环境变量。"""

    return (os.getenv(name) or _read_windows_registry_env(name) or "").strip()


def default_llm_base_url() -> str:
    """返回当前本地 LLM 入口，优先允许项目环境变量覆盖 LiteLLM 默认值。"""

    return (_get_env("FUNASR_LITE_LLM_BASE_URL") or _get_env("LITELLM_BASE_URL") or DEFAULT_LLM_BASE_URL).strip()


def default_llm_model() -> str:
    """返回默认总结模型；默认走 LiteLLM 转发的 GPT-5.6 Sol。"""

    return (_get_env("FUNASR_LITE_LLM_MODEL") or DEFAULT_LLM_MODEL).strip()


def default_llm_api_key_env() -> str:
    """返回 API key 环境变量名；默认读取 LITELLM_API_KEY。"""

    return (_get_env("FUNASR_LITE_LLM_API_KEY_ENV") or DEFAULT_LLM_API_KEY_ENV).strip()


@dataclass(frozen=True)
class LLMSettings:
    """Runtime settings for an OpenAI-compatible chat/completions endpoint."""

    enabled: bool = False
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    api_key: str = ""
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    temperature: float = 0.1
    reasoning_effort: str = DEFAULT_LLM_REASONING_EFFORT

    def resolved_api_key(self) -> str:
        key = (self.api_key or "").strip()
        if key:
            return key
        env_name = (self.api_key_env or "").strip()
        if env_name:
            return _get_env(env_name)
        return ""


@dataclass(frozen=True)
class LLMHealth:
    ok: bool
    message: str


class OpenAICompatibleClient:
    """Small stdlib-only OpenAI-compatible client.

    The project intentionally keeps this client minimal so the audio workflow can
    reuse the API integration idea from paper-workbench without making a desktop
    agent manager or adding a heavy dependency chain.
    """

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        base_url = (settings.base_url or "").strip().rstrip("/")
        if not base_url:
            raise ValueError("LLM base_url is empty.")
        if base_url.endswith("/chat/completions"):
            self.chat_url = base_url
        else:
            self.chat_url = base_url + "/chat/completions"
        self.api_key = settings.resolved_api_key()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """发送一次模型请求并返回JSON字典；payload为请求体，失败时抛出可读异常。

        由chat_text调用，只产生网络IO；云端CPA凭据配置走直连，其他后端保留原代理行为。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.chat_url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        # 新主机可直接使用CPA云端配置，无需先部署本机LiteLLM或Clash。
        open_request = urllib.request.urlopen
        if self.settings.api_key_env == "CPA_CLOUD_API_KEY":
            open_request = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        try:
            with open_request(req, timeout=max(1, int(self.settings.timeout_seconds))) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc.reason}") from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("LLM response is not a JSON object.")
        return data

    @staticmethod
    def _extract_content(data: Mapping[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response missing choices.")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise RuntimeError("LLM response choice is malformed.")
        message = first.get("message") or {}
        if not isinstance(message, Mapping):
            raise RuntimeError("LLM response message is malformed.")
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif "text" in item:
                        parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            merged = "\n".join(part for part in parts if part).strip()
            if merged:
                return merged
        raise RuntimeError("LLM response content is empty.")

    def chat_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
        }
        if not self.settings.model.startswith("gpt-5.6-"):
            payload["temperature"] = self.settings.temperature if temperature is None else temperature
        reasoning_effort = self.settings.reasoning_effort.strip()
        if self.settings.model.startswith("gpt-5.6-") and reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        data = self._post_json(payload)
        return self._extract_content(data)

    def health_check(self) -> LLMHealth:
        try:
            text = self.chat_text(
                [
                    {"role": "system", "content": "You are a health check assistant."},
                    {"role": "user", "content": "Reply with OK."},
                ],
                temperature=0,
                max_tokens=16,
            )
            return LLMHealth(ok=True, message=text[:200])
        except Exception as exc:  # noqa: BLE001 - surface exact endpoint failures to UI
            return LLMHealth(ok=False, message=str(exc))

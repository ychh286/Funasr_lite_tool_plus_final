from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMSettings:
    """Runtime settings for an OpenAI-compatible chat/completions endpoint."""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:4000/v1"
    model: str = "gpt-5.4"
    api_key: str = ""
    api_key_env: str = "LITELLM_API_KEY"
    timeout_seconds: int = 180
    temperature: float = 0.1

    def resolved_api_key(self) -> str:
        key = (self.api_key or "").strip()
        if key:
            return key
        env_name = (self.api_key_env or "").strip()
        if env_name:
            return (os.getenv(env_name, "") or "").strip()
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
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.chat_url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=max(1, int(self.settings.timeout_seconds))) as resp:
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
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
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

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BraveSearchSettings:
    """描述一次 Brave Search 调用所需的最小配置。

    功能：
        为 Audio Workbench 的追问会话提供一个轻量级网页搜索配置对象。

    关键参数：
        enabled：是否启用 Brave Search。
        api_key：可直接输入的 Brave API key。
        api_key_env：读取 API key 的环境变量名。
        base_url：Brave Search Web API 地址。
        count：返回条数上限。
        timeout_seconds：请求超时。

    返回值：
        数据类本身就是配置承载对象。

    状态影响：
        无外部副作用。

    调用时机：
        追问会话中用户显式勾选“联网补充”时。
    """

    enabled: bool = False
    api_key: str = ""
    api_key_env: str = "BRAVE_SEARCH_API_KEY"
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    count: int = 5
    timeout_seconds: int = 25

    def resolved_api_key(self) -> str:
        """解析本次调用实际使用的 Brave API key。"""

        direct = (self.api_key or "").strip()
        if direct:
            return direct
        env_name = (self.api_key_env or "").strip()
        if not env_name:
            return ""
        return os.getenv(env_name, "").strip()


@dataclass(frozen=True)
class BraveSearchResult:
    """一条网页搜索结果。"""

    title: str
    url: str
    description: str


class BraveSearchClient:
    """面向 Brave Search Web API 的最小客户端。"""

    def __init__(self, settings: BraveSearchSettings) -> None:
        self.settings = settings
        self.api_key = settings.resolved_api_key()

    @property
    def enabled(self) -> bool:
        """是否满足联网搜索的前置条件。"""

        return bool(self.settings.enabled and self.api_key)

    def search(self, query: str, *, count: int | None = None) -> list[BraveSearchResult]:
        """执行一次网页检索。

        功能：
            调用 Brave Search，并把结果规整成轻量对象，方便后续拼进 prompt。

        关键参数：
            query：检索词。
            count：本次覆盖默认返回条数。

        返回值：
            `BraveSearchResult` 列表；未启用时返回空列表。

        状态影响：
            会向外部 Brave Search API 发起网络请求。

        调用时机：
            追问会话中用户勾选联网补充时。
        """

        if not self.enabled:
            return []
        effective_count = int(count or self.settings.count)
        params = urllib.parse.urlencode({"q": query, "count": effective_count})
        request = urllib.request.Request(
            f"{self.settings.base_url}?{params}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(1, int(self.settings.timeout_seconds))) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"Brave Search HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Brave Search 连接失败：{exc.reason}") from exc

        data = json.loads(raw)
        web = data.get("web") or {}
        raw_results = web.get("results") or []
        results: list[BraveSearchResult] = []
        for item in raw_results[:effective_count]:
            if not isinstance(item, dict):
                continue
            results.append(
                BraveSearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    description=str(item.get("description") or " ".join(item.get("extra_snippets") or [])),
                )
            )
        return results


def format_web_results(results: list[BraveSearchResult]) -> str:
    """把 Brave Search 结果整理成适合 prompt 使用的纯文本。"""

    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        lines.append(f"[{idx}] {item.title}\nURL: {item.url}\n摘要: {item.description}")
    return "\n\n".join(lines).strip()

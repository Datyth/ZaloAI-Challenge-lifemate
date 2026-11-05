"""Local OpenAI-compatible chat client; rejects non-loopback URLs."""
from __future__ import annotations

import json
from socket import timeout as SocketTimeout
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LocalChat:
    def __init__(self, url: str, model: str, timeout: float = 25) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("LLM endpoint must be local loopback HTTP")
        self.url, self.model, self.timeout = url, model, timeout

    def complete(self, messages: list[dict], tools: list[dict], timeout: float | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 512,
        }
        request = Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=min(self.timeout, timeout) if timeout else self.timeout) as response:
                return json.load(response)
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, SocketTimeout)):
                raise TimeoutError("local LLM request timed out") from exc
            raise

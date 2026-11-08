"""Local OpenAI-compatible chat client; rejects non-loopback URLs."""
from __future__ import annotations

import json
import os
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def merge_system(messages: list[dict]) -> list[dict]:
    """One leading system message (templates such as Qwen3.5 reject any other): the leading system messages are joined, and a
    later system message becomes a user message. Used when LIFEMATE_MERGE_SYSTEM=1, at inference and when building SFT rows."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            if out and all(x["role"] == "system" for x in out):
                out[0] = {**out[0], "content": out[0]["content"] + "\n\n" + m["content"]}
                continue
            if out:
                m = {"role": "user", "content": m["content"]}
        out.append(m)
    return out


class LocalChat:
    def __init__(self, url: str, model: str, timeout: float = 25) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("LLM endpoint must be local loopback HTTP")
        self.url, self.model, self.timeout = url, model, timeout

    def complete(self, messages: list[dict], tools: list[dict], timeout: float | None = None) -> dict:
        if os.environ.get("LIFEMATE_MERGE_SYSTEM") == "1":
            messages = merge_system(messages)
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 512,
        }
        # Extra request fields for other base models, e.g. LIFEMATE_LLM_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'
        payload.update(json.loads(os.environ.get("LIFEMATE_LLM_EXTRA", "{}")))
        request = Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=min(self.timeout, timeout) if timeout else self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:600].decode(errors='replace')}") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, SocketTimeout)):
                raise TimeoutError("local LLM request timed out") from exc
            raise

"""Step-level ensemble of several LoRA agents on ONE ToolEnv (the kit allows exactly one ToolEnv per sample and counts every call).

At every turn each member sees the same trajectory (assistant calls + tool results so far) under its own orchestration (system
prompt and request message: E1, E1D or E1P), all members are asked in parallel, and the proposals are voted: a proposal is the
ordered list of (tool name, canonical arguments), or STOP when the member emits no call. The proposal with most votes wins;
ties go to the earliest member in the list. The agent loop (lifemate.agent.execute) then executes only the winning proposal, so
one trajectory is run, exactly as with a single model.

EnsembleChat has the LocalChat interface (complete(messages, tools, timeout)); bind() must be called once per sample with the
item, transcript and installed specs so each member's head messages can be rebuilt.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .agent import prompt_for, user_message
from .llm import LocalChat


def _proposal_key(response: dict) -> str:
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        return "STOP"
    out = []
    for call in calls:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = fn.get("arguments")
        out.append([fn.get("name"), args])
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


class EnsembleChat:
    def __init__(self, url: str, members: list[tuple[str, str]], timeout: float = 25) -> None:
        """members: [(served model name, orchestration), ...]; the first is the tie-breaker."""
        if not members:
            raise ValueError("empty ensemble")
        self.members = [(LocalChat(url, name, timeout), name, orch) for name, orch in members]
        self.pool = ThreadPoolExecutor(max_workers=len(members))
        self.heads: dict[str, tuple[dict, dict]] = {}
        self.last_votes: list[dict] = []

    def bind(self, item: dict, transcript: str, specs: list[dict]) -> None:
        self.heads = {orch: ({"role": "system", "content": prompt_for(orch)}, user_message(item, transcript, orch, specs))
                      for _, _, orch in self.members}

    def _view(self, messages: list[dict], orch: str) -> list[dict]:
        """The agent's messages with the system prompt and the request message of this member's orchestration."""
        system, user = self.heads[orch]
        out = list(messages)
        out[0] = system
        idx = next(i for i, m in enumerate(out) if m.get("role") == "user")
        out[idx] = user
        return out

    def complete(self, messages: list[dict], tools: list[dict], timeout: float | None = None) -> dict:
        futures = [self.pool.submit(chat.complete, self._view(messages, orch), tools, timeout) for chat, _, orch in self.members]
        responses, errors = [], []
        for f in futures:
            try:
                responses.append(f.result())
            except Exception as exc:   # a failed member just loses its vote
                responses.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
        keys = [(_proposal_key(r) if r is not None else None) for r in responses]
        counts = Counter(k for k in keys if k is not None)
        if not counts:
            raise RuntimeError("every ensemble member failed: " + "; ".join(errors))
        best = max(counts.values())
        winner = next(k for k in keys if k is not None and counts[k] == best)
        chosen = responses[keys.index(winner)]
        self.last_votes.append({"members": [name for _, name, _ in self.members], "keys": keys, "winner": keys.index(winner),
                                "agreement": f"{best}/{len(self.members)}", "errors": errors})
        return chosen

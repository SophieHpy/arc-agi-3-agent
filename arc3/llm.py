"""Reasoning backends. The LLM proposes; the Action Guard and heuristic fallback decide."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from arc3.memory import Memory
from arc3.perception import grid_signature, grid_stats, to_ascii
from arc3.types import ActionChoice, GridObject, Observation

logger = logging.getLogger(__name__)

MAX_PROMPT_OBJECTS = 40

SYSTEM_PROMPT = (
    "You play an unknown ARC-AGI-3 grid game. Infer its rules from evidence. "
    "Reply with ONE JSON object only: "
    '{"action": <int>, "x": <0-63, only for action 6>, "y": <0-63, only for action 6>, '
    '"reason": "<short>"}. Choose only from available_actions. '
    "Prefer actions that test an open hypothesis; avoid dead_actions."
)


class Backend(Protocol):
    def propose(
        self, obs: Observation, memory: Memory, objects: list[GridObject]
    ) -> ActionChoice | None: ...


def build_llm_context(
    obs: Observation, memory: Memory, objects: list[GridObject]
) -> dict[str, object]:
    grid = obs.grid
    return {
        "game_id": obs.game_id,
        "state": obs.state,
        "levels_completed": obs.levels_completed,
        "available_actions": obs.legal_actions,
        "grid_signature": grid_signature(grid) if grid is not None else "",
        "grid_stats": grid_stats(grid) if grid is not None else {},
        "grid_ascii": to_ascii(grid) if grid is not None else "",
        "animation_frames": len(obs.frames),
        "objects": [o.to_dict() for o in objects[:MAX_PROMPT_OBJECTS]],
        "memory": memory.to_prompt_dict(),
    }


def parse_action_json(text: str) -> ActionChoice | None:
    """Extract the first JSON object from model output; None if absent or malformed."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : end + 1])
                    break
        start = text.find("{", start + 1)
    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "action" not in data:
            continue
        try:
            action = int(data["action"])
            x = int(data["x"]) if data.get("x") is not None else None
            y = int(data["y"]) if data.get("y") is not None else None
        except (TypeError, ValueError):
            continue
        return ActionChoice(
            action, x, y, str(data.get("reason", ""))[:200], source="llm"
        )
    return None


class HeuristicBackend:
    """Placeholder backend that defers entirely to the deterministic policy."""

    def propose(
        self, obs: Observation, memory: Memory, objects: list[GridObject]
    ) -> ActionChoice | None:
        return None


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_messages(self, context: dict[str, object]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
        ]

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        return str(payload["choices"][0]["message"]["content"] or "")

    def propose(
        self, obs: Observation, memory: Memory, objects: list[GridObject]
    ) -> ActionChoice | None:
        try:
            text = self.complete(
                self.build_messages(build_llm_context(obs, memory, objects))
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            logger.warning("LLM backend failed, falling back to heuristic: %s", exc)
            return None
        return parse_action_json(text)


def backend_from_env() -> Backend:
    if os.environ.get("ARC3_USE_LLM", "0") != "1":
        return HeuristicBackend()
    return OpenAICompatibleBackend(
        base_url=os.environ.get("ARC3_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("ARC3_LLM_API_KEY", "local"),
        model=os.environ.get("ARC3_LLM_MODEL", "local-model"),
        timeout=float(os.environ.get("ARC3_LLM_TIMEOUT", "60")),
    )

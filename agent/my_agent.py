"""Adapter between the official ARC-AGI-3-Agents `Agent` ABI and `arc3.AgentCore`.

Only translation lives here: FrameData → Observation, ActionChoice → GameAction.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from agents.agent import Agent
from arcengine import FrameData, GameAction

from arc3.agent_core import AgentCore
from arc3.llm import backend_from_env
from arc3.types import Observation

HASH_BORDER = int(os.environ.get("ARC3_HASH_BORDER", "0"))


def to_observation(frame: FrameData) -> Observation:
    return Observation(
        game_id=frame.game_id,
        state=frame.state.name,
        levels_completed=frame.levels_completed,
        win_levels=frame.win_levels,
        available_actions=list(frame.available_actions or []),
        frames=[np.asarray(f, dtype=np.int16) for f in (frame.frame or [])],
    )


class MyAgent(Agent):
    MAX_ACTIONS = 2000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.core = AgentCore(backend=backend_from_env(), signature_border=HASH_BORDER)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self.core.is_done(to_observation(latest_frame))

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        choice = self.core.step(to_observation(latest_frame))
        action = GameAction.from_id(choice.action)
        reasoning: dict[str, Any] = {"why": choice.reason, "source": choice.source}
        if action.is_complex():
            action.set_data({"x": choice.x, "y": choice.y})
            reasoning.update(x=choice.x, y=choice.y)
        action.reasoning = reasoning
        return action

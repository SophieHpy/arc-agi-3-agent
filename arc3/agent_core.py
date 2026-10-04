"""Observe → record transition → update memory → propose → guard → act."""

from __future__ import annotations

from arc3.action_policy import ActionGuard, HeuristicPolicy
from arc3.llm import Backend, HeuristicBackend
from arc3.memory import Memory
from arc3.perception import find_objects
from arc3.transition import extract_transition
from arc3.types import (
    RESET_ACTION,
    STATE_GAME_OVER,
    STATE_NOT_PLAYED,
    STATE_WIN,
    ActionChoice,
    Observation,
    Transition,
)


class AgentCore:
    def __init__(
        self,
        backend: Backend | None = None,
        policy: HeuristicPolicy | None = None,
        guard: ActionGuard | None = None,
        memory: Memory | None = None,
        signature_border: int = 0,
    ) -> None:
        self.backend = backend or HeuristicBackend()
        self.policy = policy or HeuristicPolicy()
        self.guard = guard or ActionGuard()
        self.memory = memory or Memory()
        self.signature_border = signature_border
        self.level = 0
        self.prev_obs: Observation | None = None
        self.prev_choice: ActionChoice | None = None
        self.transitions: list[Transition] = []
        self.llm_rejections = 0

    def is_done(self, obs: Observation) -> bool:
        return obs.state == STATE_WIN

    def step(self, obs: Observation) -> ActionChoice:
        self._observe(obs)
        if obs.state in (STATE_NOT_PLAYED, STATE_GAME_OVER) or obs.grid is None:
            return self._remember(
                None, ActionChoice(RESET_ACTION, reason=f"reset from {obs.state}")
            )

        if obs.levels_completed != self.level:
            self.level = obs.levels_completed
            self.memory.reset_level()

        objects = find_objects(obs.grid)
        proposal = self.backend.propose(obs, self.memory, objects)
        choice = self.guard.validate(proposal, obs)
        if proposal is not None and choice is None:
            self.llm_rejections += 1
        if choice is None:
            choice = self.guard.validate(
                self.policy.choose(obs, self.memory, objects), obs
            )
        if choice is None:
            choice = ActionChoice(RESET_ACTION, reason="no valid action")
        return self._remember(obs, choice)

    def _observe(self, obs: Observation) -> None:
        if self.prev_obs is None or self.prev_choice is None:
            return
        transition = extract_transition(
            self.prev_choice, self.prev_obs, obs, self.signature_border
        )
        if transition is not None:
            self.transitions.append(transition)
            self.memory.record(transition)

    def _remember(self, obs: Observation | None, choice: ActionChoice) -> ActionChoice:
        if choice.action == RESET_ACTION:
            self.prev_obs, self.prev_choice = None, None
        else:
            self.prev_obs, self.prev_choice = obs, choice
        return choice

"""Action Guard (legality + coordinate clamping) and the deterministic heuristic policy."""

from __future__ import annotations

from arc3.memory import Memory
from arc3.types import (
    CLICK_ACTION,
    GRID_MAX,
    GRID_MIN,
    RESET_ACTION,
    ActionChoice,
    GridObject,
    Observation,
)

MAX_CLICK_TARGETS = 32
CLICK_GRID_STEP = 8


def clamp_coord(value: int) -> int:
    return max(GRID_MIN, min(GRID_MAX, int(value)))


class ActionGuard:
    """Every action, from any backend, must pass through `validate` before execution."""

    def validate(
        self, choice: ActionChoice | None, obs: Observation
    ) -> ActionChoice | None:
        if choice is None:
            return None
        if choice.action == RESET_ACTION:
            return ActionChoice(
                RESET_ACTION, reason=choice.reason, source=choice.source
            )
        if choice.action not in obs.legal_actions:
            return None
        if choice.action == CLICK_ACTION:
            if choice.x is None or choice.y is None:
                return None
            return ActionChoice(
                CLICK_ACTION,
                clamp_coord(choice.x),
                clamp_coord(choice.y),
                choice.reason,
                choice.source,
            )
        return ActionChoice(choice.action, reason=choice.reason, source=choice.source)


def click_candidates(
    objects: list[GridObject], limit: int = MAX_CLICK_TARGETS
) -> list[tuple[int, int]]:
    """Object anchor cells, smallest objects first, then a coarse grid as fallback."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for obj in sorted(objects, key=lambda o: (o.size, o.min_y, o.min_x)):
        xy = (clamp_coord(obj.anchor_x), clamp_coord(obj.anchor_y))
        if xy not in seen:
            seen.add(xy)
            out.append(xy)
        if len(out) >= limit:
            return out
    half = CLICK_GRID_STEP // 2
    for y in range(half, GRID_MAX + 1, CLICK_GRID_STEP):
        for x in range(half, GRID_MAX + 1, CLICK_GRID_STEP):
            if (x, y) not in seen:
                seen.add((x, y))
                out.append((x, y))
    return out


class HeuristicPolicy:
    """Deterministic: untried actions first, then actions that changed the grid; skip dead actions."""

    def choose(
        self, obs: Observation, memory: Memory, objects: list[GridObject]
    ) -> ActionChoice:
        candidates: list[tuple[tuple[int, int, float, int], ActionChoice]] = []
        order = 0
        for action in obs.legal_actions:
            if action in (RESET_ACTION, CLICK_ACTION):
                continue
            stats = memory.stats_for(action)
            candidates.append(
                (
                    self._score(
                        stats.attempts, stats.changes, memory.is_dead(action), order
                    ),
                    ActionChoice(action, reason="heuristic"),
                )
            )
            order += 1
        if CLICK_ACTION in obs.legal_actions:
            for x, y in click_candidates(objects):
                stats = memory.click_stats_for(x, y)
                candidates.append(
                    (
                        self._score(
                            stats.attempts,
                            stats.changes,
                            memory.is_dead_click(x, y),
                            order,
                        ),
                        ActionChoice(CLICK_ACTION, x, y, reason="heuristic-click"),
                    )
                )
                order += 1
        if not candidates:
            return ActionChoice(RESET_ACTION, reason="no legal actions")
        return min(candidates, key=lambda c: c[0])[1]

    @staticmethod
    def _score(
        attempts: int, changes: int, dead: bool, order: int
    ) -> tuple[int, int, float, int]:
        change_rate = changes / attempts if attempts else 1.0
        return (int(dead), int(attempts > 0), -change_rate, order)

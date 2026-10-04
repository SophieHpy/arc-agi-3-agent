"""Framework-independent data types shared by every arc3 module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RESET_ACTION = 0
CLICK_ACTION = 6
ALL_ACTIONS = (1, 2, 3, 4, 5, 6, 7)
GRID_MIN = 0
GRID_MAX = 63

STATE_NOT_PLAYED = "NOT_PLAYED"
STATE_NOT_FINISHED = "NOT_FINISHED"
STATE_WIN = "WIN"
STATE_GAME_OVER = "GAME_OVER"


@dataclass(frozen=True)
class GridObject:
    """A 4-connected single-colour component."""

    color: int
    size: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    anchor_x: int
    anchor_y: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    def to_dict(self) -> dict[str, int]:
        return {
            "color": self.color,
            "size": self.size,
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }


@dataclass
class Observation:
    """One environment observation; `frames` holds every animation frame."""

    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def grid(self) -> np.ndarray | None:
        return self.frames[-1] if self.frames else None

    @property
    def legal_actions(self) -> list[int]:
        return (
            list(self.available_actions)
            if self.available_actions
            else list(ALL_ACTIONS)
        )


@dataclass(frozen=True)
class ActionChoice:
    action: int
    x: int | None = None
    y: int | None = None
    reason: str = ""
    source: str = "heuristic"

    @property
    def key(self) -> tuple[int, int, int]:
        if self.action == CLICK_ACTION and self.x is not None and self.y is not None:
            return (self.action, self.x, self.y)
        return (self.action, -1, -1)


@dataclass(frozen=True)
class Transition:
    action: ActionChoice
    before_signature: str
    after_signature: str
    changed_cells: int
    level_delta: int
    game_over: bool = False
    animation_frames: int = 1

    @property
    def changed(self) -> bool:
        return self.changed_cells > 0 or self.level_delta != 0 or self.game_over

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "action": self.action.action,
            "changed_cells": self.changed_cells,
            "level_delta": self.level_delta,
        }
        if self.action.action == CLICK_ACTION:
            out["x"] = self.action.x
            out["y"] = self.action.y
        if self.game_over:
            out["game_over"] = True
        if self.animation_frames > 1:
            out["animation_frames"] = self.animation_frames
        return out

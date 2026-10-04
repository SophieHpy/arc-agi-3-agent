"""Turn (observation, action, observation) into a Transition record."""

from __future__ import annotations

from arc3.perception import frame_diff, grid_signature
from arc3.types import STATE_GAME_OVER, ActionChoice, Observation, Transition


def extract_transition(
    action: ActionChoice,
    before: Observation,
    after: Observation,
    signature_border: int = 0,
) -> Transition | None:
    if before.grid is None or after.grid is None:
        return None
    return Transition(
        action=action,
        before_signature=grid_signature(before.grid, signature_border),
        after_signature=grid_signature(after.grid, signature_border),
        changed_cells=frame_diff(before.grid, after.grid),
        level_delta=after.levels_completed - before.levels_completed,
        game_over=after.state == STATE_GAME_OVER,
        animation_frames=len(after.frames),
    )

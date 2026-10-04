"""Structured memory: per-action evidence, facts and revisable hypotheses."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from arc3.types import CLICK_ACTION, Transition


@dataclass
class ActionStats:
    attempts: int = 0
    changes: int = 0
    level_advances: int = 0
    dead_count: int = 0

    def update(self, transition: Transition) -> None:
        self.attempts += 1
        if transition.level_delta > 0:
            self.level_advances += 1
        if transition.changed:
            self.changes += 1
            self.dead_count = 0
        else:
            self.dead_count += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "changes": self.changes,
            "level_advances": self.level_advances,
            "dead_count": self.dead_count,
        }


@dataclass
class Hypothesis:
    text: str
    confidence: float = 0.5
    support: int = 0
    refute: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "support": self.support,
            "refute": self.refute,
        }


class Memory:
    """Evidence store. Per-level statistics reset on level change; facts and hypotheses persist."""

    def __init__(self, dead_threshold: int = 2, max_events: int = 20) -> None:
        self.dead_threshold = dead_threshold
        self.action_stats: dict[int, ActionStats] = {}
        self.click_stats: dict[tuple[int, int], ActionStats] = {}
        self.facts: list[str] = []
        self.hypotheses: list[Hypothesis] = []
        self.recent_events: deque[dict[str, object]] = deque(maxlen=max_events)
        self.visited_signatures: set[str] = set()

    def record(self, transition: Transition) -> None:
        action = transition.action
        self.action_stats.setdefault(action.action, ActionStats()).update(transition)
        if (
            action.action == CLICK_ACTION
            and action.x is not None
            and action.y is not None
        ):
            self.click_stats.setdefault((action.x, action.y), ActionStats()).update(
                transition
            )
        self.visited_signatures.add(transition.before_signature)
        self.visited_signatures.add(transition.after_signature)
        self.recent_events.append(transition.to_dict())
        if transition.changed_cells > 0 and action.action != CLICK_ACTION:
            self.add_fact(f"Action {action.action} changes the grid.")
        if transition.level_delta > 0:
            self.add_fact(f"Action {action.action} completed a level.")
        if transition.game_over:
            self.add_fact(f"Action {action.action} has led to GAME_OVER.")

    def reset_level(self) -> None:
        self.action_stats.clear()
        self.click_stats.clear()
        self.recent_events.clear()
        self.visited_signatures.clear()

    def stats_for(self, action: int) -> ActionStats:
        return self.action_stats.get(action, ActionStats())

    def click_stats_for(self, x: int, y: int) -> ActionStats:
        return self.click_stats.get((x, y), ActionStats())

    def is_dead(self, action: int) -> bool:
        return self.stats_for(action).dead_count >= self.dead_threshold

    def is_dead_click(self, x: int, y: int) -> bool:
        return self.click_stats_for(x, y).dead_count >= 1

    @property
    def dead_actions(self) -> list[int]:
        return sorted(
            a for a in self.action_stats if a != CLICK_ACTION and self.is_dead(a)
        )

    def add_fact(self, text: str) -> None:
        if text not in self.facts:
            self.facts.append(text)

    def add_hypothesis(self, text: str, confidence: float = 0.5) -> Hypothesis:
        for hyp in self.hypotheses:
            if hyp.text == text:
                return hyp
        hyp = Hypothesis(text=text, confidence=confidence)
        self.hypotheses.append(hyp)
        return hyp

    def update_hypothesis(self, text: str, supported: bool, step: float = 0.2) -> None:
        """Adjust confidence; hypotheses whose confidence reaches 0 are discarded."""
        hyp = self.add_hypothesis(text)
        if supported:
            hyp.support += 1
            hyp.confidence = round(min(1.0, hyp.confidence + step), 6)
        else:
            hyp.refute += 1
            hyp.confidence = round(max(0.0, hyp.confidence - step), 6)
        self.hypotheses = [h for h in self.hypotheses if h.confidence > 0.0]

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "dead_actions": self.dead_actions,
            "action_stats": {
                str(a): s.to_dict() for a, s in sorted(self.action_stats.items())
            },
            "dead_clicks": [
                list(xy) for xy, s in self.click_stats.items() if s.dead_count >= 1
            ][:20],
            "facts": list(self.facts),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "recent_events": list(self.recent_events),
        }

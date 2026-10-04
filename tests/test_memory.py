import pytest

from arc3.memory import Memory
from arc3.types import ActionChoice, Transition


def tr(action: int, changed: int, level_delta: int = 0, x=None, y=None) -> Transition:
    return Transition(
        ActionChoice(action, x, y), "a", "b" if changed else "a", changed, level_delta
    )


def test_action_stats_and_dead_actions():
    mem = Memory(dead_threshold=2)
    mem.record(tr(1, 5))
    mem.record(tr(4, 0))
    assert mem.dead_actions == []
    mem.record(tr(4, 0))
    assert mem.dead_actions == [4]
    assert mem.stats_for(1).to_dict() == {
        "attempts": 1,
        "changes": 1,
        "level_advances": 0,
        "dead_count": 0,
    }
    assert "Action 1 changes the grid." in mem.facts


def test_change_revives_dead_action():
    mem = Memory(dead_threshold=2)
    mem.record(tr(4, 0))
    mem.record(tr(4, 0))
    mem.record(tr(4, 3))
    assert mem.dead_actions == []


def test_click_stats_and_level_reset_keeps_facts():
    mem = Memory()
    mem.record(tr(6, 0, x=3, y=4))
    assert mem.is_dead_click(3, 4)
    mem.record(tr(2, 1, level_delta=1))
    assert mem.stats_for(2).level_advances == 1
    mem.reset_level()
    assert not mem.is_dead_click(3, 4)
    assert mem.action_stats == {}
    assert "Action 2 completed a level." in mem.facts


def test_hypotheses_can_be_refuted():
    mem = Memory()
    mem.add_hypothesis("red is interactive", 0.4)
    mem.update_hypothesis("red is interactive", supported=True)
    assert mem.hypotheses[0].confidence == pytest.approx(0.6)
    for _ in range(3):
        mem.update_hypothesis("red is interactive", supported=False)
    assert mem.hypotheses == []


def test_prompt_dict_shape():
    mem = Memory()
    mem.record(tr(1, 2))
    d = mem.to_prompt_dict()
    assert set(d) >= {
        "dead_actions",
        "action_stats",
        "facts",
        "hypotheses",
        "recent_events",
    }
    assert d["action_stats"]["1"]["changes"] == 1

import numpy as np

from arc3.action_policy import ActionGuard, HeuristicPolicy, clamp_coord
from arc3.agent_core import AgentCore
from arc3.llm import build_llm_context, parse_action_json
from arc3.memory import Memory
from arc3.perception import find_objects
from arc3.types import ActionChoice, Observation, Transition


def obs(actions, grid=None, state="NOT_FINISHED", level=0) -> Observation:
    g = np.zeros((8, 8), dtype=np.int16) if grid is None else grid
    g = g.copy()
    g[2, 2] = 4
    return Observation("toy", state, level, 3, list(actions), [g])


class FixedBackend:
    def __init__(self, choice):
        self.choice = choice

    def propose(self, obs, memory, objects):
        return self.choice


def test_clamp_and_guard():
    guard = ActionGuard()
    o = obs([1, 6])
    assert clamp_coord(-5) == 0 and clamp_coord(99) == 63
    assert guard.validate(ActionChoice(3), o) is None
    assert guard.validate(ActionChoice(6), o) is None
    assert guard.validate(ActionChoice(6, 80, -1), o).key == (6, 63, 0)
    assert guard.validate(ActionChoice(1, 5, 5), o).key == (1, -1, -1)
    assert guard.validate(ActionChoice(0), o).action == 0


def test_heuristic_prefers_untried_and_skips_dead():
    mem = Memory(dead_threshold=2)
    o = obs([1, 2])
    policy = HeuristicPolicy()
    assert policy.choose(o, mem, []).action == 1
    for _ in range(2):
        mem.record(Transition(ActionChoice(1), "s", "s", 0, 0))
    assert policy.choose(o, mem, []).action == 2
    mem.record(Transition(ActionChoice(2), "s", "t", 3, 0))
    assert policy.choose(o, mem, []).action == 2


def test_heuristic_clicks_objects_first():
    o = obs([6])
    choice = HeuristicPolicy().choose(o, Memory(), find_objects(o.grid))
    assert choice.key == (6, 2, 2)


def test_parse_action_json():
    assert parse_action_json(
        '```json\n{"action": 6, "x": 3, "y": 4, "reason": "t"}\n```'
    ).key == (6, 3, 4)
    assert parse_action_json('think... {"action": "2"} done').action == 2
    assert parse_action_json("not json") is None
    assert parse_action_json('{"move": 1}') is None
    assert parse_action_json('{"action": "left"}') is None


def test_illegal_llm_output_falls_back_to_heuristic():
    core = AgentCore(backend=FixedBackend(ActionChoice(5, source="llm")))
    choice = core.step(obs([1, 2]))
    assert choice.action in (1, 2) and choice.source == "heuristic"
    assert core.llm_rejections == 1


def test_valid_llm_output_is_used():
    core = AgentCore(backend=FixedBackend(ActionChoice(6, 100, 3, source="llm")))
    assert core.step(obs([6])).key == (6, 63, 3)


def test_core_records_transitions_and_resets():
    core = AgentCore()
    assert core.step(obs([1], state="NOT_PLAYED")).action == 0
    first = core.step(obs([1, 2]))
    core.step(obs([1, 2]))
    assert (
        core.transitions[-1].action == first and core.transitions[-1].changed_cells == 0
    )
    core.step(obs([1, 2], level=1))
    assert core.memory.action_stats == {}
    assert core.step(obs([1, 2], state="GAME_OVER", level=1)).action == 0


def test_llm_context_matches_spec():
    o = obs([1, 6])
    ctx = build_llm_context(o, Memory(), find_objects(o.grid))
    assert {
        "game_id",
        "state",
        "levels_completed",
        "available_actions",
        "grid_signature",
        "grid_stats",
        "objects",
        "memory",
    } <= set(ctx)

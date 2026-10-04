"""Smoke test.

Default: drive AgentCore through a synthetic two-level toy game (numpy only).
With --game: play real ARC-AGI-3 games through agent/my_agent.py; requires the
`arc-agi` package and an ARC-AGI-3-Agents checkout at $ARC3_AGENTS_DIR.

    python scripts/smoke_test.py
    ARC3_AGENTS_DIR=../ARC-AGI-3-Agents python scripts/smoke_test.py --game ls20 --max-steps 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc3.agent_core import AgentCore
from arc3.types import ActionChoice, Observation


class ToyGame:
    """A cursor on an 8x8 board; action 4 (right) moves it, reaching the right edge completes a level."""

    def __init__(self, levels: int = 2) -> None:
        self.levels = levels
        self.level = 0
        self.pos = 0
        self.started = False

    def observe(self) -> Observation:
        grid = np.zeros((8, 8), dtype=np.int16)
        grid[4, self.pos] = 3
        grid[4, 7] = 5 if self.pos != 7 else 3
        if not self.started:
            state = "NOT_PLAYED"
        elif self.level >= self.levels:
            state = "WIN"
        else:
            state = "NOT_FINISHED"
        return Observation(
            "toy", state, self.level, self.levels, [1, 2, 3, 4, 6], [grid]
        )

    def act(self, choice: ActionChoice) -> None:
        if choice.action == 0:
            self.started = True
            self.pos = 0
        elif choice.action == 4:
            self.pos += 1
            if self.pos >= 7:
                self.level += 1
                self.pos = 0


def run_toy(max_steps: int) -> int:
    game, core = ToyGame(), AgentCore()
    for step in range(max_steps):
        o = game.observe()
        if core.is_done(o):
            print(
                f"toy game WIN after {step} actions; dead_actions={core.memory.dead_actions}"
            )
            return 0
        game.act(core.step(o))
    print(f"toy game not solved within {max_steps} actions (levels={game.level})")
    return 1


def run_real(games: str, max_steps: int) -> int:
    agents_dir = os.environ.get("ARC3_AGENTS_DIR")
    if not agents_dir:
        raise SystemExit("Set ARC3_AGENTS_DIR to an ARC-AGI-3-Agents checkout.")
    sys.path.insert(0, str(Path(agents_dir).resolve()))
    import arc_agi
    from arc_agi import OperationMode

    from agent.my_agent import MyAgent

    MyAgent.MAX_ACTIONS = max_steps
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    for game_id in [g.strip() for g in games.split(",") if g.strip()]:
        env = arc.make(game_id)
        if env is None:
            print(f"{game_id}: could not create environment")
            return 1
        agent = MyAgent(
            card_id="smoke",
            game_id=game_id,
            agent_name=f"smoke.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["smoke"],
        )
        agent.main()
        final = agent.frames[-1]
        print(
            f"{game_id}: state={final.state.name} levels={final.levels_completed} "
            f"actions={agent.action_counter} llm_rejections={agent.core.llm_rejections}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--game", default=None, help="Comma-separated real game ids, e.g. ls20,vc33"
    )
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()
    return run_real(args.game, args.max_steps) if args.game else run_toy(args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())

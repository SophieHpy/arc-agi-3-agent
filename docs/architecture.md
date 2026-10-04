# Architecture (V0.1)

```text
Environment ──FrameData──▶ agent/my_agent.py (adapter)
                                 │ Observation
                                 ▼
                          arc3.agent_core.AgentCore.step
                                 │
      ┌──────────────────────────┼─────────────────────────────┐
      ▼                          ▼                             ▼
 transition.extract_transition  perception.find_objects   llm.Backend.propose
 (prev obs, prev action, obs)   (objects, stats, ascii)   (optional LLM)
      │                                                        │
      ▼                                                        ▼
 memory.Memory.record ───────────────────────────────▶ action_policy.ActionGuard
                                                               │ invalid / None
                                                               ▼
                                                     action_policy.HeuristicPolicy
                                                               │
                                                               ▼
                                              ActionChoice ──▶ GameAction
```

## Modules

| Module | Responsibility |
|--------|----------------|
| `agent/my_agent.py` | Official ABI only: `is_done`, `choose_action`; converts types. No reasoning. |
| `arc3/types.py` | `Observation`, `ActionChoice`, `Transition`, `GridObject`, constants. No framework imports. |
| `arc3/perception.py` | Grid stats, stable signature (optional HUD border), ASCII, 4-connected objects, diffs. |
| `arc3/transition.py` | `action, before, after → Transition(changed_cells, level_delta, game_over, ...)`. |
| `arc3/memory.py` | Per-action and per-click `ActionStats`, dead actions, facts, revisable hypotheses, recent events. |
| `arc3/action_policy.py` | `ActionGuard` (legal actions, ACTION6 coords clamped to 0–63) and deterministic policy. |
| `arc3/llm.py` | `Backend` protocol, `HeuristicBackend`, `OpenAICompatibleBackend`, compact context, JSON parsing. |
| `arc3/agent_core.py` | The observe → record → propose → guard → act loop. |

## Invariants

- Every action, LLM or heuristic, passes `ActionGuard.validate`; the LLM cannot bypass it.
- LLM errors (timeout, bad JSON, illegal action) fall back to `HeuristicPolicy`; rejections are counted.
- Memory stores evidence (counts, transitions), not raw history. Per-level stats reset on level change;
  facts and hypotheses persist and hypotheses are dropped when confidence reaches 0.
- `arc3` never imports `agents` or `arcengine`, so it is unit-testable with numpy alone.

## Heuristic policy (V0.1)

Candidate actions are the legal simple actions plus ACTION6 clicks on object anchor cells
(smallest objects first, then a coarse 8-pixel grid). Ranking: not dead → untried → higher change rate →
stable order. A simple action is dead after 2 consecutive no-change attempts; a click target after 1.

## Extension points for later versions

- V0.2: `Observation.frames` already carries every animation frame; add frame-sequence features in `perception`.
- V0.3: add a planner that emits short plans and checks each `Transition` against its prediction.
- V0.4: replace the policy ranking with expected utility (progress + information gain − risk − repeats).
- V0.5: plug a local Qwen server into `OpenAICompatibleBackend`.

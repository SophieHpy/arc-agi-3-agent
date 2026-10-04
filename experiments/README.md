# Experiments

Rule: one baseline vs. one change. A single score increase is not evidence;
ARC-AGI-3 runs are high-variance, so repeat runs before drawing conclusions.

| ID | Change | Status |
|----|--------|--------|
| E0 | Official starter baseline (environment, ABI, submission pipeline) | todo |
| E1 | Deterministic legal-action policy (this repo, V0.1) | todo |
| E2 | Structured memory (action → effect) | todo |
| E3 | Multi-frame animation perception | todo |
| E4 | Short verified planning | todo |
| E5 | Active exploration | todo |
| E6 | Local Qwen backend | todo |
| E7 | World model | todo |

## Run record template

Copy into `experiments/<id>-<slug>.md` for every run:

```text
Experiment:
Commit:
Model:
Prompt:
Config:
Number of games:
Runs per config:
Score (mean / per run):
Levels completed:
Runtime:
Failure cases:
Conclusion:
```

Raw logs go under `experiments/runs/` (git-ignored).

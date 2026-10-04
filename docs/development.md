# Development

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git ../ARC-AGI-3-Agents
```

## Tests and lint

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Smoke tests

```bash
# Synthetic toy game, numpy only
.venv/bin/python scripts/smoke_test.py

# Real ARC-AGI-3 games (downloads environments into ./environment_files on first run)
ARC3_AGENTS_DIR=../ARC-AGI-3-Agents .venv/bin/python scripts/smoke_test.py --game ls20,vc33 --max-steps 200
```

## Enabling the LLM backend

```bash
export ARC3_USE_LLM=1
export ARC3_LLM_BASE_URL=http://127.0.0.1:8000/v1
export ARC3_LLM_API_KEY=local
export ARC3_LLM_MODEL=your-model
```

Any OpenAI-compatible `/chat/completions` server works (vLLM, SGLang, a hosted Qwen API).

## Kaggle submission

Not wired up yet (roadmap P0/P1). The Kaggle starter splices a single `my_agent.py` into its notebook, so
the `arc3` package must be shipped alongside it (Kaggle dataset or inlined into the notebook) before submitting.

## Conventions

- `arc3/` stays framework-independent; only `agent/my_agent.py` imports `agents` / `arcengine`.
- One change per experiment; record it with the template in `experiments/README.md`.

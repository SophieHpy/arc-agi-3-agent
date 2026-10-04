# ARC-AGI-3 Agent

面向 [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) 交互式环境的 Agent 研究框架。

**当前阶段：V0.1 Research Baseline。** 目标是把"观察 → 理解变化 → 记忆证据 → 选择合法动作 → 验证"的闭环工程化，
暂不追求最高分。后续版本在同一框架内逐步替换 Perception / Memory / Planning / Model / Exploration。

```text
Observation → Perception → State → Transition → Structured Memory
            → (LLM proposal) → Action Guard → Heuristic fallback → Environment
```

## 快速开始

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                 # 单元测试
.venv/bin/python scripts/smoke_test.py        # 合成小游戏冒烟测试（只依赖 numpy）

# 真实游戏（需要官方 ARC-AGI-3-Agents 框架）
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git ../ARC-AGI-3-Agents
ARC3_AGENTS_DIR=../ARC-AGI-3-Agents .venv/bin/python scripts/smoke_test.py --game ls20 --max-steps 200
```

## 项目结构

```text
agent/my_agent.py      官方 Agent ABI 适配层（is_done / choose_action），不含推理
arc3/types.py          Observation / ActionChoice / Transition / GridObject
arc3/perception.py     网格统计、签名、ASCII、连通域对象、帧差
arc3/transition.py     动作前后变化提取
arc3/memory.py         动作统计、dead action、facts、可推翻的 hypotheses
arc3/action_policy.py  Action Guard（合法性、ACTION6 坐标 clamp）+ 确定性启发式策略
arc3/llm.py            Backend 抽象：Heuristic / OpenAI-compatible，JSON 解析
arc3/agent_core.py     主循环
configs/               基线配置（qwen_baseline.yaml）
experiments/           实验记录模板与计划（E0–E7）
docs/                  architecture.md / development.md
tests/                 perception / memory / policy 单元测试
scripts/smoke_test.py  冒烟测试
```

## LLM 后端（可选）

```bash
export ARC3_USE_LLM=1
export ARC3_LLM_BASE_URL=http://127.0.0.1:8000/v1
export ARC3_LLM_API_KEY=local
export ARC3_LLM_MODEL=your-model
```

LLM 只输出一个 JSON 动作：`{"action": 6, "x": 32, "y": 24, "reason": "..."}`。
输出非法、JSON 错误、超时或服务不可用时自动回退到确定性策略；LLM 永远不能绕过 Action Guard。

## 路线图

| 版本 | 内容 |
|------|------|
| V0.1 | 基础闭环：ABI、感知、对象、帧差、结构化记忆、Action Guard、启发式策略、可选 LLM、单元测试 |
| V0.2 | 动画感知：利用 `FrameData.frame` 多帧信息 |
| V0.3 | 假设 + 短计划验证：短计划 → 执行 → 验证 → 重规划 |
| V0.4 | 主动探索：优先能减少不确定性的动作 |
| V0.5 | 本地 LLM / 多模态推理（Qwen 系列） |
| V1.0 | 自适应 World Model Agent |

## 核心原则

1. Observe before acting
2. Action must be verified
3. Memory stores evidence
4. Hypotheses can be wrong
5. Optimize through ablation —— 每次只改一个核心因素，单次提分不算证据

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/development.md](docs/development.md)。

## License

MIT，见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。

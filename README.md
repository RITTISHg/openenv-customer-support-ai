---
title: OpenEnv Customer Support AI
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# 🎫 OpenEnv: Customer Support Ticket Resolution Environment

A production-inspired [OpenEnv](https://openenv.org) environment that simulates a customer support desk where an AI agent receives tickets, inspects prior context, and performs actions to resolve customer issues.

> **Why this environment?** It mirrors real-world work, is fully text-based, and uses deterministic grading — making it practical, explainable, and easy to benchmark.

---

## ✨ Features

- **Clean OpenEnv API** — `reset()`, `step(action)`, `state()` interface
- **Typed Pydantic Models** — Observation, Action, Reward schemas with full validation
- **3 Graded Tasks** — Easy → Medium → Hard difficulty ladder
- **Shaped Rewards** — Partial credit for classification, action choice, response quality, and resolution
- **Deterministic Grading** — Rule-based graders with keyword matching, tone analysis, and state transitions
- **Baseline Runner** — OpenAI API or rule-based dummy agent
- **Docker Ready** — Clean build with no manual steps
- **Comprehensive Tests** — Full pytest suite

---

## 🏗 Architecture

```
openenv-support/
├── openenv.yaml          # Environment metadata & task definitions
├── Dockerfile            # Deployment container
├── requirements.txt      # Python dependencies
├── src/
│   ├── env.py            # Core environment (reset/step/state)
│   ├── models.py         # Pydantic schemas
│   ├── grader.py         # Deterministic grading engine
│   ├── utils.py          # Helpers (tone, keywords, quality)
│   └── tasks/
│       ├── easy.py       # Task 1: Order delayed
│       ├── medium.py     # Task 2: Refund + missing info
│       └── hard.py       # Task 3: Multi-turn billing dispute
├── scripts/
│   ├── run_baseline.py   # Baseline agent runner
│   └── evaluate.py       # Evaluation & reporting
└── tests/
    ├── test_env.py       # Environment tests
    └── test_grader.py    # Grader tests
```

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Python API

```python
from src.env import SupportEnv
from src.models import Action, ActionType

env = SupportEnv()

# Start an easy task
obs = env.reset("easy")
print(f"Ticket: {obs.ticket_id}")
print(f"Issue: {obs.issue_type.value}")
print(f"Message: {obs.customer_message}")

# Take an action
action = Action(
    action_type=ActionType.REPLY,
    message="Thank you for reaching out. I apologize for the delay. Let me check your tracking information.",
    category="delivery",
)
obs, reward, done = env.step(action)
print(f"Reward: {reward.score:+.4f}")
print(f"Breakdown: {reward.shaping_breakdown}")

# Inspect full state
state = env.state()
print(f"Status: {state.status.value}")
```

### 3. Run Baseline

```bash
# Dummy agent (no API key needed)
python scripts/run_baseline.py --dummy

# OpenAI agent
export OPENAI_API_KEY="sk-..."
python scripts/run_baseline.py --model gpt-4o-mini
```

### 4. Run Evaluation

```bash
python scripts/evaluate.py
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

### 6. Docker

```bash
docker build -t openenv-support .
docker run openenv-support
```

---

## 🎯 Tasks

| Difficulty | Scenario | Max Steps | Success Criteria |
|-----------|----------|-----------|-----------------|
| **Easy** | Order delayed complaint | 5 | Correct category + action, no destructive close |
| **Medium** | Refund with missing order number | 7 | Right branching decision, ask for missing info |
| **Hard** | Multi-turn billing dispute | 10 | Resolution quality, tone, policy compliance |

---

## 🔄 Environment Loop

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  reset() │────▶│  Observation  │────▶│  Agent   │
└──────────┘     └──────────────┘     └────┬─────┘
                                           │
                                      Action│
                                           ▼
┌──────────┐     ┌──────────────┐     ┌──────────┐
│   done?  │◀────│ Reward + Obs │◀────│  step()  │
└────┬─────┘     └──────────────┘     └──────────┘
     │
     ├── No  → loop back to Agent
     └── Yes → episode complete
```

| Stage | What happens | Why it matters |
|-------|--------------|---------------|
| `reset(task)` | Loads fresh ticket, initializes state | Starts every episode in a known condition |
| `step(action)` | Applies action, computes reward | Creates the learning signal |
| `state()` | Returns full internal state | Makes environment inspectable & testable |
| `done` | Ends on resolution, escalation, or max steps | Prevents infinite loops |

---

## 📊 Reward Design

| Signal | Score Range |
|--------|------------|
| Correct classification | +0.2 to +0.3 |
| Correct action choice | +0.3 to +0.4 |
| High-quality response | +0.2 to +0.3 |
| Correct final resolution | +0.2 to +0.4 |
| Unsafe/destructive behavior | -0.2 to -1.0 |
| Repeated wrong action | -0.05 per repeat |

---

## 📏 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Unique identifier |
| `customer_message` | string | Latest message from customer |
| `issue_type` | enum | delivery, refund, billing, technical, account, general |
| `priority` | enum | low, medium, high, urgent |
| `customer_tone` | enum | neutral, frustrated, angry, polite, confused |
| `conversation_history` | list[dict] | All prior messages |
| `remaining_steps` | int | Steps left before forced stop |
| `metadata` | dict | Order number, account ID, etc. |

## 🎮 Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | enum | reply, request_info, escalate, close |
| `message` | string | Response to the customer |
| `category` | string | Agent's issue classification |
| `escalation_reason` | string | Reason if escalating |

---

## 📈 Baseline Scores

Scores from the built-in rule-based agent (DummyAgent):

| Task | Score | Steps | Status |
|------|-------|-------|--------|
| Easy | ~+0.65 | 2 | resolved |
| Medium | ~+0.55 | 3 | resolved |
| Hard | ~+0.60 | 4 | resolved |
| **Average** | **~+0.60** | — | — |

*Scores may vary slightly. Run `python scripts/evaluate.py` for exact values.*

---

## 🐳 Deployment

### Docker

```bash
docker build -t openenv-support .
docker run openenv-support                    # Run evaluation
docker run openenv-support python scripts/run_baseline.py --dummy  # Run baseline
```

### Hugging Face Space

1. Push this repository to a Hugging Face Space
2. Ensure the `openenv` tag is set
3. The `openenv.yaml` file defines all entry points

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

*Prepared as a structural blueprint for the OpenEnv hackathon round.*

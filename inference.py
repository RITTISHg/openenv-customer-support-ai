"""
OpenEnv Customer Support — Inference Script
============================================
MANDATORY Environment Variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Add project root to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import SupportEnv
from src.models import Action, ActionType

# ── Configuration from environment variables ──
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "openenv-support"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_MAP = {"easy": 5, "medium": 7, "hard": 10}
TEMPERATURE = 0.3
MAX_TOKENS = 300

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI customer support agent operating within a simulated environment.
You receive a support ticket observation and must respond with a structured JSON action.

Available action_types: "reply", "request_info", "escalate", "close"

Rules:
1. Always be professional, empathetic, and helpful.
2. Classify the issue correctly in the "category" field (delivery/refund/billing/technical/account/general).
3. If critical information is missing (like order number), use "request_info".
4. Only "escalate" when the issue is truly beyond your scope.
5. Only "close" when the issue is fully resolved AND the customer is satisfied.
6. Never use unprofessional language.
7. Include specific details in your response.
8. Use empathetic language like "I understand your frustration", "I sincerely apologize".

Respond with ONLY valid JSON:
{
    "action_type": "reply|request_info|escalate|close",
    "message": "your response to the customer",
    "category": "delivery|refund|billing|technical|account|general",
    "escalation_reason": "reason if escalating, empty string otherwise"
}
""").strip()


# ── Logging functions (mandatory stdout format) ──

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ──

def get_model_action(client: OpenAI, observation: dict, step: int, history: List[str]) -> Action:
    """Send observation to the LLM and parse structured JSON response into an Action."""
    history_block = "\n".join(history[-4:]) if history else "None"

    user_prompt = textwrap.dedent(f"""\
    Current ticket observation:
    - Ticket ID: {observation.get('ticket_id')}
    - Customer Message: {observation.get('customer_message')}
    - Issue Type: {observation.get('issue_type')}
    - Priority: {observation.get('priority')}
    - Customer Tone: {observation.get('customer_tone')}
    - Remaining Steps: {observation.get('remaining_steps')}
    - Metadata: {json.dumps(observation.get('metadata', {}), indent=2)}
    - Step: {step}

    Recent history:
    {history_block}

    Provide your action as JSON:
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse JSON response
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        return Action(
            action_type=ActionType(result.get("action_type", "reply")),
            message=result.get("message", ""),
            category=result.get("category", ""),
            escalation_reason=result.get("escalation_reason", ""),
        )
    except Exception as exc:
        # Fallback action on any error
        return Action(
            action_type=ActionType.REPLY,
            message=(
                "Thank you for reaching out. I understand your concern and "
                "I sincerely apologize for the inconvenience. Let me look into "
                "this for you right away."
            ),
            category=observation.get("issue_type", "general"),
        )


# ── Main inference loop ──

def run_task(env: SupportEnv, client: OpenAI, task_name: str) -> dict:
    """Run a single task and return results."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name)
        max_steps = MAX_STEPS_MAP.get(task_name, 5)

        for step in range(1, max_steps + 1):
            obs_dict = obs.model_dump()
            # Convert enums to strings for JSON serialization
            obs_dict["issue_type"] = obs_dict["issue_type"].value if hasattr(obs_dict["issue_type"], "value") else str(obs_dict["issue_type"])
            obs_dict["priority"] = obs_dict["priority"].value if hasattr(obs_dict["priority"], "value") else str(obs_dict["priority"])
            obs_dict["customer_tone"] = obs_dict["customer_tone"].value if hasattr(obs_dict["customer_tone"], "value") else str(obs_dict["customer_tone"])

            action = get_model_action(client, obs_dict, step, history)

            obs, reward, done, info = env.step(action)

            r = reward.score
            error = info.get("last_action_error")
            rewards.append(r)
            steps_taken = step

            action_str = f"{action.action_type.value}('{action.category}')"
            log_step(step=step, action=action_str, reward=r, done=done, error=error)

            history.append(
                f"Step {step}: {action.action_type.value} -> reward {r:+.2f}"
            )

            if done:
                break

        # Compute normalized score in [0, 1]
        # Average reward per step, already clamped to [-1, 1] by grader
        raw_avg = sum(rewards) / len(rewards) if rewards else 0.0
        # Map from [-1, 1] to [0, 1]
        score = (raw_avg + 1.0) / 2.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        error_msg = str(exc)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=error_msg)
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


async def main() -> None:
    """Run inference on all tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportEnv()

    all_results = []
    for task_name in TASKS:
        result = run_task(env, client, task_name)
        all_results.append(result)

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n# Summary: avg_score={avg_score:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

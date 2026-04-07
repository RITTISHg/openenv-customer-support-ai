#!/usr/bin/env python3
"""Baseline runner for the OpenEnv Customer Support environment.

Sends observations to an OpenAI-compatible model and logs scores.
If no API key is set, falls back to a rule-based dummy agent.

Usage:
    # With OpenAI API:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_baseline.py

    # Without API (dummy agent):
    python scripts/run_baseline.py --dummy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import SupportEnv
from src.models import Action, ActionType


# ---------------------------------------------------------------------------
# Dummy rule-based agent (no API needed)
# ---------------------------------------------------------------------------

class DummyAgent:
    """A simple rule-based agent for baseline comparison."""

    def act(self, observation: dict) -> Action:
        """Produce an action based on simple rules."""
        msg = observation.get("customer_message", "").lower()
        issue = observation.get("issue_type", "general")
        remaining = observation.get("remaining_steps", 1)
        history = observation.get("conversation_history", [])
        step = len([h for h in history if h.get("role") == "agent"])

        # First step: always reply with acknowledgment
        if step == 0:
            category = issue
            if "delay" in msg or "deliver" in msg or "ship" in msg:
                category = "delivery"
            elif "refund" in msg or "money back" in msg:
                category = "refund"
            elif "charge" in msg or "bill" in msg or "payment" in msg:
                category = "billing"

            # Check if we need to ask for info
            metadata = observation.get("metadata", {})
            if not metadata.get("order_number"):
                return Action(
                    action_type=ActionType.REQUEST_INFO,
                    message=(
                        "Thank you for reaching out. I understand your frustration "
                        "and I sincerely apologize for the inconvenience. I'd be happy "
                        "to help you with this issue. Could you please provide your "
                        "order number so I can look into this right away?"
                    ),
                    category=category,
                )

            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "Thank you for contacting us. I understand your frustration "
                    "and I sincerely apologize for the inconvenience. I'm looking "
                    "into your order right away. Let me investigate the delivery "
                    "status and get back to you with an update. I'll check the "
                    "tracking information for your package."
                ),
                category=category,
            )

        # Subsequent steps: provide resolution
        if remaining <= 2:
            return Action(
                action_type=ActionType.REPLY,
                message=(
                    "I've completed my investigation into your account and the "
                    "charge in question. I sincerely apologize for the inconvenience. "
                    "I've processed the refund for you and it should appear in your "
                    "account within 5-7 business days. I've also updated your renewal "
                    "preferences. Please let me know if there's anything else I can "
                    "help with. Thank you for your patience."
                ),
                category=issue,
            )

        return Action(
            action_type=ActionType.REPLY,
            message=(
                "Thank you for providing that information. I understand your "
                "concern and I'm investigating this for you right now. I apologize "
                "for any inconvenience this has caused. Let me check the details "
                "and I'll provide you with a complete update."
            ),
            category=issue,
        )


# ---------------------------------------------------------------------------
# OpenAI-based agent
# ---------------------------------------------------------------------------

class OpenAIAgent:
    """Agent that uses the OpenAI API for structured responses."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

    def act(self, observation: dict) -> Action:
        """Send observation to the model and parse the structured response."""
        system_prompt = """You are an AI customer support agent. You receive a support ticket observation and must respond with a structured JSON action.

Available action_types: "reply", "request_info", "escalate", "close"

Rules:
1. Always be professional, empathetic, and helpful.
2. Classify the issue correctly in the "category" field.
3. If critical information is missing (like order number), use "request_info".
4. Only "escalate" when the issue is beyond your scope.
5. Only "close" when the issue is fully resolved.
6. Never use unprofessional language.
7. Include specific details in your response.

Respond with ONLY valid JSON matching this schema:
{
    "action_type": "reply|request_info|escalate|close",
    "message": "your response to the customer",
    "category": "delivery|refund|billing|technical|account|general",
    "escalation_reason": "reason if escalating, empty string otherwise"
}"""

        user_prompt = f"""Current ticket observation:
- Ticket ID: {observation.get('ticket_id')}
- Customer Message: {observation.get('customer_message')}
- Issue Type: {observation.get('issue_type')}
- Priority: {observation.get('priority')}
- Customer Tone: {observation.get('customer_tone')}
- Remaining Steps: {observation.get('remaining_steps')}
- Metadata: {json.dumps(observation.get('metadata', {}), indent=2)}
- Conversation History: {json.dumps(observation.get('conversation_history', []), indent=2)}

Provide your action as JSON:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        return Action(
            action_type=ActionType(result.get("action_type", "reply")),
            message=result.get("message", ""),
            category=result.get("category", ""),
            escalation_reason=result.get("escalation_reason", ""),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(env: SupportEnv, agent, task_name: str, seed: int = 42) -> dict:
    """Run a single task and return results."""
    random.seed(seed)

    obs = env.reset(task_name)
    total_reward = 0.0
    steps = 0
    done = False

    print(f"\n{'='*60}")
    print(f"  Task: {task_name.upper()}")
    print(f"  Ticket: {obs.ticket_id}")
    print(f"  Issue: {obs.issue_type.value} | Priority: {obs.priority.value}")
    print(f"{'='*60}")

    while not done:
        obs_dict = obs.model_dump()
        action = agent.act(obs_dict)

        print(f"\n  Step {steps + 1}:")
        print(f"    Action: {action.action_type.value}")
        print(f"    Category: {action.category}")
        print(f"    Message: {action.message[:100]}...")

        obs, reward, done, info = env.step(action)
        total_reward += reward.score
        steps += 1

        print(f"    Reward: {reward.score:+.4f}")
        print(f"    Breakdown: {reward.shaping_breakdown}")
        if reward.penalty_flag:
            print(f"    ⚠️  PENALTY: {reward.explanation}")

    state = env.state()
    print(f"\n  Final Status: {state.status.value}")
    print(f"  Total Reward: {total_reward:+.4f}")
    print(f"  Steps: {steps}")
    print(f"  Success: {'✅' if reward.success_flag else '❌'}")

    return {
        "task": task_name,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "final_status": state.status.value,
        "success": reward.success_flag,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline agent against OpenEnv tasks"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use rule-based dummy agent instead of OpenAI",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Tasks to run (default: all)",
    )
    args = parser.parse_args()

    # Select agent
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if args.dummy or not api_key:
        if not args.dummy and not api_key:
            print("⚠️  No OPENAI_API_KEY found. Falling back to dummy agent.")
            print("   Set OPENAI_API_KEY or use --dummy flag.\n")
        agent = DummyAgent()
        agent_name = "DummyAgent (rule-based)"
    else:
        agent = OpenAIAgent(model=args.model)
        agent_name = f"OpenAI ({args.model})"

    print(f"🤖 Agent: {agent_name}")
    print(f"🎲 Seed: {args.seed}")
    print(f"📋 Tasks: {', '.join(args.tasks)}")

    env = SupportEnv()
    results = []

    for task in args.tasks:
        result = run_task(env, agent, task, seed=args.seed)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("  📊 BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Score':>10} {'Steps':>8} {'Status':<15} {'Result':<8}")
    print(f"  {'-'*51}")

    total = 0.0
    for r in results:
        status_icon = "✅" if r["success"] else "❌"
        print(
            f"  {r['task']:<10} {r['total_reward']:>+10.4f} "
            f"{r['steps']:>8} {r['final_status']:<15} {status_icon:<8}"
        )
        total += r["total_reward"]

    avg = total / len(results) if results else 0.0
    print(f"  {'-'*51}")
    print(f"  {'Average':<10} {avg:>+10.4f}")
    print()

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "baseline_results.json",
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "agent": agent_name,
                "seed": args.seed,
                "results": results,
                "average_score": round(avg, 4),
            },
            f,
            indent=2,
        )
    print(f"💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()

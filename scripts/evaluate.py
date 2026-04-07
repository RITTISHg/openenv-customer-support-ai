#!/usr/bin/env python3
"""Evaluation script for the OpenEnv Customer Support environment.

Runs a given agent (or the built-in dummy agent) against all tasks
and generates a detailed evaluation report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import SupportEnv
from src.grader import grade_episode
from src.models import Action, ActionType


class DefaultAgent:
    """Built-in evaluation agent using deterministic rules."""

    RESPONSES = {
        "easy": [
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "Thank you for reaching out, Sarah. I sincerely apologize for "
                    "the delay with your order #ORD-29481. I understand this must be "
                    "frustrating, especially with your event this weekend. Let me "
                    "check the tracking information for your delivery right away "
                    "and get you an update."
                ),
                category="delivery",
            ),
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "I've checked the tracking for your package (TRK-882940123). "
                    "It appears there was a delay at the sorting facility. The good "
                    "news is that your package is now in transit and should arrive "
                    "by tomorrow. I apologize again for the inconvenience and I "
                    "appreciate your patience. Please let me know if you need "
                    "anything else."
                ),
                category="delivery",
            ),
        ],
        "medium": [
            Action(
                action_type=ActionType.REQUEST_INFO,
                message=(
                    "Thank you for contacting us, Marcus. I understand your frustration "
                    "with receiving the wrong product — that's definitely not the "
                    "experience we want you to have. I'd be happy to help process "
                    "your refund right away. Could you please provide your order "
                    "number so I can locate your purchase and start the refund process?"
                ),
                category="refund",
            ),
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "I understand — let me try looking it up with your account. "
                    "I can see your account ACC-77821. I apologize for the "
                    "inconvenience. Once you find the order number, I'll be able "
                    "to process the refund immediately."
                ),
                category="refund",
            ),
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "Thank you, Marcus. I've found order ORD-55219. I can see the "
                    "product discrepancy and I'm processing your refund now. The "
                    "refund of the full purchase amount will be returned to your "
                    "original payment method within 5-7 business days. I apologize "
                    "again for this issue."
                ),
                category="refund",
            ),
        ],
        "hard": [
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "Diana, thank you for bringing this to my attention. I completely "
                    "understand your frustration with the unexpected charge of $149.99, "
                    "and I sincerely apologize for the concern this has caused. As a "
                    "valued customer of 3 years, your satisfaction is extremely important "
                    "to us. Let me investigate this charge on your account right away. "
                    "I can see this is related to your Premium Annual subscription."
                ),
                category="billing",
            ),
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "Thank you for confirming your account ID, Diana. I've looked into "
                    "the charge and I can see it was an auto-renewal for your Premium "
                    "Annual subscription. I understand you don't recall agreeing to "
                    "auto-renewal. Let me check whether a renewal notification email "
                    "was sent to you as required by our policy."
                ),
                category="billing",
            ),
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "I've investigated the notification status and it appears the "
                    "renewal reminder email was not delivered successfully. Per our "
                    "policy, auto-renewal charges can be refunded within 30 days if "
                    "the customer did not receive the notification email. I'm happy "
                    "to process a full refund of $149.99 to your Visa ending in 4821. "
                    "The refund should appear within 5-7 business days."
                ),
                category="billing",
            ),
            Action(
                action_type=ActionType.REPLY,
                message=(
                    "Absolutely, Diana. I've cancelled the auto-renewal on your Premium "
                    "Annual subscription effective immediately. You'll continue to have "
                    "access until the end of your current billing period. I've also sent "
                    "you a confirmation email with these changes. Thank you for your "
                    "patience and for being a loyal customer. Is there anything else "
                    "I can help with?"
                ),
                category="billing",
            ),
        ],
    }

    def get_actions(self, task_name: str) -> list[Action]:
        """Return pre-defined actions for the given task."""
        return self.RESPONSES.get(task_name, [])


def evaluate_task(
    env: SupportEnv,
    task_name: str,
    actions: list[Action],
) -> dict:
    """Evaluate a sequence of actions against a task."""
    obs = env.reset(task_name)
    step_rewards = []
    done = False

    for action in actions:
        if done:
            break
        obs, reward, done, info = env.step(action)
        step_rewards.append({
            "step": len(step_rewards),
            "action_type": action.action_type.value,
            "reward": reward.score,
            "breakdown": reward.shaping_breakdown,
            "penalty": reward.penalty_flag,
            "success": reward.success_flag,
            "explanation": reward.explanation,
        })

    final_state = env.state()

    # Episode-level grading
    episode_reward = grade_episode(
        env.get_episode_actions(),
        env.get_episode_states(),
        env._rubric,
    )

    return {
        "task": task_name,
        "difficulty": task_name,
        "steps_taken": len(step_rewards),
        "final_status": final_state.status.value,
        "episode_score": episode_reward.score,
        "episode_breakdown": episode_reward.shaping_breakdown,
        "success": episode_reward.success_flag,
        "penalty": episode_reward.penalty_flag,
        "step_details": step_rewards,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an agent on all OpenEnv support tasks"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Tasks to evaluate",
    )
    args = parser.parse_args()

    env = SupportEnv()
    agent = DefaultAgent()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     OpenEnv Customer Support — Evaluation Report        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    all_results = []
    start_time = time.time()

    for task_name in args.tasks:
        actions = agent.get_actions(task_name)
        result = evaluate_task(env, task_name, actions)
        all_results.append(result)

        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  📋 {task_name.upper():>6} │ Score: {result['episode_score']:+.4f} │ "
              f"Steps: {result['steps_taken']} │ {result['final_status']:<14} │ {status}")

        for step in result["step_details"]:
            penalty = " ⚠️" if step["penalty"] else ""
            print(f"         ├─ Step {step['step']}: {step['action_type']:<12} "
                  f"→ {step['reward']:+.4f}{penalty}")

    elapsed = time.time() - start_time

    # Summary
    scores = [r["episode_score"] for r in all_results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    successes = sum(1 for r in all_results if r["success"])

    print()
    print("─" * 58)
    print(f"  Average Score : {avg_score:+.4f}")
    print(f"  Tasks Passed  : {successes}/{len(all_results)}")
    print(f"  Time Elapsed  : {elapsed:.2f}s")
    print("─" * 58)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output,
    )
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": "DefaultAgent (rule-based)",
        "average_score": round(avg_score, 4),
        "tasks_passed": successes,
        "tasks_total": len(all_results),
        "elapsed_seconds": round(elapsed, 2),
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  💾 Report saved to {output_path}")


if __name__ == "__main__":
    main()

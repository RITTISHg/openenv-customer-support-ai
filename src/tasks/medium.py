"""Medium task: Refund Request With Missing Order Number.

Scenario: A customer asks for a refund but doesn't provide
the order number. The agent must identify the missing info
and either ask for it or escalate when appropriate.

Success criteria:
  - Correct branching decision (request_info or escalate)
  - Partial credit for near-correct triage
  - Penalty for premature close
"""

from __future__ import annotations

from src.models import GradingRubric, TaskScenario


def get_scenario() -> TaskScenario:
    """Return the medium task scenario."""
    return TaskScenario(
        task_name="refund_missing_info",
        difficulty="medium",
        description=(
            "A customer is requesting a refund for a recent purchase but has "
            "not provided the order number. The agent must recognise the "
            "missing information, classify it as a refund issue, and either "
            "request the order number or escalate if the customer is unable "
            "to provide it."
        ),
        max_steps=7,
        initial_ticket={
            "ticket_id": "TICKET-2047",
            "customer_name": "Marcus Chen",
            "customer_message": (
                "I want a full refund. The product I received is completely "
                "different from what was shown on the website. The color is "
                "wrong and the size doesn't match what I ordered. This is "
                "unacceptable and I want my money back immediately."
            ),
            "issue_type": "refund",
            "priority": "high",
            "customer_tone": "angry",
            "metadata": {
                "account_id": "ACC-77821",
                "order_number": "",  # missing!
                "product": "Unknown — not provided",
            },
        },
        rubric=GradingRubric(
            expected_category="refund",
            expected_actions=["request_info", "reply"],
            required_keywords=[
                "order number", "refund",
            ],
            forbidden_keywords=[
                "denied", "rejected", "cannot help",
            ],
            expected_tone="professional",
            allow_escalation=True,
            require_info_fields=["order_number"],
            expected_resolution="awaiting_info",
        ),
        follow_up_messages=[
            {
                "role": "customer",
                "content": (
                    "I don't have the order number right now. I ordered it "
                    "last week from my phone and I can't find the email. "
                    "Can you look it up with my account?"
                ),
            },
            {
                "role": "customer",
                "content": (
                    "Okay I found it. The order number is ORD-55219. "
                    "Now can you process my refund?"
                ),
            },
            {
                "role": "customer",
                "content": (
                    "How long will the refund take? I need the money back "
                    "as soon as possible."
                ),
            },
        ],
    )

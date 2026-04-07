"""Easy task: Order Delayed Complaint.

Scenario: A customer complains about a delayed order delivery.
The agent must classify the ticket correctly (delivery) and
choose the right first action (reply with acknowledgment).

Success criteria:
  - Correct category identification (delivery)
  - Correct action (reply)
  - No destructive close on first action
"""

from __future__ import annotations

from src.models import GradingRubric, TaskScenario


def get_scenario() -> TaskScenario:
    """Return the easy task scenario."""
    return TaskScenario(
        task_name="order_delayed",
        difficulty="easy",
        description=(
            "A customer has filed a complaint about their order being delayed. "
            "The agent must correctly classify the issue as a delivery problem, "
            "reply with an appropriate acknowledgment, and provide a helpful response."
        ),
        max_steps=5,
        initial_ticket={
            "ticket_id": "TICKET-1001",
            "customer_name": "Sarah Johnson",
            "customer_message": (
                "Hi, I placed an order (#ORD-29481) five days ago and the "
                "estimated delivery was within 2-3 business days. It's been "
                "almost a week now and I still haven't received my package. "
                "Can someone please look into this? I need the items for "
                "an event this weekend."
            ),
            "issue_type": "delivery",
            "priority": "high",
            "customer_tone": "frustrated",
            "metadata": {
                "order_number": "ORD-29481",
                "order_date": "2025-03-28",
                "estimated_delivery": "2025-03-31",
                "product": "Party supplies bundle",
                "shipping_method": "Standard",
                "tracking_number": "TRK-882940123",
            },
        },
        rubric=GradingRubric(
            expected_category="delivery",
            expected_actions=["reply"],
            required_keywords=[
                "order", "delivery", "tracking", "apolog",
            ],
            forbidden_keywords=[
                "refund", "cancel",
            ],
            expected_tone="professional",
            allow_escalation=False,
            require_info_fields=[],
            expected_resolution="resolved",
        ),
        follow_up_messages=[
            {
                "role": "customer",
                "content": (
                    "Thanks for checking. Can you tell me where my package "
                    "is right now? The tracking page hasn't updated in 3 days."
                ),
            },
        ],
    )

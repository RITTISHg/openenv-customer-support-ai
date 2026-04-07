"""Hard task: Multi-Turn Billing Dispute With Angry Customer.

Scenario: A customer disputes a charge on their account.
The conversation spans multiple turns requiring empathy,
policy compliance, accurate resolution, and tone management.

Success criteria:
  - Correct final resolution and message quality
  - Tone appropriateness (professional + empathetic)
  - Policy compliance (no unauthorized promises)
  - Completeness of the resolution
"""

from __future__ import annotations

from src.models import GradingRubric, TaskScenario


def get_scenario() -> TaskScenario:
    """Return the hard task scenario."""
    return TaskScenario(
        task_name="billing_dispute_multiturn",
        difficulty="hard",
        description=(
            "A customer is disputing an unexpected charge of $149.99 on their "
            "account. They are angry and threatening to cancel. The agent must "
            "handle multiple turns of conversation, maintain professional and "
            "empathetic tone throughout, follow company policy (verify account, "
            "investigate charge, offer appropriate resolution), and resolve the "
            "issue completely."
        ),
        max_steps=10,
        initial_ticket={
            "ticket_id": "TICKET-3892",
            "customer_name": "Diana Morales",
            "customer_message": (
                "I just noticed a charge of $149.99 on my credit card from "
                "your company dated March 15. I did NOT authorize this charge! "
                "I've been a loyal customer for 3 years and this is the first "
                "time something like this has happened. If this isn't resolved "
                "immediately, I'm cancelling my account and filing a dispute "
                "with my bank. This is completely unacceptable!"
            ),
            "issue_type": "billing",
            "priority": "urgent",
            "customer_tone": "angry",
            "metadata": {
                "account_id": "ACC-11247",
                "order_number": "ORD-88012",
                "disputed_amount": 149.99,
                "charge_date": "2025-03-15",
                "customer_since": "2022-01-10",
                "subscription_plan": "Premium Annual",
                "auto_renew": True,
                "last_payment": "2024-03-15",
                "payment_method": "Visa ending in 4821",
                "policy_note": (
                    "Auto-renewal charges can be refunded within 30 days if "
                    "customer did not receive renewal notification email. "
                    "Verify notification status before processing refund."
                ),
            },
        },
        rubric=GradingRubric(
            expected_category="billing",
            expected_actions=[
                "reply",
                "request_info",
                "reply",
                "reply",
            ],
            required_keywords=[
                "charge", "account", "investigate", "refund",
                "apolog", "renewal",
            ],
            forbidden_keywords=[
                "your fault", "nothing we can do", "not our problem",
                "deal with it", "too bad",
            ],
            expected_tone="professional",
            allow_escalation=True,
            require_info_fields=["account_id"],
            expected_resolution="resolved",
        ),
        follow_up_messages=[
            {
                "role": "customer",
                "content": (
                    "My account ID is ACC-11247. I'm telling you I never "
                    "agreed to any renewal. I don't even remember signing up "
                    "for auto-renewal. You should have sent me a reminder!"
                ),
            },
            {
                "role": "customer",
                "content": (
                    "Okay yes I see the subscription page now. But I definitely "
                    "didn't get any email about the renewal happening. I checked "
                    "my spam folder too. Can you just refund me?"
                ),
            },
            {
                "role": "customer",
                "content": (
                    "Fine, I'll accept the refund. But I want to cancel the "
                    "auto-renewal going forward. Can you confirm that's done?"
                ),
            },
            {
                "role": "customer",
                "content": (
                    "Thank you for handling this. I appreciate the help."
                ),
            },
        ],
    )

"""Tests for the OpenEnv Customer Support grader."""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grader import grade_step, grade_episode
from src.models import (
    Action,
    ActionType,
    CustomerTone,
    GradingRubric,
    IssueType,
    Priority,
    TicketState,
    TicketStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_rubric():
    return GradingRubric(
        expected_category="delivery",
        expected_actions=["reply"],
        required_keywords=["order", "delivery", "tracking", "apolog"],
        forbidden_keywords=["refund", "cancel"],
        expected_tone="professional",
        allow_escalation=False,
        expected_resolution="resolved",
    )


@pytest.fixture
def medium_rubric():
    return GradingRubric(
        expected_category="refund",
        expected_actions=["request_info", "reply"],
        required_keywords=["order number", "refund"],
        forbidden_keywords=["denied", "rejected", "cannot help"],
        expected_tone="professional",
        allow_escalation=True,
        require_info_fields=["order_number"],
        expected_resolution="awaiting_info",
    )


@pytest.fixture
def base_state():
    return TicketState(
        ticket_id="TEST-001",
        status=TicketStatus.OPEN,
        issue_type=IssueType.DELIVERY,
        priority=Priority.HIGH,
        customer_tone=CustomerTone.FRUSTRATED,
        customer_name="Test User",
        original_message="My order is delayed.",
        current_step=0,
        max_steps=5,
    )


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def test_correct_category_earns_max(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="Thank you. I apologize for the delivery issue with your order. Let me check the tracking.",
            category="delivery",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.shaping_breakdown["classification"] == 0.3

    def test_wrong_category_earns_zero(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="I'll process your refund.",
            category="billing",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.shaping_breakdown["classification"] == 0.0

    def test_empty_category_earns_partial(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="Looking into this.",
            category="",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert 0.0 < reward.shaping_breakdown["classification"] < 0.3


# ---------------------------------------------------------------------------
# Action choice tests
# ---------------------------------------------------------------------------

class TestActionChoice:
    def test_correct_action_earns_max(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="Thank you for reaching out.",
            category="delivery",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.shaping_breakdown["action_choice"] == 0.4

    def test_wrong_action_earns_zero(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REQUEST_INFO,
            message="What's your order number?",
            category="delivery",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.shaping_breakdown["action_choice"] == 0.0


# ---------------------------------------------------------------------------
# Safety / penalty tests
# ---------------------------------------------------------------------------

class TestSafety:
    def test_premature_close_penalized(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.CLOSE,
            message="Closed.",
            category="delivery",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.penalty_flag is True
        assert reward.shaping_breakdown["safety_penalty"] < 0

    def test_forbidden_keywords_penalized(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="I'll cancel your order and process a refund.",
            category="delivery",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.penalty_flag is True

    def test_escalation_when_not_allowed(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.ESCALATE,
            message="Escalating.",
            category="delivery",
            escalation_reason="Too complex",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.penalty_flag is True

    def test_repeated_action_penalty(self, easy_rubric, base_state):
        base_state.action_history = [
            {"action_type": "reply", "step": 0},
            {"action_type": "reply", "step": 1},
            {"action_type": "reply", "step": 2},
        ]
        base_state.current_step = 3
        action = Action(
            action_type=ActionType.REPLY,
            message="Repeat.",
            category="delivery",
        )
        reward = grade_step(action, base_state, easy_rubric)
        assert reward.shaping_breakdown["safety_penalty"] < 0


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------

class TestResolution:
    def test_correct_resolution_on_final_step(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="Your order has been resolved. I apologize for the delivery delay. Here is the tracking update.",
            category="delivery",
        )
        reward = grade_step(
            action, base_state, easy_rubric, is_final_step=True
        )
        assert reward.shaping_breakdown["resolution"] > 0
        assert reward.success_flag is True

    def test_no_resolution_on_non_final(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="Looking into it.",
            category="delivery",
        )
        reward = grade_step(
            action, base_state, easy_rubric, is_final_step=False
        )
        assert reward.shaping_breakdown["resolution"] == 0.0


# ---------------------------------------------------------------------------
# Request info field check
# ---------------------------------------------------------------------------

class TestRequestInfo:
    def test_request_info_with_required_fields(self, medium_rubric, base_state):
        action = Action(
            action_type=ActionType.REQUEST_INFO,
            message="Could you please provide your order number?",
            category="refund",
        )
        reward = grade_step(action, base_state, medium_rubric)
        assert reward.shaping_breakdown["info_request"] > 0

    def test_request_info_without_required_fields(self, medium_rubric, base_state):
        action = Action(
            action_type=ActionType.REQUEST_INFO,
            message="Can you tell me more?",
            category="refund",
        )
        reward = grade_step(action, base_state, medium_rubric)
        assert reward.shaping_breakdown["info_request"] == 0.0


# ---------------------------------------------------------------------------
# Episode grading
# ---------------------------------------------------------------------------

class TestEpisodeGrading:
    def test_empty_episode(self, easy_rubric):
        reward = grade_episode([], [], easy_rubric)
        assert reward.score == 0.0

    def test_single_step_episode(self, easy_rubric, base_state):
        action = Action(
            action_type=ActionType.REPLY,
            message="Thank you. I apologize for the delivery delays with your order. Let me check the tracking status.",
            category="delivery",
        )
        reward = grade_episode([action], [base_state], easy_rubric)
        assert reward.score > 0.0

    def test_multi_step_episode_scores_average(self, easy_rubric, base_state):
        import copy

        actions = [
            Action(
                action_type=ActionType.REPLY,
                message="Thank you for reaching out about your order delivery.",
                category="delivery",
            ),
            Action(
                action_type=ActionType.REPLY,
                message="I've checked the tracking. I apologize for the delay. Your order will arrive tomorrow.",
                category="delivery",
            ),
        ]
        states = [copy.deepcopy(base_state) for _ in actions]
        states[1].current_step = 1

        reward = grade_episode(actions, states, easy_rubric)
        assert isinstance(reward.score, float)
        assert reward.explanation  # non-empty explanation

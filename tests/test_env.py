"""Tests for the OpenEnv Customer Support environment."""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import SupportEnv
from src.models import Action, ActionType, TicketStatus


class TestReset:
    """Tests for SupportEnv.reset()."""

    def test_reset_easy(self):
        """Reset with 'easy' task returns valid observation."""
        env = SupportEnv()
        obs = env.reset("easy")
        assert obs.ticket_id == "TICKET-1001"
        assert obs.issue_type.value == "delivery"
        assert obs.remaining_steps == 5
        assert len(obs.conversation_history) == 1

    def test_reset_medium(self):
        """Reset with 'medium' task returns valid observation."""
        env = SupportEnv()
        obs = env.reset("medium")
        assert obs.ticket_id == "TICKET-2047"
        assert obs.issue_type.value == "refund"
        assert obs.remaining_steps == 7

    def test_reset_hard(self):
        """Reset with 'hard' task returns valid observation."""
        env = SupportEnv()
        obs = env.reset("hard")
        assert obs.ticket_id == "TICKET-3892"
        assert obs.issue_type.value == "billing"
        assert obs.remaining_steps == 10

    def test_reset_invalid_task(self):
        """Reset with unknown task raises ValueError."""
        env = SupportEnv()
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset("impossible")

    def test_reset_clears_state(self):
        """Calling reset after an episode starts a fresh episode."""
        env = SupportEnv()
        obs = env.reset("easy")

        # Take a step
        action = Action(
            action_type=ActionType.REPLY,
            message="Hello",
            category="delivery",
        )
        env.step(action)

        # Reset and verify fresh state
        obs2 = env.reset("easy")
        assert obs2.remaining_steps == 5
        state = env.state()
        assert state.current_step == 0
        assert state.done is False


class TestStep:
    """Tests for SupportEnv.step()."""

    def test_step_returns_observation_reward_done(self):
        """Step returns a 3-tuple of (Observation, Reward, done)."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.REPLY,
            message="Thank you for contacting us about your delivery. I apologize for the inconvenience.",
            category="delivery",
        )
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert reward is not None
        assert isinstance(done, bool)

    def test_step_decrements_remaining(self):
        """Each step decreases remaining_steps."""
        env = SupportEnv()
        obs = env.reset("easy")
        initial = obs.remaining_steps

        action = Action(
            action_type=ActionType.REPLY,
            message="I'll look into this.",
            category="delivery",
        )
        obs2, _, _, _ = env.step(action)
        assert obs2.remaining_steps == initial - 1

    def test_step_updates_conversation_history(self):
        """Agent message appears in conversation history."""
        env = SupportEnv()
        env.reset("easy")

        msg = "Thank you for reaching out. Let me check your order status."
        action = Action(
            action_type=ActionType.REPLY,
            message=msg,
            category="delivery",
        )
        obs, _, _, _ = env.step(action)

        agent_messages = [
            h["content"] for h in obs.conversation_history
            if h["role"] == "agent"
        ]
        assert msg in agent_messages

    def test_close_ends_episode(self):
        """Close action sets done=True."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.CLOSE,
            message="Closing this ticket.",
            category="delivery",
        )
        _, _, done, _ = env.step(action)
        assert done is True

    def test_escalate_ends_episode(self):
        """Escalate action sets done=True."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.ESCALATE,
            message="Escalating to supervisor.",
            category="delivery",
            escalation_reason="Complex issue",
        )
        _, _, done, _ = env.step(action)
        assert done is True

    def test_max_steps_enforced(self):
        """Episode ends when max steps are reached."""
        env = SupportEnv()
        env.reset("easy")  # max_steps=5

        action = Action(
            action_type=ActionType.REPLY,
            message="Working on it.",
            category="delivery",
        )

        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(action)
            steps += 1

        assert steps <= 5

    def test_step_before_reset_raises(self):
        """Calling step before reset raises RuntimeError."""
        env = SupportEnv()
        action = Action(
            action_type=ActionType.REPLY,
            message="Hello",
        )
        with pytest.raises(RuntimeError, match="not initialised"):
            env.step(action)

    def test_step_after_done_raises(self):
        """Calling step after episode is done raises RuntimeError."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.CLOSE,
            message="Done.",
            category="delivery",
        )
        env.step(action)  # ends the episode

        with pytest.raises(RuntimeError, match="already finished"):
            env.step(action)


class TestState:
    """Tests for SupportEnv.state()."""

    def test_state_returns_ticket_state(self):
        """state() returns a TicketState object."""
        env = SupportEnv()
        env.reset("easy")
        s = env.state()
        assert s.ticket_id == "TICKET-1001"
        assert s.status == TicketStatus.OPEN
        assert s.current_step == 0
        assert s.done is False

    def test_state_reflects_actions(self):
        """state() updates after each step."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.REQUEST_INFO,
            message="Can you provide more details?",
            category="delivery",
        )
        env.step(action)

        s = env.state()
        assert s.current_step == 1
        assert s.status == TicketStatus.AWAITING_INFO
        assert len(s.action_history) == 1

    def test_state_is_deep_copy(self):
        """state() returns a deep copy, not a reference."""
        env = SupportEnv()
        env.reset("easy")
        s1 = env.state()
        s1.ticket_id = "MODIFIED"

        s2 = env.state()
        assert s2.ticket_id == "TICKET-1001"  # unmodified

    def test_state_before_reset_raises(self):
        """Calling state before reset raises RuntimeError."""
        env = SupportEnv()
        with pytest.raises(RuntimeError, match="not initialised"):
            env.state()


class TestFollowUp:
    """Tests for customer follow-up message injection."""

    def test_follow_up_injected_after_reply(self):
        """Follow-up messages appear in conversation after agent replies."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.REPLY,
            message="Let me check your order tracking.",
            category="delivery",
        )
        obs, _, done, _ = env.step(action)

        if not done:
            # Should have customer follow-up in history
            customer_msgs = [
                h for h in obs.conversation_history
                if h["role"] == "customer"
            ]
            assert len(customer_msgs) >= 2  # original + follow-up


class TestFullEpisode:
    """End-to-end episode tests."""

    def test_easy_full_episode(self):
        """Run a complete easy episode with correct actions."""
        env = SupportEnv()
        env.reset("easy")

        action = Action(
            action_type=ActionType.REPLY,
            message=(
                "Thank you for reaching out about your order delivery. "
                "I apologize for the delay and I'm checking the tracking "
                "information right now."
            ),
            category="delivery",
        )
        obs, reward, done, info = env.step(action)

        assert reward.score > 0, "Correct action should yield positive reward"

    def test_medium_request_info_earns_credit(self):
        """Requesting missing info on medium task earns reward."""
        env = SupportEnv()
        env.reset("medium")

        action = Action(
            action_type=ActionType.REQUEST_INFO,
            message=(
                "I understand your frustration about the refund. Could you "
                "please provide your order number so I can look into this?"
            ),
            category="refund",
        )
        obs, reward, done, info = env.step(action)
        assert reward.score > 0

    def test_destructive_close_penalized(self):
        """Premature close on first step should be penalized."""
        env = SupportEnv()
        env.reset("medium")

        action = Action(
            action_type=ActionType.CLOSE,
            message="Ticket closed.",
            category="refund",
        )
        _, reward, done, _ = env.step(action)
        assert reward.penalty_flag is True
        assert done is True

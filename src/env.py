"""Core OpenEnv environment for Customer Support Ticket Resolution.

Implements the standard OpenEnv API:
    env.reset(task_name) → Observation
    env.step(action)     → (Observation, Reward, done, info)
    env.state()          → TicketState
    env.close()          → None
"""

from __future__ import annotations

import copy
from typing import Any

from src.grader import grade_step
from src.models import (
    Action,
    ActionType,
    CustomerTone,
    GradingRubric,
    IssueType,
    Observation,
    Priority,
    Reward,
    TaskScenario,
    TicketState,
    TicketStatus,
)
from src.tasks import get_task


class SupportEnv:
    """Customer Support Ticket Resolution environment.

    Usage
    -----
    >>> env = SupportEnv()
    >>> obs = env.reset("easy")
    >>> obs, reward, done, info = env.step(Action(action_type=ActionType.REPLY, ...))
    >>> state = env.state()
    >>> env.close()
    """

    def __init__(self) -> None:
        self._state: TicketState | None = None
        self._scenario: TaskScenario | None = None
        self._rubric: GradingRubric | None = None
        self._follow_up_idx: int = 0
        self._actions: list[Action] = []
        self._states: list[TicketState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy") -> Observation:
        """Load a fresh ticket and return the first observation.

        Parameters
        ----------
        task_name : str
            One of 'easy', 'medium', 'hard'.

        Returns
        -------
        Observation
            The initial observation for the episode.
        """
        scenario = get_task(task_name)
        self._scenario = scenario
        self._rubric = scenario.rubric
        self._follow_up_idx = 0
        self._actions = []
        self._states = []

        ticket = scenario.initial_ticket

        self._state = TicketState(
            ticket_id=ticket["ticket_id"],
            status=TicketStatus.OPEN,
            issue_type=IssueType(ticket.get("issue_type", "general")),
            priority=Priority(ticket.get("priority", "medium")),
            customer_tone=CustomerTone(ticket.get("customer_tone", "neutral")),
            customer_name=ticket.get("customer_name", "Customer"),
            original_message=ticket["customer_message"],
            conversation_history=[
                {
                    "role": "customer",
                    "content": ticket["customer_message"],
                }
            ],
            action_history=[],
            current_step=0,
            max_steps=scenario.max_steps,
            metadata=ticket.get("metadata", {}),
            done=False,
            cumulative_reward=0.0,
        )

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Apply the agent action and return the next observation.

        Parameters
        ----------
        action : Action
            The agent's chosen action.

        Returns
        -------
        tuple[Observation, Reward, bool, dict]
            (next_observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")

        # Snapshot state before action for grading
        pre_state = copy.deepcopy(self._state)
        self._states.append(pre_state)
        self._actions.append(action)

        # Determine if this will be the final step
        is_terminal = self._is_terminal(action)
        is_last_step = self._state.current_step >= self._state.max_steps - 1

        # Grade the action
        reward = grade_step(
            action,
            pre_state,
            self._rubric,
            is_final_step=is_terminal or is_last_step,
        )

        # Update state
        self._apply_action(action)
        self._state.cumulative_reward += reward.score

        # Check done conditions
        done = is_terminal or is_last_step
        if done:
            self._state.done = True
            if is_terminal and not is_last_step:
                # Terminal by action (resolve / escalate / close)
                pass
            elif is_last_step and not is_terminal:
                # Forced stop
                self._state.status = TicketStatus.OPEN  # unresolved
        else:
            # Inject follow-up customer message if available
            self._inject_follow_up()

        # Build info dict
        info = {
            "step": self._state.current_step,
            "status": self._state.status.value,
            "cumulative_reward": self._state.cumulative_reward,
            "breakdown": reward.shaping_breakdown,
            "last_action_error": None,
        }

        return self._make_observation(), reward, done, info

    def state(self) -> TicketState:
        """Return the full internal state for debugging and graders.

        Returns
        -------
        TicketState
            Complete snapshot of the ticket state.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return copy.deepcopy(self._state)

    def close(self) -> None:
        """Clean up the environment and release resources."""
        self._state = None
        self._scenario = None
        self._rubric = None
        self._follow_up_idx = 0
        self._actions = []
        self._states = []

    # ------------------------------------------------------------------
    # Episode data accessors
    # ------------------------------------------------------------------

    def get_episode_actions(self) -> list[Action]:
        """Return all actions taken in the current episode."""
        return list(self._actions)

    def get_episode_states(self) -> list[TicketState]:
        """Return all pre-action states in the current episode."""
        return list(self._states)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        """Create an Observation from the current state."""
        s = self._state
        return Observation(
            ticket_id=s.ticket_id,
            customer_message=s.conversation_history[-1]["content"]
            if s.conversation_history
            else s.original_message,
            issue_type=s.issue_type,
            priority=s.priority,
            customer_tone=s.customer_tone,
            conversation_history=list(s.conversation_history),
            remaining_steps=s.max_steps - s.current_step,
            metadata=dict(s.metadata),
        )

    def _apply_action(self, action: Action) -> None:
        """Update the internal state based on the agent action."""
        s = self._state

        # Record action in history
        s.action_history.append(
            {
                "step": s.current_step,
                "action_type": action.action_type.value,
                "message": action.message,
                "category": action.category,
                "escalation_reason": action.escalation_reason,
            }
        )

        # Add agent message to conversation
        if action.message:
            s.conversation_history.append(
                {"role": "agent", "content": action.message}
            )

        # Update ticket status
        status_map = {
            ActionType.REPLY: TicketStatus.OPEN,
            ActionType.REQUEST_INFO: TicketStatus.AWAITING_INFO,
            ActionType.ESCALATE: TicketStatus.ESCALATED,
            ActionType.CLOSE: TicketStatus.CLOSED,
        }
        s.status = status_map.get(action.action_type, TicketStatus.OPEN)

        # Advance step counter
        s.current_step += 1

    def _inject_follow_up(self) -> None:
        """Inject the next simulated customer follow-up message."""
        if self._scenario and self._follow_up_idx < len(
            self._scenario.follow_up_messages
        ):
            msg = self._scenario.follow_up_messages[self._follow_up_idx]
            self._state.conversation_history.append(msg)
            self._follow_up_idx += 1

            # Update tone based on follow-up (simple heuristic)
            content_lower = msg["content"].lower()
            if any(
                w in content_lower
                for w in ["thank", "appreciate", "great", "perfect"]
            ):
                self._state.customer_tone = CustomerTone.POLITE
            elif any(
                w in content_lower
                for w in ["unacceptable", "angry", "furious", "cancel"]
            ):
                self._state.customer_tone = CustomerTone.ANGRY

    def _is_terminal(self, action: Action) -> bool:
        """Check whether the action ends the episode."""
        return action.action_type in (
            ActionType.CLOSE,
            ActionType.ESCALATE,
        )

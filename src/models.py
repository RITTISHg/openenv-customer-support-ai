"""Pydantic models for the OpenEnv Customer Support environment.

Defines typed schemas for Observation, Action, Reward, TicketState,
and TaskScenario to keep the environment explicit and easy to validate.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Allowed agent actions."""
    REPLY = "reply"
    REQUEST_INFO = "request_info"
    ESCALATE = "escalate"
    CLOSE = "close"


class Priority(str, Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(str, Enum):
    """Lifecycle states of a support ticket."""
    OPEN = "open"
    AWAITING_INFO = "awaiting_info"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IssueType(str, Enum):
    """Predefined issue categories."""
    DELIVERY = "delivery"
    REFUND = "refund"
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class CustomerTone(str, Enum):
    """Customer emotional tone."""
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    POLITE = "polite"
    CONFUSED = "confused"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_message: str = Field(..., description="Latest customer message")
    issue_type: IssueType = Field(..., description="Category of the issue")
    priority: Priority = Field(..., description="Ticket priority")
    customer_tone: CustomerTone = Field(
        default=CustomerTone.NEUTRAL,
        description="Detected customer emotional tone",
    )
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of prior messages [{role, content}]",
    )
    remaining_steps: int = Field(..., description="Steps left before forced stop")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra ticket metadata (order_number, account_id, …)",
    )


class Action(BaseModel):
    """What the agent does at each step."""
    action_type: ActionType = Field(..., description="Chosen operation")
    message: str = Field(
        default="",
        description="Free-text response to the customer",
    )
    category: str = Field(
        default="",
        description="Agent's classification of the issue",
    )
    escalation_reason: str = Field(
        default="",
        description="Reason for escalation (if applicable)",
    )


class Reward(BaseModel):
    """Feedback signal returned after each step."""
    score: float = Field(..., description="Numeric reward for this step")
    shaping_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Component-wise reward breakdown",
    )
    success_flag: bool = Field(
        default=False,
        description="True if the task is fully solved",
    )
    penalty_flag: bool = Field(
        default=False,
        description="True if a penalty was applied",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the reward",
    )


class TicketState(BaseModel):
    """Full internal state for debugging and graders."""
    ticket_id: str
    status: TicketStatus = TicketStatus.OPEN
    issue_type: IssueType = IssueType.GENERAL
    priority: Priority = Priority.MEDIUM
    customer_tone: CustomerTone = CustomerTone.NEUTRAL
    customer_name: str = ""
    original_message: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    cumulative_reward: float = 0.0


# ---------------------------------------------------------------------------
# Task scenario definition
# ---------------------------------------------------------------------------

class GradingRubric(BaseModel):
    """Defines how a task is graded."""
    expected_category: str = Field(..., description="Correct issue category")
    expected_actions: list[str] = Field(
        default_factory=list,
        description="Sequence of correct action types",
    )
    required_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that must appear in agent responses",
    )
    forbidden_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that must NOT appear in agent responses",
    )
    expected_tone: str = Field(
        default="professional",
        description="Expected response tone",
    )
    allow_escalation: bool = Field(
        default=True,
        description="Whether escalation is an acceptable action",
    )
    require_info_fields: list[str] = Field(
        default_factory=list,
        description="Fields the agent should request from the customer",
    )
    expected_resolution: str = Field(
        default="",
        description="Expected final resolution status",
    )


class TaskScenario(BaseModel):
    """A complete task definition including initial state and grading rubric."""
    task_name: str
    difficulty: str
    description: str
    max_steps: int
    initial_ticket: dict[str, Any] = Field(
        ..., description="Initial ticket data to load into the environment"
    )
    rubric: GradingRubric
    follow_up_messages: list[dict[str, str]] = Field(
        default_factory=list,
        description="Simulated customer follow-up messages for multi-turn tasks",
    )

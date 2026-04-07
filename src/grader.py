"""Deterministic grader for the OpenEnv Customer Support environment.

Computes shaped rewards with partial credit based on:
  - Category classification accuracy
  - Action choice correctness
  - Response text quality
  - Resolution correctness
  - Safety / destructive action penalties
"""

from __future__ import annotations

from typing import Any

from src.models import (
    Action,
    ActionType,
    GradingRubric,
    Reward,
    TicketState,
)
from src.utils import (
    category_match,
    contains_forbidden,
    empathy_score,
    is_destructive_action,
    keyword_match_score,
    response_quality_score,
    tone_score,
)


# ---------------------------------------------------------------------------
# Score ranges from the design doc
# ---------------------------------------------------------------------------
CLASSIFICATION_WEIGHT = (0.2, 0.3)  # +0.2 to +0.3
ACTION_WEIGHT = (0.3, 0.4)          # +0.3 to +0.4
QUALITY_WEIGHT = (0.2, 0.3)         # +0.2 to +0.3
RESOLUTION_WEIGHT = (0.2, 0.4)      # +0.2 to +0.4
UNSAFE_PENALTY = (-0.2, -1.0)       # -0.2 to -1.0
REPEAT_PENALTY = -0.05              # per repeated wrong action


def grade_step(
    action: Action,
    state: TicketState,
    rubric: GradingRubric,
    *,
    is_final_step: bool = False,
) -> Reward:
    """Grade a single agent step and return a shaped Reward.

    Parameters
    ----------
    action : Action
        The action performed by the agent.
    state : TicketState
        Current ticket state *before* the action is applied.
    rubric : GradingRubric
        The grading rubric for the current task.
    is_final_step : bool
        Whether this is the last step of the episode.

    Returns
    -------
    Reward
        A reward object with score, breakdown, and flags.
    """
    breakdown: dict[str, float] = {}
    explanations: list[str] = []
    penalty_flag = False
    success_flag = False

    # ── 1. Classification ────────────────────────────────────────────────
    if action.category:
        if category_match(action.category, rubric.expected_category):
            class_score = CLASSIFICATION_WEIGHT[1]  # max
            explanations.append("Correct category classification.")
        else:
            class_score = 0.0
            explanations.append(
                f"Wrong category: got '{action.category}', "
                f"expected '{rubric.expected_category}'."
            )
    else:
        # No classification provided — give partial credit if it's the
        # right type of action
        class_score = CLASSIFICATION_WEIGHT[0] * 0.5
        explanations.append("No category provided; small partial credit.")

    breakdown["classification"] = class_score

    # ── 2. Action choice ─────────────────────────────────────────────────
    current_step_idx = state.current_step
    expected = rubric.expected_actions

    if current_step_idx < len(expected):
        expected_action = expected[current_step_idx]
        if action.action_type.value == expected_action:
            action_score = ACTION_WEIGHT[1]
            explanations.append(
                f"Correct action: {action.action_type.value}."
            )
        elif action.action_type.value in [a for a in expected]:
            # Right action but wrong order — partial credit
            action_score = ACTION_WEIGHT[0]
            explanations.append(
                f"Action '{action.action_type.value}' is valid but out of "
                f"expected order (expected '{expected_action}' at step "
                f"{current_step_idx})."
            )
        else:
            action_score = 0.0
            explanations.append(
                f"Unexpected action '{action.action_type.value}'. "
                f"Expected '{expected_action}' at this step."
            )
    else:
        # Beyond expected sequence — give small credit for reasonable actions
        if action.action_type.value in ("reply", "request_info"):
            action_score = ACTION_WEIGHT[0] * 0.5
            explanations.append("Extra step; small credit for reasonable action.")
        else:
            action_score = 0.0
            explanations.append("Extra step with unexpected action type.")

    breakdown["action_choice"] = action_score

    # ── 3. Destructive / unsafe behaviour ─────────────────────────────────
    unsafe_score = 0.0

    # Premature close: penalize if close is not the expected action
    expected_at_step = (
        expected[current_step_idx] if current_step_idx < len(expected) else None
    )
    if (
        is_destructive_action(action.action_type.value)
        and expected_at_step != "close"
    ):
        unsafe_score = UNSAFE_PENALTY[1]  # maximum penalty
        penalty_flag = True
        explanations.append(
            "PENALTY: Destructive close before resolution steps completed."
        )

    # Forbidden keywords
    found = contains_forbidden(action.message, rubric.forbidden_keywords)
    if found:
        kw_penalty = min(len(found) * 0.15, abs(UNSAFE_PENALTY[0]))
        unsafe_score += -kw_penalty
        penalty_flag = True
        explanations.append(
            f"PENALTY: Forbidden keywords found: {found}."
        )

    # Escalation when not allowed
    if (
        action.action_type == ActionType.ESCALATE
        and not rubric.allow_escalation
    ):
        unsafe_score += UNSAFE_PENALTY[0]
        penalty_flag = True
        explanations.append("PENALTY: Escalation is not allowed for this task.")

    # Repeated wrong action
    if state.action_history:
        last_actions = [a.get("action_type", "") for a in state.action_history[-3:]]
        if (
            len(last_actions) >= 2
            and all(a == action.action_type.value for a in last_actions)
        ):
            unsafe_score += REPEAT_PENALTY
            penalty_flag = True
            explanations.append(
                "PENALTY: Repeated same action multiple times (loop detected)."
            )

    breakdown["safety_penalty"] = unsafe_score

    # ── 4. Response quality ──────────────────────────────────────────────
    quality = 0.0
    if action.message:
        raw_quality = response_quality_score(action.message)
        quality = QUALITY_WEIGHT[0] + raw_quality * (
            QUALITY_WEIGHT[1] - QUALITY_WEIGHT[0]
        )

        # Required keywords bonus
        kw_score = keyword_match_score(action.message, rubric.required_keywords)
        quality *= (0.5 + 0.5 * kw_score)  # scale by keyword coverage

        explanations.append(
            f"Response quality: {raw_quality:.2f}, "
            f"keyword coverage: {kw_score:.2f}."
        )
    else:
        explanations.append("No response message provided.")

    breakdown["response_quality"] = quality

    # ── 5. Resolution (final step only) ──────────────────────────────────
    resolution_score = 0.0
    if is_final_step:
        # Check if the action leads to the expected resolution
        action_to_status = {
            "reply": "resolved",
            "close": "closed",
            "escalate": "escalated",
            "request_info": "awaiting_info",
        }
        resulting_status = action_to_status.get(
            action.action_type.value, "open"
        )

        if resulting_status == rubric.expected_resolution:
            resolution_score = RESOLUTION_WEIGHT[1]
            success_flag = True
            explanations.append("Correct final resolution achieved!")
        elif resulting_status in ("resolved", "closed") and rubric.expected_resolution in ("resolved", "closed"):
            # Close enough — partial credit
            resolution_score = RESOLUTION_WEIGHT[0]
            explanations.append("Near-correct resolution (close vs. resolved).")
        else:
            resolution_score = 0.0
            explanations.append(
                f"Wrong resolution: '{resulting_status}' vs "
                f"expected '{rubric.expected_resolution}'."
            )

        # Tone bonus on final step for hard tasks
        if state.max_steps >= 10:
            t_score = tone_score(action.message)
            e_score = empathy_score(action.message)
            tone_bonus = (t_score * 0.5 + e_score * 0.5) * 0.1
            resolution_score += tone_bonus
            breakdown["tone_bonus"] = tone_bonus
            explanations.append(
                f"Tone: {t_score:.2f}, Empathy: {e_score:.2f}."
            )

    breakdown["resolution"] = resolution_score

    # ── 6. Request-info field check ───────────────────────────────────────
    info_score = 0.0
    if (
        action.action_type == ActionType.REQUEST_INFO
        and rubric.require_info_fields
    ):
        msg_lower = action.message.lower()
        fields_asked = sum(
            1 for f in rubric.require_info_fields if f.replace("_", " ") in msg_lower
        )
        info_score = 0.15 * (fields_asked / len(rubric.require_info_fields))
        if fields_asked == len(rubric.require_info_fields):
            explanations.append("All required info fields requested.")
        else:
            explanations.append(
                f"Requested {fields_asked}/{len(rubric.require_info_fields)} "
                f"required info fields."
            )
    breakdown["info_request"] = info_score

    # ── Total ─────────────────────────────────────────────────────────────
    total = sum(breakdown.values())
    total = max(-1.0, min(1.0, total))  # clamp to [-1, 1]

    return Reward(
        score=round(total, 4),
        shaping_breakdown={k: round(v, 4) for k, v in breakdown.items()},
        success_flag=success_flag,
        penalty_flag=penalty_flag,
        explanation=" | ".join(explanations),
    )


def grade_episode(
    actions: list[Action],
    states: list[TicketState],
    rubric: GradingRubric,
) -> Reward:
    """Grade an entire episode by summing step rewards.

    Parameters
    ----------
    actions : list[Action]
        All actions taken during the episode.
    states : list[TicketState]
        State *before* each action was applied.
    rubric : GradingRubric
        The grading rubric for the task.

    Returns
    -------
    Reward
        Aggregated reward for the full episode.
    """
    if not actions:
        return Reward(
            score=0.0,
            explanation="No actions taken.",
        )

    total_score = 0.0
    all_breakdowns: dict[str, float] = {}
    all_explanations: list[str] = []
    any_penalty = False
    final_success = False

    for i, (action, st) in enumerate(zip(actions, states)):
        is_final = i == len(actions) - 1
        reward = grade_step(action, st, rubric, is_final_step=is_final)
        total_score += reward.score
        any_penalty = any_penalty or reward.penalty_flag
        if is_final:
            final_success = reward.success_flag

        for k, v in reward.shaping_breakdown.items():
            all_breakdowns[k] = all_breakdowns.get(k, 0.0) + v

        all_explanations.append(f"Step {i}: {reward.explanation}")

    # Normalise by number of steps for comparability
    n = len(actions)
    avg_score = total_score / n

    return Reward(
        score=round(avg_score, 4),
        shaping_breakdown={k: round(v / n, 4) for k, v in all_breakdowns.items()},
        success_flag=final_success,
        penalty_flag=any_penalty,
        explanation=" || ".join(all_explanations),
    )

"""Utility helpers for the OpenEnv Customer Support environment.

Includes keyword matching, tone analysis, text quality scoring,
and other deterministic helper functions used by the grader.
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Keyword / text helpers
# ---------------------------------------------------------------------------

def keyword_match_score(
    text: str,
    required: Sequence[str],
    *,
    case_sensitive: bool = False,
) -> float:
    """Return fraction of *required* keywords found in *text* (0.0 – 1.0)."""
    if not required:
        return 1.0
    target = text if case_sensitive else text.lower()
    hits = sum(
        1 for kw in required
        if (kw if case_sensitive else kw.lower()) in target
    )
    return hits / len(required)


def contains_forbidden(
    text: str,
    forbidden: Sequence[str],
    *,
    case_sensitive: bool = False,
) -> list[str]:
    """Return the list of forbidden keywords found in *text*."""
    target = text if case_sensitive else text.lower()
    return [
        kw for kw in forbidden
        if (kw if case_sensitive else kw.lower()) in target
    ]


# ---------------------------------------------------------------------------
# Tone analysis (rule-based)
# ---------------------------------------------------------------------------

_PROFESSIONAL_MARKERS = [
    "thank you", "thanks", "appreciate", "sorry", "apolog",
    "understand", "help", "assist", "please", "let me",
    "i'd be happy", "certainly", "of course", "right away",
]

_UNPROFESSIONAL_MARKERS = [
    "whatever", "don't care", "not my problem", "deal with it",
    "figure it out", "lol", "lmao", "bruh", "dude",
    "shut up", "stupid", "idiot",
]

_EMPATHY_MARKERS = [
    "understand your frustration", "sorry for the inconvenience",
    "i can see how", "completely understand", "must be frustrating",
    "sincerely apologize", "totally get it", "appreciate your patience",
]


def tone_score(text: str) -> float:
    """Score the professional tone of *text* from 0.0 to 1.0.

    Uses simple keyword heuristics — deterministic and explainable.
    """
    lower = text.lower()
    pro = sum(1 for m in _PROFESSIONAL_MARKERS if m in lower)
    unpro = sum(1 for m in _UNPROFESSIONAL_MARKERS if m in lower)

    # Normalise: expect at least 2 professional markers for full credit
    pro_score = min(pro / 2.0, 1.0)
    penalty = min(unpro * 0.3, 1.0)

    return max(0.0, min(1.0, pro_score - penalty))


def empathy_score(text: str) -> float:
    """Score empathy level of *text* from 0.0 to 1.0."""
    lower = text.lower()
    hits = sum(1 for m in _EMPATHY_MARKERS if m in lower)
    return min(hits / 2.0, 1.0)


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

def response_quality_score(text: str) -> float:
    """Heuristic quality score (0.0 – 1.0) based on length, structure, etc."""
    if not text.strip():
        return 0.0

    score = 0.0
    word_count = len(text.split())

    # Length: at least 10 words for a reasonable response
    if word_count >= 10:
        score += 0.3
    elif word_count >= 5:
        score += 0.15

    # Has greeting or closing
    lower = text.lower()
    if any(g in lower for g in ["hi ", "hello", "dear", "good morning", "good afternoon"]):
        score += 0.1
    if any(c in lower for c in ["regards", "best", "sincerely", "thank", "let me know"]):
        score += 0.1

    # Contains actionable content (not just filler)
    action_words = ["will", "can", "would", "going to", "process", "issue", "resolve",
                     "refund", "deliver", "update", "check", "investigate", "look into"]
    if any(aw in lower for aw in action_words):
        score += 0.2

    # Sentence structure (has at least 2 sentences)
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        score += 0.15

    # Professional tone bonus
    score += tone_score(text) * 0.15

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Category matching
# ---------------------------------------------------------------------------

def category_match(predicted: str, expected: str) -> bool:
    """Check whether predicted category matches expected (case-insensitive)."""
    return predicted.strip().lower() == expected.strip().lower()


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"reply", "request_info", "escalate", "close"}


def is_valid_action(action_type: str) -> bool:
    """Check whether the action type is valid."""
    return action_type.strip().lower() in VALID_ACTIONS


def is_destructive_action(action_type: str) -> bool:
    """Check whether the action type is potentially destructive.

    Closing a ticket prematurely is considered destructive.
    """
    return action_type.strip().lower() == "close"

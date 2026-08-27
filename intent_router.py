"""Deterministic intent routing that never depends on an external LLM."""

from __future__ import annotations

import re
from typing import Literal

IntentType = Literal["GREETING", "OFF_TOPIC", "PORTFOLIO_QUERY", "FOLLOW_UP"]

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|who are you|how are you)[!?.\s]*$",
    re.IGNORECASE,
)
_PORTFOLIO_TERMS = {
    "manish", "portfolio", "project", "projects", "repo", "repos", "repository",
    "github", "skill", "skills", "experience", "work", "worked", "education",
    "resume", "cv", "contact", "email", "blog", "article", "articles", "open-source",
    "opensource", "technology", "technologies", "stack", "built", "contribution",
    "contributions", "organic maps", "kubestellar", "tab story", "carbon pulse",
    "omnisupport", "availability", "hire", "developer",
}
_FOLLOW_UP = re.compile(
    r"\b(it|that|this|they|them|those|more|else|how|why|when|where|which one)\b",
    re.IGNORECASE,
)
_OFF_TOPIC = re.compile(
    r"\b(?:weather|forecast|temperature|sports score|football score|cricket score|"
    r"stock price|crypto price|bitcoin|ethereum|recipe|medical advice|diagnose|"
    r"president|election|political news|latest news|capital of|solve this equation|"
    r"calculate|translate)\b",
    re.IGNORECASE,
)


def classify_intent(message: str, history: list | None = None) -> IntentType:
    normalized = " ".join(message.lower().split())
    if _GREETING.match(normalized):
        return "GREETING"
    if history and _FOLLOW_UP.search(normalized):
        return "FOLLOW_UP"
    if any(term in normalized for term in _PORTFOLIO_TERMS):
        return "PORTFOLIO_QUERY"
    if _OFF_TOPIC.search(normalized):
        return "OFF_TOPIC"
    # Retrieval can still find exact project names not covered by this compact
    # vocabulary. The relevance gate decides whether a query is on-topic.
    return "PORTFOLIO_QUERY"

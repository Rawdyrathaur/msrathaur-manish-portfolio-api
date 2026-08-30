"""Deterministic portfolio-query understanding for retrieval."""

from __future__ import annotations

import re
from typing import Iterable


_CONTEXT_REFERENCE = re.compile(
    r"(?i)\b(?:he|his|him|it|its|that|this|they|their|them|those|"
    r"tell me more|more about|what about|how about|why does|why is|"
    r"how does|how did|which one|and why|and how)\b"
)

_EXPANSIONS = (
    (
        re.compile(r"(?i)\b(?:strongest|strengths?|best at|speciali[sz](?:e|es|ed|ation)|expertise|capabilit(?:y|ies))\b"),
        "verified skills technologies experience projects contributions and demonstrated evidence",
    ),
    (
        re.compile(r"(?i)\b(?:backend|server|api|microservices?|distributed|scalable|production(?:-ready)?|reliab(?:le|ility))\b"),
        "backend engineering Spring Boot FastAPI Java Python microservices Kafka APIs Docker Redis testing architecture production deployment",
    ),
    (
        re.compile(r"(?i)\b(?:frontend|browser|extension|user interface|ui|react)\b"),
        "frontend engineering React JavaScript TypeScript Chrome extension user interface projects",
    ),
    (
        re.compile(r"(?i)\b(?:fit|suitable|hire|role|candidate|qualified)\b"),
        "role-relevant skills experience responsibilities projects contributions and concrete evidence",
    ),
    (
        re.compile(r"(?i)\b(?:compare|comparison|difference|different|versus|vs\.?|better)\b"),
        "compare project purpose features technology stack responsibilities status and demonstrated engineering",
    ),
    (
        re.compile(r"(?i)\b(?:kind of engineer|engineering profile|professional profile|background|who is)\b"),
        "professional summary engineering strengths skills experience projects open-source contributions and current focus",
    ),
    (
        re.compile(r"(?i)\bopen[- ]source\b"),
        "open-source contributions Organic Maps KubeStellar responsibilities technologies outcomes and learnings",
    ),
    (
        re.compile(r"(?i)\b(?:impact|outcome|why it matters|value|achievement|accomplish)\b"),
        "project impact responsibilities outcomes validation learnings and concrete achievements",
    ),
)


def _message_value(item, field: str, default: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get(field, default) or default)
    return str(getattr(item, field, default) or default)


def contextualize_query(message: str, history: Iterable | None = None) -> str:
    """Attach the latest user topic only when the new message refers back to it."""
    current = " ".join(message.split())
    if not history or not _CONTEXT_REFERENCE.search(current):
        return current

    previous_user_messages = [
        " ".join(_message_value(item, "content").split())
        for item in history
        if _message_value(item, "role") == "user" and _message_value(item, "content").strip()
    ]
    if not previous_user_messages:
        return current

    return f"Previous portfolio topic: {previous_user_messages[-1]}\nCurrent follow-up: {current}"


def expand_portfolio_query(query: str) -> str:
    """Add retrieval vocabulary for abstract portfolio questions without adding facts."""
    expansions = [text for pattern, text in _EXPANSIONS if pattern.search(query)]
    if not expansions:
        return query
    return f"{query}\nRelevant portfolio evidence: {'; '.join(dict.fromkeys(expansions))}"


def build_retrieval_queries(message: str, history: Iterable | None = None) -> list[str]:
    """Return unique semantic searches: exact wording, resolved follow-up, and expansion."""
    original = " ".join(message.split())
    contextual = contextualize_query(original, history)
    expanded = expand_portfolio_query(contextual)
    return list(dict.fromkeys(query for query in (original, contextual, expanded) if query))

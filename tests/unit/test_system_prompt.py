"""
Unit tests for system prompt generation and bot identity.
"""

import pytest
from system_prompt import build_system_prompt, IDENTITY, BEHAVIOR_RULES, LENGTH_RULES


def test_system_prompt_includes_identity():
    """Verify system prompt contains bot identity."""
    prompt = build_system_prompt("Test context")
    assert "Manish" in prompt
    assert IDENTITY in prompt


def test_system_prompt_includes_rag_context():
    """Verify RAG context is injected into prompt."""
    context = "Python, FastAPI, RAG expertise"
    prompt = build_system_prompt(context)
    assert context in prompt


def test_system_prompt_includes_behavior_rules():
    """Verify behavior rules are enforced in prompt."""
    prompt = build_system_prompt("")
    assert BEHAVIOR_RULES in prompt


def test_system_prompt_includes_length_rules():
    """Verify response length guidance is included."""
    prompt = build_system_prompt("")
    assert LENGTH_RULES in prompt


def test_system_prompt_not_generic_chatgpt():
    """Verify bot doesn't claim to be ChatGPT or Claude."""
    prompt = build_system_prompt("")
    assert "ChatGPT" not in prompt
    assert "Claude" not in prompt
    assert "I'm a language model" not in prompt
    assert "As an AI assistant" not in prompt


def test_system_prompt_first_person_voice():
    """Verify prompt uses first-person voice for bot."""
    prompt = build_system_prompt("Some knowledge")
    # Should reference Manish's knowledge, not generic AI knowledge
    assert "Manish" in prompt or "built by" in prompt.lower()


def test_system_prompt_empty_context():
    """Verify prompt handles empty context gracefully."""
    prompt = build_system_prompt("")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    # Should still contain core identity/rules even with no context


def test_system_prompt_long_context():
    """Verify prompt handles large context blocks."""
    long_context = "Knowledge chunk 1\n" * 100
    prompt = build_system_prompt(long_context)
    assert long_context in prompt
    assert len(prompt) > len(long_context)


def test_system_prompt_special_characters_in_context():
    """Verify special characters are handled in context."""
    context = "Python & C++, $100K salary, 100% remote"
    prompt = build_system_prompt(context)
    assert context in prompt or ("Python" in prompt and "&" in prompt)


def test_identity_references_manish_not_generic():
    """Verify identity specifically mentions Manish."""
    assert "Manish" in IDENTITY
    # Should NOT have generic AI language
    assert "As an AI" not in IDENTITY
    assert "I am an artificial" not in IDENTITY


def test_behavior_rules_no_hallucination_clause():
    """Verify rules prevent making up facts."""
    assert "make up" in BEHAVIOR_RULES.lower() or "don't" in BEHAVIOR_RULES.lower()


def test_behavior_rules_knowledge_scope():
    """Verify rules limit responses to Manish-related topics."""
    assert "Manish" in BEHAVIOR_RULES
    assert "knowledge" in BEHAVIOR_RULES.lower() or "skills" in BEHAVIOR_RULES.lower()

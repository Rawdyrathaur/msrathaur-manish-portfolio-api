from query_understanding import (
    build_retrieval_queries,
    contextualize_query,
    expand_portfolio_query,
)


def test_standalone_query_does_not_add_conversation_history():
    history = [{"role": "user", "content": "Tell me about Tab Story"}]
    assert contextualize_query("What projects use Docker?", history) == "What projects use Docker?"


def test_follow_up_resolves_latest_user_topic():
    history = [
        {"role": "user", "content": "What open-source work has Manish done?"},
        {"role": "assistant", "content": "He contributed to two projects."},
    ]
    resolved = contextualize_query("Tell me more about that and why it matters.", history)
    assert "What open-source work has Manish done?" in resolved
    assert "Tell me more about that" in resolved


def test_strength_question_adds_evidence_vocabulary():
    expanded = expand_portfolio_query("What are his strongest areas?")
    assert "skills" in expanded
    assert "experience" in expanded
    assert "projects" in expanded


def test_backend_fit_query_adds_role_and_backend_facets():
    expanded = expand_portfolio_query("Is he a good fit for a backend role?")
    assert "microservices" in expanded
    assert "role-relevant" in expanded
    assert "concrete evidence" in expanded


def test_retrieval_queries_are_unique_and_include_expansion():
    queries = build_retrieval_queries("Which project best demonstrates production engineering?")
    assert queries[0] == "Which project best demonstrates production engineering?"
    assert len(queries) == 2
    assert "Docker" in queries[-1]

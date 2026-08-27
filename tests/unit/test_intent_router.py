from intent_router import classify_intent


def test_greeting_is_local():
    assert classify_intent("Hello!") == "GREETING"


def test_portfolio_question_is_retrieved():
    assert classify_intent("What did Manish build?") == "PORTFOLIO_QUERY"


def test_follow_up_uses_history():
    history = [type("Message", (), {"content": "Tab Story uses local AI"})()]
    assert classify_intent("How does that work?", history) == "FOLLOW_UP"


def test_clear_off_topic_query_short_circuits():
    assert classify_intent("What is the weather in Paris?") == "OFF_TOPIC"

import json
from pathlib import Path

from intent_router import classify_intent


def test_golden_set_has_minimum_coverage():
    cases = json.loads((Path(__file__).parents[2] / "evals" / "golden.json").read_text())
    assert len(cases) >= 30
    for case in cases:
        assert classify_intent(case["question"]) == case["intent"]

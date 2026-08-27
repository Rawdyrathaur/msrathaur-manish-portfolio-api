"""Run the golden portfolio answer set against a deployed or local API."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--minimum-hit-rate", type=float, default=0.80)
    args = parser.parse_args()

    cases = json.loads((Path(__file__).parent / "golden.json").read_text())
    passed = 0
    failures = []
    for case in cases:
        body = json.dumps({"message": case["question"], "history": []}).encode()
        request = urllib.request.Request(
            f"{args.base_url.rstrip('/')}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                answer = json.loads(response.read())["answer"]
        except Exception as exc:
            failures.append((case["question"], f"request failed: {exc}"))
            continue
        if any(term.lower() in answer.lower() for term in case["expected_any"]):
            passed += 1
        else:
            failures.append((case["question"], answer[:160]))

    hit_rate = passed / len(cases)
    print(f"golden_hit_rate={hit_rate:.1%} passed={passed} total={len(cases)}")
    for question, reason in failures:
        print(f"FAIL: {question} -> {reason}")
    return 0 if hit_rate >= args.minimum_hit_rate else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Compare two bench.py result files; fail on >10% latency regression.

Usage:
    python tools/bench_compare.py benchmarks/results/old.json benchmarks/results/new.json

Exit code 1 if any stage's mean latency regressed more than the threshold.
Run this before merging any branch that touches a hot path.
"""

import argparse
import json
import sys
from pathlib import Path

REGRESSION_THRESHOLD = 0.10  # 10%

# stage -> metric paths holding _stats() dicts (lower is better)
LATENCY_METRICS = {
    "vad": ["per_chunk"],
    "stt": ["transcribe_2s"],
    "llm": ["ttft", "total"],
    "tts": ["ttfb", "total"],
}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(old: dict, new: dict) -> int:
    old_stages = old.get("stages", {})
    new_stages = new.get("stages", {})

    print(f"{'stage.metric':<24} {'old mean':>12} {'new mean':>12} {'delta':>10}")
    print("-" * 62)

    regressions = []
    for stage, metrics in LATENCY_METRICS.items():
        for metric in metrics:
            old_m = (old_stages.get(stage) or {}).get(metric)
            new_m = (new_stages.get(stage) or {}).get(metric)
            label = f"{stage}.{metric}"

            if not old_m or not new_m:
                print(f"{label:<24} {'—':>12} {'—':>12} {'n/a':>10}")
                continue

            old_mean, new_mean = old_m["mean_ms"], new_m["mean_ms"]
            delta = (new_mean - old_mean) / old_mean if old_mean else 0.0
            marker = "  <-- REGRESSION" if delta > REGRESSION_THRESHOLD else ""
            print(
                f"{label:<24} {old_mean:>10.1f}ms {new_mean:>10.1f}ms {delta:>+9.1%}{marker}"
            )
            if delta > REGRESSION_THRESHOLD:
                regressions.append((label, delta))

    print("-" * 62)
    old_sha = old.get("meta", {}).get("git_sha", "?")
    new_sha = new.get("meta", {}).get("git_sha", "?")
    print(f"old: {old_sha}  new: {new_sha}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} stage(s) regressed >"
              f"{REGRESSION_THRESHOLD:.0%}: {', '.join(label for label, _ in regressions)}")
        return 1
    print("\nOK: no stage regressed beyond threshold")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two bench.py results")
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args()
    return compare(_load(args.old), _load(args.new))


if __name__ == "__main__":
    sys.exit(main())

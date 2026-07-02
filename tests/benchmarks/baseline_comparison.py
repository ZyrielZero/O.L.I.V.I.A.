"""Baseline comparison for regression detection."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator


@dataclass
class RegressionResult:
    """Result of comparing a component's old vs new metrics."""

    component: str
    old_ms: float
    new_ms: float
    delta_pct: float
    is_regression: bool


# O(1) lookup set for valid components
COMPONENTS = frozenset({"stt", "llm", "tts", "memory"})


def load_baseline(path: Path) -> Dict:
    """
    Load baseline JSON file.

    Complexity: O(n) where n = file size
    """
    with open(path) as f:
        return json.load(f)


def iter_component_comparisons(
    old: Dict, new: Dict, threshold: float
) -> Generator[tuple[str, RegressionResult], None, None]:
    """
    Generator yielding component comparisons.

    Complexity: O(n) where n = number of components
    Memory: O(1) - yields one result at a time

    Args:
        old: Previous baseline data
        new: New baseline data
        threshold: Regression threshold (e.g., 0.10 for 10%)

    Yields:
        Tuple of (component_name, RegressionResult)
    """
    for component in COMPONENTS:
        old_lat = old.get(component, {}).get("latency_avg_ms", 0)
        new_lat = new.get(component, {}).get("latency_avg_ms", 0)

        if old_lat > 0:
            delta_pct = ((new_lat - old_lat) / old_lat) * 100
            is_regression = new_lat > old_lat * (1 + threshold)
        else:
            delta_pct = 0.0
            is_regression = False

        yield component, RegressionResult(
            component=component,
            old_ms=old_lat,
            new_ms=new_lat,
            delta_pct=delta_pct,
            is_regression=is_regression,
        )


def compare_baselines(
    old_path: Path, new_path: Path, threshold: float = 0.10
) -> Dict[str, RegressionResult]:
    """
    Compare two baseline JSON files and detect regressions.

    Complexity: O(n) where n = number of components

    Args:
        old_path: Path to previous baseline JSON
        new_path: Path to new baseline JSON
        threshold: Regression threshold (default 10%)

    Returns:
        Dict of component -> RegressionResult
    """
    old = load_baseline(old_path)
    new = load_baseline(new_path)

    # Build dict from generator - O(n)
    return dict(iter_component_comparisons(old, new, threshold))


def print_comparison(results: Dict[str, RegressionResult]) -> None:
    """Print comparison results with regression warnings."""
    print("\n=== Baseline Comparison ===\n")

    for name, r in results.items():
        status = "REGRESSION" if r.is_regression else "OK"
        symbol = "[!]" if r.is_regression else "[+]"
        print(
            f"{symbol} {name}: {r.old_ms:.1f}ms -> {r.new_ms:.1f}ms "
            f"({r.delta_pct:+.1f}%) [{status}]"
        )


def has_regressions(results: Dict[str, RegressionResult]) -> bool:
    """
    Check if any component has a regression.

    Complexity: O(n) worst case, but short-circuits on first regression
    """
    return any(r.is_regression for r in results.values())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python baseline_comparison.py <old_baseline.json> <new_baseline.json>")
        sys.exit(2)

    results = compare_baselines(Path(sys.argv[1]), Path(sys.argv[2]))
    print_comparison(results)

    if has_regressions(results):
        sys.exit(1)

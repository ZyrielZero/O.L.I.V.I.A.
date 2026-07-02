"""
A/B Test Runner for O.L.I.V.I.A. Optimization Testing.

This module provides a framework for running A/B tests to compare
baseline vs optimized configurations with statistical rigor.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from .baseline_metrics import (
    LatencyTimer,
    get_gpu_memory_mb,
    get_gpu_memory_peak_mb,
    reset_gpu_peak_stats,
)
from .vram_tracker import VRAMTracker


@dataclass
class ABTestResult:
    """Results from an A/B test comparison."""

    optimization_name: str
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_pct: Dict[str, float]
    quality_score_baseline: float
    quality_score_optimized: float
    quality_regression: bool
    passed: bool
    recommendation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "optimization_name": self.optimization_name,
            "timestamp": self.timestamp,
            "baseline_metrics": self.baseline_metrics,
            "optimized_metrics": self.optimized_metrics,
            "improvement_pct": self.improvement_pct,
            "quality_score_baseline": self.quality_score_baseline,
            "quality_score_optimized": self.quality_score_optimized,
            "quality_regression": self.quality_regression,
            "passed": self.passed,
            "recommendation": self.recommendation,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Print a summary of the test results."""
        status = "PASS" if self.passed else "FAIL"
        quality = "REGRESSION" if self.quality_regression else "OK"

        print(f"\n{'='*60}")
        print(f"A/B Test: {self.optimization_name}")
        print(f"{'='*60}")
        print(f"Status: {status}")
        print(f"Quality: {quality}")
        print("\nMetric Improvements:")

        for metric, improvement in self.improvement_pct.items():
            direction = "+" if improvement > 0 else ""
            baseline_val = self.baseline_metrics.get(metric, 0)
            optimized_val = self.optimized_metrics.get(metric, 0)
            print(f"  {metric}: {baseline_val:.2f} -> {optimized_val:.2f} ({direction}{improvement:.1f}%)")

        print("\nQuality Scores:")
        print(f"  Baseline:  {self.quality_score_baseline:.2%}")
        print(f"  Optimized: {self.quality_score_optimized:.2%}")

        print(f"\nRecommendation: {self.recommendation}")
        print(f"{'='*60}\n")


class ABTestRunner:
    """
    Run A/B tests comparing baseline vs optimized configurations.

    Usage:
        runner = ABTestRunner(
            name="kv_cache_quantization",
            component="llm",
        )

        # Define test function
        async def test_llm():
            response = await llm.chat("Hello")
            return response

        # Run test
        result = await runner.run(
            test_fn=test_llm,
            quality_fn=lambda r: 1.0 if r else 0.0,
            iterations=10,
        )
    """

    def __init__(
        self,
        name: str,
        component: str = "general",
        results_dir: str = "tests/results",
    ):
        """
        Initialize A/B test runner.

        Args:
            name: Name of the optimization being tested
            component: Component being tested (stt, llm, tts, memory)
            results_dir: Directory to store results
        """
        self.name = name
        self.component = component
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.vram_tracker = VRAMTracker()

    async def run_iterations(
        self,
        test_fn: Callable[[], Awaitable[Any]],
        quality_fn: Callable[[Any], float],
        iterations: int,
        warmup: int = 1,
    ) -> Tuple[Dict[str, float], float, List[Any]]:
        """
        Run test function multiple times and collect metrics.

        Args:
            test_fn: Async function to test
            quality_fn: Function to score quality of result (0-1)
            iterations: Number of test iterations
            warmup: Number of warmup iterations (not counted)

        Returns:
            Tuple of (metrics_dict, avg_quality_score, results_list)
        """
        latencies: List[float] = []
        quality_scores: List[float] = []
        vram_samples: List[float] = []
        results: List[Any] = []

        # Warmup iterations
        for _ in range(warmup):
            try:
                await test_fn()
            except Exception:
                pass

        # Reset peak stats after warmup
        reset_gpu_peak_stats()
        initial_vram = get_gpu_memory_mb()

        # Actual iterations
        for i in range(iterations):
            try:
                timer = LatencyTimer()
                with timer:
                    result = await test_fn()

                latencies.append(timer.elapsed)
                quality_scores.append(quality_fn(result))
                vram_samples.append(get_gpu_memory_mb())
                results.append(result)

            except Exception as e:
                print(f"[ABTest] Iteration {i+1} failed: {e}")
                quality_scores.append(0.0)

        # Calculate metrics
        metrics = {}

        if latencies:
            metrics["latency_avg_ms"] = statistics.mean(latencies) * 1000
            metrics["latency_p95_ms"] = sorted(latencies)[int(len(latencies) * 0.95)] * 1000
            metrics["latency_min_ms"] = min(latencies) * 1000
            metrics["latency_max_ms"] = max(latencies) * 1000

        if vram_samples:
            metrics["vram_avg_mb"] = statistics.mean(vram_samples)
            metrics["vram_initial_mb"] = initial_vram
            metrics["vram_peak_mb"] = get_gpu_memory_peak_mb()

        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0

        return metrics, avg_quality, results

    async def run(
        self,
        test_fn: Callable[[], Awaitable[Any]],
        quality_fn: Callable[[Any], float],
        apply_optimization: Callable[[], None],
        revert_optimization: Callable[[], None],
        iterations: int = 10,
        quality_threshold: float = 0.98,
    ) -> ABTestResult:
        """
        Run A/B test comparing baseline vs optimized configuration.

        Args:
            test_fn: Async function to test
            quality_fn: Function to score quality (0-1)
            apply_optimization: Function to apply the optimization
            revert_optimization: Function to revert to baseline
            iterations: Number of iterations per configuration
            quality_threshold: Minimum quality ratio vs baseline to pass

        Returns:
            ABTestResult with comparison data
        """
        print(f"\n[ABTest] Starting: {self.name}")
        print(f"[ABTest] Iterations: {iterations} per configuration")

        # Run baseline
        print("[ABTest] Running baseline...")
        revert_optimization()  # Ensure we start at baseline
        baseline_metrics, baseline_quality, _ = await self.run_iterations(
            test_fn, quality_fn, iterations
        )
        print(f"[ABTest] Baseline complete: latency={baseline_metrics.get('latency_avg_ms', 0):.1f}ms")

        # Apply optimization and run
        print("[ABTest] Applying optimization...")
        apply_optimization()
        print("[ABTest] Running optimized...")
        optimized_metrics, optimized_quality, _ = await self.run_iterations(
            test_fn, quality_fn, iterations
        )
        print(f"[ABTest] Optimized complete: latency={optimized_metrics.get('latency_avg_ms', 0):.1f}ms")

        # Revert optimization
        revert_optimization()

        # Calculate improvements
        improvement_pct = {}
        for key in baseline_metrics:
            baseline_val = baseline_metrics[key]
            optimized_val = optimized_metrics.get(key, baseline_val)

            if baseline_val != 0:
                # For latency/VRAM, lower is better (negative improvement)
                # So we calculate (baseline - optimized) / baseline
                if "latency" in key or "vram" in key:
                    improvement_pct[key] = ((baseline_val - optimized_val) / baseline_val) * 100
                else:
                    improvement_pct[key] = ((optimized_val - baseline_val) / baseline_val) * 100

        # Determine if quality regressed
        quality_ratio = optimized_quality / baseline_quality if baseline_quality > 0 else 1.0
        quality_regression = quality_ratio < quality_threshold

        # Determine pass/fail
        # Pass if: no quality regression AND at least one meaningful improvement
        has_improvement = any(v > 5 for k, v in improvement_pct.items() if "latency" in k or "vram" in k)
        passed = not quality_regression and (has_improvement or all(v >= 0 for v in improvement_pct.values()))

        # Generate recommendation
        if quality_regression:
            recommendation = f"REJECT: Quality degraded to {quality_ratio:.1%} of baseline"
        elif has_improvement:
            best_improvement = max(improvement_pct.items(), key=lambda x: x[1])
            recommendation = f"ACCEPT: Best improvement is {best_improvement[0]} at {best_improvement[1]:.1f}%"
        else:
            recommendation = "NEUTRAL: No significant improvement or regression"

        result = ABTestResult(
            optimization_name=self.name,
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            improvement_pct=improvement_pct,
            quality_score_baseline=baseline_quality,
            quality_score_optimized=optimized_quality,
            quality_regression=quality_regression,
            passed=passed,
            recommendation=recommendation,
        )

        # Save results
        self._save_result(result)
        result.print_summary()

        return result

    def _save_result(self, result: ABTestResult) -> None:
        """Save test result to JSON file."""
        filename = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            f.write(result.to_json())

        print(f"[ABTest] Results saved to: {filepath}")


class QuickABTest:
    """
    Simplified A/B test for quick comparisons.

    Usage:
        test = QuickABTest("my_optimization")
        test.measure_baseline(lambda: do_something())
        test.measure_optimized(lambda: do_something_optimized())
        test.compare()
    """

    def __init__(self, name: str):
        self.name = name
        self.baseline_samples: List[float] = []
        self.optimized_samples: List[float] = []

    def measure_baseline(self, fn: Callable, iterations: int = 10) -> float:
        """Measure baseline performance."""
        self.baseline_samples = []

        for _ in range(iterations):
            timer = LatencyTimer()
            with timer:
                fn()
            self.baseline_samples.append(timer.elapsed)

        return statistics.mean(self.baseline_samples)

    def measure_optimized(self, fn: Callable, iterations: int = 10) -> float:
        """Measure optimized performance."""
        self.optimized_samples = []

        for _ in range(iterations):
            timer = LatencyTimer()
            with timer:
                fn()
            self.optimized_samples.append(timer.elapsed)

        return statistics.mean(self.optimized_samples)

    def compare(self) -> Dict[str, float]:
        """Compare baseline vs optimized."""
        if not self.baseline_samples or not self.optimized_samples:
            return {"error": "Missing measurements"}

        baseline_avg = statistics.mean(self.baseline_samples)
        optimized_avg = statistics.mean(self.optimized_samples)
        improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100

        result = {
            "baseline_avg_ms": baseline_avg * 1000,
            "optimized_avg_ms": optimized_avg * 1000,
            "improvement_pct": improvement,
            "baseline_p95_ms": sorted(self.baseline_samples)[int(len(self.baseline_samples) * 0.95)] * 1000,
            "optimized_p95_ms": sorted(self.optimized_samples)[int(len(self.optimized_samples) * 0.95)] * 1000,
        }

        print(f"\n[QuickABTest] {self.name}")
        print(f"  Baseline:  {result['baseline_avg_ms']:.1f}ms (p95: {result['baseline_p95_ms']:.1f}ms)")
        print(f"  Optimized: {result['optimized_avg_ms']:.1f}ms (p95: {result['optimized_p95_ms']:.1f}ms)")
        print(f"  Improvement: {improvement:.1f}%")

        return result

"""
Full experiment pipeline:
  1. Benchmark default controller params (baseline)
  2. Run AI agent to optimize params
  3. Benchmark optimized params
  4. Print improvement report and save results to results/
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from agent import optimize
from benchmark import benchmark


def run_experiment(agent_iterations: int = 5, benchmark_runs: int = 5) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)

    _header("PHASE 1 — Baseline benchmark")
    print(f"Evaluating default params {benchmark_runs} times...\n")
    baseline = benchmark(n_runs=benchmark_runs)
    print(f"\nBaseline:  {baseline['mean']:.4f} ± {baseline['stdev']:.4f}\n")

    _header("PHASE 2 — AI agent optimization")
    history = optimize(max_iterations=agent_iterations)

    _header("PHASE 3 — Post-optimization benchmark")
    print(f"Evaluating optimized params {benchmark_runs} times...\n")
    optimized = benchmark(n_runs=benchmark_runs)
    print(f"\nOptimized: {optimized['mean']:.4f} ± {optimized['stdev']:.4f}\n")

    pct = (optimized["mean"] - baseline["mean"]) / max(abs(baseline["mean"]), 1e-9) * 100
    _header("RESULTS")
    print(f"  Baseline:    {baseline['mean']:.4f} ± {baseline['stdev']:.4f}")
    print(f"  Optimized:   {optimized['mean']:.4f} ± {optimized['stdev']:.4f}")
    print(f"  Improvement: {pct:+.1f}%")

    results = {
        "timestamp": timestamp,
        "agent_iterations": agent_iterations,
        "benchmark_runs": benchmark_runs,
        "baseline": baseline,
        "optimization_history": history,
        "optimized": optimized,
        "improvement_pct": round(pct, 2),
    }
    out = Path("results") / f"experiment_{timestamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")
    return results


def _header(title: str) -> None:
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_experiment(agent_iterations=iters, benchmark_runs=runs)

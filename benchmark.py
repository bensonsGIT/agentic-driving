"""Evaluate current controller params over N runs and report mean ± std reward."""
import importlib
import statistics
import sys

import controller
from evaluator import evaluate


def benchmark(n_runs: int = 10, steps: int = 500) -> dict:
    importlib.reload(controller)
    rewards = []
    for i in range(n_runs):
        r = evaluate(steps=steps)
        rewards.append(r)
        print(f"  run {i + 1}/{n_runs}: {r:.4f}")
    mean = statistics.mean(rewards)
    stdev = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    return {"mean": round(mean, 4), "stdev": round(stdev, 4), "rewards": rewards}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    importlib.reload(controller)
    print(f"steer={controller.STEER}, accel={controller.ACCEL}  ({n} runs)\n")
    stats = benchmark(n_runs=n)
    print(f"\nmean reward: {stats['mean']:.4f} ± {stats['stdev']:.4f}")

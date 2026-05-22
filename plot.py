"""Plot optimization curve from a saved experiment JSON file."""
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib first:  pip install matplotlib")
    sys.exit(1)


def plot(result_path: str) -> None:
    data = json.loads(Path(result_path).read_text())

    history = data["optimization_history"]
    rewards = [h["reward"] for h in history]
    iterations = list(range(1, len(rewards) + 1))

    baseline_mean = data["baseline"]["mean"]
    optimized_mean = data["optimized"]["mean"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(iterations, rewards, marker="o", linewidth=2, label="Agent trial reward")
    ax.axhline(baseline_mean, color="red", linestyle="--", label=f"Baseline mean ({baseline_mean:.3f})")
    ax.axhline(optimized_mean, color="green", linestyle="--", label=f"Optimized mean ({optimized_mean:.3f})")

    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Total reward")
    ax.set_title(f"AI Agent Optimization — {data['improvement_pct']:+.1f}% improvement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(result_path).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use most recent result if no path given
        results = sorted(Path("results").glob("experiment_*.json"))
        if not results:
            print("No results found. Run experiment.py first.")
            sys.exit(1)
        path = str(results[-1])
        print(f"Using most recent result: {path}")
    else:
        path = sys.argv[1]
    plot(path)

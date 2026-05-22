"""Agentic optimization of autonomous driving controller parameters using Claude."""
import json
import re
import sys
from pathlib import Path

import anthropic
from evaluator import evaluate

CONTROLLER_PATH = Path("controller.py")
client = anthropic.Anthropic()

tools = [
    {
        "name": "read_params",
        "description": "Read the current STEER_GAIN and SPEED_TARGET from controller.py",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_params",
        "description": "Write new STEER_GAIN and SPEED_TARGET values to controller.py",
        "input_schema": {
            "type": "object",
            "properties": {
                "steer_gain": {
                    "type": "number",
                    "description": "Steering responsiveness (range 0.1–2.0)",
                },
                "speed_target": {
                    "type": "number",
                    "description": "Desired speed in km/h (range 20–40)",
                },
            },
            "required": ["steer_gain", "speed_target"],
        },
    },
    {
        "name": "run_simulation",
        "description": "Run the highway driving simulation with current controller.py params and return total reward",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of simulation steps (default 500)",
                }
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """You are a parameter optimization agent for an autonomous highway driving controller.

The controller has two tunable parameters in controller.py:
- STEER_GAIN: steering responsiveness (range 0.1–2.0; higher = more aggressive steering)
- SPEED_TARGET: desired speed in km/h (range 20–40; higher = faster driving)

Your objective is to maximize the total reward from the driving simulation.
Higher reward means safer, more efficient lane-following behaviour.

Optimization strategy:
1. Read current params and run a baseline evaluation.
2. Systematically explore the parameter space (try boundary values and promising midpoints).
3. Use results to narrow in on the optimum.
4. After all evaluations, set controller.py to the best params found.
5. Summarise what worked and why."""


def _read_params() -> dict:
    content = CONTROLLER_PATH.read_text()
    steer = float(re.search(r"STEER_GAIN\s*=\s*([\d.]+)", content).group(1))
    speed = float(re.search(r"SPEED_TARGET\s*=\s*([\d.]+)", content).group(1))
    return {"steer_gain": steer, "speed_target": speed}


def _update_params(steer_gain: float, speed_target: float) -> dict:
    content = CONTROLLER_PATH.read_text()
    content = re.sub(r"STEER_GAIN\s*=\s*[\d.]+", f"STEER_GAIN = {steer_gain}", content)
    content = re.sub(r"SPEED_TARGET\s*=\s*[\d.]+", f"SPEED_TARGET = {speed_target}", content)
    CONTROLLER_PATH.write_text(content)
    return {"steer_gain": steer_gain, "speed_target": speed_target}


def _run_simulation(steps: int = 500) -> dict:
    reward = evaluate(steps=steps)
    return {"total_reward": round(reward, 4)}


def _handle_tool(name: str, inputs: dict) -> dict:
    if name == "read_params":
        return _read_params()
    if name == "update_params":
        return _update_params(**inputs)
    if name == "run_simulation":
        return _run_simulation(**inputs)
    return {"error": f"Unknown tool: {name}"}


def optimize(max_iterations: int = 5) -> list[dict]:
    """Run the agentic optimisation loop. Returns history of (params, reward) pairs."""
    history: list[dict] = []
    messages = [
        {
            "role": "user",
            "content": (
                f"Optimise the driving controller parameters to maximise reward. "
                f"Run exactly {max_iterations} simulation evaluations, each with different "
                f"parameter values. After all runs, set controller.py to the best params "
                f"found and summarise your findings."
            ),
        }
    ]

    print(f"Starting agentic optimisation ({max_iterations} evaluations)...\n")

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"Agent: {block.text}\n")

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[Tool] {block.name}({json.dumps(block.input, separators=(',', ':'))})")
            result = _handle_tool(block.name, block.input)
            print(f"       → {json.dumps(result, separators=(',', ':'))}\n")

            if block.name == "run_simulation":
                history.append({
                    "params": _read_params(),
                    "reward": result["total_reward"],
                })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    print("\n--- Optimisation History ---")
    for i, entry in enumerate(history):
        p = entry["params"]
        print(f"  Run {i + 1:2d}: steer={p['steer_gain']:.2f}, speed={p['speed_target']:.1f} → reward={entry['reward']:.4f}")

    if history:
        best = max(history, key=lambda x: x["reward"])
        p = best["params"]
        print(f"\nBest found: steer={p['steer_gain']:.2f}, speed={p['speed_target']:.1f} → reward={best['reward']:.4f}")

    return history


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    optimize(max_iterations=iterations)

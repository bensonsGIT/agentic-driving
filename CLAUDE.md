# Agentic Driving — Claude Code Guide

## What this project does
An AI agent (Claude) autonomously tunes a highway driving controller by running simulations, observing rewards, and adjusting parameters — no human in the loop.

## Key files
| File | Purpose |
|------|---------|
| `controller.py` | The two tunable knobs: `STEER_GAIN` and `SPEED_TARGET` |
| `evaluator.py` | Runs `highway-v0` simulation, returns total reward as a float |
| `agent.py` | Claude-powered optimization loop using tool use |
| `run_eval.py` | Quick one-shot evaluation (prints reward to stdout) |
| `test.py` | Sanity-check that the gym environment loads correctly |

## Running the optimizer
```bash
export ANTHROPIC_API_KEY=sk-...
python agent.py          # 5 iterations (default)
python agent.py 10       # custom iteration count
```

## Running a quick eval (no API key needed)
```bash
python run_eval.py
```

## Environment
- Simulation: `highway-v0` (highway-env / gymnasium)
- Agent model: `claude-opus-4-7` with tool use
- Parameter space: `STEER` in [-1.0, 1.0], `ACCEL` in [-1.0, 1.0]

## Installing dependencies
```bash
pip install -r requirements.txt
```

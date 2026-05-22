# Agentic Optimization of Autonomous Driving Behavior in Simulation

A research project where an AI agent (Claude) autonomously tunes a highway driving controller by running simulations, measuring performance, and iteratively improving its parameters — no human needed in the loop.

---

## What is "Agentic Optimization"?

Traditional parameter tuning requires a human to try values, observe results, and manually adjust. In this project, **Claude acts as the optimizer**: it reads the current controller settings, runs the simulation, sees the reward, decides what to try next, and repeats — all on its own.

This is achieved using **Claude's tool use** capability. The agent is given three tools:
- `read_params` — inspect the current controller values
- `update_params` — write new values to `controller.py`
- `run_simulation` — execute the simulation and get a reward score

Claude uses these tools in a loop, reasoning about the results and converging on better parameters.

---

## Project Structure

```
agentic-driving/
├── controller.py    # The two tunable parameters (STEER_GAIN, SPEED_TARGET)
├── evaluator.py     # Runs the simulation, returns total reward
├── agent.py         # The Claude-powered optimization loop
├── run_eval.py      # Quick one-shot eval script
├── test.py          # Sanity check that the environment loads
└── requirements.txt # Python dependencies
```

---

## Beginner Setup Guide

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `gymnasium` — the standard RL simulation framework
- `highway-env` — the highway driving simulation
- `anthropic` — the Claude API client

### Step 2 — Get an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account and generate an API key
3. Set it as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-...      # Windows
```

### Step 3 — Verify the simulation works (no API key needed)

```bash
python test.py
```

You should see `working` printed. If you get an import error, re-run `pip install -r requirements.txt`.

### Step 4 — Run a single evaluation

```bash
python run_eval.py
```

This runs 500 steps of the highway simulation with the default parameters and prints the total reward. Higher is better.

### Step 5 — Run the agentic optimizer

```bash
python agent.py
```

Claude will now:
1. Read the current `STEER_GAIN` and `SPEED_TARGET`
2. Run a baseline simulation
3. Try different parameter values across 5 evaluations
4. Write the best parameters back to `controller.py`
5. Print a summary of what it found

To run more iterations (better results, more API calls):

```bash
python agent.py 10
```

---

## How It Works — Step by Step

```
agent.py starts
    │
    ├─► Claude reads current params  (read_params tool)
    ├─► Claude runs simulation        (run_simulation tool)
    ├─► Claude updates params         (update_params tool)
    ├─► Claude runs simulation again  (run_simulation tool)
    │   ... repeats N times ...
    └─► Claude sets best params found and prints summary
```

Each call to `run_simulation` triggers `evaluator.py`, which reloads `controller.py` to pick up the latest values, runs `highway-v0` for 500 steps, and returns the accumulated reward.

---

## Tunable Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `STEER_GAIN` | 0.5 | 0.1 – 2.0 | How aggressively the car steers |
| `SPEED_TARGET` | 28 | 20 – 40 | Target speed (km/h) |

---

## Extending the Research

Some directions to explore:
- **More parameters**: add lane-change threshold, safety margin, etc.
- **Different environments**: swap `highway-v0` for `merge-v0` or `roundabout-v0`
- **Logging**: save each run's params + reward to a CSV for analysis
- **Bayesian priors**: seed Claude with domain knowledge about good starting ranges
- **Multi-objective**: balance reward with energy efficiency or comfort metrics

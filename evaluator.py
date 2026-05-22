"""Evaluation wrapper — runs the simulation and returns total reward."""
import importlib
import sys

import gymnasium as gym
import highway_env  # noqa: F401 — registers highway environments


def evaluate(steps: int = 500) -> float:
    """Run the highway-v0 simulation and return total reward."""
    if "controller" in sys.modules:
        import controller
        importlib.reload(controller)
    else:
        import controller  # noqa: F811

    env = gym.make("highway-v0", config={"action": {"type": "ContinuousAction"}})
    obs, info = env.reset()
    total_reward = 0.0

    for _ in range(steps):
        action = [controller.STEER, controller.ACCEL]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    return total_reward

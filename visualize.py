"""Run the highway simulation with a live pygame window."""
import sys
import gymnasium as gym
import highway_env  # noqa: F401
import controller

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300

env = gym.make(
    "highway-v0",
    render_mode="human",
    config={"action": {"type": "ContinuousAction"}},
)

obs, info = env.reset()
total_reward = 0.0

for _ in range(steps):
    action = [controller.STEER, controller.ACCEL]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward
    if terminated or truncated:
        break

env.close()
print(f"reward: {total_reward:.4f}")

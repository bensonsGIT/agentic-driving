import gymnasium as gym
import highway_env
import controller

env = gym.make("highway-v0")

obs, info = env.reset()

total_reward = 0

for step in range(500):

    action = [controller.STEER_GAIN, controller.SPEED_TARGET]

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:
        break

print("reward:", total_reward)
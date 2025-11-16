import gymnasium as gym

# env = gym.make("CliffWalking-v1", render_mode = "human")
env = gym.make("CliffWalking-v1")

observation, info = env.reset()

print(f"Start observation: {observation}")

episode_over = False

total_reward: float = 0

while not episode_over:
    env.render()
    
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)

    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")

env.close()

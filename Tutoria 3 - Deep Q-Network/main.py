import gym
env = gym.make('LunarLander-v2')

print(env.action_space)  # Discrete(4)
print(env.observation_space)  # Box(8,)
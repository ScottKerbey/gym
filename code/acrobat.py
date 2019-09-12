import gym
env = gym.make('Acrobot-v1')
observation = env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

env.close()
print("Done")

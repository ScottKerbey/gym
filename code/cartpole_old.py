import gym
import numpy as np
import math
env = gym.make('CartPole-v0')

buckets = (1, 1, 6, 12,)

class CartPoleQ:
    def __init__(selfself, buckets, Q):
        pass

    def discretise(self, obs):
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        # new_obs[2] -= 3
        # new_obs[3] -= 6
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

observation = env.reset()
done = False

def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    new_obs[2] -= 3
    new_obs[3] -= 6
    return tuple(new_obs)


Q = np.zeros(buckets + (env.action_space.n,))
while not done:
# for _ in range(100):
    ob = discretize(observation)
    env.render()
    action = 1
    if ob[2] < 0 and ob[3] < 0:
        action = 0
    print(action)
    # print(ob)
    observation, reward, done, info = env.step(action)

env.close()
print("Done")




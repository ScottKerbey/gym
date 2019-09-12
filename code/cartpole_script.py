import gym
import numpy as np
import math
from collections import deque
env = gym.make('CartPole-v0')

# Constants
buckets = (6,4,12,12,)
velocity_max = 0.5
ang_velocity_max = 2
max_alpha = 0.1  # Learning rate
gamma = 0.9  # discount
max_epsilon = 0.3  # exploration rate
winning_reward = 195

def discretize(obs):
    '''
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    obs:
        Type: Box(4)
        Num obs                     Min     Max
        0   ignore                  0       0
        1   ignore                  0       0
        2   Pole Angle              -6      6
        3   Pole Velocity At Tip    -12     12
    '''
    upper_bounds = [env.observation_space.high[0], velocity_max, env.observation_space.high[2], ang_velocity_max]
    lower_bounds = [env.observation_space.low[0], -velocity_max, env.observation_space.low[2], -ang_velocity_max]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() < epsilon) else np.argmax(Q[state])

def update_q(state_current, state_next, action, reward, alpha):
    Q[state_current][action] += alpha * (reward + gamma * (np.max(Q[state_next]) - Q[state_current][action]))

def get_epsilon(t,total):
    return max(0.01, max_epsilon * (total - t) / total)

def get_alpha():
    return max_alpha


n_episodes = 10000
Q = np.zeros(buckets + (env.action_space.n,))

#theta_dot_history
tdh = []
scores = deque(maxlen=100)
done_done = False

for e in range(n_episodes):

    if not done_done:
        #init
        observation = env.reset()
        done = False
        alpha = get_alpha()
        epsilon = get_epsilon(e, n_episodes)
        i = 0

        while not done:
            ob = discretize(observation)
            # env.render()
            action = choose_action(ob, epsilon)
            # print(ob)
            observation, reward, done, info = env.step(action)
            update_q(ob, discretize(observation), action, reward, alpha)
            tdh.append(observation[3])
            i += 1

        scores.append(i)
        mean_score = np.mean(scores)

        if mean_score >= winning_reward and e >= 100:
            print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
            done_done = True
        if e % 100 == 0:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

done = False
observation = env.reset()
while not done:
    ob = discretize(observation)
    env.render()
    action = choose_action(ob, epsilon)
    observation, reward, done, info = env.step(action)

print(f"theta_dot properties - Max: {max(tdh)}, Min: {min(tdh)}, Average: {sum(tdh)/len(tdh)}")

env.close()
print("Done")




import gym
import numpy as np
import math
from collections import deque

# buckets=(6,4,12,12,)

class QLearnerCartPole():
    def __init__(self, buckets=(3,3,6,6,), velocity_max = 0.5, ang_velocity_max = 2, min_alpha = 0.1, gamma = 0.99, min_epsilon = 0.001, winning_reward = 195, n_episodes= 1000, max_discount=0, show=False):
        self.buckets = buckets
        self.velocity_max = velocity_max
        self.ang_velocity_max = ang_velocity_max
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.winning_reward = winning_reward
        self.n_episodes = n_episodes
        self.max_discount = max_discount
        self.show = show

        self.env = gym.make('CartPole-v0')
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
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
        upper_bounds = [self.env.observation_space.high[0], self.velocity_max, self.env.observation_space.high[2], self.ang_velocity_max]
        lower_bounds = [self.env.observation_space.low[0], -self.velocity_max, self.env.observation_space.low[2], -self.ang_velocity_max]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() < epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_current, state_next, action, reward, alpha, discount):
        update = alpha * (reward + self.gamma * (np.max(self.Q[state_next]) - self.Q[state_current][action]))
        self.Q[state_current][action] += update
        if discount > 0.01:
            for index, val in enumerate(state_current):
                # discount = 0.01
                temp_state_1 = list(state_current)
                temp_state_2 = temp_state_1
                if(val - 1 >= 0):
                    temp_state_1[index] = val - 1
                    temp_state_1 = tuple(temp_state_1)
                    self.Q[temp_state_1][action] += discount * update
                if(val + 1 < self.buckets[index]):
                    temp_state_2[index] = val + 1
                    temp_state_2 = tuple(temp_state_2)
                    self.Q[temp_state_2][action] += discount * update

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / 25)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def get_discount(self, t):
        # return max(0, self.max_discount * ((total - t) / total)**2)
        # return max(0, self.max_discount * (1 - math.tanh(10*t/total-2)) / 2)
        return max(0, min(self.max_discount, 1.0 - math.log10((t + 1) / 25)))

    def run(self):
        def close():
            # print(f"theta_dot properties - Max: {max(tdh)}, Min: {min(tdh)}, Average: {sum(tdh)/len(tdh)}")
            done = False
            if self.show:
                observation = self.env.reset()
                while not done:
                    ob = self.discretize(observation)
                    self.env.render()
                    action = self.choose_action(ob, 0)
                    observation, _, done, _ = self.env.step(action)
            self.env.close()

        #theta_dot_history
        tdh = []
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):

            #init
            observation = self.env.reset()
            done = False
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            discount = self.get_discount(e)
            i = 0

            while not done:
                ob = self.discretize(observation)
                # env.render()
                action = self.choose_action(ob, epsilon)
                # print(ob)
                observation, reward, done, _ = self.env.step(action)
                self.update_q(ob, self.discretize(observation), action, reward, alpha, discount)
                tdh.append(observation[3])
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)

            if mean_score >= self.winning_reward and e >= 100:
                print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                close()
                return e
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
        close()
        return e

        
#buckets=(4,4,6,12,)

if __name__ == "__main__":
    agent = QLearnerCartPole(buckets=(1,1,6,5,),n_episodes=1000, max_discount=0, show=True)
    agent.run()
"""
    Q implementation of the Frozen Lake environment
"""

import gym
import numpy as np


class FrozenLake:
    """
        Q-Learning with Frozen Lake

        Parameters:
        -----------
        env_name: str
            gym environment name
        lr: float
            Learning rate for agent
        min_eps: float
            Minimum exploration probability
        max_eps: float
            Max exploration probability
        gamma: float
            Rate for discounting future rewards
        decay_rate: float
            Decay rate for exploration probability
    """

    def __init__(self, env_name='FrozenLake-v0', lr=.8, min_eps=.001,
                 max_eps=1., gamma=.95, decay_rate=0.005):
        self.lr = lr
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.init_env(env_name)

    def init_env(self, env):
        """
            Initializes the gym environment
        """
        self.env = gym.make(env)

        self.action_space = self.env.action_space.n
        self.state_size = self.env.observation_space.n

        self.init_qtable()

    def init_qtable(self):
        """
            Creates the Q table to hold the states, and actions
        """
        self.qtable = np.zeros((self.state_size, self.action_space))

    def train(self, episodes=15000, steps=99, epsilon=1.0):
        """
            Selects actions for the agent to take in given
            states and updates the Q table

            Parameters
            ----------
            episodes: int
                Total number of episodes
            steps: int
                Number of steps
            epsilon: float, default=1.0
                Exploration probability
        """

        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(steps):
                explore_exploit_tradeoff = np.random.uniform()

                if explore_exploit_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, _ = self.env.step(action)

                # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]

                self.qtable[state, action] = self.qtable[state, action] + \
                    self.lr * (reward + self.gamma * np.amax(
                        self.qtable[new_state, :]
                    ) - self.qtable[state, action])

                total_reward += reward
                state = new_state

                if done:
                    break
            epsilon = self.min_eps + \
                (self.max_eps - self.min_eps) * \
                np.exp(-self.decay_rate * episode)
            rewards.append(total_reward)

        print('\nScore: ', np.sum(rewards) / episodes)

    def play(self, episodes=50, steps=99):
        """
            Plays FrozenLake with the trained agent
        """
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            disp = '\n\n===========================' + \
                f'{episode}'

            print(disp)

            for step in range(steps):
                action = np.argmax(self.qtable[state, :])

                new_state, reward, done, _ = self.env.step(action)

                total_reward += reward

                if done:
                    rewards.append(reward)
                    print(f'step: {step}\t', f'reward: {total_reward}')
                    self.env.render()
                    break
                state = new_state

        self.env.close()
        print('Reward', np.sum(rewards) / episodes)


if __name__ == '__main__':
    fl = FrozenLake()
    fl.train()
    fl.play()

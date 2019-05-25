"""
    Q-learning agent implemntation with Taxi gym env
"""

import gym
import numpy as np


class QLearning:
    """
        Implements RL agent  using taxi gym

        Parameters
        ----------
        lr: float
            Learning rate
        gamma: float
            Discount rate for future rewards
        epsilon: float
            Sets probability of taking actions that maximize future rewards
            vs exploiting unknown actions
        decay_rate: float
            Rate at which epsilon reduces with explored actions
            to encourage exploitation
        min_eps: float
            Lowest value for epsilon
        max_eps: float
            Highest value for epsilon
    """

    def __init__(self, env='Taxi-v2',
                 lr=.1,
                 gamma=0.8, epsilon=1.0,
                 decay_rate=0.001,
                 max_eps=1.0,
                 min_eps=0.001):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.gamma = gamma
        self.lr = lr

        self.init_env(env)

    def init_env(self, env_name):
        """
            Creates the environment the agent will interact with.
            The env variables create the Q table
        """
        self.env = gym.make(env_name)
        self.env.render()

        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.q_table = np.zeros((self.state_space, self.action_space))

    def train(self, episodes=2000, max_steps=99):
        """
            Takes actions and updates the q-table with
            future rewards
        """

        for episode in range(episodes):
            state = self.env.reset()

            for step in range(max_steps):
                explore_eploit_tradeoff = np.random.uniform()

                if explore_eploit_tradeoff > self.epsilon:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, info = self.env.step(action)

                self.q_table[state, action] = self.q_table[state, action] \
                    + self.lr * (reward + self.gamma * np.amax(
                        self.q_table[new_state, :]
                    ) - self.q_table[state, action]
                )

                state = new_state
                if done:
                    break
            exp_ = np.exp(-self.decay_rate * episode)
            self.epsilon = self.min_eps + exp_ * (self.max_eps - self.min_eps)

    def play(self, episodes=99, max_steps=99):
        """
            Runs the trained agent
        """
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            display = '\n----------------------------\n' + \
                f'Episode: {episode}'
            print(display)

            for step in range(max_steps):
                self.env.render()
                action = np.argmax(self.q_table[state, :])

                new_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                if done:
                    rewards.append(reward)
                    print(f'Total reward - {total_reward}')
                    break
                state = new_state
        self.env.close()
        print(f'\nReward :', np.sum(rewards) / episodes)

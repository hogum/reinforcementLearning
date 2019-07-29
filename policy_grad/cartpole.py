"""
    Cartpole policy gradient implementation
"""
import numpy as np
import tensorflow as tf
import gym


class CartPole:
    """
        Cartpole V0
    """

    def __init__(self, lr=.002, gamma=.95):
        self.gamma = gamma
        self.action_space, self.observation_space = self.setup_env()

    def preprocess_rewards(self, rewards):
        """
            Discounts and normalized the rewards
        """
        discounted_rewds = np.zeros_like(rewards)
        cumulative = 0

        for i in range(len(rewards)):
            cumulative += self.gamma + rewards[i]
            discounted_rewds[i] = cumulative

        discounted_rewds = (discounted_rewds -
                            np.mean(discounted_rewds)) \
                                    / np.std(discounted_rewds)
        return discounted_rewds

    def setup_env(self):
        """
            Initializes the game environment
        """
        env = gym.make('CartPole-v0')
        env.unwrap
        env.seed(1)
        self.env = env

        return self.env.action_space.n, self.env.observation_space
    def build_model(self):
        """
            Builds the Policy Gradient model
        """
        with tf.name_scope('inputs'):
            self.inputs = tf.compat.v1.placeholder(tf.float32,
                    (None, self.observation_space), name='inputs')
            self.actions = tf.compat.v1.placeholder(tf.float32,
                    (None, self.action_space), name='actions')
            self.dicounted_episode_rewards = tf.compat.v1.placeholder(tf.float32,
                    (None, ), name='dicounted_episode_rewards')


"""
    Holds the model
"""
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

@dataclass
class Model:
    """
        Creates the step and the training  model
    """
    policy: Any
    obsv_space: Any
    action_space: Any
    n_steps: Any
    n_envs: Any
    vf_coef: Any
    ent_coef: Any
    max_grad_norm: Any

    def __post_init__(self):
        self.build_model()
    def build_model(self):
        """
            Creates the model
        """
        sess = tf.get_default_session()
        actions = tf.compat.v1.placeholder(tf.int, [None], name='actions')
        advantages = tf.compat.v1.placeholder(tf.float32, [None], name='advantages')
        rewards = tf.compat.v1.placeholder(tf.float32, (None), name='rewards')
        lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        step_model = policy(sess, obsv_space, action_space, n_envs, 1, reuse=False)
        train_model = policy(sess, obsv_space, action_space, n_envs*n_steps, n_steps, reuse=True)

        # Loss = Policy gradient loss - entropy * entropy_coeff + value_coeff  * value los

    def save(self):
        """
            Saves the Model
        """
        pass

    def train(self):
        """
            Trains the model
            - Feed forward and retropropagates gradients
        """
        pass

    def load(self):
        """
            Loads the saved model
        """
        pass

"""
    Large scale study of curiosity driven learning
"""
import numpy as np
import tensorflow as tf

from dataclasses import dataclass
@dataclass
class Curiosity:
    """
        Intrinsic Curiosity Module
        Composed of two NNs: Inverse Model and Forward Model.
        Generates curiosity
    """
    n_states: int
    n_actions: int
    gamma: float = .99
    learning_rate: float = .99
    mem_size: float = 1000
    batch_size: float = 128

    def __post_init__(self):
        self.memory = np.zeros((self.mem_size, self.n_states * 2 + 2))
        self.build_model()
    def build_net(self):
        """
            Creates the NN model
        """
        self.states = tf.compat.v1.placeholder(tf.float32, (None, self.n_states), name='states')
        self.actions = tf.compat.v1.placeholder(tf.int32, (None,), name='actions')
        self.rewards = tf.compat.v1.placeholder(tf.float32, (None,), name='external_rewards')
        self.next_states = tf.compat.v1.placeholder(tf.float32, (None, self.n_states), name='next_states')

        self.build_dynamics_net()

    def build_dynamics_net(self):
        """
            Predicts next state
            : loss error between actual state and predicted state
        """
        with tf.compat.v1.variable_scope('dynamic_net'):
            actions = tf.expand_dims(tf.cast(self.actions, dtype=tf.float32, name='actions_float'), name='actions_2d',axis=1)
            stat_act = tf.concat((self.states, actions), axis=1, name='state_actions')
            encoded_states = self.next_states





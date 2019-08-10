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
    lr: float = .99
    mem_size: float = 1000
    batch_size: float = 128

    def __post_init__(self):
        self.memory = np.zeros((self.mem_size, self.n_states * 2 + 2))
        self.build_model()

    def build_net(self):
        """
            Creates the NN model
        """
        self.states = tf.compat.v1.placeholder(
            tf.float32, (None, self.n_states), name='states')
        self.actions = tf.compat.v1.placeholder(
            tf.int32, (None,), name='actions')
        self.rewards = tf.compat.v1.placeholder(
            tf.float32, (None,), name='external_rewards')
        self.next_states = tf.compat.v1.placeholder(
            tf.float32, (None, self.n_states), name='next_states')

        # dynamics Net
        dyn_nxt_states, curiosity, train_opt = self.build_dynamics_net()

        # RL model
        total_reward = tf.add(curiosity, self.rewards, name='reward_sum')
        q, dqn_loss, dqn_opt = self.build_dqn(total_reward)

    def build_dynamics_net(self):
        """
            Predicts next state
            : loss error between actual state and predicted state
        """
        with tf.compat.v1.variable_scope('dynamic_net'):
            actions = tf.expand_dims(tf.cast(
                self.actions, dtype=tf.float32, name='actions_float'),
                name='actions_2d', axis=1)
            stat_act = tf.concat((self.states, actions),
                                 axis=1, name='state_actions')
            encoded_states = self.next_states
            dyn_sa = tf.layers.dense(inputs=stat_act,
                                     units=32,
                                     activation=tf.nn.relu)

            # Predicted states
            dyn_next_states = tf.layers.dense(dyn_sa, self.n_states)

            with tf.name_scope('intrinsic_rew'):
                squared_diff = tf.reduce_sum(           # Intrinsic reward
                    tf.square(encoded_states - dyn_next_states),
                    axis=1)
            optimizer = tf.train.AdamOptimizer(
                self.lr, name='dyamic_opt').minimize(
                tf.reduce_mean(squared_diff)
            )
            return dyn_next_states, squared_diff, optimizer

    def build_dqn(self, reward):
        """
            Finds loss in Q value between target net
            and evaluation net
        """
        with tf.compat.v1.variable_scope('evaluation_net'):
            eval_one = tf.layers.dense(self.states,
                                       units=128,
                                       activation=tf.nn.relu,
                                       name='eval_one')
            eval_q = tf.layers.dense(eval_one, self.n_actions,
                                     activation=None,
                                     name='eval_q')

        with tf.compat.v1.variable_scope('target_net'):
            target_one = tf.layers.dense(self.next_states, 128,
                                         activation=tf.nn.relu,
                                         name='target_one')
            target_q = tf.layers.dense(target_one,
                                       self.n_actions,
                                       activation=None,
                                       name='target_q')
        with tf.compat.v1.variable_scope('q_target'):
            q_target = reward + self.gamma * \
                tf.reduce_max(target_q, axis=1, name='q_max_ns')

        with tf.compat.v1.variable_scope('q_wrt_a'):
            act_indices = tf.stack([tf.range(tf.shape(self.actions)[0],
                                             dtype=tf.int32),
                                    self.actions],
                                   axis=1,
                                   name='stacked_act_indices')
            q_wrt_ac = tf.gather_nd(params=eval_q, indices=act_indices)

        # TD error
        loss = tf.losses.mean_squared_error(
            labels=q_target,
            predictions=q_wrt_ac
        )
        opt = tf.train.AdamOptimizer(self.lr).minimize(
            loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='evaluation_net')
        )
        return eval_q, loss, opt

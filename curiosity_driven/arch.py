"""
    Simple Implementation of Large scale study of curiosity driven learning in
    Mountain Car
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
    lr: float = 1e-3
    epsilon: float = .95
    mem_size: float = 1000
    batch_size: float = 128
    write_graph: bool = True

    def __post_init__(self):
        # [s, a, r, n_s]
        self.memory = np.zeros((self.mem_size, self.n_states * 2 + 2))
        self.mem_idx = 0
        self.dyn_opt, self.dqn_opt, self.q, \
            self.intrsc_rew = self.build_model()

        self.sess = tf.compat.v1.Session()
        self.setup_replacement()

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
        dyn_nxt_states, curiosity, dyn_opt = self.build_dynamics_net()

        # RL model
        total_reward = tf.add(curiosity, self.rewards, name='reward_sum')
        q, dqn_loss, dqn_opt = self.build_dqn(total_reward)

        return dyn_opt, dqn_opt, q, curiosity

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

    def setup_replacement(self):
        """
            Assigns to target network from the evaluation net
            and writes logs
        """
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='evaluation_net')
        target_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        with tf.compat.v1.variable_scope('hard_replacement'):
            self.target_replace_op = [
                tf.assign(tg, ev) for tg, ev in zip(
                    target_params, eval_params)
            ]
        if self.write_graph:
            tf.compat.v1.summary.FileWriter(
                '/root/tensorboard/curiosity/1', graph=self.sess.graph)

    def chose_action(self, observation):
        """
            Returns an action given an observation
        """
        state = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # Feed forward the observation and get q value for each action
            action_value = self.sess.run(
                self.q, feed_dict={self.states: state})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def memorize(self, experience):
        """
            Stores an experience batch in memory
        """
        st, actions, rewards, next_st = experience
        transition = np.hstack((st, [actions, rewards], next_st))
        self.mem_idx = 0 if self.mem_idx > self.mem_size else self.mem_idx

        self.memory[self.mem_idx, :] = transition
        self.mem_idx += 1

"""
    Simple Implementation of Large scale study of curiosity driven learning in
    Mountain Car
"""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


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
    mem_size: int = 1000
    target_replace_step: int = 400
    write_graph: bool = True

    def __post_init__(self):
        # [s, a, r, n_s]
        self.memory = np.zeros((self.mem_size, self.n_states * 2 + 2))
        self.mem_idx, self.step = 0, 0
        self.dyn_opt, self.dqn_opt, self.q, \
            self.intrsc_rew = self.build_net()

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
        dyn_nxt_states, curiosity, dyn_opt = self.build_dynamics_net(
            self.states, self.actions, self.next_states)

        # RL model
        total_reward = tf.add(curiosity, self.rewards, name='reward_sum')
        q, dqn_loss, dqn_opt = self.build_dqn(
            self.states, self.actions, total_reward, self.next_states)

        return dyn_opt, dqn_opt, q, curiosity

    def build_dynamics_net(self, states, actions, next_states):
        """
            Predicts next state
            : loss error between actual state and predicted state
        """
        with tf.compat.v1.variable_scope('dynamic_net'):
            actions = tf.expand_dims(tf.cast(
                actions, dtype=tf.float32, name='actions_float'),
                name='actions_2d', axis=1)
            stat_act = tf.concat((states, actions),
                                 axis=1, name='state_actions')
            encoded_states = next_states
            dyn_sa = tf.layers.dense(inputs=stat_act,
                                     units=32,
                                     activation=tf.nn.relu)

            # Predicted states
            dyn_next_states = tf.layers.dense(dyn_sa, self.n_states)

            with tf.name_scope('intrinsic_rew'):
                squared_diff = tf.reduce_sum(           # Intrinsic reward
                    tf.square(encoded_states - dyn_next_states),
                    axis=1)
            optimizer = tf.compat.v1.train.AdamOptimizer(
                self.lr, name='dyamic_opt').minimize(
                tf.reduce_mean(squared_diff)
            )
            return dyn_next_states, squared_diff, optimizer

    def build_dqn(self, states, actions, reward, next_states):
        """
            Finds loss in Q value between target net
            and evaluation net
        """
        with tf.compat.v1.variable_scope('evaluation_net'):
            eval_one = tf.layers.dense(states,
                                       units=128,
                                       activation=tf.nn.relu,
                                       name='eval_one')
            eval_q = tf.layers.dense(eval_one, self.n_actions,
                                     activation=None,
                                     name='eval_q')

        with tf.compat.v1.variable_scope('target_net'):
            target_one = tf.layers.dense(next_states, 128,
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
            act_indices = tf.stack([tf.range(tf.shape(actions)[0],
                                             dtype=tf.int32),
                                    actions],
                                   axis=1,
                                   name='stacked_act_indices')
            q_wrt_ac = tf.gather_nd(params=eval_q, indices=act_indices)

        # TD error
        loss = tf.losses.mean_squared_error(
            labels=q_target,
            predictions=q_wrt_ac
        )
        opt = tf.train.AdamOptimizer(self.lr).minimize(
            loss, var_list=tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope='evaluation_net')
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

    def choose_action(self, observation):
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

    def learn(self, batch_size=128):
        """
            Samples experience mini batches to feed
            the model networks
        """
        # Replace target params
        if not self.step % self.target_replace_step:
            self.sess.run(self.target_replace_op)

        # sample
        last_mem_idx = self.mem_idx
        sample_batch_idx = np.random.choice(last_mem_idx, size=batch_size)
        sample_batch = self.memory[sample_batch_idx, :]

        states = sample_batch[:, :, self.n_states]
        actions = sample_batch[:, self.n_states]
        rewards = sample_batch[:, self.n_states + 1]
        next_states = sample_batch[:, -self.n_states:]

        feed_inputs = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.next_states: next_states}
        self.sess.run(self.dqn_opt, feed_dict=feed_inputs)

        if not self.step % 1000:  # delay training, stay curious
            feed_inputs.pop(self.rewards)
            self.sess.run(self.dyn_opt,
                          feed_dict=feed_inputs)
        self.step += 1

    def train(self, env, n_episodes=200, plot=True):
        """
           Trains the agent
        """
        self.episode_steps = []

        for episode in range(n_episodes):
            state = env.reset()
            step = 0

            while True:
                env.render()
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.memorize((state, action, reward, next_state))
                self.learn()

                if done:
                    print(f'Episode: {episode},  steps: {step}')
                    self.episode_steps += [step]
                    break
                state = next_state
                step += 1
        return self.plot()

    def plot(self):
        """
            Outputs the steps run for the given episodes
        """
        if not hasattr(self, 'episode_steps'):
            return
        plt.plot(self.episode_steps)
        plt.ylabel('steps')
        plt.xlabel('no. of episodes')
        plt.show()

"""
    This module contains a Deep Q Network model for Doom
    game

    - A pretrained model is in the .models directory
    The MODEL_PATH can be changed to .models/doom.ckpt
    for play.
"""

import os

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf
import numpy as np

from memory import Memory
from stack_controls import create_env, stack_frames, get_state_size, GAME


SAVE_PATH = '/home/mugoh/'
MODEL_PATH = '/tmp/.models/doom.ckpt'

STACKED_FRAMES = deque([np.zeros((84, 84), dtype=np.int)
                        for _ in range(4)], maxlen=4)
saver = None


@dataclass
class DoomDqNet:
    """
        Deep Q Network model for doom.

        Parameters
        ----------
        lr: float
            Learning rate
        gamma: float
            Discounting factor for future rewards
        eps: float
            Explore-exploit tradeoff for agent actions
        min_eps: float
            Minimum value for epsilon
        max_eps: float
            Maxumum value for epsilon
        name: str, default = 'DoomDqNet'
            Variable for tf namescope
        state_size: list, default = [84, 84, 4]
            Shape of input stack
    """
    lr: int = 0.0002
    gamma: float = 0.95
    eps: float = 0.0001
    min_eps: float = 0.01
    max_eps: float = 1.0
    memory_size: int = 1000000
    name: str = 'DoomDQNet'
    state_size: list = field(default_factory=get_state_size)
    # Left, Right, Shoot
    action_size: Any = field(default_factory=GAME.get_available_buttons_size)

    def __post_init__(self):
        self.build_model()
        self.setup_tf_writer()

    def build_model(self):
        """
            Creates the neural net model
        """
        self.memory = Memory(max_size=self.memory_size)

        with tf.compat.v1.variable_scope(self.name):
            self.inputs = tf.compat.v1.placeholder(
                tf.float32, [None, *self.state_size], name='inputs')
            self.actions = tf.compat.v1.placeholder(
                tf.float32, [None, 3], name='agent_actions')

            # target_Q:= R(s, a) + yQ^(s', a')
            self.target_Q = tf.compat.v1.placeholder(
                tf.float32, [None], name='target')
            self.build_convnet()

    def build_convnet(self):
        """
            Builds the model convolution networks
        """
        conv_one = tf.layers.conv2d(
            inputs=self.inputs,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv_one'
        )
        conv_one_batchnorm = self._batch_normalize(
            conv_one, name='batch_norm_one')
        conv_one_out = self._activate(
            conv_one_batchnorm, 'conv1_out')  # [20, 20, 32]

        conv_two = tf.layers.conv2d(
            inputs=conv_one_out,
            kernel_size=[4, 4],
            filters=64,
            strides=[2, 2],
            padding='valid',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv_two')

        conv_two_batchnorm = self._batch_normalize(conv_two, 'batch_norm_two')
        conv_two_out = self._activate(
            conv_two_batchnorm, 'conv2_out')  # -> [9, 9, 64]

        conv_three = tf.layers.conv2d(
            inputs=conv_two_out,
            filters=128,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='valid',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv_three'
        )
        conv_three_batchnorm = self._batch_normalize(
            conv_three, 'batch_norm_three')
        conv_three_out = self._activate(
            conv_three_batchnorm, 'conv3_out')

        flatten = tf.layers.flatten(conv_three_out)
        fc = tf.layers.dense(
            inputs=flatten,
            units=512,
            activation=tf.nn.elu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc_one'
        )
        self.output = tf.layers.dense(
            inputs=fc,
            units=3,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr) \
            .minimize(loss=self.loss)

    def _activate(self, layer, name=None):
        """
            Passes the layer through ELU activation function
        """
        return tf.nn.elu(layer, name=name)

    def _batch_normalize(self, layer, name=None):
        """
            Creates a batch normalization layer using input
            as the layer parameter
        """
        batch_layer = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, name=name)

        return batch_layer(inputs=layer, training=True
                           )

    def prepopulate_memory(self, episodes=64):
        """
            Takes states and actions, appending experience to memory
        """

        global STACKED_FRAMES
        self.game, actions = create_env()
        self.possible_actions = actions
        self.game.set_window_visible(False)
        self.game.new_episode()  # Render game

        #  First step
        state = self.game.get_state().screen_buffer
        state, stacked_frames = stack_frames(state, new_episode=True)
        print('\n\t==========Memorizing Experiences=========\n')

        for episode in range(episodes):
            action = actions[np.random.choice(actions.shape[0], size=1)][0]
            reward = self.game.make_action(list(action))
            done = self.game.is_episode_finished()
            print(f'{episode}  action - {action}\treward - {reward}')

            if done:  # Dead
                next_state = np.zeros(state.shape)
                self.memory + [state, action, reward, next_state, done]
                self.game.new_episode()
                state = self.game.get_state().screen_buffer
                state, stacked_frames = stack_frames(state, new_episode=True)
            else:
                next_state = self.game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(
                    next_state, stacked_frames)
                self.memory + [state, action, reward, next_state, done]
                state = next_state
        self.stacked_frames = stacked_frames
        STACKED_FRAMES = self.stacked_frames

    def train(self, episodes=500, max_steps=100, batch_size=64,
              save_interval=5, training=True):
        """
            Trains the agent: Runs episodes, collecting states,
            actions, rewards and saving as experiences to memory
        """

        if training:
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                decay_step = 0
                loss = ''
                self.game.init()

                for episode in range(episodes):
                    step = 0
                    episode_rewards = []

                    # Make new episode and observe first state
                    self.game.new_episode()
                    state = self.game.get_state().screen_buffer
                    state, stacked_frames = stack_frames(
                        state, new_episode=True)

                    while step < max_steps:
                        step += 1
                        decay_step += 1

                        # Predict action and take
                        action, explore_prob = self.predict_action(
                            sess,
                            state,
                            decay_step)
                        reward = self.game.make_action(action)
                        episode_rewards.append(reward)

                        done = self.game.is_episode_finished()

                        if done:
                            next_state = np.zeros(
                                (84, 84),
                                dtype=np.int)
                            next_state, stacked_frames = stack_frames(
                                next_state, stacked_frames)
                            # End episode
                            step = max_steps
                            total_reward = np.sum(episode_rewards)
                            self.memory + [state, action,
                                           reward, next_state, done]
                            print(f'Episode - {episode}, ' +
                                  f' Total reward - {total_reward}, ' +
                                  f'Training loss - {loss}, ' +
                                  f'Explore Prob - {explore_prob}'
                                  )
                        else:
                            next_state = self.game.get_state().screen_buffer
                            next_state, stacked_frames = stack_frames(
                                next_state, stacked_frames)
                            self.memory + [state, action,
                                           reward, next_state, done]
                            state = next_state

                        mini_batches = self._get_mini_batch(batch_size)
                        # Get Qs for next state
                        self.Qs_next_state = sess.run(
                            self.output,
                            feed_dict={
                                self.inputs: mini_batches.get(
                                    'next_states')
                            })
                        self.target_Qs_batch = self.get_target_Qs(mini_batches)
                        loss = self.find_loss(sess, mini_batches)
                        self.write_summaries(sess, episode, mini_batches)
                        self.save(sess, episode, interval=save_interval)

    def write_summaries(self, sess,  episode, mini_batches):
        """
            Flushes TF summaries
        """
        summary = sess.run(self.write_op,
                           feed_dict={
                               self.inputs: mini_batches.get('states'),
                               self.target_Q: self.target_Qs_batch,
                               self.actions: mini_batches.get('actions')
                           }
                           )
        self.writer.add_summary(summary, episode)
        self.writer.flush()

    def save(self, sess, episode, interval):
        """
            Saves the model at a given interval
        """

        if not episode % interval:
            self.saver.save(sess, MODEL_PATH)

    def find_loss(self, sess, mini_batches):
        """
            Finds the training loss
        """
        target_mini_batch = np.array(
            [m_batch for m_batch in self.target_Qs_batch])
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.inputs: mini_batches.get('states'),
                self.target_Q: target_mini_batch,
                self.actions: mini_batches.get('actions')
            })
        return loss

    def get_target_Qs(self, mini_batches):
        """
             Set Qs_tg = r if episode ends at s+1 else
             Qs_tg = r + y*max(Q[s', a'])

        """
        target_Qs_batch = []
        batch_ = mini_batches.get('batch_len') - 1  # For index

        for i in range(batch_):
            try:
                terminal = mini_batches.get('dones')[i]

                current_rewards = mini_batches.get('rewards')[i]
                if terminal:
                    # Terminal state only equals reward
                    target_Qs_batch.append(current_rewards)
                else:
                    target_ = current_rewards + self.gamma * \
                        np.max(self.Qs_next_state[i])
                    target_Qs_batch.append(target_)
            except IndexError:
                continue

        return target_Qs_batch

    def _get_mini_batch(self, batch_s):
        """
            Returns a mini batch of experiences stored in
            memory
        """
        batch, batch_len = self.memory.sample(batch_s)
        states_m_batch = self.__from_memory(batch, key='states', min_dims=3)
        actions_m_batch = self.__from_memory(batch, key='actions')
        rewards_m_batch = self.__from_memory(batch, key='rewards')
        next_state_m_batch = self.__from_memory(
            batch, key='next_states', min_dims=3)
        done_m_batch = self.__from_memory(batch, key='dones')

        return {'states':  states_m_batch,  # np.resize(states_m_batch,
                # (batch_s, 84, 84, 4))
                'actions': actions_m_batch,
                'rewards': rewards_m_batch,
                # np.resize(next_state_m_batch,
                'next_states':  next_state_m_batch,

                #        (batch_s, 84, 84, 4)),
                'dones': done_m_batch,
                'batch_len': batch_len
                }

    def __from_memory(self, batch, key, min_dims=0):
        """
            Gives states, actions, rewards, as mini
            batches from a memory sample
        """
        m_b = np.array((batch.get(key)), ndmin=min_dims)
        return m_b

    def predict_action(self, sess, state, decay_step):
        """
            Predicts the next action for the agent.

            Uses the value of epsilon to select a random value
            or action at argmax(Q[s, a])
        """
        explore_exploit_tradeoff = np.random.uniform()
        explore_prob = self.min_eps + \
            (self.max_eps - self.min_eps) * np.exp(-self.eps * decay_step)

        if explore_prob > explore_exploit_tradeoff:
            # Explore
            action = self.possible_actions[np.random.choice(
                self.possible_actions.shape[0], size=1)][0]
        else:
            # Exploit -> Estimate Q values state
            Qs = sess.run(
                self.output,
                feed_dict={
                    self.inputs: state.reshape((1, *state.shape))})
            # Best action
            choice = np.argmax(Qs)
            action = self.possible_actions[int(choice)]
        return list(action), explore_prob

    def setup_tf_writer(self):
        """
            Sets up tensorboard writer
        """
        global saver

        self.writer = tf.compat.v1.summary.FileWriter(
            os.path.join(SAVE_PATH, "tensorboard/dqn/1"))
        self.saver = tf.train.Saver()
        if not saver:
            saver = self.saver

        tf.compat.v1.summary.scalar('Loss', self.loss)
        self.write_op = tf.compat.v1.summary.merge_all()

    def play(self, episodes=25):
        """
            Play with trained agent
        """
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, MODEL_PATH)
            if not hasattr(self, 'game'):
                self.game, possible_actions = create_env(render_screen=True)
            else:
                possible_actions = self.possible_actions

            self.game.set_window_visible(True)
            self.game.init()
            total_reward = 0

            for _ in range(episodes):
                self.game.new_episode()
                while not self.game.is_episode_finished():
                    frame = self.game.get_state().screen_buffer
                    state, stacked_frames = stack_frames(
                        frame, STACKED_FRAMES)
                    # Largest Q value
                    Q_s = sess.run(
                        self.output,
                        feed_dict={
                            self.inputs: np.resize(state,
                                                   (1, *state.shape))
                        })
                    action = np.argmax(Q_s)
                    action = possible_actions[int(action)]
                    self.game.make_action(list(action))
                    reward = self.game.get_total_reward()
                print(f'score: {reward}')
                total_reward += reward
            print(f'Total reward: {total_reward/episodes}')
            self.game.close()


def main():
    """
        Runs the DQN model
    """
    create_env(render_screen=False)  # NO video on VM
    clf = DoomDqNet()
    # clf.prepopulate_memory(episodes=2000)
    # clf.train(episodes=1000, batch_size=64, max_steps=2000)
    clf.play(episodes=200)


if __name__ == '__main__':
    main()

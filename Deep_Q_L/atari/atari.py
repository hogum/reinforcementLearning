from collections import deque
from dataclasses import dataclass, field
import os

import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

from .memory import Memory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Supress INFO messages
RESOLUTON = (110, 84)

STACK_SIZE = 4
STACKED_FRAMES_ = deque(
    [np.zeros(RESOLUTON, dtype=np.int) for _ in range(STACK_SIZE)],
    maxlen=4)


def create_env():
    """
        Sets up the game environment
    """
    env = retro.make(game='SpaceInvaders-Atari2600')
    action_space = env.action_space.n
    observation_space = env.observation_space
    # One-hot encoded actions
    actions_choice = np.identity(action_space, dtype=np.int)

    return env, (actions_choice, observation_space)


def preprocess_frame(state):
    """
        Grayscales and crops out the state for
        reduced training time
    """
    state = rgb2gray(state)
    cropped_state = state[8:-12, 4:-12]
    normalized_state = cropped_state / 255
    preprocessed_state = transform.resize(normalized_state, RESOLUTON)

    return preprocessed_state


def get_state_size():
    """
        Gives the size of the state (height, width, channels)
    """
    return [*RESOLUTON, STACK_SIZE]


def stack_frames(state, stacked_frames=None, new_episode=False):
    """
        Creates a stack of four frames from previous
        state for a sense of `motion`
    """
    frame = preprocess_frame(state)

    if new_episode:
        stacked_frames = deque(
            [np.array(frame) for _ in range(STACK_SIZE)], maxlen=4)
    else:
        stacked_frames.append(frame)
    state = np.stack(stacked_frames, axis=2)

    return state, stacked_frames


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
    gamma: float = 0.99
    eps: float = 0.0001
    min_eps: float = 0.01
    max_eps: float = 1.0
    memory_size: int = 1000000
    name: str = 'DoomDQNet'
    state_size: list = field(default_factory=get_state_size)
    action_size = 8

    def __post_init__(self):
        self.build_model()
        self.memory = Memory(self.memory_size)
        self.setup_writer()

    def build_model(self):
        """
            Sets up the model for use in training the agent
        """

        with tf.variable_scope(self.name):
            self.inputs = tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, *self.state_size],
                name='inputs')
            self.actions = tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(None, self.action_size),
                name='outputs'
            )
            self.target_Q = tf.placeholder(tf.float32, [None], 'targetQ')
            self.build_conv_net()

    def build_conv_net(self):
        """
            Builds the convolutional layers used by the network
        """
        conv1 = tf.layers.conv2d(
            inputs=self.inputs,
            filters=32,
            kernel_size=[8, 8],
            strides=(4, 4),
            padding='valid',
            kerner_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv1'
        )
        conv1_out = tf.nn.relu(conv1, name='conv1_out')

        conv2 = tf.layers.conv2d(
            inputs=conv1_out,
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='valid',
            kerner_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv2'
        )
        conv2_out = tf.nn.relu(conv2, name='conv2_out')

        conv3 = tf.layers.conv2d(
            inputs=conv2_out,
            filters=64,
            strides=(2, 2),
            kernel_size=(3, 3),
            padding='valid',
            kerner_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name='conv3'
        )
        conv3_out = tf.nn.relu(conv3, name='conv3_out')

        flatten = tf.contrib.layers.flatten(conv3_out)
        fc = tf.layers.dense(
            inputs=flatten,
            units=512,
            kerner_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc'
        )
        self.output = tf.layers.dense(
            inputs=fc,
            units=self.action_size,
            activation=None,
            kerner_initializer=tf.contrib.layers.xavier_initializer(),
            name='ouput'
        )

        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def populate_memory(self, episodes):
        """
            Populates the memory with experiences received from
            random actions
            (state, action, reward, next_state, done)
        """
        self.env, env_vars = create_env()
        state = self.env.reset()
        self.actions_choice, _ = env_vars
        state, stacked_frames = stack_frames(state, new_episode=True)

        for _ in range(episodes):
            choice = np.random.randint(0, self.action_size)
            action = self.actions_choice[choice]
            next_state, reward, done, _ = self.env.step(action)

            next_state, stacked_frames = stack_frames(
                next_state, stacked_frames)

            if done:
                next_state = np.zeros(RESOLUTON, dtype=np.int)
                self.memory + [state, action, reward, next_state, done]
                state = env.reset()
                state, stacked_frames = stack_frames(state, new_episode=True)
            else:
                self.memory + [state, action, reward, next_state, done]
                state = next_state

    def setup_writer(self):
        """
            Sets up tensorboarf writer for the model
        """
        self.writer = tf.compat.v1.summary.FileWriter(
            '/root/tensorboard/dqn/2')
        tf.summary.scalar('Model', self)
        self.writer_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def predict_action(self, sess, state, decay_step):
        """
            Selects a random action for the agent for exploration
            or the action an max(Q[s', a']) for exploitation
        """
        explore_explot_tradeoff = np.random.random()

        explore_prob = self.min_eps + \
            (self.max_eps - self.min_eps) * np.exp(-self.eps * decay_step)

        if explore_prob > explore_explot_tradeoff:
            choice = np.random.randint(0, self.action_size)
            action = self.actions_choice[choice]
        else:
            Q_s = sess.run(self.output,
                           feed_dict={
                               self.inputs: state.reshape(1, *state.shape)}
                           )
            choice = np.argmax(Q_s)
            action = self.actions_choice[choice]
        return action, explore_prob

    def train(self, episodes=30, batch_size=64,
              max_steps=50000, training=True):
        """
            Trains the agent by prediction of future Qs and
            minimizing difference between Qs and future Qs [loss].
        """
        if training:
            with tf.compat.v1.Session() as sess:
                sess.run(tf.global_variables_initializer())
                decay_step = 0
                loss = acc = ''
                final_rewards = []

                for episode in range(episodes):
                    step = 0
                    episode_rewards = []
                    state = self.env.reset()
                    state, stacked_frames = stack_frames(
                        state, new_episode=True)

                    while step < max_steps:
                        step += 1
                        decay_step += 1

                        action, explore_prob = self.predict_action(
                            sess, state, decay_step)
                        next_state, reward, done, _ = self.env.step(action)

                        self.env.render()
                        episode_rewards.append(reward)

                        if done:
                            next_state = np.zeros(RESOLUTON, dtype=np.int)
                            next_state, stacked_frames = stack_frames(
                                next_state, stacked_frames)
                            step = max_steps
                            total_reward = np.sum(episode_rewards)
                            final_rewards += total_reward

                            print(f'Episode: {episode}  ' +
                                  f'Total reward: {total_reward}  ' +
                                  f'Explore Prob: {explore_prob}  ' +
                                  f'Loss: {loss}  ' +
                                  f'Accuracy: {acc}')
                            self.memory + [state, action,
                                           reward, next_state, done]
                        else:
                            next_state, stacked_frames = stack_frames(
                                next_state, stacked_frames)
                            self.memory + [state, action,
                                           reward, next_state, done]
                            state = next_state
                        loss, acc = self._learn(sess, episode, batch_size)
                        self._save(sess, episode)

    def _learn(self, sess, episode, batch_size):
        """
            Helps the agent learn from the sampled experiences
        """
        mini_batches = self.get_mini_batches(batch_size)
        target_mini_batch = self.get_q_values(sess, mini_batches)
        mini_batches.update('targets': target_mini_batch)

        loss, acc = self.find_loss(sess, mini_batches)
        self.summarize(sess, episode, mini_batches, episode)

        return loss, acc

    def get_mini_batches(self, batch_s):
        """
            Obtains random experiences from memory for
            the agent to learn from
        """

        batch, batch_len = self.memory.sample(batch_s)
        states_m_batch = self.__from_memory(
            batch,
            key='states',
            min_dims=3)
        actions_m_batch = self.__from_memory(batch, key='actions')
        rewards_m_batch = self.__from_memory(batch, key='rewards')
        next_state_m_batch = self.__from_memory(
            batch,
            key='next_states',
            min_dims=3)
        done_m_batch = self.__from_memory(batch, key='dones')

        return {
            'states':  states_m_batch,
            'actions': actions_m_batch,
            'rewards': rewards_m_batch,
            'next_states':  next_state_m_batch,
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

    def get_q_values(self, sess, m_batches):
        """
            Gets Q values
            - Q values for the  next state
            - Target Q values
        """
        _, _, rewards, next_states, dones, batch_len = m_batches
        target_Qs_batch = []

        # Next state
        Q_s = sess.run(self.output,
                       feed_dict={self.inputs: next_states})

        # Target Qs
        # r:= if episode ends at s + 1 # else gamma * max[Q(s', a')]
        for batch in range(batch_len):
            terminal = dones[batch]
            rewards_mb = rewards[batch]

            if terminal:
                target_Qs_batch.append(rewards_mb)
            else:
                target_Qs_batch.append(
                    rewards_mb + self.gamma * np.max(Q_s[batch]))

        target_mini_batch = [m_batch for m_batch in target_Qs_batch]
        return np.array(target_mini_batch)

    def find_loss(self, sess, targets, mini_batches):
        """
            Finds the training loss
        """
        return sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.inputs: mini_batches.get('states'),
                self.target_Q: mini_batches.get('targets'),
                self.actions: mini_batches.get('actions')
            }
        )

    def summarize(self, sess, mini_batches, targets, episode):
        """
            Writes tf summaries
        """
        summary = sess.run(
            self.writer_op,
            feed_dict={self.inputs: mini_batches.get('states'),
                       self.target_Q: mini_batches.get('targets'),
                       self.actions: mini_batches.get('actions')}
        )
        self.writer.add_summary(summary, episode)
        self.writer.flush()

    def _save(self, sess, count):
        """
            Saves the model checkpoints
        """
        if not count % 5:
            self.saver.save(sess, '.models/atari.ckpt')

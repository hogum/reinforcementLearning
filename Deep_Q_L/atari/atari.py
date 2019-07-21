from collections import deque
from dataclasses import dataclass, field
import random
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
        env, env_vars = create_env()
        state = env.reset()
        actions_choice, _ = env_vars
        state, stacked_frames = stack_frames(state, new_episode=True)

        for _ in range(episodes):
            choice = np.random.randint(0, self.action_size)
            action = actions_choice[choice]
            next_state, reward, done, _ = env.step(action)

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


if __name__ == '__main__':
    create_env()

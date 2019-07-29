"""
    Monte Carlo Policy Gradient implementation in
    doom Health Gathering scene
"""

import os
from collections import deque
from dataclasses import dataclass, field

import tensorflow as tf
import numpy as np

import vizdoom as vz
from skimage import transform

from memory import Memory

resolution = (84, 84)
stack_size = 4

STACKED_FRAMES_ = deque(
    [np.zeros(resolution, dtype=np.int) for _ in range(stack_size)],
    maxlen=4)


def create_env(visible=False):
    """
        Creates an instance of the game environment
    """
    path = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'

    doom = vz.DoomGame()
    doom.load_config(os.path.join(path, 'health_gathering.cfg'))
    doom.set_doom_scenario_path(os.path.join(path, 'health_gathering.wad'))

    doom.set_window_visible(visible)
    doom.init()

    actions = np.identity(doom.get_available_buttons_size())

    return doom, actions


def preprocess_frame(frame):
    """
        Preprocess the screen buffer for reduced training time
    """
    try:
        frame = np.array(frame[0] + frame[1] + frame[2])[80:, :]

    except IndexError:
        frame = frame
    frame = frame / 255

    return transform.resize(frame, resolution)


def get_state_size():
    """
        Gives the size of the state (height, width, channels)
    """
    return [*resolution, stack_size]


def stack_frames(state, stacked_frames=None, new_episode=False):
    """
        Creates a stack of four frames from previous
        state for a sense of `motion`
    """
    frame = preprocess_frame(state)

    if new_episode:
        stacked_frames = deque(
            [np.array(frame) for _ in range(stack_size)], maxlen=4)
    else:
        stacked_frames.append(frame)
    state = np.stack(stacked_frames, axis=2)

    return state, stacked_frames


def preprocess_rewards(rewards):
    """
        Discounts and normalized the rewards
    """
    discounted_rewds = np.zeros_like(rewards)
    cumulative = 0

    for i in reversed(range(len(rewards))):
        cumulative *= self.gamma
        discounted_rewds[i] = cumulative + rewards[i]

    discounted_rewds = (discounted_rewds -
                        np.mean(discounted_rewds)) \
        / np.std(discounted_rewds)
    return discounted_rewds


@dataclass
class DoomDDdqN:
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
        state_size: list, default = [100, 120, 4]
            Shape of input stack
        max_tau: int
            Max C step in updating the target network
    """
    lr: int = 0.0002
    gamma: float = 0.99
    memory_size: int = 100000
    name: str = 'DoomPG'
    state_size: list = field(default_factory=get_state_size)
    action_size = 3  # Left, Right, move Forward

    def __post_init__(self):
        self.build_model()
        self.setup_writer()

    def build_model(self):
        """
            Builds the Policy Gradient Neural Net
        """
        with tf.variable_scope(self.name):
            with tf.name_scope('inputs'):
                self.inputs = tf.compat.v1.placeholder(
                        tf.float32,
                        (None, *self.state_size),
                            name='inputs')

                self.inputs = tf.compat.v1.placeholder(
                        tf.float32,
                        (None, self.action_size),
                        name='actions'
                        )
                self.discounted_eps_rw = tf.compat.v1.placeholder(
                        tf.float32,
                        (None,),
                        name='discounted_episode_rewds'
                        )
                self.mean_reward = tf.compat.v1.placeholder(
                        tf.float32,
                        name='mean_reward')


i
                with tf.name_scope('fc_one'):
                    fc_one =

    def setup_writer(self):
        """
            Sets up the tensorboard writer
        """
        self.writer=tf.compat.v1.summary.FileWriter(
            '/root/tensorboard/policy_g/doom/1')
        tf.compat.v1.summary.scalar('Loss', self.loss)
        tf.compat.v1.summary.scalar('reward_mean', self.mean_reward)
        self.writer_op=tf.compat.v1.summary.merge_all()
        self.saver=tf.train.Saver()

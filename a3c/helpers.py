"""
    Environment and Scope setups
"""

import os

from skimage import transform, color
import numpy as np
import tensorflow as tf

from scipy import signal
import imageio

import vizdoom as vz


resolution = (84, 84)
stack_size = 1


def preprocess_frame(frame):
    """
        Preprocesses input state
        : crops, resizes, normalizes image
    """
    try:
        # This is funny. Need to find out effect of summing RGB channels
        # frame = (frame[0] + frame[1] + frame[2])[10:-10, 30:-30]
        frame = frame[0][10:-10, 30:-30]

    except IndexError:
        frame = frame[10:-10, 30:-30]
    frame = color.rgb2gray(frame)
    frame = transform.resize(frame, resolution)
    frame = np.reshape(frame, [np.prod(frame.shape)]) / 255.

    return frame


def create_env(visible=False, scene=''):
    """
        Creates an instance of the game environment
    """
    path = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'
    scene = 'defend_the_center' if not scene else scene

    doom = vz.DoomGame()
    doom.load_config(os.path.join(path, '{scene}.cfg'))
    doom.set_doom_scenario_path(os.path.join(path, '{scene}.wad'))

    doom.set_window_visible(visible)
    doom.init()

    actions = np.identity(doom.get_available_buttons_size(),
                          dtype=np.bool)

    return doom, actions


def get_state_size():
    """
        Gives the size of the state (height, width, channels)
    """
    return [*resolution, stack_size]


def update_target_graph(from_scope, worker_name):
    """
        Updates the worker network parameter with those of
        the global network
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, worker_name)

    ops = [(to_var.assign(from_var))
           for from_var, to_var in zip(from_vars, to_vars)]
    return ops


def preprocess_rewards(rewards, gamma):
    """
        Returns discounted rewards
    """
    return signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]


def create_gifs(frames, path, **kwargs):
    """
        Saves a stack of states as a GIF
    """
    with imageio.get_writer(uri=path, mode='I', **kwargs) as im_writer:
        for frame in frames:
            im_writer.append_data(frame)

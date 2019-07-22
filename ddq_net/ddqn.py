"""
    Double Duelling Deep Q Net with Prioritized Experience Replay
    : Implementation with Doom agent

     https://papers.nips.cc/paper/3964-double-q-learning
"""
import os

import tensorflow as tf
import numpy as np

import vizdoom as vz
from skimage import transfrom


def create_env():
    """
        Creates an instance of the game environment
    """
    path = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'

    doom = vz.DoomGame()
    doom.load_config(os.path.join(path, 'deadly_corridor.cfg'))
    doom.set_doom_scenario_path(os.path.join(path, 'deadly_corridor.wad'))

    doom.set_window_visible(VISIBLE)
    doom.init()

    actions = np.identity(game.get_available_buttons_size())

    return doom, actions

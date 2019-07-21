from collections import deque
import random
import os

import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Supress INFO messages
RESOLUTON = (110, 84)

STACKED_FRAMES_ = deque(
    [np.zeros(RESOLUTON, dtype=np.int) for _ in range(stack_size)],
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

    return actions_choice, observation_space


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


def stack_frames(state, stacked_frames, new_episode=False):
    """
        Creates a stack of four frames from previous
        state for a sense of `motion`
    """
    frame = preprocess_frame(state)
    stack_size = 4

    if new_episode:
        stacked_frames = deque(
            [np.array(frame) for _ in range(stack_size)], maxlen=4)
    else:
        stacked_frames.append(frame)
    state = np.stack(stacked_frames, axis=2)

    return state, stack_frames


if __name__ == '__main__':
    create_env()

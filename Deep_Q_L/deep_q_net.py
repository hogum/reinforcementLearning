import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

import vizdoom as vz

from collections import deque
import time
import os

#from Ipython.display import HTML

# HTML('<iframe>width="560" height="315" src="" frameborder="0" allow="autoplay; encrypted media" allowfullscreen > </iframe >')


def create_env():
    """
        Sets up the game environment
    """
    scenarios = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'

    doom = vz.DoomGame()
    doom.load_config(os.path.join(scenarios, 'basic.cfg'))  # Config
    doom.set_doom_scenario_path(os.path.join(
        scenarios, 'basic.wad'))  # Scenario

    return initalize_game(doom)


def initalize_game(game):
    """
        Starts the game environment with the set of
        possible actions
    """
    game.init()
    actions = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return game, actions


def test_game():
    """
        Test environment  with random action
    """
    episodes = 25
    game, actions = create_env()

    for _ in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()
            state.screen_buffer
            action = actions[np.random.choice(actions.shape[0], size=1)][0]
            print(action)
            reward = game.make_action(action)
            print(f'action: {action}\treward: {reward}')
            time.sleep(.03)
        print('Total Reward: ', game.get_total_reward())
        time.sleep(3)
    game.close()


def preprocess_frame(frame):
    """
        Crops the screen, normalizes pixel values
        and  resizes the frame for reduced computation
        time
    """
    # Grayscale frame
    # x = np.mean(frame, -1)

    # Crop screen above roof
    cropped_frame = frame[30: -10, 30: -10]
    normalized_frame = cropped_frame / 255
    resized_frame = transform.resize(normalized_frame, [84, 84])

    return resized_frame


def stack_frames(stacked_frames, state, new_episode=False):
    """
        Creates a deque stack of four frames
        removing the oldest each time a new  frame is
        appended
    """
    stack_size = 4
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int)
                            for _ in range(stack_size)], maxlen=4)
    frame = preprocess_frame(state)

    if new_episode:
        stacked_frames.append(frame)


if __name__ == '__main__':
    test_game()

"""
    This module contains functions that preprocess the game frames
    and save sets of run states
"""

import os
import time
from collections import deque
import numpy as np

import vizdoom as vz

from skimage import transform


def create_env(game_state_only=False, actions_only=False, render_screen=False):
    """
        Sets up the game environment
    """
    scenarios = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'
    # global GAME
    doom = vz.DoomGame()
    doom.load_config(os.path.join(scenarios, 'basic.cfg'))  # Config
    doom.set_doom_scenario_path(os.path.join(
        scenarios, 'basic.wad'))  # Scenario

    if game_state_only:
        return doom
    return initialize_game(doom,  render_screen, actions_only=actions_only)


def initialize_game(game, show_screen, actions_only=False):
    """
        Starts the game environment with the set of
        possible actions
    """
    if not show_screen:
        game.set_window_visible(False)

    game.init()
    actions = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return game, actions if not actions_only else actions


def test_game():
    """
        Test environment  with random action
    """
    episodes = 25
    game, actions = create_env(render_screen=True)

    for _ in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            # state = game.get_state()
            action = actions[np.random.choice(actions.shape[0], size=1)][0]
            print(action)
            reward = game.make_action(list(action))
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
    cropped_frame = frame  # frame[:, 30: -30]
    normalized_frame = cropped_frame / 255
    resized_frame = transform.resize(normalized_frame, [84, 84])

    return resized_frame


def stack_frames(state, stacked_frames=None, new_episode=False):
    """
        Creates a deque stack of four frames
        removing the oldest each time a new  frame is
        appended
    """
    stack_size = 4
    frame = preprocess_frame(state)

    if new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int)
                                for _ in range(stack_size)], maxlen=4)

        for _ in range(stack_size):
            stacked_frames.append(frame)

        stack = np.stack(stacked_frames, axis=2)
    else:
        try:
            stacked_frames.append(frame)
            stack = np.stack(stacked_frames, axis=2)
        except ValueError:
            breakpoint()
    return stack, stacked_frames


def get_empty_stack():
    """
        Creates an empty deque for frames
    """
    stack_size = 4

    return deque([np.zeros((84, 84), dtype=np.int)
                  for _ in range(stack_size)], maxlen=4)


def get_state_size():
    """
        Returns the default shape for the stack input shape

        Input stack (width, height, channel)
    """
    return [84, 84, 4]


GAME = create_env(game_state_only=True)

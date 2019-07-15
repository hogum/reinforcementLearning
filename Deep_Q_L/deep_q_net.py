import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

import vizdoom as vz

from collections import deque
from dataclasses import dataclass, field
import time
import os


def create_env():
    """
        Sets up the game environment
    """
    scenarios = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'
    global GAME
    doom = vz.DoomGame()
    doom.load_config(os.path.join(scenarios, 'basic.cfg'))  # Config
    doom.set_doom_scenario_path(os.path.join(
        scenarios, 'basic.wad'))  # Scenario

    GAME = doom
    return initalize_game(GAME)


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
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stack = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stack = np.stack(stacked_frames, axis=2)
    return stack, stacked_frames


def get_state_size():
    """
        Returns the default shape for the stack input shape

        Input stack (width, height, channel)
    """
    return [84, 84, 4]


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
    name: str = 'DoomDQN'
    state_size: list = field(default_factory=get_state_size)
    # Left, Right, Shoot
    action_size = field(default_factory=GAME.get_available_buttons_size())

    def __post_init__(self):
        self.build_model()

    def build_model(self):
        """
            Creates the neural net model
        """
        with tf.variable_scope(self.name):
            self.inputs = tf.placeholder(
                tf.float32, [None, *self.state_size], name='inputs')
            self.actions = tf.placeholder(
                tf.float32, [None, 3], name='agent_actions')

            # target_Q:= R(s, a) + yQ^(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')
            self.build_convnet()

    def build_convnet(self):
        """
            Builds the model convolution networks
        """


GAME = None

if __name__ == '__main__':
    test_game()

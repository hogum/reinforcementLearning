import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import vizdoom as vz
import time

from Ipython.display import HTML

HTML('<iframe>width="560" height="315" src="" frameborder="0" allow="autoplay; encrypted media" allowfullscreen > </iframe >')


def create_env():
    """
        Sets up the game environment
    """
    doom = vz.DoomGame()
    doom.load_config('basic.cfg')  # Config
    doom.set_doom_scenario_path('basic.wad')  # Scenario

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
            img = state.screen_buffer()
            game_vars = state.game_variables
            action = np.random.choice(actions)

            reward = game.make_reward(action)
            print(f'action: {action}\treward: {reward}')
            time.sleep(.03)
        print('Total Reward: ', game.get_total_reward())
        time.sleep(3)
    game.close()

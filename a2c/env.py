import numpy as np
import gym

import cv2
from baselines.common.atari_wrappers import FrameStack

from retro import make
from retro_contest.local import make as make_local

cv2.ocl.setUseOpenCL(False)  # No GPU use


class PreprocessFrame(gym.ObservationWrapper):
    """
        Grayscales and resizes Frame
    """

    def __init__(self, env, width=96, height=96):
        super().__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        """
            Returns preprocessed frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame


class ActionDiscretizer(gym.ActionWrapper):
    """
        Wraps a retro environment to make it use
        discrete actions for the game
    """

    def __init__(self, env):
        super(ActionDiscretizer, self).__init__(env)
        buttons = ['B', 'A', 'MODE', 'START',
                   'UP', 'DOWN', 'LEFT', 'RIGHT',
                   'C', 'Y', 'X', 'Z'
                   ]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], [
            'RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        self.actions_ = []

        """
            For each action:
                - create an array of 12[buttons] False
                For each button in action:
                    - make button index True
            Creates arrays of actions, where each True element
            is the clicked button
        """
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self.actions_.append(arr)
        self.action_space = gym.spaces.Discreet(len(self.actions_))

    def action(self, a_id):
        """
            Retrieves an action
        """
        return self.actions_[a_id].copy()


def create_env(env_idx):
    """
        Creates an environment with standard wrappers
    """
    wrappers = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}
    ]

    print(wrappers[env_idx]['game'], wrappers[env_idx]['state'], flush=True)
    env = make_local(game=wrappers[env_idx]['game'],
                     state=wrappers[env_idx]['state'],
                     bk2dir='/records')

    # Build actions array
    env = ActionDiscretizer(env)

    # Scale rewards
    env = RewardScaler(env)

    # Preprocess frames and Stack
    env = PreprocessFrame(env)
    env = FrameStack(env, 4)

    env = AllowBackTracking(env)

    return env


def make_train(env_indices=[0], all_=False):
    """
        Returns a list of environments with given indices
    """
    env_indices = np.arange(0, 13) if all_ else env_indices

    return [create_env(idx) for idx in env_indices]

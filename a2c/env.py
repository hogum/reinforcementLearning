import numpy as np
import gym

import cv2
from retro import make
from retro_contest.local import make as make_local

from baselines.common.atari_wrappers import FrameStick

cv2.oc1.setUseOpenCL(False)  # No GPU use


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

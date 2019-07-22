from collections import deque
import numpy as np


class Memory:
    """
        Controls Replay by adding experiences to deque

        Parameters
        ----------
        max_size: int
            Number of elements to hold in memory
    """

    def __init__(self, max_size=4):
        self.experiences = ['states', 'actions',
                            'rewards', 'next_states',
                            'dones']
        self.buffer_len = 0
        self.buffer = {
            "states": deque(maxlen=max_size),
            "actions": deque(maxlen=max_size),
            "rewards": deque(maxlen=max_size),
            "next_states": deque(maxlen=max_size),
            "dones": deque(maxlen=max_size)
        }

    def __add__(self, exp):

        for experience in self.experiences:
            self.buffer.get(experience).append(
                exp[self.experiences.index(experience)]
            )
        self.buffer_len += 1

    def sample(self, batch_size, replace=False):
        """
            Samples a stack of random experiences from the memory
        """
        mini_batches = {}
        idxs = np.random.choice(np.arange(self.buffer_len),
                                size=batch_size,
                                replace=replace)
        for experience in self.experiences:
            mini_batches.update(
                {
                    experience: [
                        self.buffer.get(experience)[idx] for idx in idxs
                    ]
                })

        return mini_batches, self.buffer_len

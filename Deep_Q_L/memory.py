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
        self.buffer = deque(maxlen=max_size)

    def __add__(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        """
            Samples a stack of random experiences from the memory
        """
        memory_size = len(self.buffer)
        idxs = np.random.choice(np.arange(memory_size),
                                size=batch_size,
                                replace=False)
        return (self.buffer[idx] for idx in idxs)

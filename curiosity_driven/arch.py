"""
    Large scale study of curiosity driven learning
"""
import numpy as np
import tensorflow as tf

from dataclasses import dataclass
@dataclass
class Curiosity:
    """
        Intrinsic Curiosity Module
        Composed of two NNs: Inverse Model and Forward Model.
        Generates curiosity
    """
    n_states: int
    n_actions: int
    gamma: float = .99
    learning_rate: float = .99
    mem_size: float = 1000
    batch_size: float = 128

    def __post_init__(self):
        self.memory = np.zeros((self.mem_size, self.n_states * 2 + 2))
        self.build_model()

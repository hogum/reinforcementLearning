from skimage import transform
import numpy as np


def preprocess_frame(frame):
    """
        Preprocesses input state
        : crops, resizes, normalizes image
    """
    resolution = (84, 84)
    try:
        frame = (frame[0] + frame[1] + frame[2])[10:-10, 30:-30]

    except IndexError:
        frame = frame[10:-10, 30:-30]
    frame = transform.resize(frame, resolution)
    frame = np.reshape(frame, [np.prod(frame.shape)]) / 255.

    return frame

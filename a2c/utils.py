import tensorflow as tf


def mse(pred, target):
    """
        Gives the Mean square error
    """
    return tf.square(pred - target) / 2.

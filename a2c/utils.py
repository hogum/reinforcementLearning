import tensorflow as tf
import numpy as np


def mse(pred, target):
    """
        Gives the Mean square error
    """
    return tf.square(pred - target) / 2.


def find_trainable_vars(key):
    """
        Gets variables in the given name scope
    """
    with tf.variable_scope(key):
        return tf.trainable_variables()


def swap_01(iter_):
    """
        Swaps and flattens axes 0 and 1
    """
    s = iter_.shape
    return iter_.swapaxes(0, 1).reshape(s[0] * s[1], * s[2:])


def explained_variance(y_pred, y):
    """
       Computes the fraction of variance that y pred
       explains about y

       returns 1 - var(y - y_pred) / var(y)
    """

    assert y.ndim == y_pred.ndim == 1
    var_y = np.var(y)

    return np.nan if not var_y else 1 - np.var(y - y_pred) / var_y

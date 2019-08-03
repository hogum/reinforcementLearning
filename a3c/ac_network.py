"""
    AC Network class
"""
import tensorflow as tf

resolution = (84, 84)


class AC_Network:
    """
        Holds operations to create Actor Critic Networks
    """

    def __init__(self, state_size, action_size, trainer, name='Ac_net'):
        with tf.variable_scope(name):
            self.inputs = tf.compat.v1.placeholder(dtype=tf.float32,
                                                   shape=(None, state_size),
                                                   name='inputs'
                                                   )
            self.image_in = tf.reshape(self.inputs, (-1, *resolution, 1))
            self.conv_one = tf.layers.conv2d(inputs=self.image_in,
                                             filters=64,
                                             kernel_size=(8, 8),
                                             strides=(4, 4),
                                             activation_fn=tf.nn.elu,
                                             num_outputs=16,
                                             padding='valid',
                                             name='conv1'
                                             )
            self.conv_two = tf.layers.conv2d(inputs=self.conv_one,
                                             filters=128,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding='valid',
                                             activation_fn=tf.nn.elu,
                                             name='conv2'
                                             )
            hidden = tf.contrib.layers.fully_connected(tf.layers.flatten(self.conv_two),
                                                       units=256,
                                                       activation_fn=tf.nn.elu
                                                       )

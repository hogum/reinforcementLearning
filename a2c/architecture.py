import numpy as np
import tensorflow as tf

from baselines.common.distributions import make_pdtype


class A2CPolicy:
    """
        Creates the A2C network architecture
    """

    def __init__(self, sess, obs_space, action_space,
                 n_batchs, n_steps, reuse=False):
        gain = np.sqrt(2)

        # Select probability distribution based on action space
        # Will be used to distribute actions in the stochastic
        # policy [DiagGaussianPdtype] Diagonal Gaussian, 3D normal distribution

        self.pdtype = make_pdtype(action_space)

        height, weight, channel = obs_space.shape
        obs_shape = (height, weight, channel)

        inputs_ = tf.placeholder(tf.float32, (None, *obs_shape), name='input')
        scaled_imgs = tf.cast(inputs_, tf.float32) / 255.

        """
            Build model
            3 CNN for spatial distributions
            # Temporal dependencies handles by stacking of frames

            1 common FC
            1 FC for values
            1 FC for policies
        """
        with tf.variable_scope('model', reuse=reuse):
            conv1 = conv_layer(
                inputs=scaled_imgs,
                filters=32,
                kernel_size=8,
                stride=4,
                gain=gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten, units=512, gain=gain)

            # Build FC that returns a prob distribution over
            # actions[self.pd] and pi logits [self.pi]
            self.pd, self.pi = self.pdtype.pdfromlatent(
                fc_common, init_scale=.01)

            # V(s)
            vf = fc_layer(fc_common, 1, activation_fc=None)[:, 0]

        self.inital_state = None

        # Take action in distribution
        # Stochastic policy -> Not always taking action with highest problty

        a0 = self.pd.sample()

        def value(state_in, *args_, **kwargs_):
            """
                Calculates the Value function V(s)
            """
            return sess.run(vf, {inputs_: state_in})

        def step(state_in, *args_, **kwargs_):
            """
                Takes a step

                Returns actions and V(s)
            """
            action, value = sess.run([a0, vf], {inputs_: state_in})

            return action, value

        def select_action(state_in, *args_, **kwargs_):
            """
                Outputs action to take
            """
            return sess.run(a0, {inputs_: state_in})
        self.inputs = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action


def conv_layer(inputs, filters, kernel_size, stride, gain=1.):
    """
        Returns a convolution layer with parameters provided
    """
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(stride, stride),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(
                                gain=gain)
                            )


def fc_layer(inputs, units, activation_fc=tf.nn.relu, gain=1.):
    """
        Returns a fully connected layers
    """
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fc,
                           kernel_initializer=tf.
                           orthogonal_initializer(gain=gain))

"""
    AC Network class
"""
import tensorflow as tf
import numpy as np

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
                                             activation=tf.nn.elu,
                                             padding='valid',
                                             name='conv1'
                                             )
            self.conv_two = tf.layers.conv2d(inputs=self.conv_one,
                                             filters=128,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding='valid',
                                             activation=tf.nn.elu,
                                             name='conv2'
                                             )
            hidden = tf.contrib.layers.fully_connected(
                tf.layers.flatten(self.conv_two),
                units=256,
                activation_fn=tf.nn.elu
            )
            lstm_cell = tf.keras.layers.LSTMCell(units=256)
            c_ = np.zeros((1, lstm_cell.state_size[0]), np.float32)
            h_ = np.zeros((1, lstm_cell._state_size[1]), np.float32)
            self.state_ = [c_, h_]

            c_in = tf.compat.v1.placeholder(dtype=tf.float32,
                                            shape=(1, lstm_cell.state_size[0]),
                                            name='c_hidden_state')
            h_in = tf.compat.v1.placeholder(dtype=tf.float32,
                                            shape=(1, lstm_cell.state_size[1]),
                                            name='h_output')
            self.state_in = [c_in, h_in]

            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.image_in)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_states = tf.keras.layers.RNN(
                # tf.nn.dynamic_nn
                cell=lstm_cell,
                sequence_length=step_size,
                time_major=False,
                initial_state=state_in,
                inputs=rnn_in
            )
            lstm_c, lstm_h = lstm_states
            self.state_out = lstm_outputs[:1, :], lstm_h[:1, :]
            rnn_out = tf.reshape(lstm_outputs, (-1, 256))

            self.policy = tf.contrib.layers.fully_connected(
                inputs=rnn_out,
                num_outputs=action_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.initializers.glorot_uniform(),
                biases_iniitalizer=None
            )
            self.value = tf.contrib.layers.fully_connected(
                inputs=rnn_out,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.initializers.glorot_uniform(),
                biases_iniitalizer=None
            )
            # Back propagate for target network only
            if name != 'global':
                self.create_grad_ops(name, trainer)

    def create_grad_ops(self, scope, optimizer):
        """
            Creates loss function and gradient update ops
            for the Worker network
        """
        self.actions = tf.compat.v1.placeholder(shape=(None), dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(
            indices=self.actions,
            depth=self.action_size,
            dtype=tf.float32)
        self.target_v = tf.compat.v1.placeholder(
            shape=(None), dtype=tf.float32)
        self.advatanges = tf.compat.v1.placeholder(
            shape=(None), dtype=tf.float32)

        self.outputs = tf.reduce_sum(self.policy * self.actions_one_hot,
                                     axis=1)

        # Loss functions
        self.value_loss = .5 * tf.reduce_sum(
            tf.square(
                self.target_v - tf.reshape(self.value, (-1))
            )
        )
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = - tf.reduce_sum(
            tf.log(self.outputs) * self.advatanges
        )
        self.loss = .5 * self.value_loss + \
                self.policy_loss - self.entropy * 0.01

        # gradients
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)
        self.grads = tf.gradients(self.loss, self.local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(self.grads, 40.)

        # apply gradient to network
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
        self.apply_grads = optimizer.apply_gradients(
            zip(grads, global_vars)
        )

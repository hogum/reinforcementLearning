"""
    Holds the model
"""
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from utils import mse


@dataclass
class Model:
    """
        Creates the step and the training  model
    """
    policy: Any
    obsv_space: Any
    action_space: Any
    n_steps: Any
    n_envs: Any
    vf_coef: Any
    ent_coef: Any
    max_grad_norm: Any

    def __post_init__(self):
        self.build_model()

    def build_model(self):
        """
            Creates the model
        """
        sess = tf.get_default_session()
        actions = tf.compat.v1.placeholder(tf.int, [None], name='actions')
        advantages = tf.compat.v1.placeholder(
            tf.float32, [None], name='advantages')
        rewards = tf.compat.v1.placeholder(tf.float32, (None), name='rewards')
        lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        step_model = policy(sess, obsv_space, action_space,
                            n_envs, 1, reuse=False)
        train_model = policy(sess, obsv_space, action_space,
                             n_envs*n_steps, n_steps, reuse=True)

        # Loss = (Policy gradient loss - entropy) *
        #   (entropy_coeff + value_coeff  * value loss)

        # Output - log(pi)
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi,
            labels=actions)

        # 1/n * sum A(si,ai) * -logpi(si|ai)
        pg_loss = tf.reduce_mean(advantages * neglogpac)

        # 1/2 * sum[R - V(s)] ** 2
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), rewards))

        # Limits premature convergence to subotimal policy
        entropy = tf.reduce_mean(train_model.pd.entropy())
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        params = find_trainable_vars('model')
        grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            # clip grads [Noramalize]
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params)

        optimizer=tf.train.RMSPropOptimizer(learning_rate=lr,
            decay_rate=.99,
            epsilon=1e-5)
        train_=optimizer.apply_gradients(grads)

    def save(self):
        """
            Saves the Model
        """
        pass

    def train(self):
        """
            Trains the model
            - Feed forward and retropropagates gradients
        """
        pass

    def load(self):
        """
            Loads the saved model
        """
        pass

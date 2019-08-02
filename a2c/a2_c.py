"""
    Holds the model
"""
import tensorflow as tf

from .utils import mse, find_trainable_vars


class Model:
    """
        Creates the step and the training  model

        - save/load:  Saves/Loads the Model
        - train: Trains the model
                - Feed forward and retropropagates gradients
    """

    def __init__(self, policy,
                 obsv_space,
                 action_space,
                 n_steps,
                 n_envs,
                 vf_coef,
                 ent_coef,
                 max_grad_norm
                 ):

        sess = tf.get_default_session()
        actions = tf.compat.v1.placeholder(tf.int32, [None], name='actions')
        advantages = tf.compat.v1.placeholder(
            tf.float32, [None], name='advantages')
        rewards = tf.compat.v1.placeholder(
            tf.float32, (None), name='rewards')
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
        grads = list(zip(grads, params))

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=lr,
            decay=.99,
            epsilon=1e-5)
        train_ = optimizer.apply_gradients(grads)

        def save(path):
            """
                Saves Model
            """
            saver = tf.train.Saver()
            saver.save(sess, path)

        def train(states, actions_in, returns, values, l_r):
            """
                Trains the model
            """
            # Calculate adavantage A(s, a) = R + yV(s') - V(s)
            # returns = R + yV(s')
            advantages_ = returns - values

            td_map = {train_model.inputs: states,
                      actions: actions_in,
                      advantages: advantages_,
                      rewards: returns,
                      lr: l_r}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def load(path):
            """
                Loads saved model
            """
            saver = tf.train.Saver()
            saver.restore(sess, path)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.save = save
        self.load = load
        self.initial_state = step_model.initial_state
        self.value = step_model.value
        self.step = step_model.step
        tf.global_variables_initializer().run(session=sess)

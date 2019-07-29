"""
    Cartpole policy gradient implementation
"""
import numpy as np
import tensorflow as tf
import gym


class CartPole:
    """
        Cartpole V0
    """

    def __init__(self, lr=.002, gamma=.95):
        self.gamma = gamma
        self.action_space, self.observation_space = self.setup_env()
        self.setup_writer()

    def preprocess_rewards(self, rewards):
        """
            Discounts and normalized the rewards
        """
        discounted_rewds = np.zeros_like(rewards)
        cumulative = 0

        for i in range(len(rewards)):
            cumulative += self.gamma + rewards[i]
            discounted_rewds[i] = cumulative

        discounted_rewds = (discounted_rewds -
                            np.mean(discounted_rewds)) \
            / np.std(discounted_rewds)
        return discounted_rewds

    def setup_env(self):
        """
            Initializes the game environment
        """
        env = gym.make('CartPole-v0')
        env.unwrap
        env.seed(1)
        self.env = env

        return self.env.action_space.n, self.env.observation_space

    def build_model(self):
        """
            Builds the Policy Gradient model
        """
        with tf.name_scope('inputs'):
            self.inputs = tf.compat.v1.placeholder(
                tf.float32,
                (None, self.observation_space),
                name='inputs'
            )
            self.actions = tf.compat.v1.placeholder(
                tf.float32,
                (None, self.action_space),
                name='actions'
            )
            self.discounted_episode_rewards = tf.compat.v1.placeholder(
                tf.float32,
                (None, ),
                name='dicounted_episode_rewards'
            )
            self.mean_reward = tf.compat.vi.placeholder(
                tf.float32,
                name='mean_reward'
            )

            with tf.name_scope("fc_one"):
                fc_one = tf.contrib.layers.fully_connected(
                    inputs=self.inputs,
                    num_outputs=10,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc_two"):
                fc_two = tf.contrib.layers.fully_connected(
                    inputs=fc_one,
                    num_outputs=self.action_space,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc_three"):
                fc_three = tf.contrib.layers.fully_connected(
                    inputs=fc_two,
                    num_outputs=self.action_space,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(
                    fc_three)

            with tf.name_scope("loss"):
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=fc_three, labels=self.actions)
                self.loss = tf.reduce_mean(
                    neg_log_prob * self.discounted_episode_rewards)

            with tf.name_scope("train"):
                self.optimizer = tf.train.AdamOptimizer(
                    self.lr).minimize(self.loss)

    def train(self, episodes=100):
        """
            Trains the agent
        """
        self.saver = tf.train.Saver()
        total_rewards = []

        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for episode in range(episodes):
                episode_states, episode_actions, episode_rewards = [], [], []
                state = self.env.reset()
                self.env.render()

                while True:
                    action_prob_distribution = sess.run(
                        self.action_distribution,
                        feed_dict={
                            self.inputs: state.reshape((1, 4))}
                    )
                    action = np.random.choice(range(
                        action_prob_distribution.shape[1]),
                        p=action_prob_distribution.ravel())
                    new_state, reward, done, info = self.env.step(action)
                    episode_states += [state]
                    action_s = np.zeros(self.action_space)
                    action_s[action] = 1
                    episode_actions += [action_s]
                    episode_rewards += [reward]

                    if done:
                        total_rewards += [episode_rewards]
                        rewards_ = [np.sum(x) for x in total_rewards]
                        mean_reward = np.divide(np.sum(rewards_), episode + 1)
                        max_reward = np.amax(rewards_)

                        print('\n############################' +
                              f'Episode: {episode}' +
                              f'reward: {rewards_[episode]}' +
                              f'mean r: {mean_reward}' +
                              f'max r: {max_reward}')
                        disc_rewards = self.preprocess_rewards(episode_rewards)
                        res = dict(states=episode_states,
                                   actions=episode_actions,
                                   disc_rewards=disc_rewards,
                                   mean=mean_reward
                                   )

                        self.feed_forward(sess, res)
                        self.feed_forward(sess, inputs=self.writer_op,
                                          output=res, forward=False,
                                          episode=episode)
                        break

                    state = new_state
                self.save(sess, episode)

    def feed_forward(self, sess, inputs=[], output={}, **kwargs):
        """
            Feeds inputs across the nn nodes,
            finds gradient and backpropagates
        """
        feeds = {
            self.inputs: np.vstack(np.array(output.get('states'))),
            self.actions: np.vstack(np.array(output.get('actions'))),
            self.discounted_episode_rewards: output.get('disc_rewards'),
            self.mean_reward: output.get('mean')
        }
        feeds.pop('mean') if kwargs.get('forward') else None
        res = sess.run(
            inputs or [self.loss, self.optimizer],
            feed_dict=feeds
        )

        return res[0] if kwargs.get(
            'forward') else self.summarize(sess,
                                           res,
                                           kwargs.get('episode'))

    def setup_writer(self):
        """
            Sets up tensorboard
        """
        self.writer = tf.compat.v1.summary.FileWriter(
            '/root/tensorboard/policy_g/1')
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('reward_mean', self.mean_reward)
        self.writer_op = tf.compat.v1.summary.merge_all()

    def summarize(self, sess, summary, episode):
        """
            Writes tf summaries
        """
        self.writer.add_summary(summary, episode)
        self.writer.flush

    def save(self, sess, count):
        """
            Saves model checkpoints
        """
        if not count % 50:
            self.saver.save(sess, '.models/model.ckpt')
            print('Saved')

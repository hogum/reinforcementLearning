"""
    Monte Carlo Policy Gradient implementation in
    doom Health Gathering scene
"""

import os
from collections import deque
from dataclasses import dataclass, field

import tensorflow as tf
import numpy as np

import vizdoom as vz
from skimage import transform


resolution = (100, 160)
stack_size = 4

STACKED_FRAMES_ = deque(
    [np.zeros(resolution, dtype=np.int) for _ in range(stack_size)],
    maxlen=4)


def create_env(visible=False):
    """
        Creates an instance of the game environment
    """
    path = '/usr/local/lib/python3.7/dist-packages/vizdoom/scenarios/'

    doom = vz.DoomGame()
    doom.load_config(os.path.join(path, 'defend_the_center.cfg'))
    doom.set_doom_scenario_path(os.path.join(path, 'defend_the_center.wad'))

    doom.set_window_visible(visible)
    doom.init()

    actions = np.identity(doom.get_available_buttons_size())

    return doom, actions


def preprocess_frame(frame):
    """
        Preprocess the screen buffer for reduced training time
    """
    try:
        frame = np.array(frame[0] + frame[1] + frame[2])[40:, :]

    except IndexError:
        frame = frame
    frame = frame / 255

    return transform.resize(frame, resolution)


def get_state_size():
    """
        Gives the size of the state (height, width, channels)
    """
    return [*resolution, stack_size]


def stack_frames(state, stacked_frames=None, new_episode=False):
    """
        Creates a stack of four frames from previous
        state for a sense of `motion`
    """
    frame = preprocess_frame(state)

    if new_episode:
        stacked_frames = deque(
            [np.array(frame) for _ in range(stack_size)], maxlen=4)
    else:
        stacked_frames.append(frame)
    state = np.stack(stacked_frames, axis=2)

    return state, stacked_frames


@dataclass
class DoomPG:
    """
        Deep Q Network model for doom.

        Parameters
        ----------
        lr: float
            Learning rate
        gamma: float
            Discounting factor for future rewards
        eps: float
            Explore-exploit tradeoff for agent actions
        min_eps: float
            Minimum value for epsilon
        max_eps: float
            Maxumum value for epsilon
        name: str, default = 'DoomDqNet'
            Variable for tf namescope
        state_size: list, default = [100, 120, 4]
            Shape of input stack
        max_tau: int
            Max C step in updating the target network
    """
    lr: int = 0.0002
    gamma: float = 0.99
    name: str = 'DoomPG'
    state_size: list = field(default_factory=get_state_size)
    action_size = 3  # Left, Right, move Forward

    def __post_init__(self):
        self.build_model()
        self.setup_writer()

    def build_model(self):
        """
            Builds the Policy Gradient Neural Net
        """
        with tf.variable_scope(self.name):
            with tf.name_scope('inputs'):
                self.inputs = tf.compat.v1.placeholder(
                    tf.float32,
                    (None, *self.state_size),
                    name='inputs'
                )

                self.actions = tf.compat.v1.placeholder(
                    tf.float32,
                    (None, self.action_size),
                    name='actions'
                )
                self.discounted_eps_rw = tf.compat.v1.placeholder(
                    tf.float32,
                    (None,),
                    name='discounted_episode_rewds'
                )
                self.mean_reward = tf.compat.v1.placeholder(
                    tf.float32,
                    name='mean_reward')

                with tf.name_scope('conv_one'):
                    conv_one = tf.layers.conv2d(
                        inputs=self.inputs,
                        filters=32,
                        kernel_size=[8, 8],
                        strides=(4, 4),
                        padding='valid',
                        kernel_initializer=tf.contrib.
                        layers.xavier_initializer_conv2d(),
                        name='conv_one'
                    )
                    conv_one_bn = tf.layers.batch_normalization(
                        conv_one,
                        training=True,
                        epsilon=1e-5,
                        name='batch_norm_one'
                    )
                    conv_one_out = tf.nn.relu(conv_one_bn, name='conv_one_out')
                with tf.name_scope('conv_two'):
                    conv_two = tf.layers.conv2d(
                        inputs=conv_one_out,
                        filters=64,
                        kernel_size=[4, 4],
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer=tf.contrib.
                        layers.xavier_initializer_conv2d(),
                        name='conv_two'
                    )

                    conv_two_bn = tf.layers.batch_normalization(
                        conv_two,
                        training=True,
                        epsilon=1e-5,
                        name='batch_norm_two'
                    )
                    conv_two_out = tf.nn.relu(conv_two_bn, name='conv_two_out')
                with tf.name_scope('conv_three'):

                    conv_three = tf.layers.conv2d(
                        inputs=conv_two_out,
                        filters=128,
                        kernel_size=[4, 4],
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer=tf.contrib.
                        layers.xavier_initializer_conv2d(),
                        name='conv_three'
                    )

                    conv_three_bn = tf.layers.batch_normalization(
                        conv_three,
                        training=True,
                        epsilon=1e-5,
                        name='batch_norm_three'
                    )
                    conv_three_out = tf.nn.relu(
                        conv_three_bn, name='conv_three_out')
                with tf.name_scope('flatten'):
                    flatten = tf.layers.flatten(conv_three_out)
                with tf.name_scope('fc'):
                    fc = tf.layers.dense(
                        inputs=flatten,
                        units=512,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.
                        xavier_initializer(),
                        name='fc'
                    )
                with tf.name_scope('logits'):
                    logits = tf.layers.dense(
                        inputs=fc,
                        units=3,
                        kernel_initializer=tf.contrib.layers.
                        xavier_initializer(),
                        activation=None,
                        name='logits'
                    )
                with tf.name_scope('activation'):
                    self.action_distribution = tf.nn.softmax(logits)
                with tf.name_scope('loss'):
                    neg_log_probs = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits,
                        labels=self.actions
                    )
                    self.loss = tf.reduce_mean(
                        neg_log_probs * self.discounted_eps_rw)
                with tf.name_scope('optimizer'):
                    self.optimizer = tf.train.AdamOptimizer(
                        self.lr).minimize(self.loss)

    def preprocess_rewards(self, rewards):
        """
            Discounts and normalized the rewards
        """
        discounted_rewds = np.zeros_like(rewards)
        cumulative = 0

        for i in reversed(range(len(rewards))):
            cumulative *= self.gamma
            discounted_rewds[i] = cumulative + rewards[i]

        discounted_rewds = (discounted_rewds -
                            np.mean(discounted_rewds)) \
            / np.std(discounted_rewds)
        return discounted_rewds

    def setup_writer(self):
        """
            Sets up the tensorboard writer
        """
        self.writer = tf.compat.v1.summary.FileWriter(
            '/root/tensorboard/policy_g/doom/defend')
        tf.compat.v1.summary.scalar('Loss', self.loss)
        tf.compat.v1.summary.scalar('reward_mean', self.mean_reward)
        self.writer_op = tf.compat.v1.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, batch_size=1000, n_epochs=500, training=True):
        """
            Trains the agent
        """
        epoch = 1
        total_rewards, mean_rewards,  = [], []

        if not training:
            return

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            while epoch < n_epochs + 1:
                batches, n_episode = self.create_batches(sess, batch_size)
                states_b, actions_b, batch_rewards, disc_rewards_b = batches
                summed_batch_rws = np.sum(batch_rewards)

                total_rewards.append(summed_batch_rws)
                max_batch_rw = np.amax(total_rewards)
                batch_mean = np.divide(summed_batch_rws, n_episode)
                mean_rewards += [batch_mean]

                av_training_rewd = np.divide(np.sum(mean_rewards),
                                             epoch)

                print(f'\n\nEpoch {epoch}\n\n' +
                      f'training episodes: {n_episode} ' +
                      f'reward: {summed_batch_rws} ' +
                      f'mean r: {batch_mean} ' +
                      f'max r: {max_batch_rw} ' +
                      f'average tr: {av_training_rewd}' +
                      '\n'
                      )

                batches_ = dict(
                    states=states_b,
                    actions=actions_b,
                    disc_rewards=disc_rewards_b,
                    mean=batch_mean
                )
                loss = self.feed_forward(
                    sess, output=batches_, forward=True)
                print(f'Loss: {loss}')
                self.feed_forward(sess, inputs=self.writer_op,
                                  output=batches_, forward=False,
                                  episode=epoch)
                self.save(sess, epoch)
                epoch += 1

    def feed_forward(self, sess, inputs=[], output={}, **kwargs):
        """
            Feeds inputs across the nn nodes,
            finds gradient and backpropagates
        """
        feeds = {
            self.inputs: output.get('states').reshape((
                len(output.get('states')), *self.state_size)),
            self.actions: output.get('actions'),
            self.discounted_eps_rw: output.get('disc_rewards'),
            self.mean_reward: output.get('mean')
        }

        try:  # Cant boolean tensor, meaning not None already
            inputs = [self.loss, self.optimizer] if not inputs else inputs
        except TypeError:
            pass

        _ = feeds.pop(self.mean_reward) if kwargs.get('forward') else None
        res = sess.run(inputs,
                       feed_dict=feeds
                       )

        return res[0] if kwargs.get(
            'forward') else self.summarize(sess,
                                           res,
                                           kwargs.get('episode'))

    def summarize(self, sess, summary, episode):
        """
           Writes tf summaries
        """
        self.writer.add_summary(summary, episode)
        self.writer.flush()

    def save(self, sess, epoch, interval=2):
        """
            Saves the model checkpoints
        """
        if not epoch % interval:
            self.saver.save(sess, '.models/defend/doom_pg.ckpt')

    def create_batches(self, sess, batch_size):
        """
            Gives outputs resulting from performing actions
            in a preset batch size
        """

        states, actions, episode_rewards = [], [],  []
        disc_rewards, batch_rewards, = [], []
        self.game, self.actions_choice = create_env()
        episode = 1
        self.game.new_episode()

        state = self.game.get_state().screen_buffer
        state, stacked_frames = stack_frames(state, new_episode=True)

        while True:
            action_prob = sess.run(self.action_distribution,
                                   feed_dict={
                                       self.inputs: state.reshape(
                                           (1, *self.state_size))
                                   })
            action = np.random.choice(
                range(action_prob.shape[1]),
                p=action_prob.ravel()
            )
            action = self.actions_choice[action].tolist()
            reward = self.game.make_action(action)
            states += [state]
            actions += [action]
            episode_rewards += [reward]
            done = self.game.is_episode_finished()

            if done:
                next_state = np.zeros(resolution, dtype=np.int)
                next_state, stacked_frames = stack_frames(
                    next_state, stacked_frames)
                batch_rewards += [episode_rewards]

                disc_rewards.append(
                    self.preprocess_rewards(episode_rewards))
                if len(np.concatenate(batch_rewards)) > batch_size:
                    break

                episode_rewards.clear()
                self.game.new_episode()
                episode += 1
                self.game.new_episode()
                state = self.game.get_state().screen_buffer
                state, stacked_frames = stack_frames(state,
                                                     new_episode=True)
            else:
                next_state = self.game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(next_state,
                                                          stacked_frames)
                state = next_state
        return (np.stack(np.asarray(states)),
                np.stack(np.asarray(actions)),
                np.concatenate(batch_rewards),
                np.concatenate(disc_rewards)), episode

    def play(self, episodes):
        """
            Plays trained agent
        """
        with tf.compat.v1.Session() as sess:
            game, actions_choice = create_env(visible=True)
            self.saver.restore(sess, '.models/defend/doom_pg.ckpt')

            for episode in range(episodes):
                game.new_episode()

                state = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(state, new_episode=True)

                while not game.is_episode_finished():
                    action_probability = sess.run(
                        self.action_distribution,
                        feed_dict={self.inputs: state.reshape(
                            (1, *self.state_size))}
                    )
                    action = np.random.choice(
                        range(action_probability.shape[1]),
                        p=action_probability.ravel())
                    action = actions_choice[action].tolist()

                    game.make_action(action)
                    done = game.is_episode_finished()

                    if not done:
                        next_state = game.get_state().screen_buffer
                        next_state, stacked_frames = stack_frames(
                            next_state, stacked_frames)
                        state = next_state
                    else:
                        break
                print(f'Episode {episode} ' +
                      f'score: {game.get_total_reward()}'
                      )
            game.close()


if __name__ == '__main__':
    game = DoomPG(lr=.0001, gamma=.95)
    game.train(n_epochs=30000, training=True)
    game.play(episodes=20)
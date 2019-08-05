"""
    Worker Agent
"""
import tensorflow as tf

from helpers import create_env, get_state_size, update_target_graph
from ac_network import AC_Network


class Worker:
    """
        Interacts with the environment and updates
        the global variables
    """

    def __init__(self, agent_number, optimizer, save_path, episodes):
        self.number = agent_number
        self.name = f'agent {agent_number}'
        self.save_path = save_path

        self.global_eps = episodes
        self.episode_rewards = []
        self.reward_mean = []
        self.episode_len = []

        self.game, self.actions = create_env()
        self.action_size = self.actions.shape[0]
        self.state_size = get_state_size()

        self.setup_writer()
        self.create_net(optimizer)

    def create_net(self, optimizer):
        """
            Creates a local copy of the network and operations
            to copy global parameters to the network
        """

        self.local_ac = AC_Network(
            state_size=self.state_size,
            action_size=self.action_size,
            trainer=optimizer,
            name=self.name
        )
        self.updated_ops = update_target_graph(from_scope='global',
                                               worker_name=self.name)

    def setup_writer(self):
        """
            Sets up the tensorboard writer
        """
        self.writer = tf.compat.v1.summary.FileWriter(
            '/root/tensorboard/a3c/doom/1')
        tf.compat.v1.summary.scalar('Loss', self.local_ac.loss)
        tf.compat.v1.summary.scalar('reward_mean', self.reward_mean)
        self.writer_op = tf.compat.v1.summary.merge_all()
        self.saver = tf.train.Saver()

    def save(self, sess, episode, interval=5):
        """
            Saves model checkpoints
        """
        if not episode % interval:
            self.saver.save(sess, '.models/doom/model.ckpt')

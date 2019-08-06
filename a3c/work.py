"""
    Holds the Global network and agent workers.
    Asynchronous co-ordination of the workers
"""
import multiprocessing
import threading
import time

import tensorflow as tf
import numpy as np

from ac_network import AC_Network
from helpers import get_state_size
from worker import Worker

ACTION_SIZE = 3


class Async:
    """
        Coordinates the worker processes
    """

    def __init__(self, lr=1e-4):
        with tf.device('/cpu:0'):
            global_episodes = tf.Variable(
                0, dtype=tf.int32, name='global_episodes', trainable=False)
            optimizer = tf.train.AdamOptimizer(lr)
            state_size = np.prod(get_state_size())
            _ = AC_Network(state_size=state_size,
                           action_size=ACTION_SIZE,
                           name='global'
                           )

            n_workers = multiprocessing.cpu_count()
            self.workers = [Worker(agent_number=i,
                                   optimizer=optimizer,
                                   save_path='.model',
                                   gamma=lr,
                                   episodes=global_episodes)
                            for i in range(n_workers)
                            ]

    def work(self, max_episodes=100, load_model=True):
        """
           Threads present workers to run asynchronously
        """
        with tf.Session() as sess:
            coordinator = tf.train.Coordinator()
            saver = tf.train.Saver()
            if load_model:
                model = tf.train.get_checkpoint_state('.model')
                saver.restore(sess, model.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            threads = []
            for worker in self.workers:
                t = threading.Thread(target=worker.work(
                    sess,
                    max_eps_len=max_episodes,
                    coord=coordinator,
                    saver=saver))
                t.start()
                time.sleep(.5)
                threads.append(t)
            coordinator.join(threads)

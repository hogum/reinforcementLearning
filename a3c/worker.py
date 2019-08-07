"""
    Worker Agent
"""
import tensorflow as tf
import numpy as np

from helpers import (create_env, get_state_size,
                     update_target_graph, preprocess_rewards,
                     preprocess_frame, create_gifs)
from ac_network import AC_Network


class Worker:
    """
        Interacts with the environment and updates
        the global variables
    """

    def __init__(self, agent_number, optimizer, save_path, gamma=2e-4,
                 episodes=None):
        self.number = agent_number
        self.name = f'agent_{agent_number}'
        self.save_path = save_path

        self.gamma = gamma
        self.global_eps = episodes
        self.episode_rewards = []
        self.reward_mean = []
        self.episode_len = []

        self.game, self.actions = create_env()
        self.action_size = self.actions.shape[0]
        self.state_size = np.prod(get_state_size())  # [84, 84, 1]

        self.create_net(optimizer)
        self.setup_writer()

        self.first_worker = True if self.name == 'agent_0' else False
        self.increament = self.global_eps.assign_add(1)

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

    def train(self, sess, roll_out, bootstrap_value):
        """
            Train the agent
            : Generates advantage and discounted rewards
               and updates the global network
        """
        roll_out = np.asarray(roll_out)
        states = roll_out[:, 0]
        actions = roll_out[:, 1]
        rewards = roll_out[:, 2]
        values = roll_out[:, 5]

        # Find advantage and discounted returns from rewards
        summed_rewards = np.asanyarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = preprocess_rewards(
            summed_rewards, self.gamma)[:-1]
        summed_value = np.asanyarray(values.tolist() + [bootstrap_value])

        advantages = rewards + self.gamma * \
            summed_value[1:] - summed_value[:-1]
        advantages = preprocess_rewards(advantages, self.gamma)

        # Update global network with gradients from loss and save
        feed_dict = {
            self.local_ac.target_v: discounted_rewards,
            self.local_ac.inputs: np.vstack(states),
            self.local_ac.advantages: advantages,
            self.local_ac.actions: actions,
            self.local_ac.state_in[0]: self.batch_rnn_state[0],
            self.local_ac.state_in[1]: self.batch_rnn_state[1],
        }
        * losses_n_norms, self.batch_rnn_state, _ = sess.run(
            fecthes=[self.local_ac.value_loss,
                     self.local_ac.policy_loss,
                     self.local_ac.entropy,
                     self.local_ac.grad_norms,
                     self.local_ac.var_norms,
                     self.local_ac.state_out,
                     self.local_ac.apply_grads
                     ],
            feed_dict=feed_dict)
        value_loss, policy_loss, entr_loss, \
            grad_norms, var_norms = losses_n_norms
        len_rout = len(roll_out)

        return (value_loss / len_rout,
                policy_loss / len_rout,
                entr_loss / len_rout,
                grad_norms, var_norms)

    def work(self, sess, max_eps_len, coord, buff_size=30, saver=None):
        """
            Interacts with the agent's own copy of environment
            to collect experience
        """
        n_episode = sess.run(self.global_eps)
        steps = 0
        self.saver = saver if saver else self.saver

        print(f'Starting worker {self.number}')

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.updated_ops)
                episode_buffer, episode_values, episode_frames = [], [], []
                episode_reward, episode_step, = 0, 0

                self.game.new_episode()
                done = self.game.is_episode_finished()
                state = self.game.get_state().screen_buffer
                episode_frames += [state]
                state = preprocess_frame(state)

                rnn_state = self.local_ac.state_in
                self.batch_rnn_state = rnn_state

                while not self.game.is_episode_finished():
                    # Take action using probabilities from policy net output
                    a_distribution, value, rnn_state = sess.run(
                        [self.local_ac.policy,
                         self.local_ac.value,
                         self.local_ac.state_out],
                        feed_dict={self.local_ac.inputs: [state],
                                   self.local_ac.state_in[0]: rnn_state[0],
                                   self.local_ac.state_in[1]: rnn_state[1]
                                   }
                    )
                    action = np.random.choice(
                        a_distribution[0], p=a_distribution[0])
                    action = np.argmax(a_distribution == action)

                    reward = self.game.make_action(
                        self.actions[action].tolist()) / 100
                    done = self.game.is_episode_finished()

                    if not done:
                        next_state = self.game.get_state().screen_buffer
                        episode_frames += [next_state]
                        next_state = preprocess_frame(next_state)
                    else:
                        next_state = state

                    episode_buffer.append([state, action,
                                           reward, next_state,
                                           done, value[0, 0]])
                    episode_values += [value[0, 0]]
                    episode_reward += reward
                    steps += 1
                    episode_step += 1

                    state = next_state

                    checks = [done, n_episode >= max_eps_len]
                    if len(episode_buffer) >= buff_size and not any(checks):
                        # Update step using experience mini batch
                        # Bootstrap final return from current value etimation
                        value_1 = sess.run(
                            self.local_ac.value,
                            feed_dict={
                                self.local_ac.inputs: [state],
                                self.local_ac.state_in[0]: rnn_state[0],
                                self.local_ac.state_[1]: rnn_state[1],
                            })[0, 0]
                        * losses, grad_norm, var_norm = self.train(
                            sess, episode_buffer, value_1)
                        episode_buffer.clear()
                        sess.run(self.updated_ops)
                    if done:
                        break
                self.episode_rewards.append(episode_reward)
                self.episode_len.append(episode_step)
                self.reward_mean.append(np.mean(episode_values))

                if episode_buffer:  # Update network at end of episode
                    *losses, grad_norm, var_norm = self.train(
                        sess, episode_buffer, 0.)
                self.save(sess, n_episode,
                          outputs=(losses, (grad_norm, var_norm)),
                          frames=episode_frames)
                if self.first_worker:
                    sess.run(self.increament)
                n_episode += 1

    def setup_writer(self):
        """
            Sets up the tensorboard writer
        """
        self.writer = tf.compat.v1.summary.FileWriter(
            f'/root/tensorboard/a3c/{self.number}')
        tf.compat.v1.summary.scalar('Loss', self.local_ac.loss)
        tf.compat.v1.summary.scalar('reward_mean', self.reward_mean)
        self.writer_op = tf.compat.v1.summary.merge_all()
        self.saver = tf.train.Saver()

    def save(self, sess, episode, interval=5, frame_intv=20, **kwargs):
        """
            Saves model checkpoints
        """
        outputs = kwargs.get('outputs')
        frames = kwargs.get('frames')

        if not episode % interval:
            if self.name == 'agent 0' and not episode % frame_intv:
                time_per_step = .05
                images = np.asanyarray(frames)
                create_gifs(images, f'.frames/image_{episode}.gif',
                            duration=len(images)*time_per_step,
                            true_image=True,
                            salient=False)
            self.saver.save(sess, self.save_path + f'/model-{episode}.ckpt')
            print('model saved')
            self.summarize(episode, outputs)

    def summarize(self, episode, outputs):
        """
            Writes tf summaries
        """
        losses, norms = outputs
        value_loss, policy_loss, entr_loss = losses
        grad_norm, var_norm = norms

        reward_mean = np.mean(self.episode_rewards[-5:])
        mean_len = np.mean(self.episode_len[-5:])
        mean_value = np.mean(self.reward_mean[-5:])

        summary = tf.Summary()
        summary.value.add(tag='loss/value', simple_value=float(value_loss))
        summary.value.add(tag='loss/policy', simple_value=float(policy_loss))
        summary.value.add(tag='loss/entropy', simple_value=float(entr_loss))
        summary.value.add(tag='loss/grad_norm', simple_value=float(grad_norm))
        summary.value.add(tag='loss/var_norm', simple_value=float(var_norm))

        summary.value.add(tag='performance/reward',
                          simple_value=float(reward_mean))
        summary.value.add(tag='performance/len', simple_value=float(mean_len))
        summary.value.add(tag='performance/value',
                          simple_value=float(mean_value))

        self.writer.add_summary(summary, episode)
        self.writer.flush()

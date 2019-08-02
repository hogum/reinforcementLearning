import numpy as np

from baselines.common.runners import AbstractEnvRunner

from .utils import swap_01


class Runner(AbstractEnvRunner):
    """
        Creates a mini batch of experiences

    """

    def __init__(self, env, model, n_steps, total_timesteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=n_steps)
        self.gamma = gamma

        # Used in General Advantage Estimation
        self.lam = lam

        # Total timestamps taken
        self.total_timesteps = total_timesteps

    def run(self):
        """
            Makes a mini batch of experiences
        """
        obs_mb, actions_mb, rewards_mb, values_mb = [], [], [], []
        dones_mb = []

        for _ in range(self.n_steps):
            # Observations present - AbstractEnv runs self.obs[:] = env.reset()
            # Given observations, take action and value V(s)
            actions, values = self.env.step(self.obs, self.dones)
            obs_mb.append(np.copy(self.obs))
            values_mb.append(values)
            actions_mb.append(actions)
            dones_mb.append(self.dones)

            # Take actions in the env and observe results
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)

            rewards_mb.append(rewards)

        obs_mb = np.asarray(obs_mb, dtype=np.uint8)
        actions_mb = np.asarray(actions_mb, dtype=np.int32)
        rewards_mb = np.asarray(rewards_mb, dtype=np.float32)
        values_mb = np.asarray(values_mb, dtype=np.float32)
        dones_mb = np.asarray(dones_mb, dtype=np.bool)

        final_values = self.model.value(self.obs)

        # Generalized Advantage Estimation
        mb_returns = np.zeros_like(rewards_mb)  # Advantage + value
        mb_advantages = np.zeros_like(rewards_mb)

        lastgaelam = 0

        for s in reversed(range(self.n_steps)):
            if s == self.n_steps - 1:
                # If a state is done next non-terminal = 0
                #   # delta = R - V(s) [
                #           since gamma * nextvalues * nextnon-terminal = 0]
                # else
                #   delta = R + gamma + V(st+1)

                next_nonterminal = 1. - self.dones

                # V(t+1)
                next_values = final_values
            else:
                next_nonterminal = 1. - dones_mb[s + 1]
                next_values = values_mb[s + 1]

                # delta = R(st) + gamma * V(t + 1) * nextnonterminal - V(s)
                delta = rewards_mb[s] + self.gamma * \
                    next_values * next_nonterminal - values_mb

                # advantage = delta +
                #       gamma * lambda * nextnonterminal * lastgaelam
                mb_advantages[s] = lastgaelam = delta + \
                    self.gamma * self.lam * next_nonterminal * lastgaelam

            mb_returns = mb_advantages + values_mb

            return map(swap_01, (obs_mb, actions_mb, mb_returns, values_mb))

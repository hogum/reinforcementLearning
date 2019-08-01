from baselines.common.runners import AbstractEnvRunner


class Runner:
    """
        Creates a mini batch of experiences

    """

    def __init__(self, env, model, n_steps, total_timestamps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=n_steps)
        self.gamma = gamma

        # Used in General Advantage Estimation
        self.lam = lam

        # Total timestamps taken
        self.total_timestamps = total_timestamps

    def run(self):
        """
            Makes a mini batch of experiences
        """
        obs_mb, actions_mb, rewards_mb, values_mb = [], [], [], []
        dones_mb = []

        for step in range(n_steps):
            # Observations present - AbstractEnv runs self.obs[:] = env.reset()
            # Given observations, take action and value V(s)
            actions, values = self.env.step(self.obs, self.dones)

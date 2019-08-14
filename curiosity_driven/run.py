"""
    Entry
"""
import gym

from arch import Curiosity


def run():
    """
        Instantiates and trains the Curiosity model
    """
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    cr = Curiosity(n_states=env.observation_space.shape[0],
                   n_actions=env.action_space.n)
    cr.train(env, n_episodes=100)


if __name__ == '__main__':
    run()

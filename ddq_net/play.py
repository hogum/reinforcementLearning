"""
    This module holds functions to run the Double Duelling Deep Q Net
"""

from ddqn import DoomDDdqN


def main():
    """
        Creates an instance of dddqn for training and play
    """
    clf = DoomDDdqN()
    clf.prepopulate(episodes=10000)
    clf.train(episodes=5000, batch_size=64, max_steps=500)
    clf.play()


if __name__ == '__main__':
    main()

"""
    This module holds functions to run the Double Duelling Deep Q Net
"""

from ddqn import DoomDDdqN


def main():
    """
        Creates an instance of dddqn for training and play
    """
    clf = DoomDDdqN()
    clf.prepopulate(episodes=1000)
    clf.train(episodes=64, batch_size=64, max_steps=100)
    # clf.play()


if __name__ == '__main__':
    main()

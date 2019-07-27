"""
    This module holds functions to run the Double Duelling Deep Q Net
"""

from ddqn import DoomDDdqN


def main():
    """
        Creates an instance of dddqn for training and play
    """
    clf = DoomDDdqN()
    clf.prepopulate(episodes=3)
    clf.train(episodes=3, batch_size=4, max_steps=10)
    # clf.play()


if __name__ == '__main__':
    main()

"""
    Holds the classes that create the memory instance
    for PER
"""
import numpy as np


class SumTree:
    """
        Binary tree with a max of 2 children for each node
        : Leaves hold priorities and a data array whose index
        points to the index of the leaves

        Parameters
        ----------
        capacity: int
            Number of leaf nodes
    """

    def __init__(self, capacity):
        self.pointer = 0
        self.capacity = capacity
        # leaf_nodes = capacity
        self.parent_nodes = capacity - 1
        self.tree = np.zeros(2 * self.parent_nodes)
        self.data = np.zeros(capacity, dtype=object)  # Expriences

    @property
    def root(self):
        return self.tree[0]

    def add(self, priority, items):
        """
            Adds priority score and items to data
        """
        # Update frame
        self.data[self.pointer] = items

        self.update(priority)
        self.pointer += 1

        if self.pointer >= self.capacity:
            self.pointer = 0

    def update(self, priority):
        """
            Updates the priority score, propagating
            the change throughout the tree
        """
        tree_idx = self.pointer + self.parent_nodes
        # change: = New priority score - previous score
        change = priority - self.tree[tree_idx]

        while tree_idx:
            tree_idx = (tree_idx - 1) / 2
            self.tree[tree_idx] += change

    def pluck_leaf(self, loc):
        """
            Fetches the leaf index, priority value and experience
            at this index
        """
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if loc <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    loc -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

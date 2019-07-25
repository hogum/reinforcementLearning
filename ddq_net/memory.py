"""
    Holds the classes that create the memory instance
    for PER
"""
import numpy as np

from dataclasses import dataclass, field


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
        """
            Root node
        """
        return self.tree[0]

    def add(self, priority, items):
        """
            Adds priority score and items to data
        """
        # Update frame
        self.data[self.pointer] = items
        # Index to place experience
        tree_idx = self.pointer + self.parent_nodes

        self.update(tree_idx, priority)
        self.pointer += 1

        if self.pointer >= self.capacity:
            self.pointer = 0

    def update(self, tree_idx, priority):
        """
            Updates the priority score, propagating
            the change throughout the tree
        """

        # change: = New priority score - previous score
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
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


@dataclass
class Memory:
    """
        Stores experiences from actions taken by agent

        Parameters:
        ----------
        capacity: int
            Max number of experiences the memory can hold
        per_e: float
            Ensures low priority experiences have some probability
            of being selected [Not None]
        per_a: float
            Tradesoff between random sampling and sampling of high
            priority experiences
        per_b: float
            Importance Sampling initial value
        per_b_inc: float
            Increament value for Importance Sampling(IS)
        per_b_max: float
            Maximum value for IS
        abs_err_max: float
            Clipped absolute error
    """
    capacity: int = 1000
    per_e: int = .01
    per_a: float = .6
    per_b: float = .4
    per_b_inc: float = .001
    per_b_max: float = 1.
    abs_err_max: float = 1.
    tree = SumTree(capacity)

    def add(self, exp):
        """
            Adds a new experience to the memory tree
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if not max_priority:
            max_priority = self.abs_err_max
        self.tree.add(max_priority, exp)

    def sample(self, s_size):
        """
            Samples a mini batch of experiences of size s_size
            from the memory tree
             - Subdivides the range [0.. priority_total] into s_size ranges
               and uniformly samples a priority from each
             - Exprerinces are retrieves using the sample values
             - IS weights for each element in the mini batch are calculated
        """
        mini_batch = []
        batch_idx = np.empty((s_size), dtype=np.int32)
        b_IS_weights = np.empty((s_size, 1), dtype=np.float32)

        # Priotity segement [0.. priority_max]
        priority_segment = self.tree.root / s_size
        self.per_b = np.min([self.per_b_max, self.per_b + self.per_b_inc])
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root
        max_weight = (p_min * s_size)**-self.per_b

        for i in range(s_size):
            sample_p = np.random.uniform(
                low=priority_segment * i, high=priority_segment * (i + 1))
            idx, priority, exprns = self.tree.pluck_leaf(sample_p)

            # P(j)
            sampling_probabilities = priority / self.tree.root

            # IS weights = (1/N * 1/ P(i)) ** b / max_weight_idx
            # := N *P(i)** -b / max_weight_idx
            b_IS_weights[i, 0] = np.power(
                s_size * sampling_probabilities, -self.per_b) / max_weight
            batch_idx[i] = idx
            experience = [exprns]
            mini_batch.append(experience)

        return batch_idx, mini_batch, b_IS_weights

    def update_priorities(self, tree_idx, abs_errors):
        """
            Updates priorities on the tree
        """
        abs_errors += self.per_e
        clipped_errs = np.min(abs_errors, self.abs_err_max)
        ps = np.power(clipped_errs, self.per_a)

        for tree_idx_, pr in zip(tree_idx, ps):
            self.tree.update(tree_idx_, pr)

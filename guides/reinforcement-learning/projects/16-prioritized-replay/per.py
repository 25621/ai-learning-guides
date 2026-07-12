"""Prioritized experience replay, built on a sum-tree.

Uniform replay samples every stored transition with equal probability. In a
sparse-reward task that is a terrible bet: the one transition that actually paid
out is one row in a hundred thousand, and it gets picked as often as the ninety-
nine thousand boring ones around it.

PER samples transition i with probability proportional to p_i^alpha, where p_i is
its last-seen |TD error| -- "how wrong was I about this one?". That biases the
sampler toward the transitions that still have something to teach.

Two costs come with it, and both are paid here:

  * the SAMPLER must draw from a changing, non-uniform distribution, fast. Naively
    that is O(n) per draw. A sum-tree makes it O(log n): each leaf holds a
    priority, each internal node holds the sum of its children, so a draw is a
    single root-to-leaf descent guided by one uniform number.

  * the DISTRIBUTION is now wrong. Learning from a skewed sample estimates a
    skewed expectation, so each sampled transition's loss is reweighted by an
    importance-sampling weight w_i = (1 / (N * P(i)))^beta, annealed from beta_0
    to 1 over training. beta = 0 ignores the skew entirely; beta = 1 corrects it
    fully, and full correction matters most at the end, when the network is close
    to right and a biased gradient does the most damage.
"""

import numpy as np
import torch

from dqn_lib import Batch


class SumTree:
    """A complete binary tree over `capacity` leaves; each node stores the sum of
    its subtree. Point-update and prefix-sum-search are both O(log n).

    Layout is the standard flat one: node 1 is the root, node i's children are 2i
    and 2i+1, and leaf j lives at index `capacity + j`. No pointers, no objects,
    one contiguous float array -- which is why a draw costs ~20 array reads rather
    than a Python object walk.
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    @property
    def total(self):
        return float(self.tree[1])

    def update(self, leaf, value):
        """Set one leaf's priority and repair the sums on the path to the root."""
        i = leaf + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def update_many(self, leaves, values):
        for leaf, value in zip(leaves, values):
            self.update(int(leaf), float(value))

    def find(self, prefix):
        """Return the leaf whose cumulative-priority interval contains `prefix`.

        Descend from the root: if the left subtree's mass covers the prefix, go
        left; otherwise subtract it and go right. Sampling `prefix ~ U(0, total)`
        therefore hits leaf j with probability p_j / total, exactly as intended.
        """
        i = 1
        while i < self.capacity:
            left = 2 * i
            if prefix <= self.tree[left]:
                i = left
            else:
                prefix -= self.tree[left]
                i = left + 1
        return i - self.capacity

    def find_many(self, prefixes):
        return np.array([self.find(float(p)) for p in prefixes], dtype=np.int64)


class PrioritizedReplayBuffer:
    """Same interface as dqn_lib.ReplayBuffer, so the training loop is unchanged.

    New transitions arrive with the maximum priority seen so far, guaranteeing
    every transition is replayed at least once before its priority is ever
    trusted -- otherwise a transition that happened to get a small initial error
    could sink to probability ~0 and never be looked at again.
    """

    def __init__(self, capacity, obs_shape, obs_dtype=np.float32,
                 alpha=0.6, beta0=0.4, beta_steps=50_000, eps=1e-3):
        self.capacity = int(capacity)
        self.obs_dtype = obs_dtype
        self.byte_obs = obs_dtype == np.uint8
        self.s = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.s2 = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.d = np.zeros(self.capacity, dtype=np.float32)

        self.tree = SumTree(self.capacity)
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_steps = beta_steps
        self.eps = eps                 # keeps a zero-error transition reachable
        self.max_priority = 1.0
        self.pos = 0
        self.full = False
        self.samples = 0               # drives the beta anneal
        self.sampled_count = np.zeros(self.capacity, dtype=np.int64)  # diagnostics

    def _encode(self, obs):
        return np.asarray(obs * 255.0, dtype=np.uint8) if self.byte_obs else obs

    def add(self, s, a, r, s2, d):
        i = self.pos
        self.s[i] = self._encode(s)
        self.s2[i] = self._encode(s2)
        self.a[i], self.r[i], self.d[i] = a, r, d
        self.tree.update(i, self.max_priority ** self.alpha)   # optimistic init
        self.sampled_count[i] = 0
        self.pos = (i + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.capacity if self.full else self.pos

    @property
    def beta(self):
        return min(1.0, self.beta0 + (1.0 - self.beta0) * self.samples / self.beta_steps)

    def _obs(self, arr, idx):
        t = torch.as_tensor(arr[idx])
        return t.float().div_(255.0) if self.byte_obs else t

    def sample(self, batch_size, rng):
        n = len(self)
        total = self.tree.total

        # stratified sampling: one draw from each of `batch_size` equal slices of
        # the total priority mass. Cheaper variance than batch_size iid draws, and
        # it guarantees the batch spans the whole priority range.
        edges = np.linspace(0.0, total, batch_size + 1)
        prefixes = rng.uniform(edges[:-1], edges[1:])
        idx = self.tree.find_many(prefixes)
        idx = np.clip(idx, 0, n - 1)
        self.sampled_count[idx] += 1

        probs = self.tree.tree[idx + self.capacity] / total
        weights = (n * np.maximum(probs, 1e-12)) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)  # scale so max w = 1
        self.samples += 1

        return Batch(
            self._obs(self.s, idx),
            torch.as_tensor(self.a[idx]),
            torch.as_tensor(self.r[idx]),
            self._obs(self.s2, idx),
            torch.as_tensor(self.d[idx]),
            torch.as_tensor(weights),
            idx,
        )

    def update_priorities(self, idx, td_errors):
        if idx is None:
            return
        p = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        self.max_priority = max(self.max_priority, float(p.max()))
        self.tree.update_many(idx, p ** self.alpha)

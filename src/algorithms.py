"""Policies vectorized over independent trials, never over sequential rounds."""
import numpy as np


def random_argmax(values, uniforms):
    """Uniformly choose among exact maxima in each row (no index bias)."""
    ties = values == values.max(axis=1, keepdims=True)
    rank = (uniforms * ties.sum(axis=1)).astype(int)
    return (ties.cumsum(axis=1) > rank[:, None]).argmax(axis=1)


class Policy:
    def __init__(self, n_trials, n_arms):
        if n_trials < 1 or n_arms < 1:
            raise ValueError("positive numbers of trials and arms required")
        self.counts = np.zeros((n_trials, n_arms), dtype=np.int64)
        self.means = np.zeros((n_trials, n_arms), dtype=float)
        self.rows = np.arange(n_trials)
        self.n_arms = n_arms
        self.rounds = 0

    def update(self, arms, rewards):
        self.counts[self.rows, arms] += 1
        self.means[self.rows, arms] += (
            rewards - self.means[self.rows, arms]
        ) / self.counts[self.rows, arms]
        self.rounds += 1


class UniformRandom(Policy):
    def select(self, draws):
        return (draws[:, 1] * self.n_arms).astype(int)


class EpsilonGreedy(Policy):
    def __init__(self, n_trials, n_arms, epsilon=0.1):
        super().__init__(n_trials, n_arms)
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        self.epsilon = epsilon

    def select(self, draws):
        # Forced initialization avoids silently assigning unobserved arms mean 0.
        # Randomize its order to avoid systematically favoring an arm label.
        if self.rounds < self.n_arms:
            return random_argmax((self.counts == 0).astype(float), draws[:, 2])
        greedy = random_argmax(self.means, draws[:, 2])
        explore = (draws[:, 1] * self.n_arms).astype(int)
        return np.where(draws[:, 0] < self.epsilon, explore, greedy)


class UCB1(Policy):
    def indices(self):
        """t is the upcoming 1-based round; counts are BEFORE that round."""
        if np.any(self.counts == 0):
            raise ValueError("pull every arm before evaluating UCB indices")
        t = self.rounds + 1
        return self.means + np.sqrt(2 * np.log(t) / self.counts)

    def select(self, draws):
        if self.rounds < self.n_arms:
            return random_argmax((self.counts == 0).astype(float), draws[:, 2])
        return random_argmax(self.indices(), draws[:, 2])


def make_policy(name, n_trials, n_arms):
    if name == "random":
        return UniformRandom(n_trials, n_arms)
    if name == "ucb1":
        return UCB1(n_trials, n_arms)
    if name.startswith("epsilon_"):
        return EpsilonGreedy(n_trials, n_arms, float(name.split("_", 1)[1]))
    raise ValueError(f"unknown policy: {name}")

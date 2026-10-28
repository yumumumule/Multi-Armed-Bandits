"""Stationary Bernoulli rewards; no policy can inspect future rewards."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BernoulliBandit:
    probabilities: tuple[float, ...]

    def __post_init__(self):
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
            raise ValueError("probabilities must be a nonempty finite vector")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Bernoulli probabilities must lie in [0, 1]")
        object.__setattr__(self, "probabilities", tuple(p.tolist()))

    @property
    def means(self):
        return np.array(self.probabilities)

    @property
    def gaps(self):
        return max(self.probabilities) - self.means

    def sample(self, arm: int, rng: np.random.Generator) -> int:
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < len(self.probabilities):
            raise ValueError("invalid arm")
        return int(rng.random() < self.probabilities[arm])

    def reward_table(self, horizon: int, seed: np.random.SeedSequence):
        """Independent potential outcomes indexed by arm and within-arm pull.

        Sharing this table across policies is a common-random-numbers coupling;
        each individual policy still sees independent Bernoulli reward streams.
        """
        if horizon < 1:
            raise ValueError("horizon must be positive")
        rng = np.random.Generator(np.random.PCG64(seed))
        return rng.random((len(self.probabilities), horizon)) < self.means[:, None]

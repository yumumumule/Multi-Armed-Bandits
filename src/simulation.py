"""Common random numbers, trial summaries, and explicit uncertainty estimates."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from .algorithms import make_policy
from .bandit import BernoulliBandit


def seed_sequence(seed, *keys):
    """Stable named streams: never use Python's process-randomized hash()."""
    digest = hashlib.sha256("/".join(map(str, keys)).encode()).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.SeedSequence([int(seed), *words])


def mean_ci(values):
    """Mean, standard error, and pointwise normal 95% Monte Carlo CI.

    Independent TRIALS are the units; never treat dependent rounds as samples.
    Endpoints are not clipped so the stated estimator remains transparent.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        raise ValueError("at least two independent trials required")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    return dict(mean=mean, se=se, ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se)


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z/(2*total)) / denominator
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return max(0., center-radius), min(1., center+radius)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no results to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def simulate(means, policies, horizon, trials, seed, experiment, checkpoints=()):
    """Run horizon prefixes with shared arm/pull reward tables across policies.

    Stream identities include experiment and trial, plus policy for decisions.
    Each trial has its own generator. A policy never consumes environment RNG.
    Saves trial-level checkpoint values and aggregate learning curves.
    """
    if horizon < 1 or trials < 2:
        raise ValueError("positive horizon and at least two trials required")
    bandit = BernoulliBandit(tuple(means))
    k = len(means)
    rows = np.arange(trials)
    tables = np.stack([bandit.reward_table(horizon, seed_sequence(seed, experiment, "reward", j))
                       for j in range(trials)])
    checkpoints = sorted(set([horizon, *[int(t) for t in checkpoints if 1 <= t <= horizon]]))
    curve_times = set(np.unique(np.geomspace(1, horizon, 250).astype(int)).tolist())
    curve_times.update(range(1, min(100, horizon)+1))
    curve_times.update(checkpoints)
    aggregate, raw, counts = [], [], []
    for name in policies:
        policy = make_policy(name, trials, k)
        draws = np.stack([np.random.Generator(np.random.PCG64(
            seed_sequence(seed, experiment, "policy", name, j))).random((horizon, 3))
            for j in range(trials)])
        reward_total = np.zeros(trials)
        regret_total = np.zeros(trials)
        optimal_total = np.zeros(trials)
        optimal_history = np.zeros((trials, horizon), dtype=bool)
        for step in range(horizon):
            t = step + 1
            arms = policy.select(draws[:, step, :])
            rewards = tables[rows, arms, policy.counts[rows, arms]].astype(float)
            policy.update(arms, rewards)
            reward_total += rewards
            regret_total += bandit.gaps[arms]
            optimal = bandit.gaps[arms] == 0
            optimal_total += optimal
            optimal_history[:, step] = optimal
            if t in curve_times:
                metrics = {"reward": reward_total, "pseudo_regret": regret_total,
                           "optimal_fraction": optimal_total/t,
                           "recent_optimal_fraction": optimal_history[:, max(0,t-1000):t].mean(axis=1)}
                for metric, values in metrics.items():
                    aggregate.append(dict(experiment=experiment, policy=name, t=t,
                                          metric=metric, trials=trials, **mean_ci(values)))
                if t in checkpoints:
                    for j in range(trials):
                        raw.append(dict(experiment=experiment, policy=name, trial=j, t=t,
                                        **{metric: float(v[j]) for metric,v in metrics.items()}))
        for j in range(trials):
            for arm in range(k):
                counts.append(dict(experiment=experiment, policy=name, trial=j, arm=arm,
                                   pulls=int(policy.counts[j, arm]), mean=means[arm]))
        # Accounting identities detect errors without conflating noisy rewards.
        np.testing.assert_allclose(regret_total, policy.counts @ bandit.gaps, atol=1e-7)
        assert np.all(policy.counts.sum(axis=1) == horizon)
        assert np.all((regret_total >= 0) & (regret_total <= horizon*max(bandit.gaps)+1e-7))
        print(f"{experiment:16} {name:14} T={horizon}: regret={regret_total.mean():.3f}", flush=True)
    return aggregate, raw, counts

import math
import numpy as np
from .common import arguments, metadata
from src.simulation import seed_sequence, wilson_interval, write_csv
from src.plotting import hoeffding


def exact_probability(n, mu, delta):
    """Finite binomial sum, using the SAME >= event as the experiment.

    Small tolerance includes mathematically equal lattice boundary points
    despite binary floating-point representations of decimal thresholds.
    """
    terms = []
    for k in range(n+1):
        if abs(k/n-mu) >= delta-1e-12:
            logp = (math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                    + k*math.log(mu)+(n-k)*math.log1p(-mu))
            terms.append(math.exp(logp))
    return math.fsum(terms)


def main():
    args, c = arguments("Monte Carlo deviation probabilities versus Hoeffding bounds")
    h = c['hoeffding']
    rows = []
    for mu in h['means']:
        for n in h['sample_sizes']:
            for delta in h['deltas']:
                rng = np.random.Generator(np.random.PCG64(seed_sequence(c['seed'], "hoeffding", mu, n, delta)))
                # Binomial sampling is exactly equivalent to summing n Bernoullis.
                successes = rng.binomial(n, mu, size=h['replications'])
                events = int(np.count_nonzero(np.abs(successes/n-mu) >= delta-1e-12))
                lo, hi = wilson_interval(events, h['replications'])
                raw_bound = 2*math.exp(-2*n*delta**2)
                rows.append(dict(mu=mu,n=n,delta=delta,replications=h['replications'],
                    events=events,estimate=events/h['replications'],ci_low=lo,ci_high=hi,
                    exact_probability=exact_probability(n,mu,delta),
                    raw_bound=raw_bound,bound=min(1.,raw_bound)))
    write_csv(args.output/"data"/"hoeffding.csv", rows)
    metadata(args, c, "hoeffding")
    hoeffding(args.output/"data", args.output/"figures")
    print(f"Hoeffding: {len(rows)} cells, {h['replications']} replications per cell", flush=True)


if __name__ == "__main__":
    main()

# Experimental evidence

The default results use `config.json`, seed 2018, and 300 independent trials per bandit experiment. Environment and software details are recorded in each `*_metadata.json`. Run `python -m experiments.run_all` from the repository root to regenerate simulations; run `python -m tools.build_report` to regenerate the paired contrasts and documentation tables.

| File | One row represents | Important columns |
| --- | --- | --- |
| `data/*_curves.csv` | A policy, round, and metric aggregated over trials | `mean`, `se`, `ci_low`, `ci_high`, `trials` |
| `data/*_trials.csv` | A trial and policy at a checkpoint | `reward`, `pseudo_regret`, `optimal_fraction`, `recent_optimal_fraction` |
| `data/*_counts.csv` | An arm, trial, and policy at the final horizon | `pulls`, true arm `mean` |
| `data/hoeffding.csv` | A mean, sample size, and deviation threshold | Event count, estimate, Wilson interval, exact binomial probability, raw and capped bounds |
| `data/paired_differences.csv` | A paired UCB1 minus ε = 0.1 contrast | Mean difference, standard error, pointwise confidence interval |

Rewards are cumulative observed Bernoulli rewards. Pseudo-regret is the cumulative sum of true arm gaps, not the realized reward shortfall. Frequencies include any optimal arm if there are ties; `recent_optimal_fraction` uses the last min(t,1000) rounds. Trial indices start at zero. The policy decision round starts at one. Counts in `*_counts.csv` include initialization.

Curves are retained at 250 logarithmically spaced candidate rounds, the first 100 rounds, and all relevant checkpoints, with duplicates removed. Trial checkpoint records include 100, 500, 1,000, 2,000, and 5,000 when within the horizon, plus scaling horizons and the terminal round. Full action/reward traces are not retained; the seeds and named streams regenerate them.

All bandit intervals use independent trial-level values and mean ± 1.96 standard errors. Hoeffding cells use 100,000 replications each, with Wilson probability intervals. These are pointwise intervals, not simultaneous bands. Shared reward sequences pair policies within each trial; separate experiments use separate streams. Scaling checkpoints are prefixes of the same trials.

`manifest.json` stores SHA-256 checksums of the 17 default numerical CSV files. The saved-data audit verifies them. Environment metadata and rendered figures are excluded from the checksum set because software version, operating system, fonts, and PDF creation timestamps can vary across reproductions. After an intentional, fully verified change to the default study, refresh the checksums with `python -m tools.audit_results --write-manifest`; this command performs the data and documentation checks before writing.

`figures/` contains five 300 dpi PNG figures and matching vector PDFs. All derive from saved CSV data. `custom/` is ignored by Git and may be used for separate exploratory runs without replacing the default evidence.

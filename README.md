# Exploration vs. Exploitation: A Probabilistic Study of Multi-Armed Bandits

A reproducible **ESTR2018: Probability for Engineers** course project at The Chinese University of Hong Kong. It studies how Hoeffding's inequality supports efficient exploration in stationary Bernoulli bandits and compares fixed-ε greedy with UCB1 using theory and actual simulations.

## Research question and model

How do exploration probability, reward gaps, and time horizon affect cumulative regret? Each arm i has an unknown Bernoulli mean μᵢ. A policy chooses Aₜ from past observations. With μ* = maxᵢ μᵢ and Δᵢ = μ* − μᵢ,

$$G(T)=\sum_{t=1}^T\Delta_{A_t},\qquad R(T)=\mathbb E[G(T)]=T\mu^*-\mathbb E\!\left[\sum_{t=1}^T\mu_{A_t}\right].$$

G(T) is sample-path pseudo-regret; R(T) is expected regret. Realized cumulative rewards include additional Bernoulli noise. The project estimates R(T) by averaging G(T) across independent trials.

Implemented from scratch with NumPy:

- **Uniform random:** a baseline with exact expected regret TΣᵢΔᵢ/K.
- **Fixed-ε greedy:** explores uniformly with probability ε and otherwise maximizes the empirical mean.
- **UCB1:** selects the largest index below, with natural logarithms and counts taken before round t.

$$\widehat\mu_i+\sqrt{\frac{2\ln t}{N_i(t)}}.$$

Both learning policies pull each arm once in random order and break ties uniformly. The [full report](report/report.md) derives the bonus from Hoeffding concentration, addresses adaptive sampling, and explains

$$R_{\mathrm{UCB1}}(T)\leq8\sum_{i:\Delta_i>0}\frac{\ln T}{\Delta_i}+
\left(1+\frac{\pi^2}{3}\right)\sum_{i:\Delta_i>0}\Delta_i.$$

For fixed positive gaps this is logarithmic. Fixed ε > 0 instead incurs a linear exploration cost whenever suboptimal arms exist. These guarantees concern expectations, not every individual path.

## Experimental design and findings

The defaults use seed **2018**, **300 independent trials**, and **5,000 rounds**, with horizons up to **20,000** for scaling. All values are configured in [config.json](config.json).

| Experiment | Configuration |
| --- | --- |
| Algorithm comparison | Means [0.2, 0.35, 0.5, 0.65, 0.8]; random, ε = 0.1, UCB1 |
| Exploration probability | ε ∈ {0.01, 0.05, 0.1, 0.2, 0.5} |
| Reward gaps | Easy [0.2, 0.8] versus difficult [0.49, 0.51] |
| Hoeffding deviation probabilities | Two means, seven sample sizes, three thresholds; 100,000 replications per cell |
| Regret scaling | Eight configured horizons from 100 to 20,000; repeated maximum-horizon paths |

Main comparison at T = 5,000; brackets are approximate **pointwise 95% Monte Carlo confidence intervals** based on independent trial values:

<!-- BEGIN comparison -->
| Policy | Cumulative reward (95% CI) | Pseudo-regret (95% CI) | Optimal frequency, last 1,000 (95% CI) |
| --- | --- | --- | --- |
| Uniform random | 2499.17 [2495.44, 2502.89] | 1500.19 [1498.52, 1501.86] | 0.1996 [0.1981, 0.2012] |
| ε-greedy (0.1) | 3826.09 [3817.78, 3834.41] | 175.84 [168.49, 183.20] | 0.9194 [0.9175, 0.9214] |
| UCB1 | 3850.90 [3846.93, 3854.88] | 149.95 [147.81, 152.10] | 0.9534 [0.9498, 0.9571] |
<!-- END comparison -->

![Algorithm comparison](results/figures/comparison.png)

UCB1 gives the lowest final mean regret in the main comparison. Among the five ε values, ε = 0.05 has the lowest estimated mean regret at 5,000 rounds, while ε = 0.01 shows substantial trial variability. The small-gap environment is a finite-horizon exception: ε = 0.1 outperforms UCB1 there. Smaller absolute regret in that environment does not imply easier identification, because each mistake costs less.

![Regret scaling](results/figures/regret_scaling.png)

The scaling results are consistent with UCB1's theoretical advantage over fixed exploration, but a finite simulation cannot establish asymptotic order. The [report](report/report.md) includes all five experiments, paired uncertainty intervals, exact binomial concentration probabilities, and limitations.

## Installation

Requires Python 3.10 or newer; the committed run used Python 3.12.9. Runtime dependencies are only NumPy and Matplotlib. Tests use the standard library.

```bash
git clone https://github.com/yumumumule/ESTR2018-Multi-Armed-Bandits.git
cd ESTR2018-Multi-Armed-Bandits
python -m venv .venv
```

Activate the environment on macOS/Linux with `source .venv/bin/activate`, or in Windows PowerShell with `.\.venv\Scripts\Activate.ps1`. If PowerShell activation is restricted, call `.\.venv\Scripts\python.exe` directly in place of `python`.

```bash
python -m pip install -r requirements.txt
```

[requirements-lock.txt](requirements-lock.txt) records all resolved packages from the committed Windows run. The two direct scientific dependencies are pinned in `requirements.txt`; random-number generation uses explicit PCG64. Platform-dependent font rendering can change figure bytes even when numerical results agree.

## Reproduce and verify

Run these commands **from the repository root**, after installation:

```bash
python -m unittest discover -s tests -v
python -m experiments.run_all
python -m tools.build_report
python -m tools.audit_results
```

Run any experiment individually:

```bash
python -m experiments.compare_algorithms
python -m experiments.epsilon_analysis
python -m experiments.reward_gap_analysis
python -m experiments.hoeffding_verification
python -m experiments.regret_scaling
```

Each experiment accepts `--config`, `--seed`, `--trials`, and `--output`. For example, a separate run without replacing committed evidence:

```bash
python -m experiments.compare_algorithms --seed 42 --trials 20 --output results/custom
```

The trial override applies to bandit experiments; Hoeffding replication counts come from the configuration. Changing the horizon, means, ε values, or Hoeffding settings requires editing a configuration copy and supplying `--config path/to/config.json`. Default data and report tables are kept aligned by the audit. Custom configurations can change the policy ordering and should be interpreted using their saved metadata; existing prose describes the default study.

To regenerate figures from saved CSVs without simulations:

```bash
python -m tools.replot
```

To confirm the default numerical tables are current:

```bash
python -m tools.build_report --check
```

## Reproducibility details

Every experiment, trial, and policy uses stable named random streams. Policies within a trial share arm/pull-index reward sequences for fair paired comparisons; policy decisions use separate streams. Independent trials are the units for uncertainty estimates. Changing policy execution order does not affect results. Scaling horizons are prefixes of the same independent trials and are correlated across horizons.

Results include aggregate learning curves, checkpoint observations for each trial, final arm counts, all 42 Hoeffding cells, software/configuration metadata, and paired algorithm differences. Full action histories are regenerated from seeds. Optimal-arm frequency plots use a trailing window of up to 1,000 rounds. Bands are pointwise, not simultaneous; Hoeffding probability intervals use Wilson's formula. No results are hand-adjusted or selected after inspection.

## Repository structure

```text
config.json                  Default experiment settings
requirements*.txt            Scientific dependencies and resolved environment
src/
  bandit.py                  Bernoulli environment and potential reward streams
  algorithms.py              Random, fixed-ε greedy, UCB1
  simulation.py              Independent streams, trial accounting, CSV output
  plotting.py                Figures from saved data
experiments/                 Five experiments and run_all entry point
results/
  data/                      CSV summaries, trial checkpoints, final counts
  figures/                   300 dpi PNG and vector PDF figures
  *_metadata.json            Per-experiment configuration and software versions
  manifest.json              Checksums of default numerical evidence
tests/test_bandits.py        Meaningful unit and integration tests
tools/                       Data audit, documentation generation, replotting
report/report.md             Complete academic report
```

The complete report is Markdown because no LaTeX engine was available. Figure PDFs are included. The project uses the [MIT license](LICENSE).

## References

1. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 2. MIT Press. [Publisher](https://mitpress.ublish.com/book/reinforcement-learning-an-introduction-2).
2. Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). “Finite-time Analysis of the Multiarmed Bandit Problem.” *Machine Learning*, 47(2–3), 235–256. [DOI](https://doi.org/10.1023/A:1013689704352).
3. Hoeffding, W. (1963). “Probability Inequalities for Sums of Bounded Random Variables.” *Journal of the American Statistical Association*, 58(301), 13–30. [DOI](https://doi.org/10.1080/01621459.1963.10500830).

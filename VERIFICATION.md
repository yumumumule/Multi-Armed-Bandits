# Verification record

The default study was executed on 28 October 2026 (Asia/Shanghai), using Python 3.12.9, NumPy 2.2.6, and Matplotlib 3.10.3 on Windows. See the committed metadata and dependency lock file for the resolved environment.

- `python -m unittest discover -s tests -v`: **13 tests passed**.
- `python -m pip check`: no broken requirements.
- `python -m experiments.run_all`: all five experiments completed at the full configured sizes, including 300 trials per bandit setting, horizons up to 20,000, and 42 Hoeffding cells with 100,000 replications each.
- Each of the five individual `python -m experiments.<module>` commands was rerun to a separate temporary output directory. All **16 simulation CSVs reproduced byte for byte**. The derived paired-contrast CSV was checked independently against trial differences.
- The README example with seed 42, 20 trials, and a separate output directory completed successfully.
- `python -m tools.replot`: regenerated all five PNG and five PDF figures from saved CSVs. All five PNG figures were visually inspected; a concentration-plot annotation was moved above the panels to avoid covering data.
- `python -m tools.build_report --check`: all generated tables in the report and README match the saved results.
- `python -m tools.audit_results`: checked **376 checkpoint metric summaries**, every trial's final count/regret identity, all 42 exact binomial comparisons, paired uncertainty intervals, ten figure artifacts, local documentation links, and the numerical-evidence checksums.

The full academic report is Markdown. No LaTeX executable was available, so no PDF report is claimed; vector PDF figures are included. The experiments support finite-horizon comparisons and do not establish asymptotic regret theorems empirically.

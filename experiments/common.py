import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib
from src.simulation import simulate, write_csv


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.trials is not None:
        config["trials"] = args.trials
    if config['trials'] < 2:
        parser.error("--trials must be at least 2")
    return args, config


def metadata(args, config, name):
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"experiment":name, "config":config, "python":sys.version.split()[0],
        "numpy":np.__version__, "matplotlib":matplotlib.__version__,
        "platform":platform.system(), "rng":"PCG64; SHA256-keyed SeedSequence",
        "curve_intervals":"mean +/- 1.96 across-trial standard errors; pointwise",
        "frequency_window":1000}
    (args.output/f"{name}_metadata.json").write_text(
        json.dumps(payload, indent=2)+"\n", encoding="utf-8")


def bandit_experiment(args, config, name, means, policies, horizon=None, checkpoints=()):
    horizon = horizon or config['horizon']
    checkpoints = sorted(set([100,500,1000,2000,5000,*checkpoints]))
    curves, trials, counts = simulate(means, policies, horizon, config['trials'],
                                      config['seed'], name, checkpoints)
    data = args.output/"data"
    for suffix, rows in [("curves",curves),("trials",trials),("counts",counts)]:
        write_csv(data/f"{name}_{suffix}.csv", rows)
    metadata(args, config, name)

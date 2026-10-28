from .common import arguments, bandit_experiment
from src.plotting import scaling


def main():
    args, c = arguments("Regret scaling using repeated independent maximum-horizon paths")
    bandit_experiment(args, c, "scaling", c['means'], c['comparison_policies'],
                      horizon=max(c['scaling_horizons']), checkpoints=c['scaling_horizons'])
    scaling(args.output/"data", args.output/"figures", c['means'])


if __name__ == "__main__":
    main()

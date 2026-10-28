from .common import arguments, bandit_experiment
from src.plotting import comparison


def main():
    args, c = arguments("Compare random, fixed-epsilon greedy, and UCB1")
    bandit_experiment(args, c, "comparison", c['means'], c['comparison_policies'])
    comparison(args.output/"data", args.output/"figures")


if __name__ == "__main__":
    main()

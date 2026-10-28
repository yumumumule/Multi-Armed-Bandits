from .common import arguments, bandit_experiment
from src.plotting import epsilon


def main():
    args, c = arguments("Effect of fixed exploration probabilities")
    bandit_experiment(args, c, "epsilon", c['means'], [f"epsilon_{e}" for e in c['epsilons']])
    epsilon(args.output/"data", args.output/"figures")


if __name__ == "__main__":
    main()

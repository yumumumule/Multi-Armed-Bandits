from .common import arguments, bandit_experiment
from src.plotting import gaps


def main():
    args, c = arguments("Effect of reward gaps")
    for name, means in c['gap_environments'].items():
        bandit_experiment(args, c, f"gap_{name}", means, c['comparison_policies'])
    gaps(args.output/"data", args.output/"figures")


if __name__ == "__main__":
    main()

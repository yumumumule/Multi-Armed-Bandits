from .compare_algorithms import main as comparison
from .epsilon_analysis import main as epsilon
from .reward_gap_analysis import main as gaps
from .hoeffding_verification import main as hoeffding
from .regret_scaling import main as scaling


def main():
    for experiment in [comparison, epsilon, gaps, hoeffding, scaling]:
        experiment()


if __name__ == "__main__":
    main()

"""Regenerate all default scientific figures from existing CSV files."""
import argparse
import json
from pathlib import Path
from src.plotting import comparison, epsilon, gaps, hoeffding, scaling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    data, figures = args.results/'data', args.results/'figures'
    config = json.loads((args.results/'scaling_metadata.json').read_text(encoding='utf-8'))['config']
    comparison(data,figures)
    epsilon(data,figures)
    gaps(data,figures)
    hoeffding(data,figures)
    scaling(data,figures,config['means'])
    print('Regenerated five PNG and five PDF figures from saved CSVs.')


if __name__ == '__main__':
    main()

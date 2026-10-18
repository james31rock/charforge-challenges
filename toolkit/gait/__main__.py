"""Compare two explicitly phase-labelled 3D gait traces."""
import argparse
import json
from pathlib import Path
from .core import load, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = compare(load(args.reference), load(args.candidate))
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(report['status'])


if __name__ == '__main__':
    main()

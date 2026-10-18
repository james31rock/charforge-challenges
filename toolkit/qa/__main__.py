"""Evaluate full-rate measured JSON evidence without opening or modifying a rig."""
import argparse
import json
from pathlib import Path
from .core import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(Path(args.trace).read_text(encoding='utf-8')))
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(report['status'])
    return {'PASS': 0, 'FAIL': 1, 'BLOCKED': 3}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())

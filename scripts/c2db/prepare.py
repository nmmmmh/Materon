#!/usr/bin/env python3
"""Build the C2DB metallic work-function cohort, split and graph cache (Sec. S1, S2.4).

Example:
  python scripts/c2db/prepare.py --source data/c2db/C2DB_SOURCE.json --out work/c2db/prepared
"""
import argparse
import json

from materon.attributes import feature_table
from materon.c2db.dataset import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="3,520-record C2DB snapshot (JARVIS c2db JSON)")
    parser.add_argument("--features", default=None, help="attribute table (default configs/attributes/physical20.csv)")
    parser.add_argument("--out", required=True, help="prepared directory to create")
    args = parser.parse_args()
    summary = prepare(args.source, feature_table(args.features), args.out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

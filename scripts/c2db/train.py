#!/usr/bin/env python3
"""Train one C2DB work-function predictor (one arm, one seed, one schedule).

Arms: physical20_environment_gate (Materon), physical20_nogate,
physical20_global_gate, physical20_element_gate. Seeds 42, 7, 123, 2024, 31415.
Output: <runs>/<schedule>/<arm>_s<seed>/{best.weights.h5, RESULT.json, ...}.

Example:
  python scripts/c2db/train.py --prepared work/c2db/prepared --runs work/c2db/runs \
      --arm physical20_environment_gate --seed 42 --schedule L400
"""
import argparse

from materon.c2db.training import SCHEDULES, member_dir, train
from materon.runtime import configure_tensorflow


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--arm", default="physical20_environment_gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), default="short")
    args = parser.parse_args()

    import tensorflow as tf

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    train(args.prepared, args.arm, args.seed, member_dir(args.runs, args.schedule, args.seed, args.arm),
          schedule=args.schedule)


if __name__ == "__main__":
    main()

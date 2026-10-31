#!/usr/bin/env python3
"""Fine-tune L400 members on DFT increments (Eq. S3, Sec. S3.3).

Cross-validation (leave one parent out, two seeds) chose lam = 1, 30 epochs,
learning rate 2e-4; FT1 uses the 20 pairs available before round 1 and FT2
adds the 16 round-1 pairs (36 pairs).

Pair file: JSON list of {"name", "group", "parent_id", "site", "z_sub", "dft_delta_eV"}.

Examples:
  # final ensemble member (FT1, seed 42)
  python scripts/c2db/finetune.py --prepared P --runs runs --base-tag L400 --tag FT1 --seed 42 --pairs FT_PAIRS.json
  # one cross-validation fold (hold out the Al2S2 pairs)
  python scripts/c2db/finetune.py --prepared P --runs runs --base-tag L400 --seed 42 --pairs FT_PAIRS.json \
      --holdout Al2S2 --lam 4 --epochs 10 --cv-out work/c2db/ft_cv
"""
import argparse

from materon.attributes import feature_table
from materon.c2db.dataset import read_json
from materon.c2db.finetune import finetune
from materon.c2db.training import member_dir
from materon.runtime import configure_tensorflow


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--base-tag", default="L400")
    parser.add_argument("--tag", default=None, help="output tag of the saved member (FT1, FT2)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--holdout", default=None, help="pair group to hold out (cross-validation)")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cv-out", default=None, help="output directory for cross-validation records")
    args = parser.parse_args()
    if (args.tag is None) == (args.cv_out is None):
        parser.error("give exactly one of --tag (save a member) or --cv-out (cross-validation)")

    import tensorflow as tf

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    base = member_dir(args.runs, args.base_tag, args.seed) / "best.weights.h5"
    out = member_dir(args.runs, args.tag, args.seed) if args.tag else args.cv_out
    record = finetune(args.prepared, base, read_json(args.pairs), feature_table(), args.seed, out,
                      holdout=args.holdout, lam=args.lam, epochs=args.epochs, lr=args.lr, save=args.tag is not None)
    print("validation MAE %.4f -> %.4f; held-out deltas %s -> %s (DFT %s)" % (
        record["before"]["val_mae"], record["after"]["val_mae"], record["before"]["hold_delta"],
        record["after"]["hold_delta"], record["hold_dft"]))


if __name__ == "__main__":
    main()

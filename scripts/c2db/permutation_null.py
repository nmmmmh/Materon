#!/usr/bin/env python3
"""Permutation test for channel consistency across models (SI Section S3.6).

For each channel, the statistic is the smallest Spearman correlation over all models, taken with the majority
sign, between the channel's restoration response R_j and the predicted change over the metal-site substitutions.
The null permutes the predicted changes over substitutions, with one permutation applied to all models so that
the correlation between models is kept, and records the largest statistic over the 20 channels.

Example:
  python scripts/c2db/permutation_null.py --rules data/results/GATE_RULES_SEEDS.json data/results/GATE_RULES_L400.json \
      --permutations 2000 --out work/PERMUTATION_NULL.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def rank_correlations(A, F):
    ra = np.apply_along_axis(rankdata, 0, A)
    rf = rankdata(F)
    ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-12)
    rf = (rf - rf.mean()) / rf.std()
    return (ra * rf[:, None]).mean(0)


def statistic(R):
    sign = np.sign(np.sign(R).sum(0))
    return (R * sign).min(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", nargs="+", required=True, help="channel-rules JSON files (one per schedule)")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    members, channels = [], None
    for f in args.rules:
        d = json.loads(Path(f).read_text())
        channels = d.get("channels", channels)
        jobs = [j for j in d["jobs"] if j["kind"] == "M"]
        for m in range(len(jobs[0]["native_eV"])):
            members.append((np.array([j["R_env_eV"][m] for j in jobs]), np.array([j["native_eV"][m] for j in jobs])))
    R = np.array([rank_correlations(A, F) for A, F in members])
    observed = statistic(R)
    rng = np.random.default_rng(args.seed)
    n = len(members[0][1])
    null = np.array([statistic(np.array([rank_correlations(A, F[p]) for A, F in members])).max()
                     for p in (rng.permutation(n) for _ in range(args.permutations))])
    best = int(np.argmax(observed))
    out = dict(models=len(members), substitutions=n, channels=channels, observed=observed.tolist(),
               best_channel=channels[best] if channels else best, best_statistic=float(observed[best]),
               null_quantiles={q: float(np.quantile(null, float(q))) for q in ("0.95", "0.99")},
               null_max=float(null.max()), p_value=float((null >= observed[best]).mean()),
               permutations=args.permutations)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("best_channel", "best_statistic", "null_quantiles", "null_max", "p_value")}))


if __name__ == "__main__":
    main()

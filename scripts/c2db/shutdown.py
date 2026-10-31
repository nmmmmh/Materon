#!/usr/bin/env python3
"""Ranked channel shutdown of one environment-gate checkpoint (Fig. 3, Sec. S7, Tables S5, S19-S21).

On the first 64 validation materials (split order), attributes are ranked per
material by |IG| along the gate path (32 intervals), by modulation magnitude,
by the endpoint gradient and by 20 random rankings; the top k = 1, 3, 5, 10
corrections are disabled on all atoms and the prediction change is recorded.
Means and 95 % intervals resample chemical systems (3,000 draws, seed
20261015), paired across methods. --randomization adds the parameter controls
of Sec. S7.2 on the first eight materials.

Example:
  python scripts/c2db/shutdown.py --prepared work/c2db/prepared \
      --checkpoint work/c2db/runs/L400/physical20_environment_gate_s42/best.weights.h5 \
      --out work/c2db/shutdown/L400_s42.json
"""
import argparse

import numpy as np

from materon.c2db import attribution as A
from materon.c2db.dataset import Prepared, write_json
from materon.runtime import configure_tensorflow

METHODS = ("gate_path_ig", "modulation_magnitude", "endpoint_gradient", "random")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--randomization", action="store_true", help="also run the Sec. S7.2 parameter controls")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    from materon.c2db.models import load_member

    configure_tensorflow(tf, intra=4, inter=2, tf32=False)
    data = Prepared(args.prepared)
    mean, std = data.mean, data.std
    ids = data.ids("validation")[:A.N_FIXED]
    graphs = [data.graphs[i] for i in ids]
    model = load_member(args.checkpoint)

    full = A.predict(model, graphs, mean, std)
    mask0 = A.predict(model, graphs, mean, std, mask=0.0)
    ig = {}
    for steps in (16, 32):
        atom_ig, splits = A.mask_path_ig(tf, model, graphs, std, steps)
        ig[steps] = A.per_graph_sum(atom_ig, splits)
    completeness = {n: float(np.max(np.abs(ig[n].sum(axis=1) - (full - mask0)))) for n in ig}
    if completeness[32] > A.COMPLETENESS_TOLERANCE_EV:
        raise ValueError("32-interval IG completeness exceeded 0.002 eV")
    _, correction, gate_splits = A.gate_readouts(model, graphs)
    endpoint = A.endpoint_gradient(tf, model, graphs, std)

    scores = A.ranking_scores(ig[32], correction, gate_splits, endpoint)
    rankings = {name: np.stack([A.rank(row) for row in values]) for name, values in scores.items()}
    rankings["random"] = A.nested_random_rankings(A.random5_subsets(len(graphs)))
    responses = {name: A.shutdown_responses(model, graphs, full, rankings[name], mean, std)
                 for name in ("gate_path_ig", "modulation_magnitude", "endpoint_gradient")}
    random_responses = [A.shutdown_responses(model, graphs, full, rankings["random"][r], mean, std)
                        for r in range(A.RANDOM_RANKINGS)]

    def curve(method, k):
        if method == "random":
            return np.stack([rep[k] for rep in random_responses])
        return responses[method][k][None, :]

    systems = [A.chemical_system(g) for g in graphs]
    draws = A.bootstrap_draws(len(set(systems)))
    curves, contrasts, ratios = {}, {}, {}
    for k in A.K_VALUES:
        curves[k] = {m: {"absolute": A.cluster_summary(np.abs(curve(m, k)), systems, draws),
                         "signed": A.cluster_summary(curve(m, k), systems, draws)} for m in METHODS}
        ig_abs = np.abs(curve("gate_path_ig", k)).mean(axis=0)
        contrasts[k] = {m: A.cluster_summary((ig_abs - np.abs(curve(m, k)).mean(axis=0))[None, :], systems, draws)
                        for m in METHODS if m != "gate_path_ig"}
        ratios[k] = curves[k]["gate_path_ig"]["absolute"]["mean"] / curves[k]["random"]["absolute"]["mean"]

    result = {
        "checkpoint": str(args.checkpoint), "fixed_ids": ids, "chemical_systems": systems,
        "full_eV": full.tolist(), "mask0_eV": mask0.tolist(),
        "ig16_eV": ig[16].tolist(), "ig32_eV": ig[32].tolist(), "ig_completeness_max_abs_eV": completeness,
        "rankings": {m: rankings[m].tolist() for m in rankings},
        "responses_eV": {m: {k: v.tolist() for k, v in responses[m].items()} for m in responses},
        "random_responses_eV": [{k: v.tolist() for k, v in rep.items()} for rep in random_responses],
        "curves": curves, "ig_minus_comparator_absolute": contrasts, "ig_over_random_ratio": ratios,
    }
    if args.randomization:
        n = 8
        result["randomization"] = A.randomization_controls(tf, model, args.checkpoint, graphs[:n], ig[32][:n],
                                                           mean, std)
    write_json(args.out, result)
    for k in A.K_VALUES:
        c = contrasts[k]["random"]
        print("k=%d IG/random=%.3f IG-random=%.4f [%.4f, %.4f]" % (k, ratios[k], c["mean"], c["ci95_low"],
                                                                    c["ci95_high"]))


if __name__ == "__main__":
    main()

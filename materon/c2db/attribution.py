"""Gate-path attribution and ranked channel shutdown (Sec. II B, S7, Fig. 3).

The path mask m scales all gate corrections, x~(m) = x * [1 + m * (g - 1)]
(Eq. S4). Integrated gradients along m(t) = t * 1 (Eq. S5) use the trapezoid
rule with N intervals (N + 1 gradient evaluations) and are summed over the
atoms of each material, so that sum_j A_j ~ F(1) - F(0) in eV. Shutdown sets
m_j = 0 for the k selected attributes on every atom.

Rankings compared: |IG| (gate_path_ig), sum over atoms of |x (g - 1)|
(modulation_magnitude), |dF/dm_j| at m = 1 (endpoint_gradient) and 20 random
rankings. Intervals resample chemical systems (Sec. S7.1).
"""
import numpy as np

from materon.c2db.graph import model_inputs

ARM = "physical20_environment_gate"
FEATURES = 20
K_VALUES = (1, 3, 5, 10)
N_FIXED = 64                 # first 64 validation IDs in split order
RANDOM_SEED = 20261013       # random5 subsets and nested random rankings
RANDOM_RANKINGS = 20
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_RESAMPLES = 3000
RANDOMIZATION_SEED = 20261014  # pristine weights for the parameter controls (S7.2)
COMPLETENESS_TOLERANCE_EV = 0.002


def mask_index(arm):
    return 2 if arm == ARM else 1


def predict(model, graphs, mean, std, arm=ARM, mask=None, batch_size=32):
    """Predictions in eV; ``mask`` is None, a scalar or one array per graph."""
    pieces = []
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        chunk_mask = mask if (mask is None or np.isscalar(mask)) else mask[start:start + batch_size]
        values = model(model_inputs(chunk, arm, mask=chunk_mask), training=False).numpy().reshape(-1)
        pieces.append(values.astype(np.float64) * float(std) + float(mean))
    result = np.concatenate(pieces)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite prediction")
    return result


def _mask_gradient_fn(tf, model, inputs, index):
    row_splits = inputs[index].row_splits

    @tf.function(reduce_retracing=True)
    def gradient(flat_mask):
        with tf.GradientTape() as tape:
            tape.watch(flat_mask)
            current = list(inputs)
            current[index] = tf.RaggedTensor.from_row_splits(flat_mask, row_splits, validate=False)
            objective = tf.reduce_sum(model(current, training=False))
        result = tape.gradient(objective, flat_mask)
        if result is None:
            raise ValueError("gate mask path is disconnected")
        return result

    return gradient, np.asarray(row_splits.numpy(), dtype=np.int64)


def per_graph_sum(atom_values, splits):
    return np.stack([atom_values[splits[i]:splits[i + 1]].sum(axis=0) for i in range(len(splits) - 1)])


def mask_path_ig(tf, model, graphs, std, steps=32, arm=ARM):
    """Atom-level IG along the gate path (eV) and the row splits of the batch."""
    inputs = model_inputs(graphs, arm)
    index = mask_index(arm)
    gradient, splits = _mask_gradient_fn(tf, model, inputs, index)
    template = tf.ones_like(inputs[index].flat_values)
    accumulator = np.zeros(template.shape, dtype=np.float64)
    for step in range(steps + 1):
        grad = gradient(template * tf.cast(step / steps, template.dtype))
        accumulator += (0.5 if step in (0, steps) else 1.0) * grad.numpy().astype(np.float64)
    return accumulator / steps * std, splits


def endpoint_gradient(tf, model, graphs, std, arm=ARM):
    """Per-material dF/dm_j at m = 1 (eV), summed over atoms."""
    inputs = model_inputs(graphs, arm)
    index = mask_index(arm)
    gradient, splits = _mask_gradient_fn(tf, model, inputs, index)
    values = gradient(inputs[index].flat_values).numpy().astype(np.float64) * float(std)
    return per_graph_sum(values, splits)


def gate_readouts(model, graphs, arm=ARM):
    """Gate factors g and applied corrections x (g - 1) at every atom (flat arrays) with bound checks."""
    inputs = model_inputs(graphs, arm)
    x = inputs[0]
    if arm == ARM:
        context, mask = inputs[1], inputs[2]
        gate = model.get_layer("environment_gate")
    else:
        context = model.get_layer("zero_environment")(x)
        mask = inputs[1]
        gate = model.get_layer("element_gate")
    gated, readouts = gate([x, context, mask], return_readouts=True)
    x_values = x.flat_values.numpy()
    g = readouts["gate"].flat_values.numpy()
    correction = readouts["correction"].flat_values.numpy()
    if (not np.isfinite(g).all() or g.min() < 0.5 - 1e-6 or g.max() > 1.5 + 1e-6
            or not np.allclose(correction, x_values * (g - 1.0), atol=2e-6, rtol=2e-6)
            or not np.allclose(gated.flat_values.numpy(), x_values + correction, atol=2e-6, rtol=2e-6)):
        raise ValueError("gate readout bounds or correction identity failed")
    return g, correction, np.asarray(x.row_splits.numpy(), dtype=np.int64)


def rank(scores):
    """Descending ranking; ties broken by ascending channel index."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (FEATURES,) or not np.isfinite(scores).all():
        raise ValueError("a feature ranking needs 20 finite scores")
    return np.argsort(-scores, kind="stable").astype(np.int32)


def shutdown_masks(graphs, rankings, k):
    masks = []
    for graph, ranking in zip(graphs, rankings):
        mask = np.ones_like(graph["x"], dtype=np.float32)
        mask[:, np.asarray(ranking[:k], dtype=np.int64)] = 0.0
        masks.append(mask)
    return masks


def random5_subsets(n_graphs, seed=RANDOM_SEED, replicates=RANDOM_RANKINGS):
    """Random 5-attribute subsets, drawn replicate by replicate (functional probe order)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.stack([rng.choice(FEATURES, size=5, replace=False) for _ in range(n_graphs)])
                     for _ in range(replicates)]).astype(np.int32)


def nested_random_rankings(random5, seed=RANDOM_SEED):
    """Full random rankings whose first five channels are the random5 subsets."""
    replicates, n_graphs, _ = random5.shape
    rng = np.random.default_rng(seed)
    rankings = np.empty((replicates, n_graphs, FEATURES), dtype=np.int32)
    channels = np.arange(FEATURES, dtype=np.int32)
    for r in range(replicates):
        for g in range(n_graphs):
            first = np.asarray(random5[r, g], dtype=np.int32)
            remaining = channels[~np.isin(channels, first)]
            rankings[r, g] = np.concatenate((first, rng.permutation(remaining)))
    return rankings


def shutdown_responses(model, graphs, full, rankings, mean, std, k_values=K_VALUES):
    """{k: full - masked prediction per material} after disabling the top-k ranked attributes."""
    return {int(k): full - predict(model, graphs, mean, std, mask=shutdown_masks(graphs, rankings, k))
            for k in k_values}


def ranking_scores(ig, correction, splits, endpoint):
    """Per-material ranking scores of the three non-random methods."""
    n = len(splits) - 1
    magnitude = np.stack([np.abs(correction[splits[i]:splits[i + 1]]).sum(axis=0) for i in range(n)])
    return {"gate_path_ig": np.abs(ig), "modulation_magnitude": magnitude,
            "endpoint_gradient": np.abs(endpoint)}


def chemical_system(graph):
    from pymatgen.core import Element

    return "-".join(sorted({Element.from_Z(int(z)).symbol for z in graph["z"]}))


def bootstrap_draws(n_systems, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_systems, size=(resamples, n_systems), endpoint=False)


def cluster_summary(values, systems, draws):
    """Material-weighted mean and 95 % interval from whole-chemical-system resamples.

    ``values`` has one column per material; rows (random rankings) are averaged
    within each material first. The same ``draws`` are used for every method.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    material = values.mean(axis=0)
    names = sorted(set(systems))
    rows = {name: np.flatnonzero(np.asarray(systems) == name) for name in names}
    sums = np.asarray([material[rows[name]].sum() for name in names])
    counts = np.asarray([len(rows[name]) for name in names], dtype=np.float64)
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {"mean": float(material.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def spearman(left, right):
    """Spearman correlation; None when either vector is constant."""
    from scipy.stats import rankdata

    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
    return value if np.isfinite(value) else None


def randomization_controls(tf, model, checkpoint, graphs, trained_ig, mean, std,
                           seed=RANDOMIZATION_SEED, steps=32):
    """Sec. S7.2: cumulative replacement of gate head, whole gate and whole model by pristine weights.

    The pristine model is built with ``tf.random.set_seed(seed)``; its zero-initialized
    gate head is replaced by a Glorot-uniform kernel (seed + 1) so that the gate is not
    trivially the identity. The trained model is untouched (an isolated clone is edited).
    """
    from materon.c2db.models import build_megnet

    tf.random.set_seed(seed)
    pristine = build_megnet(ARM)
    head = pristine.get_layer("environment_gate").head
    kernel_shape = head.get_weights()[0].shape
    kernel = tf.keras.initializers.GlorotUniform(seed=seed + 1)(shape=kernel_shape, dtype=tf.float32).numpy()
    head.set_weights([kernel, np.zeros_like(head.get_weights()[1])])

    clone = build_megnet(ARM)
    clone.load_weights(str(checkpoint))
    clone_gate, pristine_gate = clone.get_layer("environment_gate"), pristine.get_layer("environment_gate")
    trained_top5 = np.argsort(-np.abs(trained_ig), axis=1, kind="stable")[:, :5]
    results = {}
    for level in ("gate_head", "whole_gate", "whole_model"):
        if level == "gate_head":
            clone_gate.head.set_weights(pristine_gate.head.get_weights())
        elif level == "whole_gate":
            clone_gate.set_weights(pristine_gate.get_weights())
        else:
            clone.set_weights(pristine.get_weights())
        full = predict(clone, graphs, mean, std)
        bypass = predict(clone, graphs, mean, std, mask=[np.zeros_like(g["x"], dtype=np.float32) for g in graphs])
        atom_ig, splits = mask_path_ig(tf, clone, graphs, std, steps)
        ig = per_graph_sum(atom_ig, splits)
        top5 = np.argsort(-np.abs(ig), axis=1, kind="stable")[:, :5]
        absolute = [spearman(np.abs(trained_ig[i]), np.abs(ig[i])) for i in range(len(graphs))]
        signed = [spearman(trained_ig[i], ig[i]) for i in range(len(graphs))]
        overlap = [len(set(trained_top5[i]) & set(top5[i])) for i in range(len(graphs))]
        defined_abs = [v for v in absolute if v is not None]
        defined_signed = [v for v in signed if v is not None]
        results[level] = {
            "completeness_max_abs_eV": float(np.max(np.abs(ig.sum(axis=1) - (full - bypass)))),
            "median_signed_spearman": float(np.median(defined_signed)) if defined_signed else None,
            "median_absolute_spearman": float(np.median(defined_abs)) if defined_abs else None,
            "top5_overlap_fraction_mean": float(np.mean(overlap) / 5.0),
            "spearman_signed_by_case": signed, "spearman_absolute_by_case": absolute,
        }
    return results


def permute_graph(graph):
    """Reverse the atom order and remap edges (symmetry check, Sec. S7.2)."""
    result = dict(graph)
    order = np.arange(len(graph["z"]))[::-1]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    for key in ("z", "x", "context", "coords"):
        result[key] = np.asarray(graph[key])[order].copy()
    result["edges"] = inverse[np.asarray(graph["edges"], dtype=np.int64)]
    return result, order


def rotate_graph(graph):
    """Rotate coordinates and lattice by 90 degrees in plane (symmetry check)."""
    result = dict(graph)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    result["coords"] = np.asarray(graph["coords"]) @ rotation.T
    result["lattice"] = np.asarray(graph["lattice"]) @ rotation.T
    return result

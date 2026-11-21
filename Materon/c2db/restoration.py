"""Restoration responses and local attributions for substitutions at fixed geometry.

Restoration (Eq. 3 / Eq. S8): with candidate q and parent P at the parent
geometry, R_ij = F(q) - F(q; g_ij <- g_ij^P). Because unsubstituted atoms keep
their attributes, restoring their gate factors returns x~ exactly to the parent
value; the restored inputs x * g are passed to the nested MEGNet backbone.

Also here: the parent-to-candidate joint input IG over x and the chemical
context (Sec. S11.5), the local context and gate paths of one atom with their
IG (Eqs. S6 and S7), element-mean context replacement (Sec. S8) and the replay
of the node Set2Set attention weights (Sec. S10.3).
"""
import numpy as np

from materon.c2db.graph import model_inputs
from materon.c2db.models import backbone_of

ARM = "physical20_environment_gate"


def gate_factors(model, graphs):
    """Gate factors g, shape (n_graphs, n_atoms, 20); graphs must have equal atom counts."""
    args = model_inputs(graphs, arm=ARM, mask=1.)
    _, readouts = model.get_layer("environment_gate")(args[:3], return_readouts=True)
    return readouts["gate"].flat_values.numpy().reshape(len(graphs), len(graphs[0]["z"]), 20)


def _backbone_predict(tf, backbone, args, h_rows, mean, std):
    out = backbone([args[0].with_flat_values(tf.constant(np.concatenate(h_rows)))] + args[3:], training=False)
    return out.numpy().reshape(-1) * std + mean


def environment_restoration(tf, model, parent, candidate, site, mean, std):
    """Channel rules (Sec. S3.2): restore channel j of the gates at all unsubstituted sites.

    Returns native F(q) - F(P), R_env[j] (20), the joint restoration of all
    environment channels and |backbone(x g) - F(q)| as a replay check.
    """
    backbone = backbone_of(model)
    n_atoms = len(parent["z"])
    gate = gate_factors(model, [parent, candidate])
    env = [k for k in range(n_atoms) if k != site]
    xq = np.asarray(candidate["x"], dtype=np.float32)
    hs = [xq * gate[1]]
    for c in range(20):
        h = xq * gate[1]
        h[env, c] = xq[env, c] * gate[0, env, c]
        hs.append(h)
    h = xq * gate[1]
    h[env] = xq[env] * gate[0, env]
    hs.append(h)
    args = model_inputs([candidate] * len(hs), arm=ARM, mask=1.)
    out = _backbone_predict(tf, backbone, args, hs, mean, std)
    fp = float(model(model_inputs([parent], arm=ARM, mask=1.), training=False).numpy()[0, 0] * std + mean)
    fq = float(model(model_inputs([candidate], arm=ARM, mask=1.), training=False).numpy()[0, 0] * std + mean)
    return {"native_eV": fq - fp, "R_env_eV": (out[0] - out[1:21]).tolist(),
            "env_joint_eV": float(out[0] - out[21]), "check_eV": abs(float(out[0]) - fq)}


def single_site_restoration(tf, model, parent, candidate, site, mean, std):
    """Restore the gate vector of one unchanged atom, one channel at a time and jointly (Table S24)."""
    backbone = backbone_of(model)
    gate = gate_factors(model, [parent, candidate])
    x = np.asarray(candidate["x"], dtype=np.float32)
    args = model_inputs([candidate], arm=ARM, mask=1.)
    h = x * gate[1]
    native = float(_backbone_predict(tf, backbone, args, [h], mean, std)[0])
    restored = []
    for j in range(20):
        hj = h.copy()
        hj[site, j] = x[site, j] * gate[0, site, j]
        restored.append(float(_backbone_predict(tf, backbone, args, [hj], mean, std)[0]))
    full = h.copy()
    full[site] = x[site] * gate[0, site]
    joint = native - float(_backbone_predict(tf, backbone, args, [full], mean, std)[0])
    return {"delta_g": (gate[1, site] - gate[0, site]).astype(float).tolist(),
            "native_backbone_eV": native, "restored_eV": restored,
            "R_eV": [native - v for v in restored], "joint_gate_restore_eV": joint}


def joint_input_ig(tf, model, parent, candidates, std):
    """Parent-to-candidate IG over x and context columns 0-19 of all atoms (geometry fixed).

    Returns per-atom, per-attribute direct (x) and neighbor-context contributions
    in eV for 16 and 32 trapezoid intervals (33 gradient evaluations).
    """
    inputs = model_inputs(candidates, arm=ARM, mask=1.0)
    x_tensor, context_tensor = inputs[0], inputs[1]
    x_splits, context_splits = x_tensor.row_splits, context_tensor.row_splits
    x1 = x_tensor.flat_values
    context1 = context_tensor.flat_values[:, :20]
    x0 = tf.constant(np.concatenate([parent["x"] for _ in candidates], axis=0).astype(np.float32))
    context0 = tf.constant(np.concatenate([parent["context"][:, :20] for _ in candidates], axis=0).astype(np.float32))
    dx, dcontext = x1 - x0, context1 - context0
    geometry_context = context_tensor.flat_values[:, 20:]

    @tf.function(reduce_retracing=True)
    def gradients(current_x, current_context):
        with tf.GradientTape() as tape:
            tape.watch((current_x, current_context))
            current = list(inputs)
            current[0] = tf.RaggedTensor.from_row_splits(current_x, x_splits, validate=False)
            full_context = tf.concat((current_context, geometry_context), axis=1)
            current[1] = tf.RaggedTensor.from_row_splits(full_context, context_splits, validate=False)
            objective = tf.reduce_sum(model(current, training=False))
        return tape.gradient(objective, (current_x, current_context))

    acc = {16: [np.zeros(x1.shape), np.zeros(context1.shape)], 32: [np.zeros(x1.shape), np.zeros(context1.shape)]}
    for step in range(33):
        alpha = tf.cast(step / 32.0, x1.dtype)
        grad_x, grad_context = gradients(x0 + alpha * dx, context0 + alpha * dcontext)
        weight = 0.5 if step in (0, 32) else 1.0
        for n in ((16, 32) if step % 2 == 0 else (32,)):
            acc[n][0] += weight * grad_x.numpy().astype(np.float64)
            acc[n][1] += weight * grad_context.numpy().astype(np.float64)
    dx_np, dc_np = dx.numpy().astype(np.float64), dcontext.numpy().astype(np.float64)
    shape = (len(candidates), -1, 20)
    return {"direct%d_node_eV" % n: (dx_np * acc[n][0] * (float(std) / n)).reshape(shape) for n in (16, 32)} | \
           {"neighbor%d_node_eV" % n: (dc_np * acc[n][1] * (float(std) / n)).reshape(shape) for n in (16, 32)}


def local_path_ig(tf, model, parent, candidates, gate_parent, gate_candidates, site, mean, std, kind,
                  tolerance=0.002, l1_tolerance=0.004):
    """IG along the context path (Eq. S6) or the gate path (Eq. S7) of atom ``site``.

    kind='context': c_i(t) = c^P + t (c^q - c^P) over the 20 chemical context entries
                    (all other inputs at their candidate values);
    kind='gate':    g_i(t) = g^P + t (g^q - g^P), passed to the backbone as x * g.
    ``gate_parent`` is (n_atoms, 20); ``gate_candidates`` is (n_candidates, n_atoms, 20).
    32 intervals; refined once to 64 when the completeness error exceeds 0.002 eV
    or the 16/32 L1 difference exceeds 0.004 eV.
    """
    backbone = backbone_of(model)
    n_atoms = len(parent["z"])
    inputs = model_inputs(candidates, arm=ARM, mask=1.)
    sites = tf.constant([[site + n_atoms * k] for k in range(len(candidates))], dtype=tf.int32)
    cflat = inputs[1].flat_values
    xs = np.asarray([g["x"] for g in candidates], dtype=np.float32)
    hflat = tf.constant((xs * gate_candidates).reshape(-1, 20))
    x_site = tf.gather(inputs[0].flat_values, sites[:, 0])
    if kind == "context":
        v0 = np.stack([parent["context"][site, :20]] * len(candidates)).astype(np.float32)
        v1 = np.asarray([g["context"][site, :20] for g in candidates], dtype=np.float32)
    else:
        v0 = np.stack([gate_parent[site]] * len(candidates)).astype(np.float32)
        v1 = np.asarray(gate_candidates[:, site], dtype=np.float32)

    def evaluate(parameter):
        if kind == "context":
            chemical = tf.tensor_scatter_nd_update(cflat[:, :20], sites, parameter)
            current = list(inputs)
            current[1] = inputs[1].with_flat_values(tf.concat([chemical, cflat[:, 20:]], axis=1))
            return model(current, training=False)
        h = tf.tensor_scatter_nd_update(hflat, sites, x_site * parameter)
        return backbone([inputs[0].with_flat_values(h)] + inputs[3:], training=False)

    @tf.function(reduce_retracing=True)
    def grad(parameter):
        with tf.GradientTape() as tape:
            tape.watch(parameter)
            objective = tf.reduce_sum(evaluate(parameter))
        return tape.gradient(objective, parameter)

    endpoint0 = evaluate(tf.constant(v0)).numpy().reshape(-1).astype(float) * std + mean
    endpoint1 = evaluate(tf.constant(v1)).numpy().reshape(-1).astype(float) * std + mean
    delta = v1 - v0
    integrals = {16: np.zeros(delta.shape), 32: np.zeros(delta.shape)}
    for step in range(33):
        gradient = grad(tf.constant(v0 + (step / 32.) * delta)).numpy().astype(float)
        weight = .5 if step in (0, 32) else 1.
        integrals[32] += weight * gradient
        if step % 2 == 0:
            integrals[16] += weight * gradient
    contributions = {n: delta.astype(float) * value * (std / n) for n, value in integrals.items()}
    effects = endpoint1 - endpoint0
    l1 = np.abs(contributions[16] - contributions[32]).sum(axis=1)
    error = contributions[32].sum(axis=1) - effects
    refine = (abs(error) > tolerance) | (l1 > l1_tolerance)
    refined = None
    if refine.any():
        total = np.zeros(delta.shape)
        for step in range(65):
            gradient = grad(tf.constant(v0 + (step / 64.) * delta)).numpy().astype(float)
            total += (.5 if step in (0, 64) else 1.) * gradient
        refined = delta.astype(float) * total * (std / 64)
    results = []
    for k in range(len(candidates)):
        chosen = refined[k] if refine[k] else contributions[32][k]
        results.append({"endpoints_eV": [float(endpoint0[k]), float(endpoint1[k])],
                        "contributions16_eV": contributions[16][k].tolist(),
                        "contributions32_eV": contributions[32][k].tolist(),
                        "selected_intervals": 64 if refine[k] else 32,
                        "selected_contributions_eV": chosen.tolist(),
                        "completeness_error_eV": float(chosen.sum() - effects[k]),
                        "L1_16_32_eV": float(l1[k])})
    return results


def context_references(graphs, ids):
    """Atom-weighted training-set mean context of each central element, and the global mean."""
    sums, counts = {}, {}
    total = np.zeros(37, dtype=np.float64)
    n_atoms = 0
    for uid in ids:
        z = np.asarray(graphs[uid]["z"], dtype=np.int32)
        context = np.asarray(graphs[uid]["context"], dtype=np.float64)
        total += context.sum(axis=0)
        n_atoms += len(context)
        for element in np.unique(z):
            selected = context[z == element]
            sums[element] = sums.get(element, np.zeros(37, dtype=np.float64)) + selected.sum(axis=0)
            counts[element] = counts.get(element, 0) + len(selected)
    return {int(z): sums[z] / counts[z] for z in sums}, total / n_atoms


def replace_context(graph, means, global_mean, section="all"):
    """Replace the context of each atom by its element's training mean ('all', 'chemistry' or 'geometry')."""
    result = dict(graph)
    context = np.array(graph["context"], dtype=np.float32, copy=True)
    z = np.asarray(graph["z"], dtype=np.int32)
    for element in np.unique(z):
        mean = means.get(int(element), global_mean).astype(np.float32)
        rows = z == element
        if section == "all":
            context[rows] = mean
        elif section == "chemistry":
            context[rows, :20] = mean[:20]
        else:
            context[rows, 20:] = mean[20:]
    result["context"] = context
    return result


def set2set_node_attention(tf, backbone, backbone_args, n_atoms, n_graphs):
    """Replay the node Set2Set attention weights of all T iterations (Table S29)."""
    from kgcnn.layers.pool.set2set import PoolingSet2Set

    pools = [layer for layer in backbone.layers if isinstance(layer, PoolingSet2Set)]
    extracted = []
    for layer in pools:
        submodel = tf.keras.Model(backbone.inputs, [layer.input, layer.output])
        memory, pooled = submodel(backbone_args, training=False)
        extracted.append((layer, memory, pooled))
    node = [item for item in extracted
            if np.array_equal(item[1].row_lengths().numpy(), [n_atoms] * n_graphs)]
    if len(node) != 1:
        raise ValueError("could not identify the node Set2Set pool")
    layer, memory, pooled = node[0]
    m, lengths, index = memory.values, memory.row_lengths(), memory.value_rowids()
    qstar = layer.qstar0(m, index, lengths)
    weights = []
    for _ in range(layer.T):
        q = layer.lay_lstm(qstar, training=False)
        et = layer.f_et(m, tf.repeat(q, lengths, axis=0))
        a = tf.exp(et - layer.get_scale_per_sample(et, index, lengths))
        a = a * layer.get_norm(a, index, lengths)
        weights.append(a.numpy().reshape(n_graphs, n_atoms))
        r = tf.math.segment_sum(m * a[:, None], index)
        qstar = tf.concat([q, r], axis=1)[:, None, :]
    attention = np.stack(weights, axis=1)
    return attention, float(tf.reduce_max(tf.abs(qstar - pooled)).numpy())

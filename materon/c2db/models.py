"""kgcnn 2.1 MEGNet predictors of the C2DB study and their paired initialization.

Four predictors are compared (Sec. S2.3): no gate, global static gate,
element-conditioned gate (context set to zero) and environment gate. The
atomic-number-embedding arm is built only so that the paired initialization
reproduces the random-number sequence of the original runs.
"""
import numpy as np
import tensorflow as tf
from kgcnn.literature.Megnet import make_crystal_model

from materon.c2db.graph import ARM_NAMES, DEFAULT_ARM, GATE_ARMS
from materon.gates.environment import EnvironmentGate, GlobalFeatureGate, ZeroEnvironment

PARITY_ATOL = 2e-5
PARITY_RTOL = 2e-5


def _check_arm(arm):
    if arm not in ARM_NAMES:
        raise ValueError("arm must be one of %s" % (", ".join(ARM_NAMES),))


def _node_branch(arm):
    """Role-specific node inputs and the node representation passed to MEGNet."""
    if arm == "atomic_number_embedding":
        atomic_number = tf.keras.Input((None,), ragged=True, dtype="int32", name="atomic_number")
        embedded = tf.keras.layers.Embedding(
            input_dim=101, output_dim=64, name="atomic_number_embedding")(atomic_number)
        return [atomic_number], embedded
    node = tf.keras.Input((None, 20), ragged=True, name="physical20")
    if arm == "physical20_nogate":
        return [node], node
    if arm == "physical20_global_gate":
        mask = tf.keras.Input((None, 20), ragged=True, name="path_mask")
        return [node, mask], GlobalFeatureGate(name="global_feature_gate")([node, mask])
    if arm == "physical20_element_gate":
        mask = tf.keras.Input((None, 20), ragged=True, name="path_mask")
        zero_context = ZeroEnvironment(name="zero_environment")(node)
        return [node, mask], EnvironmentGate(name="element_gate")([node, zero_context, mask])
    context = tf.keras.Input((None, 37), ragged=True, name="environment")
    mask = tf.keras.Input((None, 20), ragged=True, name="path_mask")
    return [node, context, mask], EnvironmentGate(name="environment_gate")([node, context, mask])


def _megnet_specs():
    return [
        {"shape": (None, 20), "name": "node_features", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "edges", "dtype": "int64", "ragged": True},
        {"shape": (1,), "name": "state", "dtype": "float32", "ragged": False},
        {"shape": (None, 3), "name": "images", "dtype": "int64", "ragged": True},
        {"shape": (3, 3), "name": "lattice", "dtype": "float32", "ragged": False},
    ]


def _build_megnet_backbone(specs):
    # Three MEGNet blocks (64/32/32), node and edge Set2Set (16 channels, T = 3),
    # 25-term Gaussian distance expansion to 5 A (sigma 0.4), 32/16/1 output MLP.
    return make_crystal_model(
        name="MEGNet_backbone", inputs=specs,
        input_embedding={"node": {"input_dim": 95, "output_dim": 64},
                         "graph": {"input_dim": 100, "output_dim": 64}},
        make_distance=True, expand_distance=True,
        gauss_args={"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4},
        meg_block_args={"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32],
                        "env_embed": [64, 32, 32], "activation": "kgcnn>softplus2"},
        set2set_args={"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        node_ff_args={"units": [64, 32], "activation": "kgcnn>softplus2"},
        edge_ff_args={"units": [64, 32], "activation": "kgcnn>softplus2"},
        state_ff_args={"units": [64, 32], "activation": "kgcnn>softplus2"},
        nblocks=3, has_ff=True, dropout=None, use_set2set=True, verbose=0,
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, True], "units": [32, 16, 1],
                    "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]},
    )


def build_megnet(arm=DEFAULT_ARM):
    """kgcnn 2.1 MEGNet with the requested node-input role."""
    _check_arm(arm)
    inputs, node_features = _node_branch(arm)
    if arm == "atomic_number_embedding":
        node_features = tf.keras.layers.Dense(20, name="atomic_to_physical20")(node_features)
    specs = _megnet_specs()
    extra = [tf.keras.Input(**spec) for spec in specs[1:]]
    out = _build_megnet_backbone(specs)([node_features] + extra)
    return tf.keras.Model(inputs + extra, out, name="MEGNet_" + arm)


def backbone_of(model):
    """The nested MEGNet backbone (input: gated attributes x~ plus geometry)."""
    nested = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(nested) != 1:
        raise ValueError("expected exactly one nested kgcnn MEGNet backbone")
    return nested[0]


def load_member(checkpoint, arm=DEFAULT_ARM):
    model = build_megnet(arm)
    model.load_weights(str(checkpoint))
    return model


def _megnet_mlp_role(index):
    roles = ("node_input_ff", "edge_input_ff", "state_input_ff",
             "node_update_ff_0", "edge_update_ff_0", "state_update_ff_0",
             "node_update_ff_1", "edge_update_ff_1", "state_update_ff_1", "output_mlp")
    if index >= len(roles):
        raise ValueError("unexpected kgcnn 2.1 MEGNet MLP ordering")
    return roles[index]


def _megnet_weight_roles(model):
    """Map every backbone variable to a semantic role shared by all arms."""
    backbone = backbone_of(model)
    counts = {"MLP": 0, "MEGnetBlock": 0, "DenseEmbedding": 0, "PoolingSet2Set": 0}
    result = {}

    def add_dense_weights(prefix, layer):
        if len(layer.weights) % 2:
            raise ValueError("unexpected kgcnn DenseEmbedding weight layout at %s" % prefix)
        for index, variable in enumerate(layer.weights):
            parameter = "kernel" if index % 2 == 0 else "bias"
            result["%s/dense_%d/%s" % (prefix, index // 2, parameter)] = variable

    for layer in backbone.layers:
        if not layer.weights:
            continue
        layer_type = type(layer).__name__
        if layer_type == "MLP":
            add_dense_weights("megnet/%s" % _megnet_mlp_role(counts[layer_type]), layer)
            counts[layer_type] += 1
        elif layer_type == "MEGnetBlock":
            block_index = counts[layer_type]
            counts[layer_type] += 1
            for role_name, child_stem in (("node_update", "lay_phi_n"), ("edge_update", "lay_phi_e"),
                                          ("state_update", "lay_phi_u")):
                for stage in range(3):
                    child_name = child_stem if stage == 0 else child_stem + "_%d" % stage
                    child = getattr(layer, child_name, None)
                    if child is None or len(child.weights) != 2:
                        raise ValueError("unexpected kgcnn MEGnetBlock layout at %s" % child_name)
                    prefix = "megnet/block_%d/%s/stage_%d" % (block_index, role_name, stage)
                    result[prefix + "/kernel"] = child.weights[0]
                    result[prefix + "/bias"] = child.weights[1]
        elif layer_type == "DenseEmbedding":
            add_dense_weights("megnet/readout_embedding_%d" % counts[layer_type], layer)
            counts[layer_type] += 1
        elif layer_type == "PoolingSet2Set":
            for index, variable in enumerate(layer.weights):
                result["megnet/set2set_pool_%d/weight[%d]" % (counts[layer_type], index)] = variable
            counts[layer_type] += 1
        else:
            raise ValueError("unmapped trainable kgcnn layer type: %s" % layer_type)
    if counts["MEGnetBlock"] != 3 or counts["MLP"] != 10:
        raise ValueError("kgcnn 2.1 MEGNet layer roles differ from the inspected architecture")
    return result


def _copy_roles(source_roles, target_roles):
    if set(source_roles) != set(target_roles):
        raise ValueError("weight role mismatch between arms")
    for path in sorted(source_roles):
        source, target = source_roles[path], target_roles[path]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError("weight shape mismatch at %s" % path)
        target.assign(source)


def build_paired_models(seed=42):
    """Build all arms and give them identical initial backbone (and gate) weights.

    The arms are built in ARM_NAMES order after seeding; the environment-gate
    arm is the reference whose backbone weights are copied to every other arm,
    and whose gate weights are copied to the element-conditioned gate.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    models = {arm: build_megnet(arm=arm) for arm in ARM_NAMES}
    role_maps = {arm: _megnet_weight_roles(model) for arm, model in models.items()}
    for arm in ARM_NAMES:
        if arm != DEFAULT_ARM:
            _copy_roles(role_maps[DEFAULT_ARM], role_maps[arm])
    source = models["physical20_environment_gate"].get_layer("environment_gate")
    target = models["physical20_element_gate"].get_layer("element_gate")
    if len(source.weights) != len(target.weights):
        raise ValueError("element and environment gate weight counts differ")
    for source_variable, target_variable in zip(source.weights, target.weights):
        target_variable.assign(source_variable)
    return models


def validate_initial_parity(models, sample_graphs, model_inputs):
    """Before training, every gate arm equals the no-gate arm (identity gate) and so does mask 0."""
    reference = models["physical20_nogate"](
        model_inputs(sample_graphs, "physical20_nogate"), training=False).numpy()
    zero_masks = [np.zeros_like(graph["x"], dtype=np.float32) for graph in sample_graphs]
    errors = {}
    for arm in GATE_ARMS:
        identity = models[arm](model_inputs(sample_graphs, arm), training=False).numpy()
        bypass = models[arm](model_inputs(sample_graphs, arm, mask=zero_masks), training=False).numpy()
        errors[arm] = (float(np.max(np.abs(identity - reference))), float(np.max(np.abs(bypass - reference))))
    tolerance = PARITY_ATOL + PARITY_RTOL * max(1.0, float(np.max(np.abs(reference))))
    failures = [arm for arm, values in errors.items() if not all(np.isfinite(v) and v <= tolerance for v in values)]
    if failures:
        raise ValueError("initial identity/bypass parity failed: " + ", ".join(failures))
    return errors

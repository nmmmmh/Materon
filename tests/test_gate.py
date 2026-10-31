"""Shape and bound tests of the Materon gates on random tensors (no data, no training).

Boundary conditions covered:
* environment gate: every factor in [0.5, 1.5] even for extreme head outputs;
  g = 1 and output = input when the correction is zero (untrained zero head,
  or path mask 0); correction = x (g - 1);
* E/I gate: output shape equals input shape; with the correction disabled
  (feature mask 0) the output equals the input; CGCNN and MEGNet variable
  layouts compute the same function at native masks.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from materon.gates.ei import BiomimeticEIGate  # noqa: E402
from materon.gates.environment import EnvironmentGate  # noqa: E402

ROWS = [3, 5, 1]


def _ragged(width, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    values = (scale * rng.normal(size=(sum(ROWS), width))).astype(np.float32)
    return tf.RaggedTensor.from_row_lengths(values, ROWS)


def _environment_inputs(mask_value=1.0, scale=1.0):
    x = _ragged(20, 0, scale)
    context = _ragged(37, 1, scale)
    mask = x.with_flat_values(tf.fill(tf.shape(x.flat_values), mask_value))
    return x, context, mask


def test_environment_gate_identity_at_initialization():
    gate = EnvironmentGate()
    x, context, mask = _environment_inputs()
    out, readouts = gate([x, context, mask], return_readouts=True)
    assert out.shape.as_list()[-1] == 20
    np.testing.assert_array_equal(out.row_lengths().numpy(), ROWS)
    np.testing.assert_allclose(readouts["gate"].flat_values.numpy(), 1.0)
    np.testing.assert_allclose(readouts["correction"].flat_values.numpy(), 0.0)
    np.testing.assert_allclose(out.flat_values.numpy(), x.flat_values.numpy())


def test_environment_gate_bounds_and_correction():
    gate = EnvironmentGate()
    x, context, mask = _environment_inputs(scale=50.0)
    gate([x, context, mask])
    rng = np.random.default_rng(2)
    gate.head.set_weights([(100.0 * rng.normal(size=w.shape)).astype(np.float32) for w in gate.head.get_weights()])
    out, readouts = gate([x, context, mask], return_readouts=True)
    g = readouts["gate"].flat_values.numpy()
    assert g.min() >= 0.5 and g.max() <= 1.5
    assert np.any(np.abs(g - 1.0) > 0.1)
    xv = x.flat_values.numpy()
    np.testing.assert_allclose(readouts["correction"].flat_values.numpy(), xv * (g - 1.0), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(out.flat_values.numpy(), xv * g, rtol=1e-5, atol=1e-4)


def test_environment_gate_zero_mask_is_identity():
    gate = EnvironmentGate()
    x, context, mask = _environment_inputs(mask_value=0.0)
    gate([x, context, mask])
    gate.head.set_weights([np.ones(w.shape, np.float32) for w in gate.head.get_weights()])
    out, readouts = gate([x, context, mask], return_readouts=True)
    np.testing.assert_allclose(readouts["correction"].flat_values.numpy(), 0.0)
    np.testing.assert_allclose(out.flat_values.numpy(), x.flat_values.numpy())


def _ei_config():
    return dict(reduced_dim=8, comb_dropout=0.01, use_shared_expert=True, low_rank_gate=False,
                gate_low_rank_dim=16, use_dimension_gating=True, dimension_gating_type="soft",
                synergy_init=0.1, synergy_min=0.005, synergy_max=0.5, gate_seed=42)


def test_ei_gate_shape_and_disabled_correction():
    x = _ragged(20, 3)
    gate = BiomimeticEIGate(20, **_ei_config())
    out, analysis = gate(x, return_analysis=True)
    np.testing.assert_array_equal(out.row_lengths().numpy(), ROWS)
    assert out.flat_values.shape == x.flat_values.shape
    assert 0.005 <= float(analysis["synergy"]) <= 0.5
    gate.set_feature_mask(np.zeros(20))
    np.testing.assert_allclose(gate(x).flat_values.numpy(), x.flat_values.numpy())
    gate.reset_feature_mask()


def test_ei_gate_layouts_agree():
    x = _ragged(20, 4)
    cgcnn = BiomimeticEIGate(20, pathway_controls=False, **_ei_config())
    megnet = BiomimeticEIGate(20, pathway_controls=True, **_ei_config())
    node_mask = x.with_flat_values(tf.ones((sum(ROWS), 1)))
    cgcnn(x)
    megnet([x, node_mask])
    assert len(cgcnn.non_trainable_weights) == 1 and len(megnet.non_trainable_weights) == 5
    megnet.set_weights(cgcnn.get_weights()[:-1] + megnet.get_weights()[-5:])
    np.testing.assert_allclose(megnet([x, node_mask]).flat_values.numpy(), cgcnn(x).flat_values.numpy(),
                               rtol=1e-6, atol=1e-6)
    megnet.set_pathway_state(dimension=0.0, synergy=0.0)
    np.testing.assert_allclose(megnet([x, node_mask]).flat_values.numpy(), x.flat_values.numpy())

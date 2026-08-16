"""Contract and regression checks for test cgcnn input contract static in the TF2.9 multi-backbone engine."""

from materon_tf29_engine.backbones.configs import CGCNN_INPUTS, MEGNET_INPUTS

def _spec(inputs, name):
    matches = [x for x in inputs if x["name"] == name]
    assert len(matches) == 1
    return matches[0]

def test_cgcnn_periodic_translation_dtype():
    # CGCNN adds cell translations to float32 fractional coordinates.
    assert _spec(CGCNN_INPUTS, "node_frac_coordinates")["dtype"] == "float32"
    assert _spec(CGCNN_INPUTS, "range_image")["dtype"] == "float32"
    assert _spec(CGCNN_INPUTS, "range_indices")["dtype"] == "int64"

def test_megnet_contract_unchanged():
    # MEGNet's previously validated official contract must not be altered.
    assert _spec(MEGNET_INPUTS, "node_coordinates")["dtype"] == "float32"
    assert _spec(MEGNET_INPUTS, "range_image")["dtype"] == "int64"

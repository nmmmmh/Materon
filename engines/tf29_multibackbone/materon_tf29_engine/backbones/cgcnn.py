"""Cgcnn utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
from .configs import CGCNN_CONFIG
from .utils import supported_kwargs,classification_output
from materon_tf29_engine.features.element_prior import load_prior_bundle
from materon_tf29_engine.integration.official_builder_patch import build_official_with_materon
from materon_tf29_engine.integration.weight_pairing import copy_backbone_weights

def _builder():
    from kgcnn.literature.CGCNN import make_crystal_model
    return make_crystal_model

def _attempts(fn,c):
    out=[]
    # kgcnn 2.1 nested input_embedding, later 2.x input_node_embedding.
    out.append(dict(c))
    d=dict(c); emb=d.pop("input_embedding",None); d["input_node_embedding"]=(emb or {}).get("node",{"input_dim":95,"output_dim":64}); out.append(d)
    return [supported_kwargs(fn,x) for x in out]

def build_baseline(task_type="regression",config=None):
    fn=_builder(); c=classification_output(CGCNN_CONFIG,"cgcnn",task_type); c.update(config or {}); errors=[]
    for a in _attempts(fn,c):
        try: return fn(**a)
        except Exception as e: errors.append(repr(e))
    raise RuntimeError("CGCNN official builder failed for supported signatures: %s"%errors)

def build_pair(task_type="regression",feature_mode="grouped_pca",shuffle_seed=None,gate_config=None,config=None):
    baseline=build_baseline(task_type,config); fn=_builder(); c=classification_output(CGCNN_CONFIG,"cgcnn",task_type); c.update(config or {})
    prior=load_prior_bundle(feature_mode,shuffle_seed); gate={"rank":4,"alpha_max":0.10,"kernel_l2":1e-4}; gate.update(gate_config or {})
    errors=[]
    for a in _attempts(fn,c):
        try:
            materon=build_official_with_materon(fn,a,prior,95,64,gate); copy_backbone_weights(baseline,materon); return baseline,materon,"materon_node_embedding"
        except Exception as e: errors.append(repr(e))
    raise RuntimeError("CGCNN Materon builder failed: %s"%errors)

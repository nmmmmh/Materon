"""Megnet utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import tensorflow as tf
from .configs import MEGNET_CONFIG
from .utils import supported_kwargs,classification_output
from materon_tf29_engine.features.element_prior import load_prior_bundle
from materon_tf29_engine.integration.official_builder_patch import build_official_with_materon
from materon_tf29_engine.integration.weight_pairing import copy_backbone_weights

def _builder():
    from kgcnn.literature.Megnet import make_crystal_model
    return make_crystal_model

def build_baseline(task_type="regression",config=None):
    fn=_builder(); c=classification_output(MEGNET_CONFIG,"megnet",task_type); c.update(config or {})
    return fn(**supported_kwargs(fn,c))

def build_pair(task_type="regression",feature_mode="grouped_pca",shuffle_seed=None,gate_config=None,config=None):
    baseline=build_baseline(task_type,config); fn=_builder(); c=classification_output(MEGNET_CONFIG,"megnet",task_type); c.update(config or {})
    prior=load_prior_bundle(feature_mode,shuffle_seed)
    gate={"rank":4,"alpha_max":0.15,"kernel_l2":1e-4}; gate.update(gate_config or {})
    materon=build_official_with_materon(fn,supported_kwargs(fn,c),prior,95,64,gate)
    copy_backbone_weights(baseline,materon)
    return baseline,materon,"materon_node_embedding"

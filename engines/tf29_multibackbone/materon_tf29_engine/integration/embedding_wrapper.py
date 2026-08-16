# -*- coding: utf-8 -*-
"""Embedding wrapper utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
from typing import Any
import tensorflow as tf
from materon_tf29_engine.core.small_data_ei import SmallDataEIGate
from materon_tf29_engine.features.element_prior import ElementPriorLookup

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class PriorAwareNodeEmbedding(tf.keras.layers.Layer):
    """Wrap the exact embedding layer created by the installed kgcnn builder."""
    def __init__(self, base_layer, prior_table, feature_names, rank=4, alpha_max=0.15, kernel_l2=1e-4, **kwargs:Any):
        super().__init__(**kwargs); self.base_layer=base_layer; self.prior_table=prior_table; self.feature_names=list(feature_names)
        self.lookup=ElementPriorLookup(prior_table,feature_names,name="physical_prior")
        output_dim=getattr(base_layer,"output_dim",None)
        if output_dim is None:
            cfg=base_layer.get_config() if hasattr(base_layer,"get_config") else {}; output_dim=cfg.get("output_dim")
        if output_dim is None: raise ValueError("Cannot determine base embedding output_dim")
        self.gate=SmallDataEIGate(feature_dim=int(output_dim),context_dim=len(self.feature_names),rank=rank,alpha_max=alpha_max,kernel_l2=kernel_l2,name="ei_gate")
    def call(self,atomic_numbers,training=False):
        emb=self.base_layer(atomic_numbers); context=self.lookup(atomic_numbers); return self.gate([emb,context],training=training)
    @property
    def adapter_only_weights(self): return list(self.lookup.weights)+list(self.gate.weights)
    def get_analysis_tensors(self,symbolic=False): return self.gate.get_analysis_tensors(symbolic=symbolic)
    def get_config(self):
        # Full-model serialization is not relied upon because some kgcnn-2.1 layers have incomplete configs.
        c=super().get_config(); c.update({"prior_table":self.prior_table,"feature_names":self.feature_names,"rank":self.gate.rank,"alpha_max":self.gate.alpha_max,"kernel_l2":self.gate.kernel_l2,"base_layer_class":self.base_layer.__class__.__name__}); return c

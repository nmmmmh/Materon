# -*- coding: utf-8 -*-
"""Element prior utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from .schema import RAW_FEATURE_COLUMNS

def _root(): return Path(__file__).resolve().parent

def load_prior_bundle(mode:str="grouped_pca", shuffle_seed:Optional[int]=None, max_atomic_number:int=100)->Dict:
    raw=pd.read_csv(_root()/"physic_features_pool.csv").sort_values("Z")
    if int(raw.Z.max())<max_atomic_number: raise ValueError("Prior table does not cover requested atomic number")
    if mode in {"grouped_pca","shuffled_grouped_pca"}:
        frame=pd.read_csv(_root()/"grouped_pca_context.csv").sort_values("Z")
        names=[c for c in frame.columns if c!="Z"]
    elif mode=="full20": frame=raw[["Z"]+RAW_FEATURE_COLUMNS]; names=list(RAW_FEATURE_COLUMNS)
    elif mode=="z_only":
        z=raw.Z.to_numpy(np.float32); z=(z-z.mean())/z.std(); frame=pd.DataFrame({"Z":raw.Z,"Z_standardized":z}); names=["Z_standardized"]
    elif mode=="none":
        return {"table":np.zeros((max_atomic_number+1,0),np.float32),"names":[],"mode":mode,"metadata":{}}
    else: raise ValueError("Unknown prior mode: %s"%mode)
    values=frame[names].to_numpy(np.float32)
    if mode=="shuffled_grouped_pca":
        rng=np.random.RandomState(0 if shuffle_seed is None else int(shuffle_seed)); values=values[rng.permutation(len(values))]
    table=np.zeros((max_atomic_number+1,len(names)),np.float32)
    for z,row in zip(frame.Z.astype(int),values):
        if 1<=z<=max_atomic_number: table[z]=row
    metadata=json.loads((_root()/"feature_metadata.json").read_text(encoding="utf-8"))
    return {"table":table,"names":names,"mode":mode,"metadata":metadata}

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class ElementPriorLookup(tf.keras.layers.Layer):
    def __init__(self, table, feature_names=None, **kwargs):
        super().__init__(**kwargs); a=np.asarray(table,np.float32)
        if a.ndim!=2: raise ValueError("table must be rank 2")
        self.table_list=a.tolist(); self.max_z=a.shape[0]-1; self.context_dim=a.shape[1]; self.feature_names=list(feature_names or [])
    def build(self,input_shape):
        # Fixed physical prior is data, not a model parameter. Keeping it as a
        # constant ensures added-parameter accounting measures only the adapter.
        self.table=tf.constant(self.table_list,dtype=tf.float32); super().build(input_shape)
    def _lookup(self,z):
        z=tf.cast(z,tf.int32); tf.debugging.assert_greater_equal(z,0); tf.debugging.assert_less_equal(z,self.max_z,message="Atomic number exceeds prior table; no clipping is allowed")
        return tf.gather(self.table,z)
    def call(self,z):
        is_ragged=isinstance(z,tf.RaggedTensor) or (hasattr(z,"flat_values") and hasattr(z,"with_flat_values"))
        return z.with_flat_values(self._lookup(z.flat_values)) if is_ragged else self._lookup(z)
    def get_config(self):
        c=super().get_config(); c.update({"table":self.table_list,"feature_names":self.feature_names}); return c

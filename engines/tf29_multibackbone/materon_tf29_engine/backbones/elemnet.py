"""Elemnet utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import tensorflow as tf
from materon_tf29_engine.features.element_prior import load_prior_bundle
from materon_tf29_engine.core.composition_ei import CompositionSimplexEIGate

def _head(x,task_type,hidden_units,name_prefix="elemnet"):
    for idx,u in enumerate(hidden_units): x=tf.keras.layers.Dense(u,activation="relu",name="%s_dense_%02d"%(name_prefix,idx))(x)
    act="sigmoid" if task_type=="classification" else "linear"
    return tf.keras.layers.Dense(1,activation=act,name="%s_output"%name_prefix)(x)

def build_baseline(task_type="regression",max_atomic_number=100,hidden_units=(256,256,128,128,64)):
    inp=tf.keras.layers.Input((max_atomic_number,),dtype="float32",name="composition")
    return tf.keras.Model(inp,_head(inp,task_type,hidden_units),name="ElemNetSmall")

def build_pair(task_type="regression",max_atomic_number=100,hidden_units=(256,256,128,128,64),feature_mode="grouped_pca",shuffle_seed=None,gate_config=None):
    baseline=build_baseline(task_type,max_atomic_number,hidden_units)
    prior=load_prior_bundle(feature_mode,shuffle_seed,max_atomic_number)
    inp=tf.keras.layers.Input((max_atomic_number,),dtype="float32",name="composition")
    cfg={"rank":4,"alpha_max":0.10}; cfg.update(gate_config or {})
    gate=CompositionSimplexEIGate(prior["table"],feature_names=prior["names"],name="materon_composition_gate",**cfg)
    out=_head(gate(inp),task_type,hidden_units)
    materon=tf.keras.Model(inp,out,name="ElemNetSmall_Materon")
    # Adapter has deterministic initializers; copy dense backbone explicitly by matching suffix names/order.
    src=[l for l in baseline.layers if l.weights]; dst=[l for l in materon.layers if l.weights and l.name!="materon_composition_gate"]
    if len(src)!=len(dst): raise RuntimeError("ElemNet backbone layer count mismatch")
    for a,b in zip(src,dst):
        if [tuple(x.shape) for x in a.weights]!=[tuple(x.shape) for x in b.weights]: raise RuntimeError("ElemNet backbone shape mismatch")
        b.set_weights(a.get_weights())
    return baseline,materon,"materon_composition_gate"

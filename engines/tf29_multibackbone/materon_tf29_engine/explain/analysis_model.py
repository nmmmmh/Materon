"""Analysis model utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from materon_tf29_engine.integration.weight_pairing import find_adapter_layer

def analysis_tensors(model,symbolic=False):
    adapter=find_adapter_layer(model)
    if not hasattr(adapter,"get_analysis_tensors"):
        raise RuntimeError("Adapter does not expose analysis tensors")
    try: tensors=adapter.get_analysis_tensors(symbolic=symbolic)
    except TypeError: tensors=adapter.get_analysis_tensors()
    if not tensors:
        raise RuntimeError("Analysis tensors are empty; build/call the model first")
    return tensors

def build_analysis_model(model):
    tensors=analysis_tensors(model,symbolic=True)
    names=list(tensors)
    outputs=[model.output]+[tensors[k] for k in names]
    return tf.keras.Model(model.inputs,outputs,name=model.name+"_analysis"),["prediction"]+names

def tensor_to_serializable(value):
    if isinstance(value,tf.RaggedTensor):
        return {"flat_values":value.flat_values.numpy(),"row_splits":value.row_splits.numpy()}
    return np.asarray(value)

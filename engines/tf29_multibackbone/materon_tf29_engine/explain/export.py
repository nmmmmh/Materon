"""Export utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from materon_tf29_engine.integration.embedding_wrapper import PriorAwareNodeEmbedding
from materon_tf29_engine.core.composition_ei import CompositionSimplexEIGate

def adapter_state(model):
    rows=[]
    for l in model.submodules:
        if isinstance(l,PriorAwareNodeEmbedding): rows.append({"type":"node_embedding","name":l.name,"alpha":float(l.gate.alpha.numpy()),"raw_alpha":float(l.gate.raw_alpha.numpy()),"feature_names":l.feature_names,"adapter_params":int(sum(np.prod(v.shape) for v in l.adapter_only_weights))})
        elif isinstance(l,CompositionSimplexEIGate): rows.append({"type":"composition_simplex","name":l.name,"alpha":float(l.alpha.numpy()),"raw_alpha":float(l.raw_alpha.numpy()),"context_dim":l.context_dim,"feature_names":l.feature_names})
    return rows
def save_adapter_state(model,path): Path(path).write_text(json.dumps(adapter_state(model),indent=2,ensure_ascii=False),encoding="utf-8")

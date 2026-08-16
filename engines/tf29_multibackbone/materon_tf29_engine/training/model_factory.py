"""Model factory utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
from materon_tf29_engine.backbones import megnet,cgcnn,elemnet

def build_model(backbone,variant,task_type,max_atomic_number=100,feature_mode="grouped_pca",shuffle_seed=None):
    module={"megnet":megnet,"cgcnn":cgcnn,"elemnet":elemnet}[backbone]
    if variant=="baseline":
        model=module.build_baseline(task_type=task_type,**({"max_atomic_number":max_atomic_number} if backbone=="elemnet" else {})); return model,None,None
    if variant!="materon": raise ValueError("variant must be baseline or materon")
    kwargs={"task_type":task_type,"feature_mode":feature_mode,"shuffle_seed":shuffle_seed}
    if backbone=="elemnet": kwargs["max_atomic_number"]=max_atomic_number
    baseline,model,layer=module.build_pair(**kwargs); return model,baseline,layer

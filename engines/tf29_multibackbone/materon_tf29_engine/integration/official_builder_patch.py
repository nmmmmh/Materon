# -*- coding: utf-8 -*-
"""Inject Materon while the installed official kgcnn builder creates its node embedding.

No clone_model and no reimplementation of MEGNet/CGCNN are used.
"""
from __future__ import annotations
from contextlib import contextmanager
import inspect, sys, threading
from typing import Dict, List, Tuple
from .embedding_wrapper import PriorAwareNodeEmbedding

_LOCK=threading.RLock()

def _related_modules(builder):
    modules=[]
    funcs=[builder, inspect.unwrap(builder)]
    for f in funcs:
        m=inspect.getmodule(f)
        if m and m not in modules: modules.append(m)
    prefixes=set()
    for m in modules:
        parts=m.__name__.split('.')
        if len(parts)>=3: prefixes.add('.'.join(parts[:3]))
    for name,m in list(sys.modules.items()):
        if m and any(name.startswith(p) for p in prefixes) and m not in modules: modules.append(m)
    return modules

def _dims(layer,args,kwargs):
    inp=kwargs.get("input_dim",getattr(layer,"input_dim",None)); out=kwargs.get("output_dim",getattr(layer,"output_dim",None))
    if inp is None or out is None:
        try:
            cfg=layer.get_config(); inp=cfg.get("input_dim",inp); out=cfg.get("output_dim",out)
        except Exception: pass
    if (inp is None or out is None) and args:
        try:
            sig=inspect.signature(layer.__class__.__init__); b=sig.bind_partial(None,*args,**kwargs); inp=b.arguments.get("input_dim",inp); out=b.arguments.get("output_dim",out)
        except Exception: pass
    return inp,out

def discover_embedding_symbols(builder)->List[Tuple[object,str,object]]:
    found=[]
    for module in _related_modules(builder):
        for name in ("OptionalInputEmbedding","Embedding"):
            obj=getattr(module,name,None)
            if obj is not None and callable(obj): found.append((module,name,obj))
    # Deduplicate module/name.
    unique=[]; seen=set()
    for item in found:
        key=(id(item[0]),item[1])
        if key not in seen: seen.add(key); unique.append(item)
    return unique

def inspect_builder_contract(builder)->Dict:
    modules=_related_modules(builder); symbols=discover_embedding_symbols(builder)
    return {"builder_module":getattr(inspect.getmodule(builder),"__name__",None),"builder_file":inspect.getsourcefile(inspect.unwrap(builder)),"related_modules":[m.__name__ for m in modules],"patch_symbols":[m.__name__+"."+n for m,n,_ in symbols]}

@contextmanager
def _patched_embedding(builder, prior_bundle, expected_input_dim:int, expected_output_dim:int, gate_config:Dict):
    candidates=discover_embedding_symbols(builder)
    if not candidates: raise RuntimeError("No embedding constructor was found in the installed kgcnn builder. Contract: %s"%inspect_builder_contract(builder))
    state={"count":0,"layer_names":[]}; originals=[]
    def make_proxy(original):
        def proxy(*args,**kwargs):
            base=original(*args,**kwargs); inp,out=_dims(base,args,kwargs)
            if state["count"]==0 and inp is not None and out is not None and int(inp)==int(expected_input_dim) and int(out)==int(expected_output_dim):
                state["count"]+=1
                wrapped=PriorAwareNodeEmbedding(base_layer=base,prior_table=prior_bundle["table"],feature_names=prior_bundle["names"],name="materon_node_embedding",**gate_config)
                state["layer_names"].append(wrapped.name); return wrapped
            return base
        return proxy
    with _LOCK:
        try:
            for module,name,original in candidates:
                originals.append((module,name,original)); setattr(module,name,make_proxy(original))
            yield state
        finally:
            for module,name,original in originals: setattr(module,name,original)

def build_official_with_materon(builder, model_config:Dict, prior_bundle:Dict, expected_input_dim:int=95, expected_output_dim:int=64, gate_config:Dict=None):
    cfg=dict(gate_config or {})
    with _patched_embedding(builder,prior_bundle,expected_input_dim,expected_output_dim,cfg) as state:
        model=builder(**model_config)
    if state["count"]!=1: raise RuntimeError("Expected exactly one node embedding injection, got %d. Contract: %s"%(state["count"],inspect_builder_contract(builder)))
    model._materon_inserted_layer=state["layer_names"][0]
    model._materon_prior_names=list(prior_bundle["names"])
    return model

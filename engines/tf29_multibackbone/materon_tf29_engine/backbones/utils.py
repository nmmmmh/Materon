"""Utils utilities for the TF2.9 multi-backbone engine."""

import inspect

def supported_kwargs(fn,config):
    sig=inspect.signature(fn); has_var=any(p.kind==p.VAR_KEYWORD for p in sig.parameters.values())
    return dict(config) if has_var else {k:v for k,v in config.items() if k in sig.parameters}

def classification_output(config,backbone,task_type):
    c=dict(config); mlp=dict(c["output_mlp"])
    if task_type=="classification":
        acts=list(mlp["activation"]); acts[-1]="sigmoid"; mlp["activation"]=acts
    c["output_mlp"]=mlp; return c

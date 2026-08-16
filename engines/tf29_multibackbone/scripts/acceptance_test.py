#!/usr/bin/env python3
"""Command-line workflow for acceptance test in the TF2.9 multi-backbone engine."""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))
import argparse,gc,json,tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from matbench.bench import MatbenchBenchmark
from materon_tf29_engine.training.cache import GraphTaskCache,CompositionTaskCache
from materon_tf29_engine.training.model_factory import build_model
from materon_tf29_engine.training.identity import assert_identity,compare_float_outputs
from materon_tf29_engine.integration.weight_pairing import (find_adapter_layer,assert_full_model_weight_parity,snapshot_weights,assert_snapshot_restored,perturb_first_floating_weight)
from materon_tf29_engine.explain.analysis_model import analysis_tensors

def sample(x,n=4): return [v[:n] for v in x] if isinstance(x,(list,tuple)) else x[:n]
def _ref(v):
    try: return v.ref()
    except Exception: return id(v)
def activate(model,value=0.01):
    count=0
    for layer in model.submodules:
        if hasattr(layer,"set_activation") and hasattr(layer,"raw_alpha"):
            layer.set_activation(value); count+=1
    if count==0: raise AssertionError("No activatable E/I layer")
def gradient_check(model,x,y):
    activate(model,0.01)
    target=tf.convert_to_tensor(np.asarray(y[:4],np.float32).reshape(-1,1))
    with tf.GradientTape() as tape:
        pred=model(x,training=True)
        regularization=tf.add_n(model.losses) if model.losses else tf.constant(0.0)
        loss=tf.reduce_mean(tf.abs(tf.cast(pred,tf.float32)-target))+regularization
    grads=tape.gradient(loss,model.trainable_weights)
    finite=0; grad_by_ref={}
    for var,grad in zip(model.trainable_weights,grads):
        if grad is None: continue
        dense=tf.convert_to_tensor(grad)
        tf.debugging.assert_all_finite(dense,"Non-finite gradient: %s"%var.name)
        grad_by_ref[_ref(var)]=dense; finite+=1
    if finite==0: raise AssertionError("No finite trainable gradients")
    adapter=find_adapter_layer(model); non_alpha=[]
    for var in adapter.adapter_only_weights:
        if not getattr(var,"trainable",False) or "raw_alpha" in var.name: continue
        grad=grad_by_ref.get(_ref(var))
        if grad is not None: non_alpha.append(float(tf.reduce_max(tf.abs(grad)).numpy()))
    if not non_alpha or max(non_alpha)<=0.0:
        raise AssertionError("Adapter has no non-zero internal gradient after activation")
    return {"loss":float(loss.numpy()),"finite_gradient_tensors":finite,"max_adapter_internal_gradient":max(non_alpha)}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--backbones",default="megnet,cgcnn,elemnet"); p.add_argument("--graph-task",default="matbench_phonons"); p.add_argument("--composition-task",default="matbench_expt_gap"); p.add_argument("--atol",type=float,default=1e-6); p.add_argument("--rtol",type=float,default=5e-7); p.add_argument("--max-ulp",type=int,default=4); a=p.parse_args()
 report={}
 for backbone in [x.strip() for x in a.backbones.split(',') if x.strip()]:
  tf.keras.backend.clear_session(); gc.collect(); tf.random.set_seed(0); np.random.seed(0)
  task_name=a.composition_task if backbone=="elemnet" else a.graph_task
  mb=MatbenchBenchmark(subset=[task_name],autoload=False); task=next(iter(mb.tasks)); task.load(); fold=list(task.folds)[0]
  cache=CompositionTaskCache(task) if backbone=="elemnet" else GraphTaskCache(task,backbone)
  xtr,ytr,xt,yt=cache.fold(fold); xs=sample(xtr)
  if backbone=="cgcnn":
   print("CGCNN INPUT DTYPES:", [str(x.dtype) for x in xs], flush=True)
   if str(xs[1].dtype)!="<dtype: 'float32'>" or str(xs[4].dtype)!="<dtype: 'float32'>":
    raise AssertionError("CGCNN fractional coordinates and range_image must both be float32")
  model,base,layer=build_model(backbone,"materon",str(task.metadata["task_type"]))
  diff,parity=assert_identity(base,model,xs,a.atol,a.rtol,a.max_ulp)
  total=model.count_params(); basep=base.count_params(); added=total-basep
  if added<=0 or added/basep>=0.01: raise AssertionError("Adapter parameter ratio must be in (0,1%%): %s"%(added/basep))
  _=model(xs,training=False); components=analysis_tensors(model)
  required={"alpha","competition","excitation","inhibition"}
  required.update({"signed_score","physical_context","simplex_input_sum","simplex_output_sum"} if backbone=="elemnet" else {"competition_logit","context_contribution","physical_context","signed_latent","delta"})
  if not required.issubset(set(components)): raise AssertionError("Missing analysis tensors: %s"%(required-set(components)))
  grad_report=gradient_check(model,xs,ytr)
  pred=np.asarray(model(xs,training=False))
  with tempfile.TemporaryDirectory() as d:
   path=str(Path(d)/"weights.h5")
   original_snapshot=snapshot_weights(model)
   model.save_weights(path)

   # Same-object reload: perturb one tensor, reload the checkpoint, and require
   # every weight to return bit-exactly to the pre-save snapshot.
   perturbation=perturb_first_floating_weight(model,amount=0.125)
   model.load_weights(path)
   restored_ok,restored_report=assert_snapshot_restored(model,original_snapshot,atol=0.0)
   if not restored_ok:
    raise AssertionError("Same-model checkpoint restore failed: %s"%restored_report)
   pred_same=np.asarray(model(xs,training=False))
   same_graph_numerics=compare_float_outputs(pred,pred_same,atol=a.atol,rtol=a.rtol,max_ulp=a.max_ulp)
   if not same_graph_numerics["passed"]:
    raise AssertionError("Same-model output restore failed: %s"%same_graph_numerics)

   # Fresh-model reload: reconstruct the complete architecture, load the same
   # checkpoint, and require every model weight to be bit-exactly equal.
   model2,base2,_=build_model(backbone,"materon",str(task.metadata["task_type"]))
   model2.load_weights(path)
   fresh_ok,fresh_weight_report=assert_full_model_weight_parity(model,model2,atol=0.0)
   if not fresh_ok:
    raise AssertionError("Fresh-model checkpoint weights failed: %s"%fresh_weight_report)
   pred2=np.asarray(model2(xs,training=False))

  # Two separately built GPU graphs can follow different TF32/fusion paths.
  # Record their output difference for diagnosis, but checkpoint correctness
  # is determined by exact all-weight parity above.
  cross_graph_numerics=compare_float_outputs(
   pred,pred2,atol=a.atol,rtol=a.rtol,max_ulp=max(a.max_ulp,128)
  )
  rt=fresh_weight_report["max_abs_difference"]
  if backbone=="elemnet":
   gate=find_adapter_layer(model)
   gate.reset_identity()
   out=gate(xs)
   arr=out.numpy()
   identity_error=float(np.max(np.abs(arr-np.asarray(xs,np.float32))))
   sum_error=float(np.max(np.abs(arr.sum(1)-np.asarray(xs,np.float32).sum(1))))
   if identity_error!=0.0:
    raise AssertionError("ElemNet gate identity is not exact: %.3e"%identity_error)
   if sum_error!=0.0 or np.any(arr<0) or np.any((xs==0)&(arr!=0)):
    raise AssertionError("Simplex invariant failed")
  report[backbone]={"identity_max_abs":diff,"backbone_parity":parity,"baseline_params":basep,"materon_params":total,"added_params":added,"added_ratio":added/basep,"weight_roundtrip_max_abs":rt,"checkpoint_same_model_restore":restored_report,"checkpoint_same_model_perturbation":perturbation,"checkpoint_same_graph_numerics":same_graph_numerics,"checkpoint_fresh_model_weight_parity":fresh_weight_report,"checkpoint_cross_graph_output_diagnostic":cross_graph_numerics,"inserted_layer":layer,"analysis_keys":sorted(components),"gradient_check":grad_report}
 print(json.dumps(report,indent=2)); print("ACCEPTANCE TEST: PASS")
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Command-line workflow for export batch explanations in the TF2.9 multi-backbone engine."""

from __future__ import annotations

# Make the isolated engine package importable when this script is executed
# directly from the release tree without a prior editable install.
from pathlib import Path as _MateronPath
import sys as _materon_sys
_MATERON_ENGINE_ROOT = _MateronPath(__file__).resolve().parents[1]
if str(_MATERON_ENGINE_ROOT) not in _materon_sys.path:
    _materon_sys.path.insert(0, str(_MATERON_ENGINE_ROOT))
import argparse,json
from pathlib import Path
import numpy as np
import tensorflow as tf
from matbench.bench import MatbenchBenchmark
from materon_tf29_engine.training.cache import GraphTaskCache,CompositionTaskCache
from materon_tf29_engine.training.model_factory import build_model
from materon_tf29_engine.explain.analysis_model import analysis_tensors,tensor_to_serializable
from materon_tf29_engine.integration.weight_pairing import find_adapter_layer

def take(x,start,stop): return [v[start:stop] for v in x] if isinstance(x,(list,tuple)) else x[start:stop]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--backbone',choices=['megnet','cgcnn','elemnet'],required=True); p.add_argument('--task',required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--weights',type=Path,required=True); p.add_argument('--split',choices=['train','test'],default='test'); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--feature-mode',choices=['grouped_pca','full20','none','z_only','shuffled_grouped_pca'],default='grouped_pca'); p.add_argument('--outdir',type=Path,required=True); a=p.parse_args()
 mb=MatbenchBenchmark(subset=[a.task],autoload=False); task=next(iter(mb.tasks)); task.load(); fold_id=list(task.folds)[a.fold]
 cache=CompositionTaskCache(task) if a.backbone=='elemnet' else GraphTaskCache(task,a.backbone); xtr,ytr,xt,yt=cache.fold(fold_id); x,y=(xtr,ytr) if a.split=='train' else (xt,yt)
 model,base,layer=build_model(a.backbone,'materon',str(task.metadata['task_type']),feature_mode=a.feature_mode); model.load_weights(str(a.weights)); a.outdir.mkdir(parents=True,exist_ok=True)
 adapter=find_adapter_layer(model); feature_names=list(getattr(adapter,'feature_names',[])); manifest={'backbone':a.backbone,'task':a.task,'fold':a.fold,'split':a.split,'weights':str(a.weights),'inserted_layer':layer,'feature_mode':a.feature_mode,'feature_names':feature_names,'batches':[]}
 for start in range(0,len(y),a.batch_size):
  stop=min(start+a.batch_size,len(y)); xb=take(x,start,stop); pred=np.asarray(model(xb,training=False)); comp=analysis_tensors(model); payload={'prediction':pred,'target':np.asarray(y[start:stop])}
  for name,value in comp.items():
   item=tensor_to_serializable(value)
   if isinstance(item,dict): payload[name+'__flat_values']=item['flat_values']; payload[name+'__row_splits']=item['row_splits']
   else: payload[name]=item
  file='batch_%05d_%05d.npz'%(start,stop); np.savez_compressed(a.outdir/file,**payload); manifest['batches'].append({'file':file,'start':start,'stop':stop})
 (a.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=='__main__': main()

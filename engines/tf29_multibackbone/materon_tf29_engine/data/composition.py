"""Composition utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import numpy as np
from pymatgen.core import Composition

def as_composition(v): return v if isinstance(v,Composition) else (v.composition if hasattr(v,"composition") else Composition(v))
def composition_vector(v,max_atomic_number=100):
    c=as_composition(v).fractional_composition; x=np.zeros(max_atomic_number,np.float32)
    for el,a in c.items():
        z=int(el.Z)
        if not 1<=z<=max_atomic_number: raise ValueError("Element Z=%d outside width %d"%(z,max_atomic_number))
        x[z-1]=float(a)
    if x.sum()<=0: raise ValueError("Empty composition")
    return x/x.sum()
def composition_matrix(values,max_atomic_number=100): return np.stack([composition_vector(v,max_atomic_number) for v in values]).astype(np.float32)

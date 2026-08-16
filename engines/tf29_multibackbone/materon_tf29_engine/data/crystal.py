"""Crystal utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
import numpy as np
from materon_tf29_engine.backbones.configs import MEGNET_INPUTS,CGCNN_INPUTS
CRYSTAL_CALLBACKS={
"graph_labels":lambda st,y:np.expand_dims(y,axis=-1),
"node_coordinates":lambda st,y:np.asarray(st.cart_coords,dtype=np.float32),
"node_frac_coordinates":lambda st,y:np.asarray(st.frac_coords,dtype=np.float32),
"graph_lattice":lambda st,y:np.ascontiguousarray(np.asarray(st.lattice.matrix,dtype=np.float32)),
"charge":lambda st,y:np.asarray([st.charge],dtype=np.float32),
"node_number":lambda st,y:np.asarray(st.atomic_numbers,dtype=np.int64)}
def range_method(cutoff): return [{"map_list":{"method":"set_range_periodic","max_distance":float(cutoff)}}]

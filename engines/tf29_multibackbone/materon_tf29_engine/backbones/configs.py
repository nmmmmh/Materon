"""Configs utilities for the TF2.9 multi-backbone engine."""

MEGNET_INPUTS=[
{"shape":(None,),"name":"node_number","dtype":"float32","ragged":True},
{"shape":(None,3),"name":"node_coordinates","dtype":"float32","ragged":True},
{"shape":(None,2),"name":"range_indices","dtype":"int64","ragged":True},
{"shape":(1,),"name":"charge","dtype":"float32","ragged":False},
{"shape":(None,3),"name":"range_image","dtype":"int64","ragged":True},
{"shape":(3,3),"name":"graph_lattice","dtype":"float32","ragged":False}]
MEGNET_CONFIG={"name":"Megnet","inputs":MEGNET_INPUTS,"input_embedding":{"node":{"input_dim":95,"output_dim":64},"graph":{"input_dim":100,"output_dim":64}},"make_distance":True,"expand_distance":True,"gauss_args":{"bins":25,"distance":5,"offset":0.0,"sigma":0.4},"meg_block_args":{"node_embed":[64,32,32],"edge_embed":[64,32,32],"env_embed":[64,32,32],"activation":"kgcnn>softplus2"},"set2set_args":{"channels":16,"T":3,"pooling_method":"sum","init_qstar":"0"},"node_ff_args":{"units":[64,32],"activation":"kgcnn>softplus2"},"edge_ff_args":{"units":[64,32],"activation":"kgcnn>softplus2"},"state_ff_args":{"units":[64,32],"activation":"kgcnn>softplus2"},"nblocks":3,"has_ff":True,"dropout":None,"use_set2set":True,"verbose":10,"output_embedding":"graph","output_mlp":{"use_bias":[True,True,True],"units":[32,16,1],"activation":["kgcnn>softplus2","kgcnn>softplus2","linear"]}}
CGCNN_INPUTS=[
{"shape":(None,),"name":"node_number","dtype":"int64","ragged":True},
{"shape":(None,3),"name":"node_frac_coordinates","dtype":"float32","ragged":True},
{"shape":(None,2),"name":"range_indices","dtype":"int64","ragged":True},
{"shape":(3,3),"name":"graph_lattice","dtype":"float32","ragged":False},
{"shape":(None,3),"name":"range_image","dtype":"float32","ragged":True}]
CGCNN_CONFIG={"name":"CGCNN","inputs":CGCNN_INPUTS,"input_embedding":{"node":{"input_dim":95,"output_dim":64}},"representation":"unit","make_distances":True,"expand_distance":True,"gauss_args":{"bins":40,"distance":8,"offset":0.0,"sigma":0.4},"depth":3,"conv_layer_args":{"units":64,"activation_s":"softplus","activation_out":"softplus","batch_normalization":True},"node_pooling_args":{"pooling_method":"mean"},"output_embedding":"graph","output_mlp":{"use_bias":[True,False],"units":[64,1],"activation":["softplus","linear"]},"verbose":10}

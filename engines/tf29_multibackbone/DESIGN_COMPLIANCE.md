# 设计需求对应

| 需求 | 实现 |
|---|---|
| TF2.9/kgcnn2.1 | 只用tf.keras 2.9公开API；图backbone调用安装的kgcnn官方builder |
| 不重写backbone | builder-time embedding wrapper；不clone、不复制MEGNet/CGCNN网络 |
| 小数据 | rank4、单门控、约700参数、轻L2、微小alpha激活 |
| E/I语义 | excitation/inhibition均softplus非负；competition与1-competition互补 |
| 恒等退化 | raw_alpha=0时严格identity；权重及预测双重验收 |
| 特征 | 20维源表保留；12个物理一致context；label-free PCA；具名导出 |
| CGCNN/MEGNet | 同一PriorAwareNodeEmbedding核心，分别使用原生builder |
| ElemNet | simplex保持乘性重加权；ElemNet-small明确命名 |
| 公平比较 | 同seed/fold/GPU；同backbone初始权重；同shuffle seed |
| VASP友好 | 4 worker×2主CPU线程，nice=10，一GPU一进程 |
| 解释 | 可按batch导出E/I、signed latent、delta及每个物理context对competition logit的实际加性贡献 |
| 统计 | seed级推断，fold-seed配对作描述 |

## v3.1 CGCNN dtype compliance

The periodic image is physically an integer lattice shift, but kgcnn 2.1.0
adds it directly to float32 fractional coordinates. Materon therefore preserves
the integer values while representing the model input as float32. This is a
compatibility cast only and does not alter the graph topology or distances.


## v3.2 pairing compliance

The baseline and Materon backbone are now paired by semantic layer location,
not by a flattened variable list.  This is essential for CGCNN because repeated
convolution and BatchNorm layers can have equal-shaped tensors that are not
interchangeable.


## v3.3 numerical-compliance distinction

Materon now distinguishes exact model identity from backend numerical
equivalence.  Exactness is enforced where it is mathematically and
architecturally meaningful: adapter output and paired backbone tensors.
Separately traced GPU graphs are checked in ULP and relative units so that a
one-representable-float difference is not misclassified as a changed model.


## v3.4 checkpoint compliance

Checkpoint fidelity is now tested at the state level. Exact equality of every
trainable and non-trainable tensor is the hard criterion. This includes
BatchNorm moving statistics and Materon adapter parameters. Cross-graph output
differences are retained for transparency but cannot falsely reject an exact
checkpoint restoration.


## v3.5 representation-geometry compliance

Graph and composition models now use different integration topology while
sharing the same E/I principle. Graph adapters replace an embedding operation;
the composition adapter is a new simplex-preserving transformation before the
Dense backbone. The pairing and identity tests explicitly respect this
difference instead of forcing both representations into one layer-count rule.


## v3.6 ElemNet scope

The `elemnet` entry is a reduced element-fraction MLP (`ElemNetSmall`), not a
crystal-structure GNN and not a claim of reproducing the full historical
17-layer ElemNet architecture. On structure-input Matbench tasks it uses only
the fractional composition obtained from the structure. Such results are valid
as composition-only controls but must not be presented as structure-aware
predictions.

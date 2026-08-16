# 验收标准

| 项目 | 硬阈值 |
|---|---|
| Python/TF/kgcnn | 3.9 / 2.9.0 / 2.1.0 |
| Matbench协议 | matbench_v0.1、phonons 1265样本、5 folds |
| 官方backbone | 使用安装的make_crystal_model；无clone_model |
| 注入次数 | 每个图模型恰好1次节点embedding |
| backbone权重 | 数量、shape、数值逐项相同 |
| epoch-0恒等 | max abs prediction diff <= 1e-6 |
| adapter规模 | added params / baseline params < 1% |
| 特征 | 20维保留；主context为12组label-free PCA；无固定算术平均 |
| ElemNet | 非负、和为1、零支持集保持 |
| checkpoint | same-model reload and fresh-model load: every weight tensor exactly restored |
| 反向传播 | adapter内部至少一个非alpha参数梯度非零且全部已用梯度有限 |
| 解释tensor | E/I、competition logit、physical context、context contribution、delta均可导出 |
| 统计 | 5个seed级差为主；25 pairs为描述 |

## CGCNN periodic geometry contract

For kgcnn 2.1.0 with `representation="unit"` and `make_distances=True`:

- `node_frac_coordinates` must be float32.
- `range_image` (periodic cell translations) must be float32 at model input.
- `range_indices` remains int64.
- The preflight must print and verify these dtypes before CGCNN construction.


## Semantic backbone pairing

- Global flattened `model.weights` positional copying is forbidden.
- Official and Materon direct graph layers must be paired topologically.
- The official embedding must pair only with
  `PriorAwareNodeEmbedding.base_layer`.
- Every official backbone tensor and every Materon backbone tensor must be
  covered exactly once.
- Layer class, local weight count, and every local weight shape must match.
- Adapter-level identity at raw_alpha=0 must be exactly zero.


## Full-model numerical equivalence

Do not use a fixed `1e-6` absolute threshold alone for two separately built
float32 GPU graphs.

Exact structural requirements remain:

- backbone tensor parity: maximum absolute difference exactly 0;
- adapter identity at alpha=0: maximum absolute difference exactly 0.

Complete outputs pass only under the documented combined relative/absolute
criterion or a maximum float32 ULP distance of 4.  All diagnostics must be
written to the acceptance report.


## Checkpoint hard criteria

- Save a complete Materon weight checkpoint.
- Perturb one floating trainable tensor in the same model object.
- Reload and require every weight to equal its pre-save snapshot exactly.
- Rebuild a fresh model, load the checkpoint and require every model weight to
  equal the saved model exactly.
- The same graph's restored prediction must pass numerical equivalence.
- Outputs from separately constructed GPU graphs are diagnostic only; they do
  not define whether a checkpoint restored correctly.


## ElemNet inserted-adapter criteria

- Materon ElemNet has exactly one more direct layer than baseline.
- The only additional direct layer is `CompositionSimplexEIGate`.
- Removing that adapter must leave a one-to-one ordered Input/Dense backbone.
- All paired Dense weights must be exactly equal at initialization.
- At raw_alpha=0, adapter output must equal input exactly.
- Composition sums must be unchanged exactly on the acceptance batch.
- Outputs must remain non-negative.
- An element absent in the input must remain absent in the output.

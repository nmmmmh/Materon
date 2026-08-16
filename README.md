# Materon
Materon: A bio-inspired excitation–inhibition gating module (EIGate) for interpretable machine learning of two-dimensional materials — a plug-and-play add-on for crystal graph neural networks (CGCNN, MEGNet, etc.).
[README.md](https://github.com/user-attachments/files/31114337/README.md)
# Materon Core Release v2.0.0

This release fixes the project architecture around a **single canonical EIGate**.
Scientific protocols and frozen result files are preserved; the change is code organization and compatibility routing, not a new experiment.

## Canonical public API

```python
from materon import EIGate
```

The only graph E/I implementation is `materon/core/eigate.py`. CGCNN and MEGNet historical gate names are thin compatibility subclasses, and workflow-local gate files are re-export shims only. See `EIGATE_ARCHITECTURE.md`.

## Recommended server installation

```bash
cd /data/materon/core/Materon_Core_Release_v2_0_0_20260813
pip install -e . --no-deps
```

This installs only the local Materon package metadata and does not change the frozen TensorFlow/kgcnn environment.

After installation, the lightweight target-server gate routing check is:

```bash
python tools/eigate_runtime_preflight.py
```

It performs forward-only compatibility checks and does not retrain any model.

---


这是从 `/data/materon` 全树冻结包中重新整理出的**单一干净核心发布包**。目标是解决历史版本、hotfix、旧结果、重复 add-on 和多个输出根混杂的问题。

## 目录

```text
engines/
  tf29_multibackbone/       通用 TF2.9 多骨架 Materon 引擎
workflows/
  cgcnn_matbench/           CGCNN 小数据 Matbench/JCIS 正式流程
  megnet_matbench/          MEGNet 官方固定训练流程
  explainability_2d/        HOIP + JDFT2D 二维可解释性流程
  evidence_collector/       JCIS 证据收集与 phonons 后验解释
  elemnet/                  ElemNet V15 独立流程
reference/
  cgcnn_phonons_protocol/   冻结的 CGCNN phonons 协议元数据
results/
  ...                       与上述流程对应的已验证核心结果
data/
  ...                       HOIP 原始数据；各工作流的冻结物理特征表随工作流自包含
provenance/
  ...                       原服务器全量文件索引、来源映射、排除清单
```

## 规范

1. **不再以日期/Hotfix命名代码目录。** 版本信息只记录在 provenance 和协议文件中。
2. **科学协议 ID、模型 ID、随机种子、fold 和结果字段不改名。** 这些属于可复现性证据，不做美化重命名。
3. **只保留当前 canonical 代码。** 历史 v1.0.0/v1.0.1/v1.0.2、MEGNet 错误 LR 分支、EvidenceCollector v1.0.0 等不重复放入代码树。
4. **2D role-matched forward 已并入主流程。** 不再保留单独 add-on 副本。
5. **结果目录只收 validated/current 数据。** 已知错误的 `materon_megnet_outputs` 永久排除。
6. **缓存和模型权重不打包。** 这些是服务器生成物，核心结果与代码包保持轻量、可审计。

## 验证

```bash
python tools/validate_release.py
bash tools/verify_sha256.sh
```

`provenance/source_inventory_all.csv` 仍保留了原 `/data/materon` 的完整索引，因此被排除的历史文件仍然可追溯。

## 推荐服务器目录

将本包作为唯一代码根解压到：

```text
/data/materon/core/Materon_Core_Release_20260813
```

历史压缩包放在 `/data/materon/packages/`，运行结果继续保存在独立的 `/data/materon/results/` 或既有正式输出根；不要再把压缩包、代码树、缓存和结果混在同一层。

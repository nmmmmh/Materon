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


This is a **single, clean core release package** reorganized from the frozen `/data/materon` full tree. Its goal is to resolve the problem of historical versions, hotfixes, stale results, duplicated add-ons, and multiple output roots all being mixed together.

## Catalogue

```text
engines/
  tf29_multibackbone/       TF2.9 Multi-backbones Materon engine
workflows/
  cgcnn_matbench/           CGCNN finite-data Matbench/JCIS workflows
  megnet_matbench/          MEGNet officially fixed training schedule
  explainability_2d/        HOIP + JDFT2D 2D explainable schedule
  evidence_collector/       JCIS evidence collection &  phonons post-hoc 
  elemnet/                  ElemNet V15 independent schedule
reference/
  cgcnn_phonons_protocol/   frozen CGCNN phonons protocal data
results/
  ...                       
data/
  ...                       HOIP original data；
provenance/
  ...                       
```


## Validation

```bash
python tools/validate_release.py
bash tools/verify_sha256.sh
```


## Sever catalogue

unzip

```text
/data/materon/core/Materon_Core_Release_20260813
```


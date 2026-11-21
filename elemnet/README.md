# ElemNet with the E/I gate (Matbench steels)

Composition-only control of SI Section S2.3 (SI Table S4, Figure S3b). The Materon E/I gate is placed in front of
an unchanged ElemNet network: 258 engineered composition features (stoichiometry, Magpie and Meredig descriptors),
a 1024-512-256-128 multilayer perceptron, MAE loss, Adam (learning rate 0.001), batch size 32, early stopping on
inner validation, five official Matbench folds and seeds 42, 123 and 2024, on the 312 yield strengths of
`matbench_steels`.

Four models are compared: ElemNet; ElemNet with the E/I gate; a parameter-matched plain blend; and the gated blend
y = (1 - lambda) y_ElemNet + lambda y_gate with a fixed baseline-anchored weight lambda = 0.50, selected on
inner-validation predictions only. The results of the paper are in `../data/results/elemnet_steels/`.

## Run

```bash
pip install -r requirements.txt          # Python 3.9, TensorFlow 2.9.0, matminer 0.7.4
bash scripts/RUN_ALL_FOREGROUND.sh       # cache, protocol, inner baseline, blend tuning, formal runs, summary
```

The stages can also be run one by one (`scripts/00_*.sh` ... `scripts/11_*.sh`). Everything is written to
`workspace/` inside this directory; the formal summary is `workspace/results/formal_summary/FORMAL_SUMMARY.json`.
The scripts use four GPUs by default.

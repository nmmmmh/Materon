# Materon

Code for "Environment-Modulated Material Neurons for Interpretable Learning of Two-Dimensional Materials".

Materon (Material Neuron) represents each atom by 20 named, standardized elemental attributes and keeps a separate
channel for each attribute. A learned gate enhances or suppresses every channel before message passing, so each
channel can be read, disabled or restored inside a trained model and compared across independently trained models.
The gate replaces only the elemental node features of a standard crystal graph network (CGCNN or MEGNet). Two
implementations are provided: the excitation/inhibition (E/I) gate used in the Matbench and fixed-split benchmarks,
Eq. (2), and the bounded environment gate g = 1 + 0.5 tanh h(x, c) of the C2DB work-function study, Eq. (1).

## Installation

The code was run with Python 3.9, TensorFlow 2.9.0 and kgcnn 2.1.0 (Linux, CUDA GPUs). Use an editable install so
that the `configs/` directory is found (or set `MATERON_CONFIG_DIR`):

```bash
conda create -n materon python=3.9 && conda activate materon
pip install -e ".[benchmarks,test]"
pytest tests
```

`benchmarks` adds matbench and `test` adds pytest. The phonon scripts (`scripts/dft/make_stability_jobs.py`,
`make_phonon_q_jobs.py`, `phonon_analysis.py`) were run with phonopy 4.8.4 and NumPy 2.4 in a separate
environment (`pip install phonopy==4.8.4 pymatgen`); every other DFT script runs in the main environment.
VASP 6.5.1 and the PBE PAW datasets are licensed and not included.

## Layout

| Path | Content |
|---|---|
| `materon/attributes.py` | the 20 named attributes (Table S2), attribute tables, grouped PCA |
| `materon/gates/` | E/I gate and search candidates (`ei.py`); bounded environment gate (`environment.py`) |
| `materon/cgcnn/` | CGCNN graphs, caches, models, training protocol, gate readouts |
| `materon/megnet/` | kgcnn MEGNet caches, models, training, interpretation helpers |
| `materon/fixed_split/` | HOIP/JDFT2D fixed-split study: datasets, group split, callbacks, explanation utilities |
| `materon/c2db/` | C2DB cohort and split, 37-entry context, gated MEGNet, training, fine-tuning, gate-path IG and shutdown, restoration, screens |
| `materon/dft/` | VASP inputs, work function (Eq. S9), DOS and rigid band, mechanism decomposition (Eq. S10), phonons, stability |
| `scripts/` | one entry point per stage (below); every data, cache and output path is a command-line argument |
| `configs/` | attribute tables, model configurations, training recipes and protocols |

## Data

Nothing is redistributed here. Place the inputs anywhere and pass their paths.

| Data | Source | Used by |
|---|---|---|
| Matbench v0.1 `matbench_phonons`, `matbench_dielectric`, `matbench_jdft2d` | downloaded by the `matbench` package (Dunn et al., npj Comput. Mater. 6, 138, 2020) | CGCNN and MEGNet benchmarks, JDFT2D fixed split |
| CoRE 2D-HOIP PBE band gaps, 499 structures | CoRE 2D-HOIP DB; a gzipped pandas "split" JSON with columns `structure` (pymatgen dict) and `band_gap_pbe` | fixed-split study (`--hoip-path`) |
| C2DB snapshot, 3,520 records | JARVIS-Tools `jarvis.db.figshare.data("c2db")` (Figshare file 28682010), saved as a JSON list of records | C2DB study (`--source`) |
| DFT increments for fine-tuning | your own calculations (format in `materon/c2db/finetune.py`) | `scripts/c2db/finetune.py` |

The composition-only steels/ElemNet control (Sec. S5) is not part of this package.

## Reproducing the paper

`P` is a plan JSON written by a `make_plan` script; plans are executed with
`python scripts/run_plan.py --plan P --script <entry point> --gpus 0,1,2,3 --log-dir logs/` (sequential without `--gpus`).
Variables: `$C` CGCNN cache root, `$M` MEGNet cache root, `$W` working directory.

**CGCNN Matbench phonons, configuration selection inside each outer training pool (Table I, Table S7).**
```bash
python scripts/cgcnn/build_caches.py --cache-root $C
# for KIND in backbone_r1 backbone_r2 backbone_r3 gate_core gate_context gate_capacity gate_confirm final:
python scripts/cgcnn/make_plan.py --kind $KIND --root $W/phonons --cache-root $C --output $W/plans/$KIND.json [--selection S | --formal-settings F]
python scripts/run_plan.py --plan $W/plans/$KIND.json --script scripts/cgcnn/run_trial.py --gpus 0,1,2,3 --log-dir $W/logs/$KIND
# between rounds: aggregate_results.py --phase inner --output $W/inner.csv, then
#   after backbone_r1: select_backbone_recipes.py --stages backbone_round1 --top-n 4
#   after backbone_r2: select_backbone_recipes.py --stages backbone_round2 --top-n 2
#   after backbone_r3: select_backbone_recipes.py --stages backbone_round2,backbone_round3 --top-n 2
#   after the gate searches: select_search.py --stages gate_core,gate_context,gate_capacity --top-n 4
#   after gate_confirm: select_search.py --stages gate_confirm --top-n 1
python scripts/cgcnn/select_formal_settings.py --inner-csv ... --backbone ... --winner ... --output $W/formal_settings.json
python scripts/cgcnn/aggregate_results.py --root $W/phonons --phase final --output $W/final.csv
python scripts/cgcnn/phonons_tables.py --inner-csv $W/inner.csv --final-csv $W/final.csv --settings $W/formal_settings.json --output-dir $W/tables
python scripts/cgcnn/feature_stability.py --root $W/phonons --cache-root $C --output-dir $W/interp
```

**CGCNN dielectric and JDFT2D (calibrated transfer protocol).**
```bash
python scripts/cgcnn/make_transfer_plan.py --cache-root $C --outdir $W/transfer --output $W/plans/transfer.json
python scripts/run_plan.py --plan $W/plans/transfer.json --script scripts/cgcnn/run_transfer.py --gpus 0,1,2,3 --log-dir $W/logs/transfer
python scripts/cgcnn/aggregate_transfer.py --root $W/transfer --output-dir $W/transfer/reports
```

**MEGNet Matbench (kgcnn 2.1.0, seed 42) and E/I interpretation (Table I, Sec. S6).**
```bash
python scripts/megnet/build_caches.py --cache-root $M
python scripts/megnet/make_plan.py --cache-root $M --outdir $W/megnet --output $W/plans/megnet.json
python scripts/run_plan.py --plan $W/plans/megnet.json --script scripts/megnet/train_job.py --gpus 0,1,2,3 --log-dir $W/logs/megnet
python scripts/megnet/interpret.py --task matbench_phonons --fold 0 --cache-root $M --outdir $W/megnet   # each task and fold
python scripts/megnet/aggregate.py --outdir $W/megnet --reports $W/megnet/reports
python scripts/megnet/export_cgcnn_detailed.py --cgcnn-output-root $W/transfer --cgcnn-cache-root $C --export-root $W/cgcnn_detail
python scripts/megnet/cross_backbone.py --megnet-output-root $W/megnet --cgcnn-transfer-root $W/transfer --cgcnn-detailed-root $W/cgcnn_detail --cgcnn-phonons-stability $W/interp/materon_best_consensus_feature_stability.csv --outdir $W/megnet/reports/cross_backbone
```

**Fixed-split HOIP and JDFT2D study (Tables S1, S13-S18) and null rankings (Table S15).**
```bash
python scripts/fixed_split/audit_and_split.py --hoip-path HOIP.json.gz --megnet-cache-root $M --output-root $W/2d
python scripts/fixed_split/build_caches.py --hoip-path HOIP.json.gz --cgcnn-cache-root $C --megnet-cache-root $M
python scripts/fixed_split/make_plan.py --output-root $W/2d --cgcnn-cache-root $C --megnet-cache-root $M --output $W/plans/2d.json
python scripts/run_plan.py --plan $W/plans/2d.json --script scripts/fixed_split/train_job.py --gpus 0,1,2,3 --log-dir $W/logs/2d
python scripts/fixed_split/explain_job.py --dataset hoip_bandgap --backbone cgcnn --output-root $W/2d --cgcnn-cache-root $C --megnet-cache-root $M   # each dataset and backbone
python scripts/fixed_split/aggregate.py --output-root $W/2d
python scripts/fixed_split/null_ranking.py --dataset hoip_bandgap --backbone megnet --mode untrained --seeds 42 43 44 45 46 --output-root $W/2d --cgcnn-cache-root $C --megnet-cache-root $M --out $W/null/hoip_bandgap__megnet__untrained.json
python scripts/fixed_split/aggregate_null.py --raw-dir $W/null --interpretability-root $W/2d/interpretability --output $W/NULL_RANKING.json
```

**C2DB work-function models (Sec. III B; Tables S5, S20, S24, S26).** Arms: `physical20_environment_gate`
(Materon), `physical20_nogate`, `physical20_global_gate`, `physical20_element_gate`; seeds 42, 7, 123, 2024, 31415;
schedules `short` (80 epochs) and `L400`.
```bash
python scripts/c2db/prepare.py --source C2DB_SOURCE.json --out $W/c2db/prepared
python scripts/c2db/train.py --prepared $W/c2db/prepared --runs $W/c2db/runs --arm physical20_environment_gate --seed 42 --schedule L400
python scripts/c2db/evaluate.py --prepared $W/c2db/prepared --runs $W/c2db/runs --tag L400 --out $W/c2db/EVAL_L400.json
python scripts/c2db/shutdown.py --prepared $W/c2db/prepared --checkpoint $W/c2db/runs/L400/physical20_environment_gate_s42/best.weights.h5 --out $W/c2db/shutdown/L400_s42.json
python scripts/c2db/channel_rules.py --prepared $W/c2db/prepared --runs $W/c2db/runs --tag L400 --out $W/c2db/RULES_L400.json
python scripts/c2db/environment_probe.py --prepared $W/c2db/prepared --env-checkpoint .../short/physical20_environment_gate_s42/best.weights.h5 --element-checkpoint .../short/physical20_element_gate_s42/best.weights.h5 --out $W/c2db/environment
python scripts/c2db/al2s2_probe.py --prepared $W/c2db/prepared --checkpoint .../short/physical20_environment_gate_s42/best.weights.h5 --out $W/c2db/al2s2.json
```
`shutdown.py --randomization` adds the parameter controls of Table S21.

**Locked screens, fine-tuning and designs (Sec. III D, III E; Tables S34, S35).** Each lock is written with the
SHA-256 digest of its text before the corresponding DFT calculations.
```bash
python scripts/c2db/screen.py --kind contrast --prepared $W/c2db/prepared --checkpoint .../short/physical20_environment_gate_s42/best.weights.h5 --out $W/locks/contrast
python scripts/c2db/screen.py --kind contrast-seeds --prepared $W/c2db/prepared --runs $W/c2db/runs --tag short --previous $W/locks/contrast/CONTRAST_PREREGISTERED.json --out $W/locks/contrast_seeds
python scripts/c2db/finetune.py --prepared $W/c2db/prepared --runs $W/c2db/runs --seed 42 --pairs FT_PAIRS.json --holdout Al2S2 --lam 1 --epochs 30 --cv-out $W/c2db/ft_cv   # cross-validation fold
python scripts/c2db/finetune.py --prepared $W/c2db/prepared --runs $W/c2db/runs --seed 42 --pairs FT_PAIRS.json --tag FT1   # each seed
python scripts/c2db/screen.py --kind round1 --prepared $W/c2db/prepared --runs $W/c2db/runs --tag FT1 --previous $W/locks/contrast/CONTRAST_PREREGISTERED.json --out $W/locks/round1
python scripts/dft/lock_mechanism.py $W/locks/mechanism
python scripts/c2db/screen.py --kind round2 --prepared $W/c2db/prepared --runs $W/c2db/runs --tag FT2 --previous $W/locks/contrast/CONTRAST_PREREGISTERED.json $W/locks/round1/ROUND1_PREREGISTERED.json --out $W/locks/round2
python scripts/c2db/lock_design.py --round 1 --prepared $W/c2db/prepared --runs $W/c2db/runs --tags L400 FT1 FT2 --parent-doscar contrast_dft/Ta2Te2__14789ef267c9/static24/DOSCAR --out $W/locks/design1
python scripts/c2db/lock_design.py --round 2 --prepared $W/c2db/prepared --runs $W/c2db/runs --tags FT2 --first-lock $W/locks/design1/DESIGN_PREREGISTERED.json --out $W/locks/design2
python scripts/c2db/predict_pairs.py --prepared $W/c2db/prepared --runs $W/c2db/runs --tags L400 FT1 FT2 --candidates $W/locks/round2/ROUND2_PREREGISTERED.json --out $W/c2db/ROUND2_MODELS.json
```

**DFT inputs and post-processing (Sec. VI D, S11, S12, S14).** `<PAW>` is the root of the PBE PAW datasets.
```bash
python scripts/dft/make_al2s2_jobs.py --out al2s2_dft --potcar-root <PAW>
python scripts/dft/make_screen_jobs.py --lock $W/locks/contrast/CONTRAST_PREREGISTERED.json --set contrast --out contrast_dft --potcar-root <PAW>
python scripts/dft/make_screen_jobs.py --lock $W/locks/round1/ROUND1_PREREGISTERED.json --set screen --contrast-root contrast_dft --out round1_dft --potcar-root <PAW>
python scripts/dft/make_design_jobs.py --lock $W/locks/design2/DESIGN2_PREREGISTERED.json --out design2_dft --potcar-root <PAW> --references Mo,I
python scripts/dft/make_mos2_job.py --out mos2_ref --potcar-root <PAW> && python scripts/dft/band_edges.py mos2_ref/static24
python scripts/dft/extract_workfunctions.py --layout nested --root contrast_dft --out contrast_dft/RESULTS.json
python scripts/dft/mechanism_pairs.py pairs.json MECH_PAIRS.json
python scripts/dft/decompose_workfunction.py --method core --out PHI_CORE.json contrast_dft round1_dft
python scripts/dft/stability_energetics.py --out STABILITY.json --ref Mo=... --ref Ta=... --ref Te=... --phase MoTaTe2=... --phase Ta2Te2=...
python scripts/dft/make_stability_jobs.py --out stability_dft --potcar-root <PAW> --phonon MoTaTe2=.../POSCAR --phonon Ta2Te2=.../POSCAR
python scripts/dft/phonon_analysis.py dispersion PHONONS.json stability_dft MoTaTe2 Ta2Te2
```
Each script documents its arguments (`--help`). VASP is run as `mpirun -np 1 vasp_gpu` in each job directory;
static jobs start from the CONTCAR of the preceding relaxation.

## Citation

M. Ni, H. Zhou, Z. Wang, Z. Meng, Z. Li, H. Zheng, X. Zhang, X. Liu, Z. Chen and J. Lin, Environment-Modulated
Material Neurons for Interpretable Learning of Two-Dimensional Materials (2026). [Journal, volume, pages and DOI to
be added on publication.]

## License

MIT, see `LICENSE`.

# Materon

Code and data for "Neuron-Inspired Gating Reveals How Neighbors Control Substitution in 2D Materials".

Materon (Material Neuron) represents each atom by 20 named, standardized elemental attributes and keeps a separate
channel for each attribute. A learned gate, conditioned on the atom and its neighborhood, enhances or suppresses
every channel within fixed bounds before message passing, in the way the balance of excitation and inhibition sets
the response of a neuron. Each channel can therefore be read, switched off or restored inside a trained model and
compared across independently trained models.

The gate is a drop-in front end: it replaces only the elemental node features and leaves the backbone unchanged.
This repository uses it with two crystal graph networks (CGCNN and MEGNet) and with a composition-only network
(ElemNet). Two gate forms are provided: the excitation/inhibition (E/I) gate of the Matbench and fixed-split
benchmarks, and the bounded gate g = 1 + 0.5 tanh h(x, c) of the C2DB work-function study.

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
environment (`pip install phonopy==4.8.4 pymatgen`); every other DFT script runs in the main environment. The
plotting code in `figures/` has its own requirements (`figures/README.md`). VASP 6.5.1 and the PBE PAW datasets are
licensed and not included.

## Layout

| Path | Content |
|---|---|
| `materon/attributes.py` | the 20 named attributes (SI Table S3), attribute tables, grouped PCA |
| `materon/gates/` | E/I gate and search candidates (`ei.py`); bounded environment gate (`environment.py`) |
| `materon/cgcnn/` | CGCNN graphs, caches, models, training protocol, gate readouts |
| `materon/megnet/` | kgcnn MEGNet caches, models, training, interpretation helpers |
| `materon/fixed_split/` | HOIP/JDFT2D fixed-split study: datasets, group split, callbacks, explanation utilities |
| `materon/c2db/` | C2DB cohort and split, 37-entry context, gated MEGNet, training, fine-tuning, gate-path IG and shutdown, restoration, screens |
| `materon/dft/` | VASP inputs, work function, DOS and rigid band, mechanism decomposition, phonons, stability |
| `elemnet/` | ElemNet with the E/I gate on Matbench steels (composition-only control, SI Section S2.3) |
| `scripts/` | one entry point per stage (below); every data, cache and output path is a command-line argument |
| `scripts/interface/` | metal/MoS2 heterostructures: structures, analysis plan, barriers, charge-density difference |
| `configs/` | attribute tables, model configurations, training recipes and protocols |
| `data/` | prepared C2DB data, all trained C2DB checkpoints, locked screens, results and DFT inputs (`data/README.md`) |
| `figures/` | plotting scripts and plotting data for every figure of the Letter and the SI (`figures/README.md`) |

## Data

`data/` contains everything needed to reproduce the C2DB study without retraining: the prepared data set and split,
the 40 trained checkpoints (four predictors with five seeds, the ungated and gated L400 models, and two fine-tuned
ensembles), the six locked screening files with their SHA-256 digests, the model and DFT results behind the tables,
and the inputs and final structures of the DFT calculations. Contents and provenance are listed in `data/README.md`.

The following public inputs are not redistributed:

| Data | Source | Used by |
|---|---|---|
| Matbench v0.1 `matbench_phonons`, `matbench_dielectric`, `matbench_jdft2d`, `matbench_steels` | downloaded by the `matbench` package (Dunn et al., npj Comput. Mater. 6, 138, 2020) | CGCNN, MEGNet and ElemNet benchmarks, JDFT2D fixed split |
| CoRE 2D-HOIP PBE band gaps, 499 structures | CoRE 2D-HOIP DB; a gzipped pandas "split" JSON with columns `structure` (pymatgen dict) and `band_gap_pbe` | fixed-split study (`--hoip-path`) |
| C2DB snapshot, 3,520 records | JARVIS-Tools `jarvis.db.figshare.data("c2db")` (Figshare file 28682010), saved as a JSON list of records | `scripts/c2db/prepare.py` (`data/c2db_prepared` is its output) |

## Quick start: the neighbor readout from the included checkpoints

These commands use only `data/` and run inference only (a GPU is optional).

```bash
# channel rules over the 592 metal-site substitutions (SI Section S3.6, Table S11a)
python scripts/c2db/channel_rules.py --prepared data/c2db_prepared --runs data/checkpoints --tag L400 --out work/RULES_L400.json
# permutation test of channel consistency across the ten models
python scripts/c2db/permutation_null.py --rules data/results/GATE_RULES_SEEDS.json data/results/GATE_RULES_L400.json --out work/PERMUTATION_NULL.json
# ungated baselines: input IG and the element-gate control (Table S11c), layer IG on hidden states (Table S11d)
python scripts/c2db/ungated_baselines.py --prepared data/c2db_prepared --runs data/checkpoints --rules data/results/GATE_RULES_L400.json --out-dir work/baselines
python scripts/c2db/layer_ig.py --prepared data/c2db_prepared --runs data/checkpoints --rules data/results/GATE_RULES_L400.json --out work/baselines/HIDDEN_IG.json
# every figure of the Letter and the SI
cd figures && python run_all.py
```

The outputs can be compared with `data/results/GATE_RULES_*.json` and `data/results/ungated_baselines/`.

## Reproducing the paper


`P` is a plan JSON written by a `make_plan` script; plans are executed with
`python scripts/run_plan.py --plan P --script <entry point> --gpus 0,1,2,3 --log-dir logs/` (sequential without `--gpus`).
Variables: `$C` CGCNN cache root, `$M` MEGNet cache root, `$W` working directory.

**CGCNN Matbench phonons, configuration selection inside each outer training pool (SI Section S2.2, Tables S4–S6).**
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

**CGCNN dielectric and JDFT2D, calibrated transfer protocol (SI Section S2.1, Tables S4, S5).**
```bash
python scripts/cgcnn/make_transfer_plan.py --cache-root $C --outdir $W/transfer --output $W/plans/transfer.json
python scripts/run_plan.py --plan $W/plans/transfer.json --script scripts/cgcnn/run_transfer.py --gpus 0,1,2,3 --log-dir $W/logs/transfer
python scripts/cgcnn/aggregate_transfer.py --root $W/transfer --output-dir $W/transfer/reports
```

**MEGNet Matbench (kgcnn 2.1.0, seed 42) and E/I interpretation (SI Sections S2.1, S3.2; Tables S4, S5, S7, S8).**
```bash
python scripts/megnet/build_caches.py --cache-root $M
python scripts/megnet/make_plan.py --cache-root $M --outdir $W/megnet --output $W/plans/megnet.json
python scripts/run_plan.py --plan $W/plans/megnet.json --script scripts/megnet/train_job.py --gpus 0,1,2,3 --log-dir $W/logs/megnet
python scripts/megnet/interpret.py --task matbench_phonons --fold 0 --cache-root $M --outdir $W/megnet   # each task and fold
python scripts/megnet/aggregate.py --outdir $W/megnet --reports $W/megnet/reports
python scripts/megnet/export_cgcnn_detailed.py --cgcnn-output-root $W/transfer --cgcnn-cache-root $C --export-root $W/cgcnn_detail
python scripts/megnet/cross_backbone.py --megnet-output-root $W/megnet --cgcnn-transfer-root $W/transfer --cgcnn-detailed-root $W/cgcnn_detail --cgcnn-phonons-stability $W/interp/materon_best_consensus_feature_stability.csv --outdir $W/megnet/reports/cross_backbone
```

**Fixed-split HOIP and JDFT2D study (SI Section S1.1, Tables S1, S2) and null rankings (Table S7).**
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

**C2DB work-function models (SI Sections S1.4, S3.3–S3.6; Tables S9–S11).** Arms: `physical20_environment_gate`
(Materon), `physical20_nogate`, `physical20_global_gate`, `physical20_element_gate`; seeds 42, 7, 123, 2024, 31415;
schedules `short` (80 epochs) and `L400`.
```bash
python scripts/c2db/prepare.py --source C2DB_SOURCE.json --out $W/c2db/prepared   # optional: data/c2db_prepared is this output
python scripts/c2db/train.py --prepared data/c2db_prepared --runs $W/c2db/runs --arm physical20_environment_gate --seed 42 --schedule L400   # optional: data/checkpoints holds every trained member
python scripts/c2db/evaluate.py --prepared data/c2db_prepared --runs data/checkpoints --tag L400 --out $W/c2db/EVAL_L400.json
python scripts/c2db/shutdown.py --prepared data/c2db_prepared --checkpoint data/checkpoints/L400/physical20_environment_gate_s42/best.weights.h5 --out $W/c2db/shutdown/L400_s42.json
python scripts/c2db/channel_rules.py --prepared data/c2db_prepared --runs data/checkpoints --tag L400 --out $W/c2db/RULES_L400.json
python scripts/c2db/environment_probe.py --prepared data/c2db_prepared --env-checkpoint data/checkpoints/short/physical20_environment_gate_s42/best.weights.h5 --element-checkpoint data/checkpoints/short/physical20_element_gate_s42/best.weights.h5 --out $W/c2db/environment
python scripts/c2db/al2s2_probe.py --prepared data/c2db_prepared --checkpoint data/checkpoints/short/physical20_environment_gate_s42/best.weights.h5 --out $W/c2db/al2s2.json
```
`shutdown.py --randomization` adds the parameter controls of Table S11b.

**Locked screens, fine-tuning and designs (SI Sections S6, S7; Tables S14, S15).** Each lock is written with the
SHA-256 digest of its text before the corresponding DFT calculations.
```bash
python scripts/c2db/screen.py --kind contrast --prepared data/c2db_prepared --checkpoint data/checkpoints/short/physical20_environment_gate_s42/best.weights.h5 --out $W/locks/contrast
python scripts/c2db/screen.py --kind contrast-seeds --prepared data/c2db_prepared --runs data/checkpoints --tag short --previous $W/locks/contrast/CONTRAST_PREREGISTERED.json --out $W/locks/contrast_seeds
python scripts/c2db/finetune.py --prepared data/c2db_prepared --runs data/checkpoints --seed 42 --pairs data/results/FT_PAIRS.json --holdout Al2S2 --lam 1 --epochs 30 --cv-out $W/c2db/ft_cv   # cross-validation fold
python scripts/c2db/finetune.py --prepared data/c2db_prepared --runs $W/c2db/runs --seed 42 --pairs data/results/FT_PAIRS.json --tag FT1   # each seed; data/checkpoints/FT1 and FT2 hold the results
python scripts/c2db/screen.py --kind round1 --prepared data/c2db_prepared --runs data/checkpoints --tag FT1 --previous $W/locks/contrast/CONTRAST_PREREGISTERED.json --out $W/locks/round1
python scripts/dft/lock_mechanism.py $W/locks/mechanism
python scripts/c2db/screen.py --kind round2 --prepared data/c2db_prepared --runs data/checkpoints --tag FT2 --previous $W/locks/contrast/CONTRAST_PREREGISTERED.json $W/locks/round1/ROUND1_PREREGISTERED.json --out $W/locks/round2
python scripts/c2db/lock_design.py --round 1 --prepared data/c2db_prepared --runs data/checkpoints --tags L400 FT1 FT2 --parent-doscar contrast_dft/Ta2Te2__14789ef267c9/static24/DOSCAR --out $W/locks/design1
python scripts/c2db/lock_design.py --round 2 --prepared data/c2db_prepared --runs data/checkpoints --tags FT2 --first-lock $W/locks/design1/DESIGN_PREREGISTERED.json --out $W/locks/design2
python scripts/c2db/predict_pairs.py --prepared data/c2db_prepared --runs data/checkpoints --tags L400 FT1 FT2 --candidates $W/locks/round2/ROUND2_PREREGISTERED.json --out $W/c2db/ROUND2_MODELS.json
```

**DFT inputs and post-processing (SI Sections S4, S5, S7; Tables S12, S13, S15).** `<PAW>` is the root of the PBE PAW datasets.
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

**Metal/MoS2 interfaces (SI Section S7.3, Table S16).** The analysis plan `scripts/interface/ANALYSIS_PLAN.md` was
fixed before any interface result was read. `build.py` writes the 74-atom MoTaTe2/MoS2 and Ta2Te2/MoS2
heterostructures for three registries and the isolated layers in the same cell; the inputs and final structures are
in `data/dft/interface/`. Set `MATERON_INTERFACE_ROOT` to the directory that holds `jobs/` (one directory per system
with `relax/` and `static/`).
```bash
python scripts/interface/build.py jobs
python scripts/interface/analyze.py MoTaTe2_MoS2_s0 Ta2Te2_MoS2_s1   # barriers, interface dipole, work functions
python scripts/interface/setup_cdd.py MoTaTe2_MoS2_s0 Ta2Te2_MoS2_s1 # layers at fixed geometry (jobs_cdd/)
python scripts/interface/cdd.py MoTaTe2_MoS2_s0 Ta2Te2_MoS2_s1       # plane-integrated charge-density difference
python scripts/interface/kanalysis.py MoTaTe2_MoS2_s0 Ta2Te2_MoS2_s1 # MoS2 band edges at K (non-SCF runs in nscfK/)
python scripts/interface/extract.py MoTaTe2_MoS2_s0 Ta2Te2_MoS2_s1   # figure data
```

**ElemNet on Matbench steels (SI Section S2.3, Table S4).** See `elemnet/README.md`.

Before fine-tuning, copy `data/checkpoints` to `$W/c2db/runs`, so that the fine-tuned members are written next to the
L400 members they start from.

## Citation

M. Ni, H. Zhou, Z. Wang, Z. Meng, Z. Li, H. Zheng, X. Zhang, X. Liu, Z. Chen and J. Lin, Neuron-Inspired Gating
Reveals How Neighbors Control Substitution in 2D Materials (2026). [Journal, volume, pages and DOI to be added on
publication.]

## License

MIT, see `LICENSE`. Data in `data/` derived from C2DB, Matbench and CoRE 2D-HOIP remain subject to the terms of
those sources.

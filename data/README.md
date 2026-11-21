# Data

Data needed to reproduce the C2DB work-function study, the neighbor readouts, the locked screens and the DFT
analysis without retraining. SI references are to the Supporting Information of the paper.

## `c2db_prepared/`

Output of `scripts/c2db/prepare.py` on the 3,520-record C2DB snapshot, read by `materon.c2db.dataset.Prepared`:
the 2,379 retained metallic monolayers as graphs (`GRAPHS.pkl.gz`), work-function labels per partition
(`TRAIN_LABELS.json`, `VALIDATION_LABELS.json`, `TEST_LABELS.json`), the split by chemical system (`SPLIT.json`,
1,653/378/348), the excluded records with reasons (`EXCLUSIONS.json`) and the target scaling (`PREPARED.json`).

## `checkpoints/`

All trained C2DB members, `<tag>/<arm>_s<seed>/best.weights.h5` (`materon.c2db.training.member_dir`), seeds 42, 7,
123, 2024 and 31415:

| tag | arms | role |
|---|---|---|
| `short` (at most 80 epochs) | `physical20_environment_gate`, `physical20_nogate`, `physical20_global_gate`, `physical20_element_gate` | four predictors (SI Table S9) |
| `L400` (at most 400 epochs) | `physical20_environment_gate`, `physical20_nogate` | long schedule; ten gated models with `short` |
| `FT1`, `FT2` | `physical20_environment_gate` | fine-tuned ensembles of the locked rounds (SI Section S6.2) |

## `locks/`

The six screening files written before the corresponding DFT calculations, in order: contrast set, round 1,
mechanism hypotheses, round 2, design round 1, design round 2. `SHA256SUMS` lists their digests, which match those
recorded in SI Section S6.1.

## `results/`

| Path | Content |
|---|---|
| `EVAL_L400.json`, `SEED_EVAL.json` | C2DB metrics of the L400 and short-schedule models (SI Table S9) |
| `GATE_RULES_SEEDS.json`, `GATE_RULES_L400.json` | restoration responses of all channels for every substitution and model (SI Section S3.6, Table S11a) |
| `shutdown_seeds/` | channel shutdown and parameter randomization (SI Tables S10, S11b) |
| `ungated_baselines/` | input IG of the ungated models, element-gate control, layer IG on hidden states (SI Table S11c,d) |
| `null_ranking/` | null rankings of the fixed-split study (SI Table S7) |
| `FT_PAIRS.json`, `FT_PAIRS_R2.json`, `ft_cv/` | DFT increments used for fine-tuning and the fine-tuning cross-validation |
| `mechanism/` | mechanism hypotheses, pair quantities and analyses (SI Section S5, Table S13) |
| `elemnet_steels/` | ElemNet steels results: all runs, blend predictions and the summary (SI Table S4) |

## `dft/`

Inputs (`INCAR`, `KPOINTS`, `POSCAR`), final structures (`CONTCAR`), electronic-step logs (`OSZICAR`) and job
metadata of the calculations in SI Sections S4–S7: `contrast/`, `round1/`, `round2_and_design1/` (round 2 and the
first design round), `design2/`, `stability/` (references, mixing and phonon supercells), `mos2_reference/`, and
`interface/` (heterostructures for three registries, isolated layers, layer-split runs for the charge-density
difference, and the result files of SI Table S16). PAW potentials (`POTCAR`) are licensed and not included; the
PBE datasets used are named in SI Table S12. Wavefunctions, densities and potentials are not included because of
their size.

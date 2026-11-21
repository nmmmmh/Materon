# Figure code

Plotting code and plotting input data for every figure of the Letter (Figures 1–4 and the TOC graphic) and the
Supporting Information (Figures S1–S11 and A1–A5). The scripts only plot stored values; they run no model training,
inference or DFT.

## Requirements

Tested with these versions (Windows 11, Python 3.12.7):

| package | version |
|---|---|
| matplotlib | 3.11.1 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| Pillow | 12.3.0 |
| pymatgen | 2026.5.4 (`Element`, `Structure` only) |

All text is set in Arial (Windows `arial.ttf`, `ariali.ttf`, `arialbd.ttf`, version 7.06). Matplotlib must be
able to find Arial; another Arial build or a fallback font changes the text rendering.

Optional, only to re-render the structure images: OVITO 3.17.0 Python module (`render_structures.py`).

## Run

```
python run_all.py
```

writes every figure to `output/` as PNG (600 dpi) and PDF, and copies the Supporting Information figures to their
numbers (`Figure_SI_S1` ... `Figure_SI_A5`). A single script can be run on its own, e.g. `python fig2_jpcl.py`.

| Figure | script | file written by the script |
|---|---|---|
| Figure 1 | `fig1_jpcl.py` | Figure_1_JPCL |
| Figure 2 | `fig2_jpcl.py` | Figure_2_JPCL |
| Figure 3 | `fig3_jpcl.py` | Figure_3_JPCL |
| Figure 4 | `fig4_jpcl.py` | Figure_4_JPCL |
| TOC graphic | `toc_jpcl.py` | TOC_JPCL |
| Figure S1 | `figS1_S4_S12.py` | Figure_S1 |
| Figure S2 | `fig2.py` | Figure_2 |
| Figure S3 | `figS11.py` | Figure_S11 |
| Figure S4 | `figS3.py` | Figure_S3 |
| Figure S5 | `figS1_S4_S12.py` | Figure_S12 |
| Figure S6 | `figS6_S7.py` | Figure_S6 |
| Figure S7 | `figS14.py` | Figure_S14 |
| Figure S8 | `figS10.py` | Figure_S10 |
| Figure S9 | `fig5.py` | Figure_5 (panels a and b) |
| Figure S10 | `fig6.py` | Figure_6 |
| Figure S11 | `figS11_jpcl.py` | Figure_SI_11 |
| Figure A1 | `figS8.py` | Figure_S8 |
| Figure A2 | `figS2_S9.py` | Figure_S2 |
| Figure A3 | `figS6_S7.py` | Figure_S7 |
| Figure A4 | `figS2_S9.py` | Figure_S9 |
| Figure A5 | `figS13.py` | Figure_S13 |

The file names written by the scripts follow an earlier numbering of the figures; `run_all.py` maps them to the
numbers above. `figS5.py` writes a figure that is not part of the current Supporting Information.

Helpers: `symbols.py` (attribute symbols, palette, shared style), `ballstick.py` and `raster3d.py` (structure
rendering). Some scripts print the summary numbers quoted in the captions.

## Data

`data/SOURCES.json` lists the input files with their SHA-256 and origin.

- `data/evidence/`: model and DFT result files used by the benchmark and interpretation figures.
- `data/mechanism/`: DFT and model summaries of the mechanism, locked screens and designs (Figures 3, 4, S8–S10).
- `data/interface/`: metal/MoS2 interface results (Figure S11, SI Table S16).
- `data/candidate_dft_v3/`: exported data of the DFT application figure (Figure A5).
- `data/original_artwork/Manuscript_image1.png`: the neuron artwork of Figure 1 and the TOC graphic, as a lossless
  RGBA PNG.
- `data/structures/`: structure images read by Figures S9, S10 and A5, and the POSCAR files they were rendered from.

## Re-rendering the structure images (optional)

```
python render_structures.py [OUTDIR]
```

runs `render_structure_ovito.py` with the OVITO interpreter (environment variable `OVITO_PYTHON`, default: the current
interpreter) using the settings of the published images. The default `OUTDIR` is `data/structures`, which
overwrites the stored images. With OVITO 3.17.0 the output is byte-identical to the stored files; other OVITO
versions may differ slightly.

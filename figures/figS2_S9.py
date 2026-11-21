"""SI Figs. S2 and S9 redrawn with symbol labels from saved evidence files (no new evaluation).
S2: case-level mean signed modulation <Δx> of hoip_0275 and mb-jdft2d-541 (original_2d sample_feature_scores.csv).
S9: joint parent-to-B input IG (physics_figure/source_data/ig_channels.csv) and Al1 PAW s/p populations
(physics_figure/source_data/Al_index1_projected_populations.csv). Files under data/evidence/sources/... keep their
paths of the evidence package."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in PDF output
sys.path.insert(0, str(Path(__file__).parent))
from symbols import C_CGCNN, C_MEGNET, TEAL, C_B, SYM, GROUP_ORDER, GREY, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
EV = DATA / 'evidence' / 'sources'
PHYS = EV / 'research/eigate_v2/c2db_workfunction_v1_20261013/evidence_strengthening_v1/physics_figure/source_data'
POP = PHYS / 'Al_index1_projected_populations.csv'
style(); y = np.arange(20)

def case(task, sample):
    out = {}
    for net in ('cgcnn', 'megnet'):
        p = EV / f'original_2d/results/interpretability/{task}/{net}/sample_feature_scores.csv'
        out[net] = {r['feature']: float(r['mean_signed_actual_delta']) for r in csv.DictReader(open(p, encoding='utf-8')) if r['sample_id'] == sample}
    return out
fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.4), sharey=True, gridspec_kw={'wspace': 0.12})
for ax, (task, sample), tag in zip(axs, (('hoip_bandgap', 'hoip_0275'), ('jdft2d_exfoliation', 'mb-jdft2d-541')), ('(a)', '(b)')):
    d = case(task, sample); assert len(d['cgcnn']) == 20 and len(d['megnet']) == 20
    cg = [d['cgcnn'][f] for f in GROUP_ORDER]; mg = [d['megnet'][f] for f in GROUP_ORDER]
    for yy, a_, b_ in zip(y, cg, mg):
        ax.plot([0, a_ if abs(a_) > abs(b_) else b_], [yy, yy], color='#d0d0d0', lw=0.7, zorder=1)
    ax.axvline(0, color='#9a9a9a', lw=0.6, ls=(0, (3, 2)), zorder=0)
    for sep in (4.5, 10.5, 14.5): ax.axhline(sep, color='#e2e2e2', lw=0.5, zorder=0)
    ax.scatter(cg, y, s=18, color=C_CGCNN, lw=0, zorder=3, label='CGCNN')
    ax.scatter(mg, y, s=18, facecolor='white', edgecolor=C_MEGNET, lw=0.9, zorder=4, label='MEGNet')
    ax.set_xlabel(r'Mean $\Delta x_j$'); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    label(ax, tag, x=-0.04 if tag == '(b)' else -0.22)
axs[0].set_yticks(y, [SYM[f] for f in GROUP_ORDER], fontsize=8); axs[0].set_ylim(19.7, -0.7)
axs[1].legend(loc='upper left', fontsize=7.5, handletextpad=0.2)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S2.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

src = PHYS / 'ig_channels.csv'
ig = {r['feature']: (float(r['direct_node_feature_eV_32']), float(r['neighbor_context_feature_eV_32'])) for r in csv.DictReader(open(src, encoding='utf-8'))}
pop = list(csv.DictReader(open(POP, encoding='utf-8')))
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 3.2), gridspec_kw={'width_ratios': [1.5, 1], 'wspace': 0.45})
a.axvline(0, color=GREY, lw=0.7)
a.barh(y - 0.2, [ig[f][0] for f in GROUP_ORDER], height=0.38, color=TEAL, label=r'$x$')
a.barh(y + 0.2, [ig[f][1] for f in GROUP_ORDER], height=0.38, color='#a8d5c6', label=r'$c$')
a.set_yticks(y, [SYM[f] for f in GROUP_ORDER], fontsize=8); a.set_ylim(19.7, -0.7); a.tick_params(axis='y', length=0)
a.set_xlabel('IG (eV)'); a.legend(loc='center right', fontsize=8); label(a, '(a)', x=-0.2)
for k, (mat, c) in enumerate((('parent', GREY), ('design', C_B))):
    vals = [float(r['projected_population_e']) for r in pop if mat in r['material'] for o in ('s', 'p') if r['orbital'] == o]
    b.bar(np.arange(2) + (k - 0.5) * 0.36, vals, width=0.34, color=c, label=('Al$_2$S$_2$', 'AlBS$_2$')[k])
    for x, v in zip(np.arange(2) + (k - 0.5) * 0.36, vals):
        b.text(x, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
b.set_xticks([0, 1], [r'$s$', r'$p$']); b.set_ylabel(r'Al$_1$ population (electrons)'); b.set_ylim(0, 0.6)
b.legend(fontsize=7.5, loc='upper right'); label(b, '(b)', x=-0.35)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S9.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

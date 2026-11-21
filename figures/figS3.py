"""Fig. S3: cross-network and cross-fold reproducibility of attribute readouts (original E/I models), in the 2x2 layout of the
original submission Fig. 3, with symbol labels; data from the saved feature_scores.csv files (evidence archive) and
interpretability_stability.csv. Files under data/evidence/sources/... keep their paths of the evidence package."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in PDF output
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from symbols import SYM, GROUP_ORDER, NAVY, TEAL, GREY, INK, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
EV = DATA / 'evidence' / 'sources'           # hash-verified copies of the canonical evidence files
rd = lambda p: {r['feature']: float(r['mean_abs_actual_delta']) for r in csv.DictReader(open(EV / p, encoding='utf-8'))}

style()
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), gridspec_kw={'hspace': 0.36, 'wspace': 0.32})
axs = axs.ravel()
for ax, task, tag in ((axs[0], 'hoip_bandgap', '(a)'), (axs[1], 'jdft2d_exfoliation', '(b)')):
    OFF = {'hoip_bandgap': {'atom_specific_heat_capacity': (8, -8), 'atomic_covalent Radius': (-10, 8),
                            'atomic_fusion_heat': (14, 6), 'atom_melting_point': (14, -12)},
           'jdft2d_exfoliation': {'atomic_lfumo': (-12, 6), 'atomic_npval': (-14, 8),
                                  'X': (14, -6), 'atomic_ndval': (14, -18)}}[task]
    c = rd(f'original_2d/results/interpretability/{task}/cgcnn/feature_scores.csv')
    m = rd(f'original_2d/results/interpretability/{task}/megnet/feature_scores.csv')
    k = list(c); tc = set(sorted(k, key=lambda f: -c[f])[:10]); tm = set(sorted(k, key=lambda f: -m[f])[:10])
    lim = 1.08 * max(max(c.values()), max(m.values()))
    ax.plot([0, lim], [0, lim], ls='--', lw=0.8, color=GREY, zorder=1)
    for f in k:
        ax.scatter(c[f], m[f], s=22, color=NAVY if (f in tc and f in tm) else '#b4b9bd', zorder=3, lw=0)
    for f in sorted(k, key=lambda f: -(c[f] + m[f]))[:4]:
        dx, dy = OFF.get(f, (-7, 5) if m[f] > c[f] else (7, -6))   # label away from the diagonal
        ax.annotate(SYM[f], (c[f], m[f]), xytext=(dx, dy), textcoords='offset points', fontsize=8.5,
                    ha='left' if dx > 0 else 'right', va='center', color=NAVY,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=GREY, shrinkA=1, shrinkB=3))
    rho = spearmanr([c[f] for f in k], [m[f] for f in k])[0]
    ax.text(0.05, 0.92, rf'$\rho_{{\mathrm{{s}}}}={rho:.3f}$', transform=ax.transAxes, fontsize=8.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r'CGCNN, mean $|\Delta x|$'); ax.set_ylabel(r'MEGNet, mean $|\Delta x|$')
    label(ax, tag, x=-0.2)
# (c) cross-fold stability of MEGNet rankings
st = {r['task']: r for r in csv.DictReader(open(EV / 'original_megnet/results/reports/interpretability_stability.csv'))}
tasks = [('matbench_phonons', 'Phonons'), ('matbench_dielectric', 'Dielectric'), ('matbench_jdft2d', 'JDFT2D')]
ax = axs[2]
for i, (t, s) in enumerate(tasks):
    mu, mn = float(st[t]['mean_pairwise_spearman']), float(st[t]['min_pairwise_spearman'])
    ax.bar(i, mu, width=0.55, color=TEAL)
    ax.errorbar(i, mu, yerr=[[mu - mn], [0]], fmt='none', ecolor=INK, elinewidth=0.8, capsize=3, capthick=0.8)
    ax.text(i, mu + 0.002, f'{mu:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.text(i, 0.866, f'min\n{mn:.3f}', ha='center', va='bottom', fontsize=7, color='white', linespacing=1.3)
ax.set_xticks(range(3), [s for _, s in tasks]); ax.set_ylim(0.86, 1.0); ax.set_ylabel(r'Fold-pair Spearman $\rho_{\mathrm{s}}$')
label(ax, '(c)', x=-0.2)
# (d) Top-10 membership: Matbench tasks from the five-fold MEGNet mean, HOIP from the CGCNN/MEGNet mean rank
cols = []
for t in ('phonons', 'dielectric', 'jdft2d'):
    folds = [rd(f'original_megnet/results/matbench_{t}/materon_explain_megnet/outer_fold_{f}/seed_42/interpretability/feature_scores.csv') for f in range(5)]
    mean = {x: np.mean([d[x] for d in folds]) for x in folds[0]}
    cols.append(set(sorted(mean, key=lambda f: -mean[f])[:10]))
c = rd('original_2d/results/interpretability/hoip_bandgap/cgcnn/feature_scores.csv'); m = rd('original_2d/results/interpretability/hoip_bandgap/megnet/feature_scores.csv')
rc = {f: i for i, f in enumerate(sorted(c, key=lambda f: -c[f]))}; rm = {f: i for i, f in enumerate(sorted(m, key=lambda f: -m[f]))}
cols.append(set(sorted(c, key=lambda f: (rc[f] + rm[f]) / 2)[:10]))
rows = [f for f in GROUP_ORDER if any(f in s for s in cols)]
ax = axs[3]
for i in range(len(rows)): ax.axhline(i, color='#cfd3d6', lw=0.5, ls=(0, (1, 2)), zorder=0)
for j in range(4): ax.axvline(j, color='#cfd3d6', lw=0.5, ls=(0, (1, 2)), zorder=0)
for j, s in enumerate(cols):
    for i, f in enumerate(rows):
        if f in s: ax.scatter(j, i, s=26, color=NAVY, lw=0, zorder=3)
ax.set_xticks(range(4), ['Phonons', 'Dielectric', 'JDFT2D', '2D HOIP'])
ax.set_yticks(range(len(rows)), [SYM[f] for f in rows], fontsize=8)
ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(-0.5, 3.5); ax.tick_params(axis='y', length=0)
label(ax, '(d)', x=-0.2)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S3.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

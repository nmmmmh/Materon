"""SI Fig. S11 in the style of the original submission Fig. 2(b),(c): (a) seed-averaged outer-fold MAE
(original_phonons fold_level_results.csv), (b) composition-only ElemNet control (reported means, Table S25).
The ablation of the original Fig. 2(a) is now main-text Fig. 2(b)."""
import csv, sys
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in PDF output
sys.path.insert(0, str(Path(__file__).parent))
from symbols import C_CONV, C_MATERON, C_NOGATE, PURPLE, RUST, INK, style, label
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
fold = {}
for r in csv.DictReader(open(DATA / 'evidence/sources/original_phonons/results/paper_tables/fold_level_results.csv')):
    if r['config'] in ('cgcnn_baseline', 'materon_best_nested'):
        fold.setdefault(r['config'], {})[int(r['outer_fold'])] = float(r['three_seed_mean_mae'])
elem = [('ElemNet', 111.391, C_CONV), ('Gated', 109.9, C_NOGATE), ('Blend', 108.7, '#c4c8cb'), ('Gated\nblend', 106.673, C_MATERON)]
style()
fig, (b, c) = plt.subplots(1, 2, figsize=(5.0, 2.3), gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.5})
for x in (0, 1): b.axvline(x, color='#b0b0b0', lw=0.6, ls=(0, (3, 2)), zorder=0)
for f in range(5):
    b.plot([0, 1], [fold['cgcnn_baseline'][f], fold['materon_best_nested'][f]], color=INK, lw=0.8, marker='o', ms=3, zorder=2)
b.set_xticks([0, 1], ['Conventional', 'Materon']); b.set_xlim(-0.3, 1.3); b.set_ylabel(r'MAE (cm$^{-1}$)')
c.scatter(range(4), [v for _, v, _ in elem], s=26, color=[k for _, _, k in elem], zorder=3, lw=0)
c.grid(axis='y', color='#e3e3e3', lw=0.5); c.set_axisbelow(True); c.set_xlim(-0.5, 3.5)
c.set_ylim(104, 114); c.set_yticks([104, 106, 108, 110, 112, 114]); c.set_xticks(range(4), [n for n, _, _ in elem], fontsize=7); c.set_ylabel('MAE (MPa)')
for x, (_, v, _) in enumerate(elem): c.text(x, v + 0.45, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
label(b, '(a)', x=-0.42); label(c, '(b)', x=-0.36)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S11.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print({k: round(sum(v.values()) / 5, 3) for k, v in fold.items()})

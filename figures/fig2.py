"""Fig. 2 from the reported values (Table II and SI Table S1), in the style of the original submission Fig. 2.
(a) relative reduction of mean test MAE of Materon versus the conventional network; (b) CGCNN phonons ablation."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import C_CGCNN, C_MEGNET, C_CONV, C_NOGATE, C_MATERON, INK, style, label
HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
# (baseline MAE, Materon MAE) from Table II
mae = {('Phonons', 'CGCNN'): (51.976, 38.263), ('Phonons', 'MEGNet'): (29.438, 26.568),
       ('Dielectric', 'CGCNN'): (0.380, 0.315), ('Dielectric', 'MEGNet'): (0.3326, 0.3269),
       ('JDFT2D', 'CGCNN'): (41.731, 36.435), ('JDFT2D', 'MEGNet'): (57.168, 46.961)}
red = {k: 100 * (1 - m / b) for k, (b, m) in mae.items()}
# CGCNN phonons ablation: mean MAE and 95% interval, Table S1 (no interval for the shuffled control)
abl = [('Conventional', 51.976, (41.38, 62.57), C_CONV), ('Ungated', 38.728, (30.48, 46.98), C_NOGATE),
       ('Materon', 38.263, (31.98, 44.54), C_MATERON), ('Materon-PCA', 36.749, (31.13, 42.37), '#c4c8cb'),
       ('Consensus', 41.516, (35.10, 47.93), '#c4c8cb'), ('Parameter-matched', 42.769, (35.90, 49.64), '#c4c8cb'),
       ('Attribute-shuffled', 46.711, None, '#c4c8cb')]
style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={'width_ratios': [1, 1.1], 'wspace': 0.45})
tasks = ['Phonons', 'Dielectric', 'JDFT2D']; y = np.arange(3); h = 0.34
for net, dy, c in (('CGCNN', -h / 2, C_CGCNN), ('MEGNet', h / 2, C_MEGNET)):
    v = [red[(t, net)] for t in tasks]
    a.barh(y + dy, v, height=h, color=c, lw=0, label=net)
    for yy, vv in zip(y + dy, v):
        a.text(vv + 0.6, yy, f'{vv:.1f}', va='center', fontsize=7, color=c)
a.set_yticks(y, tasks); a.set_ylim(2.6, -0.6); a.tick_params(axis='y', length=0)
a.set_xlim(0, 31); a.set_xlabel('MAE reduction (%)')
a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7.5, borderaxespad=0.2)
for i, (nm, mu, ci, c) in enumerate(abl):
    b.barh(i, mu, height=0.58, color=c, lw=0)
    if ci:
        b.errorbar(mu, i, xerr=[[mu - ci[0]], [ci[1] - mu]], fmt='none', ecolor=INK, elinewidth=0.7, capsize=2, capthick=0.7)
b.set_yticks(range(len(abl)), [x[0] for x in abl]); b.set_ylim(len(abl) - 0.4, -0.6); b.tick_params(axis='y', length=0)
b.set_xlabel(r'Phonons MAE (cm$^{-1}$)'); b.set_xlim(0, 70)
label(a, '(a)', x=-0.27, y=1.09); label(b, '(b)', x=-0.42, y=1.09)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_2.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print({k: round(v, 2) for k, v in red.items()})

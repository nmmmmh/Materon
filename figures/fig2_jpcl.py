"""Fig. 3: channel shutdown and channel-level rules are reproduced across ten independently trained
environment-gate models. (a) long-schedule ensemble (L400, five seeds): mean absolute prediction change after disabling
the k top-ranked gate corrections in 64 fixed validation materials (45 chemical systems), mean over seeds with the seed
range as band; (b) ratio of IG-ranked to random shutdown for the short-schedule seeds (Table S17b) and L400;
(c) Spearman correlation between the environment-gate restoration response of a channel and the predicted change over
592 M-site substitutions, per model. Data: data/mechanism/SHUTDOWN_L400.json, GATE_RULES_*.json; short-schedule
shutdown values as in figS14.py (Table S17b). Inference only."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, TEAL, GOLD, GREY, LIGHT, INK, RUST, SYM, C2DB_ORDER, GROUP_ORDER, style, label
C_CGCNN, C_MEGNET, SAGE = NAVY, TEAL, GOLD

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
D = DATA / 'mechanism'
L = json.loads((D / 'SHUTDOWN_L400.json').read_text())
SEEDS = ['42', '7', '123', '2024', '31415']
k = L['42']['k']
METH = [('gate_path_ig', 'IG', NAVY, 'o', '-'), ('modulation_magnitude', 'Gate magnitude', SAGE, '^', '-'), ('random', 'Random', GREY, 'D', '-')]
# short-schedule seeds (Table S17b): mean absolute response per seed (rows 42, 7, 123, 2024, 31415)
SHORT = {'gate_path_ig': [[0.143372, 0.155714, 0.197464, 0.235285], [0.115204, 0.185936, 0.211883, 0.259611], [0.175532, 0.278669, 0.296141, 0.308993], [0.150312, 0.259045, 0.267784, 0.320349], [0.177914, 0.236332, 0.25364, 0.299993]],
         'random': [[0.034807, 0.073252, 0.102939, 0.160163], [0.028483, 0.062607, 0.090054, 0.151186], [0.034613, 0.078331, 0.116525, 0.188512], [0.042668, 0.088884, 0.126388, 0.204838], [0.043456, 0.087204, 0.117674, 0.181728]]}

style(); plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.0, 4.35))
gs = fig.add_gridspec(2, 3, width_ratios=[0.85, 1.0, 0.9], height_ratios=[1, 0.78], wspace=0.42, hspace=0.42, left=0.07, right=0.99, top=0.95, bottom=0.08)
mb, a, b = (fig.add_subplot(gs[0, i]) for i in range(3))
r4 = fig.add_subplot(gs[1, :])
# (a) Matbench: relative reduction of the mean test MAE, Materon vs conventional network (Table S4)
TASKS = ['Phonons', 'Dielectric', 'JDFT2D']
RED = {'CGCNN': [26.4, 17.1, 12.7], 'MEGNet': [9.7, 1.7, 17.9]}
yy = np.arange(len(TASKS))
for n, (net, col) in enumerate((('CGCNN', C_CGCNN), ('MEGNet', C_MEGNET))):
    mb.barh(yy + (n - 0.5) * 0.36, RED[net], height=0.34, color=col, label=net)
mb.set_yticks(yy); mb.set_yticklabels(TASKS); mb.invert_yaxis(); mb.set_xlim(0, 36); mb.set_xticks([0, 10, 20, 30])
mb.set_xlabel('MAE reduction (%)')
mb.legend(loc='center right', bbox_to_anchor=(1.03, 0.5), fontsize=6.5, handlelength=1.0, handletextpad=0.3)

for key, name, col, mk, ls in METH:
    y = np.array([[r['mean'] for r in L[s]['methods'][key]] for s in SEEDS])
    if key in ('gate_path_ig', 'random'):
        a.fill_between(k, y.min(0), y.max(0), color=col, alpha=0.15, lw=0)
    a.plot(k, y.mean(0), marker=mk, ms=3.5, lw=1.1, color=col, ls=ls, mec='white', mew=0.4, label=name)
a.set_xticks(k); a.set_ylim(0, None)
a.set_xlabel('Channels switched off, $k$'); a.set_ylabel(r'Mean $|\Delta F|$ (eV)')

ig_l = np.array([[r['mean'] for r in L[s]['methods']['gate_path_ig']] for s in SEEDS])
rn_l = np.array([[r['mean'] for r in L[s]['methods']['random']] for s in SEEDS])
for n, (ratio, col, mk) in enumerate(((np.array(SHORT['gate_path_ig']) / np.array(SHORT['random']), GREY, 'o'),
                                      (ig_l / rn_l, NAVY, 'o'))):
    dx = (n - 0.5) * 0.35
    for row in ratio: b.plot([x + dx for x in k], row, ls='none', marker=mk, ms=2.5, color=col, alpha=0.5, mew=0)
    b.errorbar([x + dx for x in k], ratio.mean(0), yerr=ratio.std(0, ddof=1), marker=mk, ms=3.5, lw=1.1, color=col,
               mec='white', mew=0.4, capsize=1.5, elinewidth=0.7, mfc=col if n else 'white')
b.axhline(1, color=GREY, lw=0.6, ls=(0, (2, 2)))
b.set_xticks(k); b.set_ylim(0.8, 5.6); b.set_xlabel('Channels switched off, $k$'); b.set_ylabel('IG / random')

h = [plt.Line2D([], [], ls='none', marker='o', mfc='white', mec=GREY, ms=4), plt.Line2D([], [], ls='none', marker='o', color=NAVY, ms=4)]
b.legend(h, ['80 epochs', '400 epochs'], loc='upper right', fontsize=6.5)
a.legend(loc='upper left', fontsize=6.3, handlelength=1.4, handletextpad=0.3, borderaxespad=0.2)
# (d) per-channel Spearman correlation between the neighbour restoration response R_j and the predicted change, ten models
from scipy.stats import spearmanr
order = [C2DB_ORDER.index(c) for c in GROUP_ORDER]
for n, (tag, col) in enumerate((('SEEDS', GREY), ('L400', NAVY))):
    J = [x for x in json.loads((D / f'GATE_RULES_{tag}.json').read_text())['jobs'] if x['kind'] == 'M']
    nat = np.array([x['native_eV'] for x in J]); R = np.array([x['R_env_eV'] for x in J])
    rho = np.array([[spearmanr(R[:, m, j], nat[:, m])[0] for j in range(20)] for m in range(nat.shape[1])])
    allr = rho if n == 0 else np.vstack([allr, rho])
    for p, j in enumerate(order):
        xs = p + (n - 0.5) * 0.26 + np.linspace(-0.05, 0.05, len(rho))
        r4.scatter(xs, rho[:, j], s=7, marker='o', facecolors='white' if n == 0 else col, edgecolors=col, lw=0.6, zorder=3)
for p, j in enumerate(order):
    lo, hi = allr[:, j].min(), allr[:, j].max()
    r4.plot([p, p], [lo, hi], color=NAVY if C2DB_ORDER[j] == 'atomic_ndval' else LIGHT, lw=2.6, solid_capstyle='round', zorder=1, alpha=0.9 if C2DB_ORDER[j] == 'atomic_ndval' else 1)
jd = GROUP_ORDER.index('atomic_ndval')
r4.axvspan(jd - 0.48, jd + 0.48, color=LIGHT, alpha=0.45, lw=0, zorder=0)
r4.axhline(0, color=INK, lw=0.6)
r4.axhline(0.025, color=GREY, lw=0.8, ls=(0, (3, 2)))
r4.set_xticks(range(20)); r4.set_xticklabels([SYM[c] for c in GROUP_ORDER], fontsize=6.5)
r4.get_xticklabels()[jd].set_color(NAVY); r4.get_xticklabels()[jd].set_fontweight('bold')
r4.set_xlim(-0.6, 19.6); r4.set_ylim(-0.5, 0.75); r4.set_yticks([-0.4, 0, 0.4]); r4.set_ylabel('Spearman correlation')
r4.tick_params(axis='x', length=0, pad=3)
for ax, t in zip((mb, a, b), ('(a)', '(b)', '(c)')): label(ax, t, x=-0.26 if ax is not mb else -0.5, y=1.0)
label(r4, '(d)', x=-0.05, y=1.02)
print('N_d min over ten models', round(allr[:, C2DB_ORDER.index('atomic_ndval')].min(), 3), 'others max-min', round(max(allr[:, j].min() for j in range(20) if j != C2DB_ORDER.index('atomic_ndval')), 3))
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_2_JPCL.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
r = ig_l / rn_l
print('L400 IG/random by k mean', np.round(r.mean(0), 2), 'sd', np.round(r.std(0, ddof=1), 2))

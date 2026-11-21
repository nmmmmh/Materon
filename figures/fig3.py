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
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, OCHRE, SAGE, GREY, LIGHT, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
D = DATA / 'mechanism'
L = json.loads((D / 'SHUTDOWN_L400.json').read_text())
SEEDS = ['42', '7', '123', '2024', '31415']
k = L['42']['k']
METH = [('gate_path_ig', 'Gate-path IG', NAVY, 'o', '-'), ('endpoint_gradient', 'Endpoint gradient', OCHRE, 's', '--'),
        ('modulation_magnitude', 'Modulation magnitude', SAGE, '^', '-'), ('random', 'Random', GREY, 'D', '-')]
# short-schedule seeds (Table S17b): mean absolute response per seed (rows 42, 7, 123, 2024, 31415)
SHORT = {'gate_path_ig': [[0.143372, 0.155714, 0.197464, 0.235285], [0.115204, 0.185936, 0.211883, 0.259611], [0.175532, 0.278669, 0.296141, 0.308993], [0.150312, 0.259045, 0.267784, 0.320349], [0.177914, 0.236332, 0.25364, 0.299993]],
         'random': [[0.034807, 0.073252, 0.102939, 0.160163], [0.028483, 0.062607, 0.090054, 0.151186], [0.034613, 0.078331, 0.116525, 0.188512], [0.042668, 0.088884, 0.126388, 0.204838], [0.043456, 0.087204, 0.117674, 0.181728]]}

style(); plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.0, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.05], wspace=0.42, left=0.07, right=0.99, top=0.84, bottom=0.2)
a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

for key, name, col, mk, ls in METH:
    y = np.array([[r['mean'] for r in L[s]['methods'][key]] for s in SEEDS])
    if key in ('gate_path_ig', 'random'):
        a.fill_between(k, y.min(0), y.max(0), color=col, alpha=0.15, lw=0)
    a.plot(k, y.mean(0), marker=mk, ms=3.5, lw=1.1, color=col, ls=ls, mec='white', mew=0.4, label=name)
a.set_xticks(k); a.set_ylim(0, None)
a.set_xlabel('Disabled channels, $k$'); a.set_ylabel(r'Mean $|\Delta F|$ (eV)')

ig_l = np.array([[r['mean'] for r in L[s]['methods']['gate_path_ig']] for s in SEEDS])
rn_l = np.array([[r['mean'] for r in L[s]['methods']['random']] for s in SEEDS])
for n, (ratio, col, mk) in enumerate(((np.array(SHORT['gate_path_ig']) / np.array(SHORT['random']), GREY, 'o'),
                                      (ig_l / rn_l, NAVY, 'o'))):
    dx = (n - 0.5) * 0.35
    for row in ratio: b.plot([x + dx for x in k], row, ls='none', marker=mk, ms=2.5, color=col, alpha=0.5, mew=0)
    b.errorbar([x + dx for x in k], ratio.mean(0), yerr=ratio.std(0, ddof=1), marker=mk, ms=3.5, lw=1.1, color=col,
               mec='white', mew=0.4, capsize=1.5, elinewidth=0.7, mfc=col if n else 'white')
b.axhline(1, color=GREY, lw=0.6, ls=(0, (2, 2)))
b.set_xticks(k); b.set_ylim(0.8, 5.6); b.set_xlabel('Disabled channels, $k$'); b.set_ylabel('IG / random')

CH = [(13, r'$N_{d}$'), (8, r'$\varepsilon_{\mathrm{H}}$'), (6, r'$E_{\mathrm{IP}}$'), (12, r'$N_{\mathrm{v}}$')]
for n, (tag, col, mfc) in enumerate((('SEEDS', GREY, 'white'), ('L400', NAVY, NAVY))):
    J = [j for j in json.loads((D / f'GATE_RULES_{tag}.json').read_text())['jobs'] if j['kind'] == 'M']
    nat = np.array([j['native_eV'] for j in J]); R = np.array([j['R_env_eV'] for j in J])
    for m, (ch, _) in enumerate(CH):
        rho = [spearmanr(R[:, s, ch], nat[:, s])[0] for s in range(5)]
        c.scatter(m + (n - 0.5) * 0.28 + np.linspace(-0.06, 0.06, 5), rho, s=14, color=col, facecolors=mfc, lw=0.8, zorder=3)
c.axhline(0, color=LIGHT, lw=0.6, zorder=0)
c.set_xticks(range(len(CH))); c.set_xticklabels([s for _, s in CH]); c.set_ylim(-0.6, 0.8)
c.set_ylabel(r'$\rho_{\mathrm{s}}(R_{j}, \Delta F)$')
h = [plt.Line2D([], [], ls='none', marker='o', mfc='white', mec=GREY, ms=4), plt.Line2D([], [], ls='none', marker='o', color=NAVY, ms=4)]
b.legend(h, ['Short schedule', 'L400'], loc='upper right', fontsize=6.5)
fig.legend(*a.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.36, 0.86), ncol=4, fontsize=6.5,
           columnspacing=0.8, handletextpad=0.25, handlelength=1.2)
for ax, t in zip((a, b, c), ('(a)', '(b)', '(c)')): label(ax, t, x=-0.24, y=1.0)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_3.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
r = ig_l / rn_l
print('L400 IG/random by k mean', np.round(r.mean(0), 2), 'sd', np.round(r.std(0, ddof=1), 2))

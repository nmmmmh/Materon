"""Fig. S14: attribution-ranked channel shutdown, replotted from the saved summary values in SI Table S17
(64 validation materials, 45 chemical systems; random = mean of 20 rankings per material). Panel (a): reference
model (seed 42); panel (b): five training seeds 42, 7, 123, 2024, 31415 (Table S17b). No new evaluation."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import TEAL, NAVY, GOLD, GREY, style, label
OCHRE, SAGE = TEAL, GOLD

OUT = Path(__file__).resolve().parent / 'output'; OUT.mkdir(exist_ok=True)
k = [1, 3, 5, 10]
# mean absolute response (eV) and chemical-system-grouped 95% interval, Table S17
data = {
    'Gate-path IG':         ([0.143372, 0.155714, 0.197464, 0.235285], [(0.124569, 0.163384), (0.109862, 0.209623), (0.141813, 0.264674), (0.173665, 0.301412)], NAVY, 'o'),
    'Endpoint gradient':    ([0.146224, 0.149934, 0.182731, 0.234842], [(0.126628, 0.167336), (0.108467, 0.198987), (0.135521, 0.235398), (0.174740, 0.303120)], OCHRE, 's'),
    'Gate magnitude': ([0.087830, 0.145373, 0.169184, 0.201203], [(0.067482, 0.111833), (0.107828, 0.186554), (0.122120, 0.221230), (0.150783, 0.256869)], SAGE, '^'),
    'Random':               ([0.034807, 0.073252, 0.102939, 0.160163], [(0.028677, 0.042781), (0.061967, 0.085648), (0.086367, 0.120576), (0.128747, 0.194051)], GREY, 'D'),
}
# mean absolute response (eV) per seed (rows: 42, 7, 123, 2024, 31415; columns: k), Table S17b
seeds = {
    'Gate-path IG': [[0.143372, 0.155714, 0.197464, 0.235285], [0.115204, 0.185936, 0.211883, 0.259611], [0.175532, 0.278669, 0.296141, 0.308993], [0.150312, 0.259045, 0.267784, 0.320349], [0.177914, 0.236332, 0.25364, 0.299993]],
    'Endpoint gradient': [[0.146224, 0.149934, 0.182731, 0.234842], [0.115776, 0.196321, 0.216697, 0.257352], [0.174503, 0.273485, 0.289002, 0.299514], [0.153883, 0.246461, 0.279748, 0.319653], [0.184221, 0.249816, 0.271536, 0.285848]],
    'Gate magnitude': [[0.08783, 0.145373, 0.169184, 0.201203], [0.051152, 0.114387, 0.175318, 0.221023], [0.113544, 0.199526, 0.228214, 0.294653], [0.098736, 0.176053, 0.217771, 0.266115], [0.104954, 0.159692, 0.233872, 0.278875]],
    'Random': [[0.034807, 0.073252, 0.102939, 0.160163], [0.028483, 0.062607, 0.090054, 0.151186], [0.034613, 0.078331, 0.116525, 0.188512], [0.042668, 0.088884, 0.126388, 0.204838], [0.043456, 0.087204, 0.117674, 0.181728]],
}
style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'wspace': 0.3})
for name, (m, ci, c, mk) in data.items():
    ls = '--' if name == 'Endpoint gradient' else '-'
    if name in ('Gate-path IG', 'Random'):
        a.fill_between(k, [x[0] for x in ci], [x[1] for x in ci], color=c, alpha=0.15, lw=0)
    a.plot(k, m, marker=mk, ms=3.5, lw=1.1, color=c, label=name, mec='white', mew=0.4, linestyle=ls)
a.set_xlabel('Channels switched off, $k$'); a.set_ylabel(r'Mean $|\Delta F|$ (eV)')
a.set_xticks(k); a.set_ylim(0, 0.32)
a.annotate('1.92×', xy=(5, 0.197464), xytext=(3.3, 0.245), fontsize=7.5, color=NAVY,
           arrowprops=dict(arrowstyle='-', lw=0.5, color=NAVY, shrinkB=3))
rnd = np.array(seeds['Random'])
for n, name in enumerate(('Gate-path IG', 'Endpoint gradient', 'Gate magnitude')):
    c, mk = data[name][2], data[name][3]
    r = np.array(seeds[name]) / rnd
    dx = (n - 1) * 0.18
    for row in r:
        b.plot([x + dx for x in k], row, ls='none', marker=mk, ms=2.5, color=c, alpha=0.5, mew=0)
    b.errorbar([x + dx for x in k], r.mean(0), yerr=r.std(0, ddof=1), marker=mk, ms=3.5, lw=1.1, color=c,
               mec='white', mew=0.4, capsize=1.5, elinewidth=0.7,
               linestyle='--' if name == 'Endpoint gradient' else '-')
b.axhline(1, color=GREY, lw=0.6, ls=(0, (2, 2)))
b.set_xlabel('Channels switched off, $k$'); b.set_ylabel('Ratio to random'); b.set_xticks(k); b.set_ylim(0.8, 5.6)
b.annotate('2.22×', xy=(4.82, 2.22), xytext=(5.45, 2.72), fontsize=7.2, color=NAVY,
           arrowprops=dict(arrowstyle='-', lw=0.5, color=NAVY, shrinkB=3))
a.set_title('Reference model', fontsize=8, pad=4)
b.set_title('Five training seeds', fontsize=8, pad=4)
fig.legend(*a.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=4,
           fontsize=6.5, columnspacing=0.8, handletextpad=0.25, handlelength=1.2)
fig.subplots_adjust(top=0.87, bottom=0.19, wspace=0.3)
label(a, '(a)', x=-0.2, y=1.0); label(b, '(b)', x=-0.2, y=1.0)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_S14.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
r = np.array(seeds['Gate-path IG']) / rnd
print(r.mean(0).round(3), r.std(0, ddof=1).round(3))

"""SI Fig. S5: full channel-shutdown curves with chemical-system-grouped 95% intervals for every ranking,
from explanation_controls/outputs/ANALYSIS.json (evidence archive; same values as Table S17). No new evaluation."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in PDF output
sys.path.insert(0, str(Path(__file__).parent))
from symbols import SAGE, NAVY, OCHRE, GREY, INK, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
d = json.loads((DATA / 'evidence/sources/research/eigate_v2/c2db_workfunction_v1_20261013/evidence_strengthening_v1/explanation_controls/outputs/ANALYSIS.json').read_text(encoding='utf-8'))
K = [1, 3, 5, 10]; x = np.arange(4)
M = [('gate_path_ig', 'Gate-path IG', NAVY, 'o'), ('endpoint_gradient', 'Endpoint gradient', OCHRE, 's'),
     ('modulation_magnitude', 'Modulation magnitude', SAGE, '^'), ('random', 'Random', GREY, 'D')]
style()
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={'wspace': 0.32})
for ax, key, yl in ((axs[0], 'absolute_delta_prediction_eV', r'Mean $|\Delta F|$ (eV)'),
                    (axs[1], 'delta_prediction_eV', r'Mean $\Delta F$ (eV)')):
    for n, (m, lab, c, mk) in enumerate(M):
        v = [d['curves'][m][str(k)][key] for k in K]
        mu = np.array([e['mean'] for e in v]); lo = np.array([e['ci95_low'] for e in v]); hi = np.array([e['ci95_high'] for e in v])
        xx = x + (n - 1.5) * 0.15
        ax.errorbar(xx, mu, yerr=[mu - lo, hi - mu], fmt=mk, ms=3.5, color=c, mec='white', mew=0.4,
                    elinewidth=0.8, capsize=0, label=lab, zorder=3)
    ax.set_xticks(x, K); ax.set_xlabel('Disabled channels, $k$'); ax.set_ylabel(yl)
axs[0].set_ylim(0, 0.32)
axs[1].axhline(0, color=INK, lw=0.6, zorder=1)
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.93), ncol=4, fontsize=7.5)
label(axs[0], '(a)', x=-0.2, y=1.0); label(axs[1], '(b)', x=-0.2, y=1.0)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S5.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print(round(d['curves']['gate_path_ig']['5']['absolute_delta_prediction_eV']['mean'], 6),
      round(d['curves']['random']['5']['absolute_delta_prediction_eV']['mean'], 6))

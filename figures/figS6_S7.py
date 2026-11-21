"""SI Figs. S6 and S7: same plotting logic as cross_environment_explain_v1/render_figures_v2.py (evidence archive),
with symbol channel labels, panel letters and no in-figure titles. Data: FIGURE_SOURCE.json, SAME_ELEMENT_CONTEXT_DISPERSION.json
(paths of the evidence package under data/evidence/sources/research/...)."""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Element
sys.path.insert(0, str(Path(__file__).parent))
from symbols import seq_cmap, SYM, C2DB_ORDER, GROUP_ORDER, NAVY, OCHRE, INK, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
SRC = DATA / 'evidence/sources/research/eigate_v2/c2db_workfunction_v1_20261013/cross_environment_explain_v1'
NAMES = [SYM[f] for f in GROUP_ORDER]
IDX = [C2DB_ORDER.index(f) for f in GROUP_ORDER]  # display in grouped order
ro = lambda v: np.asarray(v)[IDX]
read = lambda n: json.loads((SRC / n).read_text(encoding='utf-8'))
style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
# ---------------------------------------------------------------- Fig. S7: fixed cross-environment cases
src = read('FIGURE_SOURCE.json')
assert len(src['feature_names']) == 20
cases = src['cases']; selected = []
for j in range(3):
    pair = cases[2 * j:2 * j + 2]
    z = min(int(k) for k in pair[0]['contributions'])
    central = [next(r for r in row['element_readouts'] if r['Z'] == z) for row in pair]
    selected.append((pair, z, central))
gmax = 1.1 * max(np.max(abs(np.array(row[arm + '_gate']) - 1)) for _, _, c in selected for row in c for arm in ('env', 'element'))
imax = 1.1 * max(np.max(abs(np.array(row['contributions'][str(z)][arm]))) for p, z, _ in selected for row in p for arm in ('env', 'element'))
fig, axes = plt.subplots(2, 3, figsize=(7.0, 7.4), sharey=True, gridspec_kw={'hspace': 0.28, 'wspace': 0.12})
fig.subplots_adjust(right=0.83)
y = np.arange(20)
for j, (pair, z, central) in enumerate(selected):
    sym = Element.from_Z(z).symbol; formula = pair[0]['id'].split('-')[0]
    ax = axes[0, j]
    for k, (c, m) in enumerate(((NAVY, 'o'), (OCHRE, 's'))):
        ax.plot(ro(central[k]['env_gate']) - 1, y + (k - 0.5) * 0.36, ls='none', color=c, marker=m, ms=3.2, mec='white', mew=0.3, label=('A', 'B')[k], zorder=3)
    ax.plot(ro(central[0]['element_gate']) - 1, y, 'x', color=INK, ms=3.2, mew=0.8, label='el.', zorder=2)
    ax.set_xlim(-gmax, gmax); ax.set_xticks([-0.4, 0, 0.4]); ax.set_xlabel(r'$g-1$')
    ax.set_title(sym + ' in ' + re.sub(r'(\d+)', lambda m: '$_{' + m.group(1) + '}$', formula), fontsize=8.5)
    ax = axes[1, j]
    for k, row in enumerate(pair):
        c = (NAVY, OCHRE)[k]
        for arm, fc in (('env', c), ('element', 'white')):
            ax.plot(ro(row['contributions'][str(z)][arm]), y + (k - 0.5) * 0.36, ls='none', color=c, marker=('o', 's')[k], ms=3.2, mfc=fc, mew=0.7,
                    label=('A', 'B')[k] + (' env.' if arm == 'env' else ' el.'), zorder=3)
    ax.set_xlim(-imax, imax); ax.set_xlabel('IG (eV)')
    for ax in axes[:, j]:
        ax.axvline(0, color=INK, lw=0.6); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
        for yy in y: ax.axhline(yy, color='#eef1f4', lw=0.5, zorder=0)
        for sep in (4.5, 10.5, 14.5): ax.axhline(sep, color='#c9d1d9', lw=0.5, zorder=1)
for r in range(2):
    axes[r, 0].set_yticks(y, NAMES, fontsize=8); axes[r, 0].set_ylim(19.7, -0.7)
for i, ax in enumerate(axes.flat):
    label(ax, '(' + 'abcdef'[i] + ')', x=-0.3 if i % 3 == 0 else -0.08, y=1.03)
handles, labels_ = axes[0, 2].get_legend_handles_labels()
fig.legend(handles, labels_, fontsize=6.5, loc='center left', bbox_to_anchor=(0.845, 0.74), handletextpad=0.2, labelspacing=0.25)
handles, labels_ = axes[1, 2].get_legend_handles_labels()
fig.legend(handles, labels_, fontsize=6.5, loc='center left', bbox_to_anchor=(0.845, 0.24), handletextpad=0.2, labelspacing=0.25)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_S7.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

# ---------------------------------------------------------------- Fig. S6: same-element context dispersion
groups = read('SAME_ELEMENT_CONTEXT_DISPERSION.json')['groups']
mat = np.array([g['env_gate_ptp'] for g in groups])[:, IDX]
fig, ax = plt.subplots(figsize=(5.4, 7.6))
im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=seq_cmap(), vmin=0, vmax=float(mat.max()))
ax.set_xticks(np.arange(20), NAMES, fontsize=8); ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(groups)), [f"{g['element']} ({g['materials']})" for g in groups], fontsize=6.5)
ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(r'Range of $\bar{g}$')
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S6.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

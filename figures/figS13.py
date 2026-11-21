"""Fig. S13: re-typeset copy of the DFT application figure (candidate_dft_v3) in the unified style.
Reads the exported data of candidate_dft_v3 (data/candidate_dft_v3: GAINS, WORK_FUNCTIONS, POTENTIAL, PDOS) and the OVITO
render of the parent geometry (data/structures/structure_Al2S2.png with its atom marks structure_Al2S2.json)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbols import style, plt, label, C_B, C_GA, INK, GREY

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
V3 = DATA / 'candidate_dft_v3'
style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
C_P = '#4f5763'
COL = {'parent': C_P, 'B': C_B, 'Ga': C_GA}
NAME = {'parent': r'Al$_2$S$_2$', 'B': r'AlBS$_2$', 'Ga': r'AlGaS$_2$'}
gains = pd.read_csv(V3 / 'GAINS.csv').set_index('system')
wf = pd.read_csv(V3 / 'WORK_FUNCTIONS.csv').set_index('system')
pot = pd.read_csv(V3 / 'POTENTIAL.csv')
pdos = pd.read_csv(V3 / 'PDOS.csv')

fig = plt.figure(figsize=(7.0, 5.4))
# ---------------------------------------------------------------- (a) structure and design response
# parent geometry (dft_design_closed_loop_v1/structures/parent.json as data/structures/Al2S2_parent.vasp) rendered with
# OVITO (render_structures.py):
# OV_EDGE=1 OV_DIR=-0.35,0.55,-1.0 OV_SIZE=1400,1200 OV_MARK="0,0,0;1,0,0" render_structure_ovito.py ... structure_Al2S2.png 2 2
from matplotlib.patches import Circle
img = plt.imread(DATA / 'structures' / 'structure_Al2S2.png')
mk = json.loads((DATA / 'structures' / 'structure_Al2S2.json').read_text())['marks']
ys, xs = np.nonzero(img[..., 3] > 0.02); x0, x1, y0, y1 = xs.min() - 70, xs.max() + 30, ys.min() - 10, ys.max() + 10
ax = fig.add_axes([-0.01, 0.56, 0.205, 0.38])
ax.imshow(img[y0:y1, x0:x1], interpolation='lanczos', zorder=1); ax.axis('off')
for key, sub, dxy in (('0,0,0', '0', (-40, 150)), ('1,0,0', '1', (-215, -160))):
    x, y = mk[key][0] - x0, mk[key][1] - y0
    ax.add_patch(Circle((x, y), 54, fc='none', ec=INK, lw=0.8, zorder=5))
    ax.annotate(rf'Al$_{sub}$', xy=(x, y), xytext=(x + dxy[0], y + dxy[1]), fontsize=7.5, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK, shrinkA=2, shrinkB=5), zorder=6)
ax.text((x1 - x0) / 2, -15, r'Al$_2$S$_2$', ha='center', va='bottom', fontsize=8)
ax.text((x1 - x0) / 2, y1 - y0 + 15, r'Al$_0$ $\rightarrow$ B, Ga', ha='center', va='top', fontsize=7.5)
ax.set_ylim(y1 - y0 + 150, -120); ax.set_xlim(-10, x1 - x0 + 10)
fig.text(0.0, 0.955, '(a)', fontsize=9, fontweight='bold', va='bottom')
ax2 = fig.add_axes([0.22, 0.6, 0.25, 0.32])
for k, s in enumerate(('B', 'Ga')):
    m, d = gains.loc[s, 'model_gain_eV'], gains.loc[s, 'DFT_gain_eV']
    model_y, dft_y = k - 0.08, k + 0.08
    ax2.plot([m, d], [model_y, dft_y], color=COL[s], lw=1.0, zorder=1)
    ax2.scatter([m], [model_y], s=26, facecolor='white', edgecolor=COL[s], lw=1.0, zorder=3, label='Materon' if k == 0 else None)
    ax2.scatter([d], [dft_y], s=26, color=COL[s], zorder=3, label='DFT' if k == 0 else None)
    ax2.text(d, dft_y + 0.14, f'{d:.3f}', ha='center', va='top', fontsize=6.5, color=COL[s])
    ax2.text(m, model_y - 0.14, f'{m:.3f}', ha='center', va='bottom', fontsize=6.5, color=COL[s])
ax2.set_yticks([0, 1], ['B', 'Ga']); ax2.set_ylim(1.6, -0.7); ax2.set_xlim(0, 1.4); ax2.set_xticks([0, 0.4, 0.8, 1.2])
ax2.set_xlabel(r'$\Delta\Phi$ (eV)'); ax2.tick_params(axis='y', length=0)
lg = ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=1, fontsize=6.5, handletextpad=0.1, labelspacing=0.3)
for h in lg.legend_handles: h.set_edgecolor(INK); h.set_facecolor('white' if h.get_label() == 'Materon' else INK)

# ---------------------------------------------------------------- (b) planar-averaged potential
axb = fig.add_axes([0.53, 0.6, 0.45, 0.33])
for s in ('parent', 'Ga', 'B'):
    d = pot[pot.system == s]
    axb.plot(d.z_A, d.Vbar_minus_EF_eV, color=COL[s], lw=1.0, label=NAME[s])
axb.axhline(0, color=GREY, lw=0.6, ls='--')
axb.text(23.6, -0.8, r'$E_\mathrm{F}$', ha='right', va='top', fontsize=7, color='#666666')
axb.set_xlim(0, 24); axb.set_xticks([0, 6, 12, 18, 24]); axb.set_ylim(-24, 7)
axb.set_xlabel(r'$z$ (Å)'); axb.set_ylabel(r'$\bar{V}(z) - E_\mathrm{F}$ (eV)')
fig.text(0.47, 0.955, '(b)', fontsize=9, fontweight='bold', va='bottom')
ins = axb.inset_axes([0.07, 0.1, 0.33, 0.48])
c = 24.0
for s in ('parent', 'Ga', 'B'):
    d = pot[pot.system == s]
    L = d[(d.z_A / c >= 0.1) & (d.z_A / c < 0.3)].Vbar_minus_EF_eV.mean()
    R = d[(d.z_A / c >= 0.7) & (d.z_A / c < 0.9)].Vbar_minus_EF_eV.mean()
    assert abs((L + R) / 2 - wf.loc[s, 'DFT_Phi_eV']) < 2e-3, (s, (L + R) / 2, wf.loc[s, 'DFT_Phi_eV'])
    ins.plot([0, 1], [L, L], color=COL[s], lw=1.2); ins.plot([2, 3], [R, R], color=COL[s], lw=1.2)
    ins.text(3.25, R, f"{wf.loc[s, 'DFT_Phi_eV']:.3f}", va='center', fontsize=6.5, color=COL[s])
ins.set_xlim(-0.3, 4.6); ins.set_ylim(2.9, 4.5)
ins.set_xticks([0.5, 2.5], ['Left', 'Right']); ins.set_yticks([3, 4])
ins.tick_params(labelsize=6.5, length=2); ins.set_title(r'$\Phi$ (eV)', fontsize=6.5, pad=2)
ins.spines['top'].set_visible(False); ins.spines['right'].set_visible(False)
axb.legend(loc='lower right', fontsize=6.5, handlelength=1.4)

# ---------------------------------------------------------------- (c),(d) Al1 PDOS
ymax = 0.42
SIG = 0.1   # eV, Gaussian broadening of the PDOS
for col, (orb, x0, lab) in enumerate((('s', 0.07, '(c)'), ('p', 0.57, '(d)'))):
    for r, s in enumerate(('parent', 'B', 'Ga')):
        ax = fig.add_axes([x0, 0.33 - r * 0.135, 0.41, 0.12])
        d = pdos[(pdos.system == s) & (pdos.orbital == orb)]
        e, y = d.E_minus_Evac_eV.to_numpy(), d.PDOS_states_per_eV.to_numpy()
        y = np.array([np.sum(y * np.exp(-0.5 * ((e - x) / SIG) ** 2)) for x in e]) * np.mean(np.diff(e)) / (SIG * np.sqrt(2 * np.pi))
        ax.plot(e, y, color=COL[s], lw=0.8)
        ax.fill_between(e, 0, y, color=COL[s], alpha=0.12, lw=0)
        ef = d.EF_minus_Evac_eV.iloc[0]
        ax.axvline(ef, color=COL[s], lw=0.7, ls='--')
        ax.text(0.015, 0.92, NAME[s], transform=ax.transAxes, va='top', fontsize=7, color=COL[s])
        ax.set_xlim(-21, 2); ax.set_ylim(-0.01, ymax); ax.set_yticks([0, 0.2, 0.4])
        if r < 2: ax.tick_params(axis='x', labelbottom=False)
        else: ax.set_xlabel(r'$E - E_\mathrm{vac}$ (eV)')
        if r == 0:
            label(ax, lab, x=-0.16, y=1.12)
            ax.text(1.0, 1.08, rf'Al$_1$: ${orb}$', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5)
            ax.text(ef + 0.3, ymax * 0.95, r'$E_\mathrm{F}$', fontsize=6.5, color=COL[s], va='top')
        if r == 1: ax.set_ylabel(rf'$D_{{{orb}}}(E)$ (states eV$^{{-1}}$)')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_S13.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

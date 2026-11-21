"""JPCL Figure S11: charge rearrangement at the Ta2Te2/MoS2 and MoTaTe2/MoS2 interfaces (lowest-energy registries).
(a) plane-integrated charge-density difference dlam(z) = lam(het) - lam(metal) - lam(MoS2), electrons per Angstrom;
(b) its integral Q(z) from the bottom of the cell. Shading marks the metal (left) and MoS2 (right) layers.
Data: data/interface/cdd.json and <job>_fig.json (written by scripts/interface/cdd.py and extract.py)."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, GREY, INK, LIGHT, style, label

HERE = Path(__file__).resolve().parent
I = HERE / 'data' / 'interface'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
Q = json.loads((I / 'cdd.json').read_text())
IF = {'Ta2Te2': ('Ta2Te2_MoS2_s1', GREY), 'MoTaTe2': ('MoTaTe2_MoS2_s0', NAVY)}

style(); plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.32, left=0.08, right=0.985, top=0.9, bottom=0.18))
for nm, (job, col) in IF.items():
    r = json.loads((I / f'{job}_fig.json').read_text()); zat = np.array(r['atoms_z']); sp = r['species']; z0 = zat.min()
    q = Q[job]; z = np.array(q['z']) - z0; m = (z > -4) & (z < 15)
    lw = 0.9 if col == GREY else 1.1
    a.plot(z[m], np.array(q['dlam'])[m], color=col, lw=lw, label=nm.replace('2', '$_2$') + '/MoS$_2$')
    b.plot(z[m], np.array(q['Q'])[m], color=col, lw=lw)
    if col == NAVY:
        zTe = [zat[k] for k, e in enumerate(sp) if e == 'Te']; zS = [zat[k] for k, e in enumerate(sp) if e == 'S']
        for ax in (a, b):
            for lo, hi in ((min(zTe), max(zTe)), (min(zS), max(zS))):
                ax.axvspan(lo - z0, hi - z0, color=LIGHT, alpha=0.35, lw=0, zorder=0)
for ax in (a, b):
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (2, 2))); ax.set_xlim(-4, 15); ax.set_xlabel(r'$z$ ($\mathrm{\AA}$)')
a.set_ylabel(r'$\Delta\lambda$ ($e\,\mathrm{\AA}^{-1}$)'); b.set_ylabel(r'$Q$ ($e$ per cell)')
a.legend(loc='lower left', fontsize=6.0, frameon=False, handlelength=0.9)
for ax, t in zip((a, b), ('(a)', '(b)')): label(ax, t, x=-0.16, y=1.02)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_SI_11.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print({k: round(Q[j]['Q_mid_e_per_cell'], 4) for k, (j, _) in IF.items()})

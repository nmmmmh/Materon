"""Fig. S10: core-referenced decomposition of the work-function change of early-transition-metal hosts.
dPhi = -(level shift of the reference core vs vacuum) - (E_F shift vs that core). Points on the diagonal are pure band filling;
vertical distance from it is the electrostatic (dipole and level) term. The two axes share the E_F term, so their correlation is
not independent evidence; the reference test is the size of the level term and the sign agreement of dN_eff with the nominal
valence change (printed, for the caption). (a) chalcogen-core reference; (b) unsubstituted
host-metal core reference. Same pairs and markers as Fig. 4 (filled: contrast set and round 1; open: round 2; open stars: designs;
Ta2TeI is Janus, face-mean dPhi). Data: data/mechanism."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, TEAL, RUST, PURPLE, GOLD, GREY, INK, LIGHT, style, label
from pymatgen.core import Element

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
D = DATA / 'mechanism'
J = lambda f: json.loads((D / f).read_text())
src = [dict(p, src='known') for p in J('MECH_PAIRS.json')] + [dict(p, src='new') for p in J('MECH_PAIRS_R2.json')] + \
      [dict(p, src='design') for f in ('DESIGN_MECH.json', 'DESIGN2_MECH.json') for p in J(f)]


def mech_class(host, sub, anion):
    g = Element(sub).group
    if host == 'Ta' and anion == 'Te': return 'gap'
    if g in (1, 2, 13, 14, 15): return 'ionic'
    if g >= 9: return 'filled'
    return 'local'


COL = {'ionic': RUST, 'filled': GOLD, 'local': TEAL, 'gap': NAVY}
MK = {'ionic': 's', 'filled': 'D', 'local': 'o', 'gap': '*'}
rows = []
for p in src:
    if 'error' in p or not 3 <= Element(p['host']).group <= 6: continue
    anion = [x for x in ('Te', 'Se', 'S') if x in p['parent'].split('__')[0]][0]
    rows.append(dict(p, cls=mech_class(p['host'], p['sub'], anion)))

style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), gridspec_kw=dict(wspace=0.32, left=0.1, right=0.985, top=0.84, bottom=0.16))
lim = (-2.4, 2.4)
for ax, ref in zip(axes, ('X', 'M')):
    ax.plot(lim, lim, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axhline(0, color=LIGHT, lw=0.6, zorder=0); ax.axvline(0, color=LIGHT, lw=0.6, zorder=0)
    for r in rows:
        x, y, big = -r[f'EF_vs_core_{ref}_eV'], r['dPhi'], r['cls'] == 'gap'
        if r['src'] == 'known':
            ax.scatter(x, y, s=40 if big else 20, marker=MK[r['cls']], color=COL[r['cls']], lw=0, zorder=3)
        else:
            ax.scatter(x, y, s=46 if big else 22, marker=MK[r['cls']], facecolors='white', edgecolors=COL[r['cls']], lw=0.9, zorder=4)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect('equal', adjustable='box')
    sub = r'\mathrm{X}' if ref == 'X' else r'\mathrm{M}'
    ax.set_xlabel(rf'$-\Delta(E_{{\mathrm{{F}}}}-V_{{{sub}}})$ (eV)'); ax.set_ylabel(r'$\Delta\Phi$ (eV)')
    x = np.array([-r[f'EF_vs_core_{ref}_eV'] for r in rows]); y = np.array([r['dPhi'] for r in rows])
    ok = np.isfinite(x); x, y = x[ok], y[ok]  # no unsubstituted host metal left (e.g. W2Te2) for the M reference
    rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
    print(ref, 'n', len(x), 'spearman', round(rho, 3), 'median |level term|', round(float(np.median(np.abs(y - x))), 3))
handles = [plt.Line2D([], [], ls='none', marker=MK[k], color=COL[k], ms=5 if k != 'gap' else 7) for k in ('ionic', 'filled', 'local', 'gap')]
handles += [plt.Line2D([], [], ls='none', marker='o', mfc=INK, mec=INK, ms=4.5), plt.Line2D([], [], ls='none', marker='o', mfc='white', mec=INK, mew=0.9, ms=4.5)]
fig.legend(handles, [r'$s/p$', r'filled-$d$', r'shared-$d$', r'Ta$_2$Te$_2$ host', 'Defining set', 'Predicted'],
           loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=6, fontsize=6.5, columnspacing=0.9, handletextpad=0.25)
for ax, t in zip(axes, ('(a)', '(b)')):
    label(ax, t, x=-0.2, y=1.02)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_S10.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

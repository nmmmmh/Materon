"""Fig. 4: how a single M-site substitution moves the Fermi level of early-transition-metal hosts.
(a) total DOS of Ta2Te2 and Ti2Se2 (parent lattice) with the rigid-band Fermi levels for +1 and +2 electrons per cell;
(b) d-projected DOS of the substituent site: W in Ta2S2 and in Ta2Te2, Pd in Sc2Se2;
(c) nominal valence change dN against the effective doping dN_eff (rigid-band inversion, chalcogen-core reference);
(d) DFT dPhi against dN_eff, with rigid-band curves of Ta2Te2 and Ti2Se2.
Data: data/mechanism. Filled markers: contrast set and round 1 (used to identify the mechanism classes);
open markers: round 2 (computed after the classes were written into this script); open stars: locked Ta2Te2 designs.
All static24 at the parent in-plane lattice."""
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
dos = json.loads((D / 'DOS_FIG.json').read_text())
pairs = [dict(p, src='known') for p in json.loads((D / 'MECH_PAIRS.json').read_text())] +         [dict(p, src='new') for p in json.loads((D / 'MECH_PAIRS_R2.json').read_text())]
DESIGN_DN = {'ReTaTe2': 2, 'OsTaTe2': 3, 'W2Te2': 2, 'WReTe2': 3, 'NbTaTe2': 0, 'HfTaTe2': -1, 'MoTaTe2': 1, 'Ta2TeI': 1}
designs = [dict(p, src='design', dN_nominal=DESIGN_DN[p['candidate'].split('__')[0]])
           for f in ('DESIGN_MECH.json', 'DESIGN2_MECH.json')  # Ta2TeI: Janus, face-mean dPhi
           for p in json.loads((D / f).read_text())]


def chem(s):
    import re
    return re.sub(r'(\d+)', r'$_{\1}$', s)


def mech_class(host, sub, anion):
    g = Element(sub).group
    if host == 'Ta' and anion == 'Te': return 'gap'
    if g in (1, 2, 13, 14, 15): return 'ionic'
    if g >= 9: return 'filled'
    return 'local'


COL = {'ionic': RUST, 'filled': GOLD, 'local': TEAL, 'gap': NAVY}
MK = {'ionic': 's', 'filled': 'D', 'local': 'o', 'gap': '*'}
rows = []
for p in pairs:
    if 'error' in p: continue
    host, sub = p['host'], p['sub']
    if not 3 <= Element(host).group <= 6: continue
    anion = [x for x in ('Te', 'Se', 'S') if x in p['parent'].split('__')[0]][0]
    g = lambda e: Element(e).group if Element(e).group <= 12 else Element(e).group - 10
    rows.append(dict(name=p['candidate'].split('__')[0], host=host, sub=sub, dN=g(sub) - g(host), neff=p['dN_eff_X'],
                     dphi=p['dPhi'], cls=mech_class(host, sub, anion), src=p['src']))
for p in designs:
    rows.append(dict(name=p['candidate'].split('__')[0], host='Ta', sub=p['sub'], dN=p['dN_nominal'], neff=p['dN_eff_X'],
                     dphi=p['dPhi'], cls='gap', src='design'))


def mark(ax, r):
    big = r['cls'] == 'gap'
    if r['src'] == 'known':
        ax.scatter(r['x'], r['y'], s=40 if big else 20, marker=MK[r['cls']], color=COL[r['cls']], lw=0, zorder=3)
    else:
        ax.scatter(r['x'], r['y'], s=46 if big else 22, marker=MK[r['cls']], facecolors='white', edgecolors=COL[r['cls']],
                   lw=0.9, zorder=4)


def smooth(e, y, sigma=0.06):
    '''Gaussian broadening for display on a 0.01 eV grid (DOSCAR spacing ~0.1 eV).'''
    g = np.arange(e.min(), e.max(), 0.01); yi = np.interp(g, e, y)
    k = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + 1e-9, 0.01) / sigma) ** 2); k /= k.sum()
    return g, np.convolve(yi, k, mode='same')


def rigid(lab, dn):
    r = dos[lab]; e = np.array(r['E_minus_EF']); n = np.array(r['idos'])
    return np.interp(r['n_at_EF'] + np.asarray(dn), n, e)


style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.28, left=0.08, right=0.985, top=0.96, bottom=0.085)
a, b, c, d = (fig.add_subplot(gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))

# (a) total DOS
for lab, col, fill in (('Ti2Se2', GREY, False), ('Ta2Te2', NAVY, True)):
    r = dos[lab]; e, t = smooth(np.array(r['E_minus_EF']), np.array(r['total'])); m = (e > -2.0) & (e < 3.0)
    a.plot(e[m], t[m], color=col, lw=1.0, label=chem(lab))
    if fill: a.fill_between(e[m], 0, t[m], color=col, alpha=0.12, lw=0)
a.axvline(0, color=INK, lw=0.6, ls=(0, (2, 2)))
for dn in (1, 2):
    x = float(rigid('Ta2Te2', dn)); a.plot([x], [8.9], marker='v', color=NAVY, ms=4)
    a.text(x, 9.3, f'+{dn}', ha='center', fontsize=7, color=NAVY)
a.axvspan(0.7, 2.3, color=LIGHT, alpha=0.4, lw=0)
a.set_xlim(-2, 3); a.set_ylim(0, 10)
a.set_xlabel(r'$E-E_{\mathrm{F}}$ (eV)'); a.set_ylabel('DOS (states eV$^{-1}$ cell$^{-1}$)')
a.legend(loc='upper left', fontsize=7)

# (b) substituent d-PDOS (stacked)
for k, (lab, col) in enumerate((('WTaS2', TEAL), ('WTaTe2', NAVY), ('PdScSe2', GOLD))):
    r = dos[lab]; e, s = smooth(np.array(r['E_minus_EF']), np.array(r['site_d'])); m = (e > -5) & (e < 2.5)
    s = s / s[m].max() * 0.85; off = 1.25 * (2 - k)      # each trace scaled to its own maximum
    b.fill_between(e[m], off, off + s[m], color=col, alpha=0.25, lw=0)
    b.plot(e[m], off + s[m], color=col, lw=0.9)
    b.text(-4.85, off + 0.93, f"{r['site']}@{chem({'WTaS2': 'Ta2S2', 'WTaTe2': 'Ta2Te2', 'PdScSe2': 'Sc2Se2'}[lab])}",
           ha='left', va='bottom', fontsize=7, color=col)
b.axvline(0, color=INK, lw=0.6, ls=(0, (2, 2)))
b.set_xlim(-5, 2.5); b.set_ylim(-0.05, 3.75); b.set_yticks([])
b.spines['left'].set_visible(False)
b.set_xlabel(r'$E-E_{\mathrm{F}}$ (eV)'); b.set_ylabel(r'$D_{d}(E)$ (arb. units)')

# (c) nominal vs effective doping
lim = (-5.6, 7.6)
c.plot(lim, lim, color=GREY, lw=0.6, ls=(0, (2, 2)))
c.axhline(0, color=LIGHT, lw=0.6); c.axvline(0, color=LIGHT, lw=0.6)
for r in rows:
    mark(c, dict(r, x=r['dN'], y=r['neff']))
for r in rows:
    if r['src'] == 'known' and r['sub'] in ('Pd', 'Rh', 'Na', 'Ir', 'W', 'Mg') and (r['sub'] != 'W' or r['cls'] == 'gap') or (r['sub'] == 'Ru' and r['cls'] == 'local'):
        c.annotate(r['sub'], (r['dN'], r['neff']), xytext=(4, 2), textcoords='offset points', fontsize=6.5, color=COL[r['cls']])
c.set_xlim(*lim); c.set_ylim(-3.8, 2.0)
c.set_xlabel(r'$\Delta N$'); c.set_ylabel(r'$\Delta N_{\mathrm{eff}}$')

# (d) dPhi vs effective doping with rigid-band curves
x = np.linspace(-3.2, 2.2, 200)
for lab, col in (('Ta2Te2', NAVY), ('Ti2Se2', GREY)):
    d.plot(x, -rigid(lab, x), color=col, lw=0.9, ls='-' if lab == 'Ta2Te2' else '--')
d.axhline(0, color=LIGHT, lw=0.6); d.axvline(0, color=LIGHT, lw=0.6)
for r in rows:
    mark(d, dict(r, x=r['neff'], y=r['dphi']))
d.set_xlim(-3.2, 2.2); d.set_ylim(-2.4, 2.4)
d.set_xlabel(r'$\Delta N_{\mathrm{eff}}$'); d.set_ylabel(r'$\Delta\Phi$ (eV)')
handles = [plt.Line2D([], [], ls='none', marker=MK[k], color=COL[k], ms=5 if k != 'gap' else 7) for k in ('ionic', 'filled', 'local', 'gap')]
handles += [plt.Line2D([], [], ls='none', marker='o', mfc=INK, mec=INK, ms=4.5), plt.Line2D([], [], ls='none', marker='o', mfc='white', mec=INK, mew=0.9, ms=4.5)]
fig.legend(handles, [r'$s/p$', r'filled-$d$', r'shared-$d$', chem('Ta2Te2') + ' host', 'Defining set', 'Predicted'],
           loc='center', bbox_to_anchor=(0.53, 0.505), ncol=6, fontsize=7, columnspacing=1.3, handletextpad=0.25)
for ax, t in zip((a, b, c, d), ('(a)', '(b)', '(c)', '(d)')):
    label(ax, t, x=-0.16, y=1.02)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_3_JPCL.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print(len(rows), 'early-host pairs;', {k: sum(r['cls'] == k for r in rows) for k in COL})

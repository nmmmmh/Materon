"""JPCL Fig. 4: mechanism-guided design of MoTaTe2 as an n-type contact to monolayer MoS2.
Combines panels (c),(d) of the PR Applied Fig. 5 and panels (b),(c) of Fig. 6 with unchanged data and styling:
(a) Ta2Te2 substitutions, DFT change against nominal added electrons, rigid-band curve and locked FT2 predictions;
(b) work function against formation energy (in-plane relaxed where computed), MoS2 electron affinity dashed;
(c) plane-averaged potential of the Ta2Te2/MoS2 and MoTaTe2/MoS2 interfaces (lowest-energy registries) with the MoS2 CBM at K;
(d) MoS2-layer DOS of both interfaces. Interface data: data/interface.
Data: data/mechanism; inset image data/structures/structure_Ta2Te2_W.png."""
from pathlib import Path
import json, re, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, TEAL, GREY, INK, LIGHT, GOLD, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
D = DATA / 'mechanism'
J = lambda f: json.loads((D / f).read_text())
EA, IE = 4.253, 5.919
chem = lambda s: re.sub(r'(\d+)', r'$_{\1}$', s)

summ = J('DESIGN_SUMMARY.json')['rows']; lock = J('DESIGN_PREREGISTERED.json')
d2 = J('DESIGN2_SUMMARY.json')['rows']; lock2 = J('DESIGN2_PREREGISTERED.json')
dosd = J('DOS_DESIGN.json')


def smooth(e, y, sigma=0.08):
    g = np.arange(e.min(), e.max(), 0.01); yi = np.interp(g, e, y)
    k = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + 1e-9, 0.01) / sigma) ** 2); k /= k.sum()
    return g, np.convolve(yi, k, mode='same')


style(); plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.0, 5.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.0], hspace=0.34, wspace=0.3, left=0.08, right=0.985, top=0.96, bottom=0.07)
c, d, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
gl = gs[1, 0].subgridspec(2, 1, height_ratios=[0.58, 0.42], hspace=0.18)
a = fig.add_subplot(gl[0]); q = fig.add_subplot(gl[1], sharex=a)

# ---------------- (a) Ta2Te2 series (PR Applied Fig. 5c)
dos = J('DOS_FIG.json')['Ta2Te2']; e = np.array(dos['E_minus_EF']); n = np.array(dos['idos'])
x = np.linspace(-1.2, 5.3, 300)
c.plot(x, -(np.interp(dos['n_at_EF'] + x, n, e)), color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
c.axhline(0, color=LIGHT, lw=0.6); c.axhline(EA - 5.009, color=TEAL, lw=0.8, ls=(0, (4, 2)), zorder=1)
known = {'WTaTe2': (1, 3.951 - 5.009), 'IrTaTe2': (4, 3.820 - 5.009), 'PtTaTe2': (5, 3.921 - 5.009)}
for nm, (dn, dp) in known.items():
    c.scatter(dn, dp, s=18, color=GREY, lw=0, zorder=3)
DN = {'ReTaTe2': 2, 'OsTaTe2': 3, 'W2Te2': 2, 'WReTe2': 3, 'NbTaTe2': 0, 'HfTaTe2': -1}
off = {'ReTaTe2': -0.08, 'W2Te2': 0.08, 'OsTaTe2': -0.08, 'WReTe2': 0.08}
for r in lock['designs']:
    nm = r['name']; xx = DN[nm] + off.get(nm, 0)
    col = GREY if nm in ('NbTaTe2', 'HfTaTe2') else NAVY
    c.scatter(xx, summ[nm]['phi_clamped_eV'] - 5.009, s=46, marker='*', color=col, lw=0, zorder=4)
    c.scatter(xx, r['model_dPhi_eV']['FT2']['mean'], s=18, marker='o', facecolors='white', edgecolors=col, lw=0.8, zorder=4)
for r in lock2['designs']:
    nm = r['name']
    if nm in d2:
        xx = 1 + (0.12 if 'I' in nm else -0.12); faces = np.array(d2[nm]['phi_faces_clamped_eV']) - 5.009
        if np.ptp(faces) > 0.1:
            c.plot([xx, xx], faces, color=TEAL, lw=0.8, zorder=4)
        c.scatter([xx] * len(set(faces.round(3))), sorted(set(faces.round(3))), s=46, marker='*', color=TEAL, lw=0, zorder=5)
        c.scatter(xx, r['model_dPhi_eV']['FT2']['mean'], s=18, facecolors='white', edgecolors=TEAL, lw=0.8, zorder=5)
c.set_xlim(-1.5, 5.5); c.set_ylim(-2.8, 1.3); c.set_xticks(range(-1, 6))
c.set_xlabel(r'$\Delta N$'); c.set_ylabel(r'$\Delta\Phi$ (eV)')
hc = [plt.Line2D([], [], ls='none', marker='*', color=NAVY, ms=7), plt.Line2D([], [], ls='none', marker='*', color=TEAL, ms=7),
      plt.Line2D([], [], ls='none', marker='o', mfc='white', mec=INK, ms=4.5), plt.Line2D([], [], ls='none', marker='o', color=GREY, ms=4.5),
      plt.Line2D([], [], color=GREY, ls=(0, (3, 2)), lw=0.9)]
c.legend(hc, ['Design round 1', 'Design round 2', 'Ensemble', 'Other substitutions', 'Rigid band'], loc='upper right', fontsize=6.3,
         handletextpad=0.3, ncol=2, columnspacing=0.8, frameon=True, framealpha=1, facecolor='white', edgecolor='none', borderaxespad=0)

# ---------------- (b) work function vs formation energy (PR Applied Fig. 5d)
ins = d.inset_axes([0.6, 0.6, 0.4, 0.4]); ins.imshow(plt.imread(DATA / 'structures' / 'structure_Ta2Te2_W.png'), interpolation='lanczos'); ins.axis('off')
d.fill_between([-0.25, 0.0], 2.2, EA, color=TEAL, alpha=0.10, lw=0)
d.axhline(EA, color=TEAL, lw=0.8, ls=(0, (4, 2))); d.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
pts = []
for nm, v in summ.items():
    phi = v.get('phi_relaxed_eV', v['phi_clamped_eV']); ef = v.get('Ef_relaxed_eV_atom', v['Ef_clamped_eV_atom'])
    kind = 'parent' if nm == 'Ta2Te2' else 'control' if nm in ('NbTaTe2', 'HfTaTe2') else 'design'
    pts.append((nm, ef, phi, kind))
pts.append(('WTaTe2', -0.028, 3.951, 'known'))
for nm, v in d2.items():
    ef = v.get('Ef_relaxed_eV_atom', v['Ef_clamped_eV_atom']); faces = v.get('phi_faces_relaxed_eV', v['phi_faces_clamped_eV'])
    if abs(faces[0] - faces[1]) > 0.1:
        d.plot([ef, ef], faces, color=TEAL, lw=0.8, zorder=3)
        pts.append((nm, ef, min(faces), 'design2')); d.scatter(ef, max(faces), marker='*', color=TEAL, s=70, lw=0, zorder=4)
    else:
        pts.append((nm, ef, v.get('phi_relaxed_eV', v['phi_clamped_eV']), 'design2'))
sty = {'parent': dict(marker='o', color=INK, s=24), 'control': dict(marker='*', color=GREY, s=60),
       'design': dict(marker='*', color=NAVY, s=60), 'known': dict(marker='o', color=GREY, s=20), 'design2': dict(marker='*', color=TEAL, s=70)}
lab_off = {'Ta2Te2': (-10, -12), 'NbTaTe2': (-14, 5), 'HfTaTe2': (4, 4), 'ReTaTe2': (7, -7), 'OsTaTe2': (7, -3), 'W2Te2': (-6, -10),
           'WReTe2': (-14, 6), 'WTaTe2': (-16, -11), 'MoTaTe2': (-14, -12), 'Ta2TeI': (6, -3)}
for nm, ef, phi, kind in pts:
    st = dict(sty[kind]); d.scatter(ef, phi, zorder=4, **({'lw': 0} | st if 'lw' not in st else st))
    d.annotate(chem(nm), (ef, phi), xytext=lab_off.get(nm, (4, 2)), textcoords='offset points', fontsize=6.3, color=INK)
d.set_xlim(-0.25, 0.45); d.set_ylim(2.2, 5.7)
d.set_xlabel(r'$E_{\mathrm{form}}$ (eV atom$^{-1}$)'); d.set_ylabel(r'$\Phi$ (eV)')

# ---------------- (c) plane-averaged potential of the two interfaces, aligned at E_F; bars mark the MoS2 CBM at K
I = DATA / 'interface'
res = json.loads((I / 'interface_results.json').read_text())
IF = {'Ta2Te2': ('Ta2Te2_MoS2_s1', GREY), 'MoTaTe2': ('MoTaTe2_MoS2_s0', TEAL)}
for nm, (job, col) in IF.items():
    r = json.loads((I / f'{job}_fig.json').read_text())
    z = np.array(r['z']); v = np.array(r['V_minus_EF']); zat = np.array(r['atoms_z'])
    z0 = zat.min(); m = (z > z0 - 9) & (z < zat.max() + 9)
    a.plot(z[m] - z0, v[m], color=col, lw=0.9 if col == GREY else 1.1)
    sp = r['species']; zS = np.array([zat[k] for k, e in enumerate(sp) if e == 'S'])
    if nm == 'MoTaTe2':  # light bands mark the metal and MoS2 layers (atomic planes of the MoTaTe2 cell)
        zTe = np.array([zat[k] for k, e in enumerate(sp) if e == 'Te'])
        for lo, hi in ((zTe.min(), zTe.max()), (zS.min(), zS.max())):
            a.axvspan(lo - z0, hi - z0, color=LIGHT, alpha=0.35, lw=0, zorder=0)
    cbm = res[nm]['Phi_B_n_K']
    a.plot([zS.min() - z0 - 0.6, zS.max() - z0 + 0.6], [cbm] * 2, color=col, lw=2.2, solid_capstyle='butt')
a.axhline(0, color=INK, lw=0.6, ls=(0, (2, 2)))
a.set_xlim(-6, 18); a.set_ylim(-2.5, 6.0)
a.set_ylabel(r'$\bar V - E_{\mathrm{F}}$ (eV)'); a.tick_params(labelbottom=False)

# ---------------- (d) plane-integrated charge-density difference of the two interfaces (cdd.json)
Q = json.loads((I / 'cdd.json').read_text())
for nm, (job, col) in IF.items():
    r = json.loads((I / f'{job}_fig.json').read_text()); zat = np.array(r['atoms_z']); sp = r['species']; z0 = zat.min()
    zq = np.array(Q[job]['z']) - z0; m = (zq > -6) & (zq < 18)
    q.plot(zq[m], np.array(Q[job]['dlam'])[m], color=col, lw=0.9 if col == GREY else 1.1)
    if nm == 'MoTaTe2':
        zS = [zat[k] for k, e in enumerate(sp) if e == 'S']; zTe = [zat[k] for k, e in enumerate(sp) if e == 'Te']
        for lo, hi in ((min(zTe), max(zTe)), (min(zS), max(zS))):
            q.axvspan(lo - z0, hi - z0, color=LIGHT, alpha=0.35, lw=0, zorder=0)
q.axhline(0, color=INK, lw=0.5, ls=(0, (2, 2))); q.set_ylim(-0.15, 0.15); q.set_yticks([-0.1, 0, 0.1])
q.set_xlabel(r'$z$ ($\mathrm{\AA}$)'); q.set_ylabel(r'$\Delta\lambda$ ($e\,\mathrm{\AA}^{-1}$)')

# ---------------- (d) MoS2-layer DOS of the two interfaces near E_F
for nm, (job, col) in IF.items():
    r = json.loads((I / f'{job}_fig.json').read_text())
    e, t = smooth(np.array(r['E']), np.clip(np.array(r['dos_MoS2']), 0, None), sigma=0.05)
    m = (e > -2.2) & (e < 2.2)
    if col == GREY: b.fill_between(e[m], 0, t[m], color=col, alpha=0.3, lw=0)
    b.plot(e[m], t[m], color=col, lw=0.9 if col == GREY else 1.1, label=chem(nm) + r'/MoS$_2$')
    b.axvline(res[nm]['Phi_B_n_K'], color=col, lw=0.8, ls=(0, (4, 2)))
b.axvline(0, color=INK, lw=0.6, ls=(0, (2, 2)))
b.set_xlim(-2.2, 2.2); b.set_ylim(0, 82)
b.set_xlabel(r'$E - E_{\mathrm{F}}$ (eV)'); b.set_ylabel(r'MoS$_2$ DOS (states eV$^{-1}$ cell$^{-1}$)')
b.legend(loc='upper left', fontsize=6.0, handlelength=0.9, handletextpad=0.4, labelspacing=0.3, frameon=False, borderaxespad=0.1)

for ax, t in zip((c, d, a, b), ('(a)', '(b)', '(c)', '(e)')): label(ax, t, x=-0.16, y=1.02)
label(q, '(d)', x=-0.16, y=1.04)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_4_JPCL.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print({k: round(res[k]['Phi_B_n_K'], 3) for k in IF}, round(res['summary']['S_K'], 3))

"""Fig. 6: the MoTaTe2 design as an n-type contact and its stability. All structures in-plane relaxed, PBE.
(a) relaxed MoTaTe2 in a 4 x 4 x 1 computational cell (24 A including vacuum), ray-traced with OVITO (render_structures.py; Mo green, Ta grey, Te gold);
(b) energy levels relative to vacuum: MoS2 band edges (same protocol) and the Fermi levels of Ta2Te2 and MoTaTe2;
(c) vacuum-aligned total DOS of Ta2Te2 and MoTaTe2 with the Mo d projection;
(d) formation energy per atom along Ta2Te2 - MoTaTe2 - Mo2Te2 (same prototype) with the tie line of the end members and
    the convex hull of the Mo-Ta-Te system along this composition line (competing monolayers and elemental bulk);
(e) harmonic phonon dispersion and (f) phonon DOS of MoTaTe2 (4x4x1 supercell), with Ta2Te2 (grey) for comparison.
Data: data/mechanism/DOS_DESIGN.json, STABILITY_ENERGETICS.json, PHONONS.json, PHONONS_Q.json;
image data/structures/structure_MoTaTe2_cell.png."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, TEAL, GREY, INK, LIGHT, GOLD, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
D = DATA / 'mechanism'
J = lambda f: json.loads((D / f).read_text())
EA, IE = 4.253, 5.919
dos = J('DOS_DESIGN.json'); st = J('STABILITY_ENERGETICS.json')
ph = J('PHONONS.json')


def smooth(e, y, sigma=0.08):
    g = np.arange(e.min(), e.max(), 0.01); yi = np.interp(g, e, y)
    k = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + 1e-9, 0.01) / sigma) ** 2); k /= k.sum()
    return g, np.convolve(yi, k, mode='same')


style(); plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.42, left=0.075, right=0.985, top=0.96, bottom=0.09)
s_ax = fig.add_subplot(gs[0, 0]); a = fig.add_subplot(gs[0, 1]); b = fig.add_subplot(gs[0, 2]); c = fig.add_subplot(gs[1, 0])
gph = gs[1, 1:].subgridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.06)
d = fig.add_subplot(gph[0]); f = fig.add_subplot(gph[1], sharey=d)
s_ax.imshow(plt.imread(DATA / 'structures' / 'structure_MoTaTe2_cell.png'), interpolation='lanczos'); s_ax.axis('off')   # OVITO render (OV_CELL=1)

# (a) energy levels
lo = -6.6
a.fill_between([0.15, 0.85], lo, -IE, color=GOLD, alpha=0.22, lw=0)
a.fill_between([0.15, 0.85], -EA, -3.3, color=NAVY, alpha=0.18, lw=0)
a.plot([0.15, 0.85], [-IE] * 2, color=GOLD, lw=1.0); a.plot([0.15, 0.85], [-EA] * 2, color=NAVY, lw=1.0)
for x0, nm, col in ((1.5, 'Ta2Te2', GREY), (2.5, 'MoTaTe2', TEAL)):
    ef = -dos[nm]['phi_eV']
    a.fill_between([x0 - 0.35, x0 + 0.35], lo, ef, color=col, alpha=0.35 if col == GREY else 0.3, lw=0)
    a.plot([x0 - 0.35, x0 + 0.35], [ef] * 2, color=col, lw=1.2)
a.axhline(-EA, color=NAVY, lw=0.7, ls=(0, (4, 2)), zorder=0)
ef_t = -dos['Ta2Te2']['phi_eV']
a.annotate('', xy=(1.5, -EA), xytext=(1.5, ef_t), arrowprops=dict(arrowstyle='<->', color=INK, lw=0.7, shrinkA=0, shrinkB=0))
a.text(1.56, (ef_t - EA) / 2, r'$\Phi_{\mathrm{B}}^{\mathrm{SM}}$', fontsize=7, va='center')
a.set_xlim(0, 3.0); a.set_ylim(lo, -3.3)
a.set_xticks([0.5, 1.5, 2.5]); a.set_xticklabels([r'MoS$_2$', r'Ta$_2$Te$_2$', r'MoTaTe$_2$'])
a.set_ylabel(r'$E - E_{\mathrm{vac}}$ (eV)')

# (b) vacuum-aligned DOS
for nm, col, fill in (('Ta2Te2', GREY, True), ('MoTaTe2', TEAL, False)):
    r = dos[nm]; e, t = smooth(np.array(r['E_minus_vac']), np.array(r['total']))
    m = (e > -7.5) & (e < -2.0)
    if fill: b.fill_between(e[m], 0, t[m], color=col, alpha=0.3, lw=0)
    b.plot(e[m], t[m], color=col, lw=0.9 if fill else 1.1, label=r'Ta$_2$Te$_2$' if fill else r'MoTaTe$_2$')
    b.axvline(r['EF_minus_vac'], color=col, lw=0.8, ls=(0, (2, 2)))
e, md = smooth(np.array(dos['MoTaTe2']['E_minus_vac']), np.array(dos['MoTaTe2']['d_proj']['Mo']))
m = (e > -7.5) & (e < -2.0)
b.fill_between(e[m], 0, md[m], color=TEAL, alpha=0.35, lw=0, label=r'Mo $d$')
b.axvline(-EA, color=NAVY, lw=0.7, ls=(0, (4, 2)))
b.set_xlim(-7.5, -2.0); b.set_ylim(0, None)
b.set_xlabel(r'$E - E_{\mathrm{vac}}$ (eV)'); b.set_ylabel('DOS (states eV$^{-1}$ cell$^{-1}$)')
b.legend(loc='upper left', fontsize=6.0, handlelength=0.9, handletextpad=0.4, labelspacing=0.3, frameon=False, borderaxespad=0.1)

# (d) formation energy along the Ta2Te2 - Mo2Te2 line
P = st['phases']; xs = [0, 0.5, 1]; ys = [P['Ta2Te2']['Ef_eV_atom'], P['MoTaTe2']['Ef_eV_atom'], P['Mo2Te2']['Ef_eV_atom']]
hull0 = st['Ta2Te2']['hull_eV_atom']
hull1 = 0.75 * P['MoTe2-2H']['Ef_eV_atom']           # 3/4 MoTe2 + 1/4 Mo at x = 1
c.plot([0, 1], [ys[0], ys[2]], color=GREY, lw=0.8, ls=(0, (3, 2)))
c.plot([0, 0.5, 1], [hull0, st['MoTaTe2']['hull_eV_atom'], hull1], color=INK, lw=0.8, ls=(0, (1, 1.5)))  # TaTe2 + metal up to x = 1/2
c.annotate('', xy=(0.5, (ys[0] + ys[2]) / 2), xytext=(0.5, ys[1]), arrowprops=dict(arrowstyle='<->', color=TEAL, lw=0.8, shrinkA=2, shrinkB=0))
c.annotate('', xy=(0.5, st['MoTaTe2']['hull_eV_atom']), xytext=(0.5, ys[1]), arrowprops=dict(arrowstyle='<->', color=INK, lw=0.6, shrinkA=2, shrinkB=0))
c.scatter([0, 1], [ys[0], ys[2]], s=26, color=GREY, zorder=3)
c.scatter([0.5], [ys[1]], s=70, marker='*', color=TEAL, lw=0, zorder=4)
c.text(0.53, (ys[1] + (ys[0] + ys[2]) / 2) / 2, r'$\Delta E_{\mathrm{mix}}$', fontsize=7, va='center', color=TEAL)
c.text(0.53, (ys[1] + st['MoTaTe2']['hull_eV_atom']) / 2, r'$\Delta E_{\mathrm{hull}}$', fontsize=7, va='center')
c.set_xticks(xs); c.set_xticklabels([r'Ta$_2$Te$_2$', r'MoTaTe$_2$', r'Mo$_2$Te$_2$'])
c.set_xlim(-0.12, 1.12); c.set_ylabel(r'$E_{\mathrm{form}}$ (eV atom$^{-1}$)')
# (e),(f) phonons: dispersion and DOS
for nm, col, lw in (('Ta2Te2', '#a9a9a9', 0.7), ('MoTaTe2', TEAL, 0.75)):
    r = ph[nm]; dist = r['distances']; scale = dist[-1][-1]
    for seg, fr in zip(dist, r['frequencies_THz']):
        fr = np.array(fr)
        # branches connected by eigenvector overlap; parent: per segment the three branches reaching lowest (acoustic + M-point instability)
        ks = np.argsort(fr.min(0))[:3] if nm == 'Ta2Te2' else range(fr.shape[1])
        for k in ks: d.plot(np.array(seg) / scale, fr[:, k], color=col, lw=lw, zorder=3 if nm == 'MoTaTe2' else 2)
    g = np.array(r['dos']); f.plot(g / g.max(), r['dos_freq_THz'], color=col, lw=lw)
    if nm == 'MoTaTe2': f.fill_betweenx(r['dos_freq_THz'], 0, g / g.max(), color=col, alpha=0.15, lw=0)
r = ph['MoTaTe2']; ticks = [sg[0] / r['distances'][-1][-1] for sg in r['distances']] + [1.0]
# exact frequencies at q = (k/5, k/5) from the slanted supercell, on M-Gamma
seg = r['distances'][2]; x0, x1, sc_ = seg[0], seg[-1], r['distances'][-1][-1]
for key, fr in J('PHONONS_Q.json').items():
    if float(key) > 0.5: continue    # (3/5,3/5), (4/5,4/5) are equivalent to (2/5,2/5), (1/5,1/5)
    x = (x0 + (1 - float(key) / 0.5) * (x1 - x0)) / sc_
    d.scatter([x], [min(fr)], s=16, facecolor='white', edgecolor=TEAL, lw=0.8, zorder=5)   # lowest branch only
for t in ticks[1:-1]: d.axvline(t, color=LIGHT, lw=0.6)
d.axhline(0, color=LIGHT, lw=0.6); f.axhline(0, color=LIGHT, lw=0.6)
d.set_xticks(ticks); d.set_xticklabels(r['labels']); d.set_xlim(0, 1)
d.set_ylim(min(-0.3, min(v['min_freq_path_THz'] for v in ph.values()) - 0.2), None)   # imaginary frequencies plotted as negative
d.set_ylabel('Frequency (THz)')
f.set_xlim(0, 1.15); f.set_xticks([]); f.set_xlabel('DOS'); plt.setp(f.get_yticklabels(), visible=False)
for ax, t, x in zip((a, b, c, d, f), ('(b)', '(c)', '(d)', '(e)', '(f)'), (-0.36, -0.3, -0.36, -0.12, -0.12)): label(ax, t, x=x, y=1.02)
fig.canvas.draw()   # (a): image centred in its cell, label aligned with the other top-row labels
cell = gs[0, 0].get_position(fig); pb0 = a.get_position()
s_ax.set_position([cell.x0 - 0.045, pb0.y0 - 0.055, cell.width + 0.07, pb0.height + 0.085]); s_ax.set_anchor('C')
pc, pb = c.get_position(), a.get_position()     # same column as (d), same height as (b)
fig.text(pc.x0 - 0.36 * pc.width, pb.y1 + 0.02 * pb.height, '(a)', fontsize=9, fontweight='bold', va='bottom', ha='left')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_6.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
print('Phi_B Ta2Te2', round(dos['Ta2Te2']['phi_eV'] - EA, 3), 'MoTaTe2', round(dos['MoTaTe2']['phi_eV'] - EA, 3),
      'Emix', round(st['E_mix_eV_atom'], 4), 'Ehull', round(st['MoTaTe2']['E_above_hull_eV_atom'], 4))

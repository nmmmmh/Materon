"""Fig. 5: closing the loop with locked predictions and DFT.
(a) mean absolute error of the predicted work-function change on each locked set, for the C2DB-trained ensemble (L400)
    and the ensembles fine-tuned on the DFT available before each set (FT1, FT2); open bars: zero-change predictor.
    The contrast-set values for L400 are post hoc (its DFT preceded L400 training of the long schedule but was never used).
(b) FT1 (round 1) and FT2 (round 2) predictions against DFT; open markers: round-2 candidates chosen for uncertainty.
(c) Ta2Te2 host: DFT change against the nominal number of released electrons, with the rigid-band curve of the parent
    and the FT2 predictions locked for the designs; inset: Ta2Te2 3 x 3 with one Ta replaced (OVITO render).
(d) work function against formation energy; dashed line: electron affinity of monolayer MoS2 (same protocol); shaded:
    n-type ohmic contact (Schottky-Mott) with negative formation energy. Lattice-relaxed values where computed.
Data: data/mechanism; inset image data/structures/structure_Ta2Te2_W.png."""
from pathlib import Path
import json, re, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, TEAL, GREY, INK, LIGHT, style, label

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
D = DATA / 'mechanism'
J = lambda f: json.loads((D / f).read_text())
EA = 4.253
chem = lambda s: re.sub(r'(\d+)', r'$_{\1}$', s)
C_L, C_F1, C_F2 = GREY, '#6f9fbe', NAVY

# ---------------- data
con = J('CONTRAST_ANALYSIS.json')['rows']; ev = J('EVAL_L400.json')['ensemble']['physical20_environment_gate']
y_con = np.array([r['dft_change_eV'] for r in con]); l_con = np.array(ev['contrast_delta_eV_post_hoc'])
r1 = J('ROUND1_ANALYSIS.json')['rows']; r2 = J('ROUND2_ANALYSIS.json')['rows']
mae = lambda p, y: float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))
sets = [('Contrast', {'L400': mae(l_con, y_con)}, float(np.mean(np.abs(y_con))))]
y1 = [r['dft_change_eV'] for r in r1]
sets.append(('Round 1', {t: mae([r['pred_eV'][t] for r in r1], y1) for t in ('L400', 'FT1')}, float(np.mean(np.abs(y1)))))
for sel, lab in (('confident', 'Round 2\nconfident'), ('uncertain', 'Round 2\nuncertain')):
    rr = [r for r in r2 if r['selection'] == sel]; yy = [r['dft_change_eV'] for r in rr]
    sets.append((lab, {t: mae([r['pred_eV'][t] for r in rr], yy) for t in ('L400', 'FT1', 'FT2')}, float(np.mean(np.abs(yy)))))

summ = J('DESIGN_SUMMARY.json')['rows']; lock = J('DESIGN_PREREGISTERED.json')
d2 = J('DESIGN2_SUMMARY.json')['rows']
lock2 = J('DESIGN2_PREREGISTERED.json')

style(); plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(7.0, 2.6))
gs = fig.add_gridspec(1, 2, wspace=0.3, left=0.08, right=0.985, top=0.92, bottom=0.17)
a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
_fx = plt.figure(); c, d = _fx.add_subplot(121), _fx.add_subplot(122)   # (c),(d) now in main-text Figure 4a,b; drawn off-figure

# ---------------- (a) error ladder
w = 0.2
for i, (name, m, zero) in enumerate(sets):
    a.bar(i - 1.5 * w, zero, w, color='white', ec=GREY, lw=0.7)
    for k, (t, col) in enumerate((('L400', C_L), ('FT1', C_F1), ('FT2', C_F2))):
        if t in m: a.bar(i + (k - 0.5) * w, m[t], w, color=col, lw=0)
a.set_xticks(range(len(sets))); a.set_xticklabels([s[0] for s in sets], fontsize=7)
a.set_ylabel('MAE of $\\Delta\\Phi$ (eV)'); a.set_ylim(0, 1.2)
h = [plt.Rectangle((0, 0), 1, 1, fc='white', ec=GREY, lw=0.7)] + [plt.Rectangle((0, 0), 1, 1, fc=c_, lw=0) for c_ in (C_L, C_F1, C_F2)]
a.legend(h, ['Zero', 'L400', 'FT1', 'FT2'], loc='upper left', ncol=4, fontsize=6.5, columnspacing=0.8, handlelength=1.0)

# ---------------- (b) parity
lim = (-2.6, 2.4)
b.plot(lim, lim, color=GREY, lw=0.6, ls=(0, (2, 2)))
b.axhline(0, color=LIGHT, lw=0.6); b.axvline(0, color=LIGHT, lw=0.6)
b.scatter([r['pred_eV']['FT1'] for r in r1], y1, s=18, color=C_F1, lw=0, zorder=3, label='FT1, round 1')
for sel, fc in (('confident', C_F2), ('uncertain', 'white')):
    rr = [r for r in r2 if r['selection'] == sel]
    b.scatter([r['pred_eV']['FT2'] for r in rr], [r['dft_change_eV'] for r in rr], s=20, facecolors=fc, edgecolors=C_F2, lw=0.9,
              zorder=4, label='FT2, round 2' + ('' if sel == 'confident' else ' (uncertain)'))
b.set_xlim(*lim); b.set_ylim(*lim); b.set_aspect('equal', adjustable='box')
b.set_xlabel(r'Predicted $\Delta\Phi$ (eV)'); b.set_ylabel(r'DFT $\Delta\Phi$ (eV)')
b.legend(loc='lower right', fontsize=6.3, handletextpad=0.2, borderaxespad=0.2)

# ---------------- (c) Ta2Te2 series
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
    c.scatter(xx, summ[nm]['phi_clamped_eV'] - 5.009, s=46, marker='*', color=NAVY, lw=0, zorder=4)
    c.scatter(xx, r['model_dPhi_eV']['FT2']['mean'], s=18, marker='o', facecolors='white', edgecolors=NAVY, lw=0.8, zorder=4)
for r in lock2['designs']:
    nm = r['name']
    if nm in d2:
        xx = 1 + (0.12 if 'I' in nm else -0.12); faces = np.array(d2[nm]['phi_faces_clamped_eV']) - 5.009
        if np.ptp(faces) > 0.1:  # Janus: both faces, joined
            c.plot([xx, xx], faces, color=TEAL, lw=0.8, zorder=4)
        c.scatter([xx] * len(set(faces.round(3))), sorted(set(faces.round(3))), s=46, marker='*', color=TEAL, lw=0, zorder=5)
        c.scatter(xx, r['model_dPhi_eV']['FT2']['mean'], s=18, facecolors='white', edgecolors=TEAL, lw=0.8, zorder=5)
c.set_xlim(-1.5, 5.5); c.set_ylim(-2.8, 0.5); c.set_xticks(range(-1, 6))
c.set_xlabel(r'$\Delta N$'); c.set_ylabel(r'$\Delta\Phi$ (eV)')
hc = [plt.Line2D([], [], ls='none', marker='*', color=NAVY, ms=7), plt.Line2D([], [], ls='none', marker='*', color=TEAL, ms=7),
      plt.Line2D([], [], ls='none', marker='o', mfc='white', mec=INK, ms=4.5), plt.Line2D([], [], ls='none', marker='o', color=GREY, ms=4.5),
      plt.Line2D([], [], color=GREY, ls=(0, (3, 2)), lw=0.9)]
c.legend(hc, ['Design round 1', 'Design round 2', 'FT2, locked', 'Other substitutions', 'Rigid band'], loc='upper right', fontsize=6.3,
         handletextpad=0.3, ncol=2, columnspacing=0.8, frameon=True, framealpha=1, facecolor='white', edgecolor='none', borderaxespad=0)
ins = d.inset_axes([0.6, 0.6, 0.4, 0.4]); ins.imshow(plt.imread(DATA / 'structures' / 'structure_Ta2Te2_W.png'), interpolation='lanczos'); ins.axis('off')  # OVITO render

# ---------------- (d) work function vs formation energy
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
    if abs(faces[0] - faces[1]) > 0.1:  # Janus: plot both faces, joined
        d.plot([ef, ef], faces, color=TEAL, lw=0.8, zorder=3)
        pts.append((nm, ef, min(faces), 'design2')); d.scatter(ef, max(faces), marker='*', color=TEAL, s=70, lw=0, zorder=4)
    else:
        pts.append((nm, ef, v.get('phi_relaxed_eV', v['phi_clamped_eV']), 'design2'))
sty = {'parent': dict(marker='o', color=INK, s=24), 'control': dict(marker='o', facecolors='white', edgecolors=GREY, s=22, lw=0.9),
       'design': dict(marker='*', color=NAVY, s=60), 'known': dict(marker='o', color=GREY, s=20), 'design2': dict(marker='*', color=TEAL, s=70)}
lab_off = {'Ta2Te2': (-10, -12), 'NbTaTe2': (-14, 5), 'HfTaTe2': (4, 4), 'ReTaTe2': (7, -7), 'OsTaTe2': (7, -3), 'W2Te2': (-6, -10),
           'WReTe2': (-14, 6), 'WTaTe2': (-16, -11), 'MoTaTe2': (-14, -12), 'Ta2TeI': (6, -3)}
for nm, ef, phi, kind in pts:
    st = dict(sty[kind]); d.scatter(ef, phi, zorder=4, **({'lw': 0} | st if 'lw' not in st else st))
    d.annotate(chem(nm), (ef, phi), xytext=lab_off.get(nm, (4, 2)), textcoords='offset points', fontsize=6.3, color=INK)
d.set_xlim(-0.25, 0.45); d.set_ylim(2.2, 5.7)
d.set_xlabel(r'$E_{\mathrm{form}}$ (eV atom$^{-1}$)'); d.set_ylabel(r'$\Phi$ (eV)')
for ax, t in zip((a, b), ('(a)', '(b)')): label(ax, t, x=-0.16, y=1.02)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_5.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
for s_ in sets: print(s_[0].replace('\n', ' '), {k: round(v, 3) for k, v in s_[1].items()}, 'zero', round(s_[2], 3))

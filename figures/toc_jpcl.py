"""JPCL TOC graphic (3.25 x 1.75 in): neuron-inspired channel gating -> band filling of Ta2Te2 -> MoTaTe2 below the MoS2
conduction-band edge. Vacuum-aligned DOS from data/mechanism/DOS_DESIGN.json; neuron artwork as in Fig. 1(a)."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from symbols import NAVY, TEAL, GREY, INK, PURPLE, RUST, GOLD, LIGHT, style
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
EA = 4.253
dos = json.loads((DATA / 'mechanism' / 'DOS_DESIGN.json').read_text())

style(); plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 7})
fig = plt.figure(figsize=(3.25, 1.75))

# neuron artwork (same crop as Fig. 1a, labels removed)
src = Image.open(DATA / 'original_artwork' / 'Manuscript_image1.png').convert('RGBA')
bg = Image.new('RGBA', src.size, 'white'); bg.alpha_composite(src); src = bg.convert('RGB')
k = 1.1725
p1 = src.crop((int(15 * k), int(250 * k), int(15 * k) + 515, int(800 * k)))
a = np.asarray(p1).astype(int); sat = a.max(axis=2) - a.min(axis=2); lo = a.min(axis=2)
w = (np.clip((lo - 214) / 14, 0, 1) * np.clip((14 - sat) / 10, 0, 1))[..., None]
p1 = Image.fromarray((a * (1 - w) + 255 * w).round().astype('uint8'))
from PIL import ImageDraw
d1 = ImageDraw.Draw(p1)
d1.rectangle((0, 0, 352, 114), fill='white')                         # serif labels
_b = np.asarray(p1).copy()                                            # leader lines: fill from the neighboring pixels
for x0, x1, y1 in ((337, 342, 162), (389, 394, 158)):
    for y in range(95, y1):
        l, r = _b[y, x0 - 2].astype(float), _b[y, x1 + 2].astype(float)
        for xx in range(x0, x1 + 1):
            _b[y, xx] = (l + (r - l) * (xx - x0 + 1) / (x1 - x0 + 2)).round()
p1 = Image.fromarray(_b); d1 = ImageDraw.Draw(p1)
d1.rectangle((274, 243, 298, 261), fill='white'); d1.rectangle((424, 243, 451, 261), fill='white')   # serif (-) and (+)
_a = np.asarray(p1).astype(float)
_ring = np.concatenate([_a[316:335, 354:357].reshape(-1, 3), _a[316:335, 374:377].reshape(-1, 3)])
d1.rectangle((356, 317, 375, 334), fill=tuple(int(v) for v in _ring.mean(axis=0)))   # serif Sigma
p1 = p1.crop((0, 105, p1.width, p1.height))
ai = fig.add_axes([0.03, 0.47, 0.33, 0.53]); ai.imshow(p1, interpolation='lanczos'); ai.axis('off')
ai.text(286, 252 - 105, '\u2212', fontsize=6, ha='center', va='center', color='#3f6e9a')
ai.text(437, 252 - 105, '+', fontsize=6, ha='center', va='center', color=RUST)
ai.text(365, 325 - 105, r'$\Sigma$', fontsize=5.5, ha='center', va='center', color=INK)
# atomic picture: substituent (teal) and its host-metal neighbours (soft navy glow: gated N_d channel)
aa = fig.add_axes([0.04, 0.0, 0.33, 0.53]); aa.set_xlim(-1.75, 1.75); aa.set_ylim(-1.75, 1.75); aa.set_aspect('equal'); aa.axis('off')
for i in (-1, 0, 1):
    for j in (-1, 0, 1):
        if i == j == 0:
            for rr, al in ((0.52, 0.08), (0.42, 0.14)):
                aa.add_patch(Circle((0, 0), rr, color=TEAL, alpha=al, lw=0, zorder=2))
            aa.add_patch(Circle((0, 0), 0.30, color=TEAL, zorder=3))
        elif abs(i) + abs(j) == 1:
            for rr, al in ((0.56, 0.06), (0.47, 0.10), (0.38, 0.16)):
                aa.add_patch(Circle((i, j), rr, color=NAVY, alpha=al, lw=0, zorder=2))
            aa.add_patch(Circle((i, j), 0.27, color='#7d8a96', zorder=3))
for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    aa.plot([0, 0.62 * i], [0, 0.62 * j], color=TEAL, lw=0.6, alpha=0.6, zorder=1)
aa.text(1.0, 0.0, r'$N_d$', ha='center', va='center', fontsize=5.5, color='white', zorder=4)
# vacuum-aligned DOS, energy on the vertical axis
ax = fig.add_axes([0.535, 0.12, 0.455, 0.85])


def smooth(e, y, sigma=0.08):
    g = np.arange(e.min(), e.max(), 0.01); yi = np.interp(g, e, y)
    kk = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + 1e-9, 0.01) / sigma) ** 2); kk /= kk.sum()
    return g, np.convolve(yi, kk, mode='same')


ef0, ef1 = dos['Ta2Te2']['EF_minus_vac'], dos['MoTaTe2']['EF_minus_vac']
for nm, col in (('Ta2Te2', GREY), ('MoTaTe2', TEAL)):
    r = dos[nm]; e, t = smooth(np.array(r['E_minus_vac']), np.array(r['total']))
    m = (e > -6.2) & (e < -3.0)
    if nm == 'Ta2Te2':   # occupied parent states (grey) and the states filled by the released electron (navy, rigid band)
        mo = m & (e <= ef0); ax.fill_betweenx(e[mo], 0, t[mo], color=GREY, alpha=0.3, lw=0)
        mf = m & (e >= ef0) & (e <= ef1); ax.fill_betweenx(e[mf], 0, t[mf], color=NAVY, alpha=0.55, lw=0)
    ax.plot(t[m], e[m], color=col, lw=0.9 if nm == 'Ta2Te2' else 1.1)
    ax.axhline(r['EF_minus_vac'], color=col, lw=0.9, ls=(0, (2, 1.5)))
ax.add_patch(FancyArrowPatch((3.6, ef0), (3.6, ef1), shrinkA=0, shrinkB=0, arrowstyle='-|>', mutation_scale=7, color=INK, lw=1.0))
ax.text(8.6, ef0 - 0.08, r'Ta$_2$Te$_2$', va='top', ha='right', fontsize=6.5, color=GREY)
ax.text(8.6, ef1 + 0.13, r'MoTaTe$_2$', va='bottom', ha='right', fontsize=6.5, color=TEAL)
ax.text(3.85, -4.5, r'$\Delta\Phi = -1.02$ eV', ha='left', va='center', fontsize=6, color=INK)
ax.set_ylim(-6.2, -3.0); ax.set_xlim(0, 8.7)
ax.set_xticks([]); ax.set_xlabel('DOS', labelpad=1, fontsize=6.5)
ax.set_ylabel(r'$E - E_{\mathrm{vac}}$ (eV)', fontsize=6.5, labelpad=1); ax.tick_params(labelsize=6)
fig.add_artist(FancyArrowPatch((0.37, 0.265), (0.42, 0.265), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9, color=INK, lw=1.0))
for ext in ('png', 'pdf', 'tif'):
    fig.savefig(OUT / f'TOC_JPCL.{ext}', dpi=600)
print('EF Ta2Te2', round(ef0, 3), 'MoTaTe2', round(ef1, 3))

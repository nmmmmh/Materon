"""Fig. 1 built on the original submission Fig. 1: panels (a) and (b) reuse the original artwork (neuron and E/I feature
modulation) with the serif labels replaced in Arial and the spelling corrected; panels (c) and (d) add the environment
gate, prediction and interventions as vector graphics.
Original artwork: original_submission/Manuscript.docx, word/media/image1.tiff, stored losslessly as
data/original_artwork/Manuscript_image1.png (RGBA, pixel-identical)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in PDF output
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'font.size': 8})
INK, AL, S_, SUB = '#2b2b2b', '#9aa5b1', '#e0b04a', '#7a5ba6'
EXC, INH, MOD = '#3b8f46', '#2f56a6', '#ef8a2c'        # arrow colors of the original artwork

# --- original artwork (from the frozen original submission) -----------------------------------------------
src = Image.open(DATA / 'original_artwork' / 'Manuscript_image1.png').convert('RGBA')
bg = Image.new('RGBA', src.size, 'white'); bg.alpha_composite(src); src = bg.convert('RGB')
k = 1.1725
p1 = src.crop((int(15 * k), int(250 * k), int(15 * k) + 515, int(800 * k)))
p2 = src.crop((int(540 * k), int(275 * k), int(540 * k) + 396, int(785 * k)))
a_ = np.asarray(p1).astype(int)                                # flatten the grey panel gradient of the original box
sat = a_.max(axis=2) - a_.min(axis=2); lo = a_.min(axis=2)
w = (np.clip((lo - 214) / 14, 0, 1) * np.clip((14 - sat) / 10, 0, 1))[..., None]   # soft mask, keeps the glows
a_ = a_ * (1 - w) + 255 * w; p1 = Image.fromarray(a_.round().astype('uint8'))
d1 = ImageDraw.Draw(p1)
d1.rectangle((0, 18, 352, 114), fill='white')   # serif labels: misspelt and attached to the wrong leader lines
d1.rectangle((274, 243, 298, 261), fill='white'); d1.rectangle((424, 243, 451, 261), fill='white')   # serif (-) and (+)
_a = np.asarray(p1).astype(float)                                # serif Sigma: fill with the local soma color
_ring = np.concatenate([_a[316:335, 354:357].reshape(-1, 3), _a[316:335, 374:377].reshape(-1, 3)])
d1.rectangle((356, 317, 375, 334), fill=tuple(int(v) for v in _ring.mean(axis=0)))
d2 = ImageDraw.Draw(p2)
for box, fill in (((168, 58, 203, 94), 'white'),               # sigma
                  ((176, 192, 230, 224), 'white'),             # att
                  ((78, 278, 214, 306), 'white'),              # promo/inhib
                  ((56, 310, 84, 420), 'white'),               # 1 2 ... N
                  ((62, 488, 374, 580), '#f2f2f2')):           # Synergy Modulation
    d2.rectangle(box, fill=fill)

fig = plt.figure(figsize=(7.0, 2.45))
def img_ax(rect, im):
    a = fig.add_axes(rect); a.imshow(im, interpolation='lanczos'); a.axis('off'); return a
a1 = img_ax([0.0, 0.02, 0.215, 0.86], p1)
a1.text(350, 50, 'Excitatory', fontsize=7.5, ha='right', va='center')
a1.text(324, 97, 'Inhibitory', fontsize=7.5, ha='right', va='center')
a1.plot([329, 341], [97, 97], color=INK, lw=0.5); a1.plot([341, 341], [97, 125], color=INK, lw=0.5)
a1.text(286, 252, '\u2212', fontsize=7.5, ha='center', va='center', color=INH)
a1.text(437, 252, '+', fontsize=7.5, ha='center', va='center', color='#c8423b')
a1.text(365, 325, r'$\Sigma$', fontsize=7, ha='center', va='center', color=INK)
a2 = img_ax([0.225, 0.13, 0.19, 0.75], p2)
a2.text(191, 76, r'$\sigma$', fontsize=8, ha='center', va='center')
a2.text(203, 208, r'$u$', fontsize=9, ha='center', va='center')                       # node confidence u (Fig. S6)
a2.text(116, 299, r'$v^{+}$', fontsize=7.5, ha='center', va='center', color=EXC)        # excitatory / inhibitory candidates
a2.text(180, 299, r'$v^{-}$', fontsize=7.5, ha='center', va='center', color=INH)
for yy, s in ((322, '1'), (349, '2'), (406, r'$N$')):
    a2.text(70, yy, s, fontsize=7, ha='center', va='center')
a2.text(70, 378, '\u2026', fontsize=7, ha='center', va='center', rotation=90)   # vertical ellipsis (Arial has no U+22EE)
a2.text(218, 532, 'Synergy\nmodulation', fontsize=7.5, ha='center', va='center', linespacing=1.1)
lg = fig.add_axes([0.225, 0.0, 0.19, 0.12]); lg.axis('off'); lg.set_xlim(0, 3); lg.set_ylim(0, 1)
for n, (c, s) in enumerate(((EXC, 'Exc.'), (INH, 'Inh.'), (MOD, 'Mod.'))):
    lg.add_patch(FancyArrowPatch((n + 0.05, 0.5), (n + 0.35, 0.5), arrowstyle='-|>', mutation_scale=6, color=c, lw=1.2))
    lg.text(n + 0.42, 0.5, s, fontsize=7, va='center')

# --- vector panels (c) and (d) ----------------------------------------------------------------------------
ax = fig.add_axes([0.43, 0.0, 0.57, 1.0]); ax.set_xlim(0, 160); ax.set_ylim(0, 70); ax.axis('off'); ax.set_aspect('equal')
def arrow(p0, p1_, color=INK, lw=0.8, rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(p0, p1_, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))
ox, oy, d = 6, 30, 8.0
pos = []
for r in range(5):
    for c in range(6):
        x = ox + c * d + (r % 2) * d / 2; y = oy + r * d * 0.866
        if x > 52: continue
        pos.append((x, y, 'S' if (r + c) % 2 else 'Al'))
ci = (ox + 3 * d, oy + 2 * d * 0.866)
ax.add_patch(Circle(ci, 1.75 * d, fc='#eef3f6', ec=INK, lw=0.7, ls=(0, (3, 2)), zorder=0))
for x, y, el in pos:
    near = np.hypot(x - ci[0], y - ci[1])
    fc = AL if el == 'Al' else S_
    if abs(near - d) < 0.5 and x < ci[0] and y > ci[1]: fc = SUB          # one substituted neighbor
    if near < 0.5: fc = AL
    ax.add_patch(Circle((x, y), 2.6 if el == 'Al' else 2.2, fc=fc, ec='white', lw=0.5, zorder=2))
ax.add_patch(Circle(ci, 3.8, fc='none', ec=INK, lw=1.0, zorder=3))
ax.text(ci[0] + 3.2, ci[1] - 5.0, r'$i$', fontsize=8.5, zorder=4)
ax.text(ci[0] + 1.75 * d * 0.7, ci[1] + 1.75 * d * 0.78, r'$c_i$', fontsize=8.5)
sx, sy = 8, 9
for kk in range(20):
    ax.add_patch(Rectangle((sx + kk * 2.1, sy), 1.6, 5, fc=AL, ec='none'))
ax.text(sx + 20 * 2.1 + 1.5, sy + 2.5, r'$x_i$', va='center', fontsize=8.5)
arrow((ci[0], ci[1] - 1.75 * d - 1), (ci[0], sy + 6.5))
gx, gy = 70, 48
rng = np.random.default_rng(3); g = np.clip(rng.normal(0, 0.45, 20), -0.9, 0.9)
ax.plot([gx - 1, gx + 20 * 1.9], [gy, gy], color=INK, lw=0.6, ls='--')
for kk, v in enumerate(g):
    ax.add_patch(Rectangle((gx + kk * 1.9, gy if v > 0 else gy + v * 10), 1.4, abs(v) * 10, fc=EXC if v > 0 else INH, ec='none'))
ax.text(gx + 20 * 1.9 + 1.5, gy, r'$g_{ij}$', va='center', fontsize=8.5)
arrow((sx + 47, sy + 2.5), (gx - 2, gy - 8), rad=0.25)
arrow((ci[0] + 1.75 * d + 1, ci[1] + 2), (gx - 2, gy + 3), rad=-0.15)
nx, ny = 138, 48
nodes = [(nx, ny)] + [(nx + 6 * np.cos(a), ny + 6 * np.sin(a)) for a in np.radians([90, 162, 234, 306, 18])]
for a_ in range(len(nodes)):
    for b_ in range(a_ + 1, len(nodes)):
        if a_ == 0 or abs(a_ - b_) in (1, 4):
            ax.plot([nodes[a_][0], nodes[b_][0]], [nodes[a_][1], nodes[b_][1]], color=INK, lw=0.6)
for p in nodes: ax.add_patch(Circle(p, 1.6, fc=AL, ec='none'))
arrow((gx + 20 * 1.9 + 8, gy), (nx - 8.5, gy))
arrow((nx + 8.5, ny), (nx + 14, ny))
ax.text(nx + 15, ny, r'$F$', va='center', fontsize=9.5)
iy = 17
ax.add_patch(FancyBboxPatch((70, iy - 7), 84, 14, boxstyle='round,pad=0.6,rounding_size=2.5', fc='#f2f2f2', ec='none'))
ax.add_patch(Rectangle((75, iy - 4), 1.7, 7, fc=EXC, ec='none'))
ax.add_patch(Rectangle((75, iy - 4), 1.7, 3, fc='white', ec=INK, lw=0.4, ls=(0, (1, 1))))
ax.text(79.5, iy, r'$g_{ij}\leftarrow g^{\mathrm{P}}_{ij}$', va='center', fontsize=8.5)
for kk in range(3):
    ax.add_patch(Rectangle((116 + kk * 2.4, iy - 4), 1.7, 7, fc='#c9cfd6', ec='none'))
    ax.plot([115.6 + kk * 2.4, 118.1 + kk * 2.4], [iy - 4, iy + 3], color=INK, lw=0.7)
ax.text(124.5, iy, r'$g_{ij}\rightarrow 1$', va='center', fontsize=8.5)
arrow((nx, ny - 8), (nx - 8, iy + 8.5), rad=-0.2, color='#8a8a8a', lw=0.6)
for x, s in ((0.005, '(a)'), (0.225, '(b)'), (0.435, '(c)'), (0.73, '(d)')):
    fig.text(x, 0.93, s, fontsize=9, fontweight='bold', va='bottom')
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_1.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

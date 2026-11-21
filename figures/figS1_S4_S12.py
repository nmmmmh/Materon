"""Figs. S12 (functional tests), S4 (dimension gate versus largest branch) and S1 (computational structure of the original
E/I update), redrawn in the unified style. Values are the reported original results; no new evaluation."""
from pathlib import Path
import sys
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbols import style, plt, label, C_CGCNN, C_MEGNET, NAVY, INK, GREY
OUT = Path(__file__).resolve().parent / 'output'; OUT.mkdir(exist_ok=True)
style()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
EXC, INH, MOD = '#c25e3a', '#3f6e9a', '#d49a2e'          # pathway colors of Fig. 1(b)
PATH = ['Dim.\ngate', 'Exc.', 'Inh.', 'Syn.']

def bars(ax, vals, colors, fmt):
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=colors, lw=0)
    for xx, v in zip(x, vals):
        ax.text(xx, v + 0.015 * ax.get_ylim()[1], fmt.format(v), ha='center', va='bottom', fontsize=6.5)
    ax.tick_params(axis='x', length=0)

# ---------------------------------------------------------------- Fig. S12: functional tests
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.2), gridspec_kw={'width_ratios': [1.35, 1.35, 0.75, 0.75], 'wspace': 0.55})
a, b, c, d = axs
a.set_ylim(0, 45); bars(a, [39.28, 0.63, 0.57, 0.32], [NAVY, EXC, INH, MOD], '{:.2f}')
a.set_xticks(range(4), PATH); a.set_ylabel(r'$\Delta$MAE (cm$^{-1}$)'); label(a, '(a)', x=-0.42, y=1.04)
b.set_ylim(0, 27); bars(b, [23.5833, 0.5989, 0.4696, 0.7971], [NAVY, EXC, INH, MOD], '{:.2f}')
b.set_xticks(range(4), PATH); b.set_ylabel(r'Mean $|\Delta F|$ (meV atom$^{-1}$)'); label(b, '(b)', x=-0.42, y=1.04)
c.set_ylim(0, 2.0); bars(c, [1.781, 1.35], [NAVY, GREY], '{:.2f}')
c.set_xticks(range(2), ['Mean', 'Median']); c.set_ylabel('Top-5 / Random-5'); c.axhline(1, color=GREY, lw=0.6, ls='--')
label(c, '(c)', x=-0.75, y=1.04)
d.set_ylim(0, 18); bars(d, [16.16, 9.78], [NAVY, GREY], '{:.2f}')
d.set_xticks(range(2), ['Top-5', 'Bottom-5']); d.set_ylabel(r'$\Delta$MAE (cm$^{-1}$)'); label(d, '(d)', x=-0.75, y=1.04)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S12.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

# ---------------------------------------------------------------- Fig. S4: dimension gate versus largest branch
ratios = {('HOIP', 'CGCNN'): 25.7, ('HOIP', 'MEGNet'): 1.4, ('JDFT2D', 'CGCNN'): 280.5, ('JDFT2D', 'MEGNet'): 29.6}
fig, ax = plt.subplots(figsize=(3.4, 1.6))
for k, (bb, col, fill) in enumerate((('CGCNN', C_CGCNN, True), ('MEGNet', C_MEGNET, False))):
    for t, task in enumerate(('HOIP', 'JDFT2D')):
        v = ratios[(task, bb)]; y = t + (k - 0.5) * 0.3
        ax.plot([1, v], [y, y], color=col, lw=0.8, alpha=0.5, zorder=2)
        ax.scatter([v], [y], s=24, color=col if fill else 'white', edgecolor=col, lw=1.0, zorder=3, label=bb if t == 0 else None)
        ax.text(v * 1.3, y, f'{v:.1f}', va='center', fontsize=6.5, color=col)
ax.set_xscale('log'); ax.set_xlim(0.8, 1000)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])); ax.xaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1000']))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks([0, 1], ['HOIP', 'JDFT2D']); ax.set_ylim(1.6, -0.6); ax.tick_params(axis='y', length=0)
ax.set_xlabel('Dimension gate / largest branch response')
ax.legend(loc='lower right', fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S4.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

# ---------------------------------------------------------------- Fig. S1: computational structure of the original E/I update
fig, ax = plt.subplots(figsize=(5.6, 1.9)); ax.set_xlim(0, 80); ax.set_ylim(0, 27); ax.axis('off')
def box(x, y, w, h, txt, fc='#eef2f4'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec='#6b7780', lw=0.6))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=7.5, linespacing=1.35)
def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, color='#55606a', lw=0.7,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1))
R_ = fig.canvas.get_renderer()
def odot_line(cx, y, left, right, fs=7.5):
    """Math line containing the Hadamard product; the operator is drawn because Arial has no U+2299 glyph."""
    tl = ax.text(0, y, left, fontsize=fs, va='center'); tr = ax.text(0, y, right, fontsize=fs, va='center')
    inv = ax.transData.inverted()
    wl = inv.transform((tl.get_window_extent(R_).width, 0))[0] - inv.transform((0, 0))[0]
    wr = inv.transform((tr.get_window_extent(R_).width, 0))[0] - inv.transform((0, 0))[0]
    gap = 1.1; x0 = cx - (wl + wr + 2 * gap) / 2
    tl.set_x(x0); tr.set_x(x0 + wl + 2 * gap); xc = x0 + wl + gap
    ax.plot([xc], [y], marker='o', ms=4.2, mfc='white', mec=INK, mew=0.6, zorder=5)
    ax.plot([xc], [y], marker='o', ms=0.9, color=INK, zorder=6)
box(1, 16, 11, 7, 'Elemental\n' + r'attributes $x$')
box(17, 16, 15, 7, 'Dimension gate\n ')
odot_line(24.5, 18.2, r'$x_d = x$', r'$d$')
box(37, 16, 15, 7, 'Node confidence\n' + r'$u = a\,x_d$')
box(57, 19.2, 18, 5.3, 'Shared expert\n' + r'$v^{+},\ v^{-}$')
box(57, 12.2, 18, 5.3, 'Softmax\n' + r'$w^{+},\ w^{-}$')
odot_line(48.5, 3.0, r'$\Delta x = \tilde{x} - x = (d-1)$', r'$x + s\,(w^{+}v^{+} - w^{-}v^{-})$')
box(22, 0.8, 53, 8.2, r'$\tilde{x} = x_d + s\,(w^{+}v^{+} - w^{-}v^{-})$' + '\n ',
    fc='#f4efe6')
arrow((12.6, 19.5), (16.4, 19.5)); arrow((32.6, 19.5), (36.4, 19.5))
arrow((52.6, 20.5), (56.4, 21.8)); arrow((52.6, 18.5), (56.4, 14.9))
arrow((24.5, 15.4), (32, 9.6), rad=0.15); arrow((66, 11.6), (64, 9.6)); arrow((75.8, 21.8), (75.6, 8.0), rad=-0.6)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'Figure_S1.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

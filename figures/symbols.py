"""Canonical mathematical symbols for the 20 named elemental attributes (glossary: SI Table S13) and shared figure style.
Keys are the source field names; C2DB_ORDER is the channel order of the C2DB model (Table S19 index)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SYM = {
    'density': r'$\rho$',
    'atom_melting_point': r'$T_{\mathrm{m}}$',
    'atom_specific_heat_capacity': r'$c_{p}$',
    'atomic_boiling_point': r'$T_{\mathrm{b}}$',
    'atomic_fusion_heat': r'$\Delta H_{\mathrm{f}}$',
    'atomic_thermal_cond': r'$\kappa$',
    'atomic_ip': r'$E_{\mathrm{IP}}$',
    'atomic_ea': r'$E_{\mathrm{EA}}$',
    'atomic_hfomo': r'$\varepsilon_{\mathrm{H}}$',
    'atomic_lfumo': r'$\varepsilon_{\mathrm{L}}$',
    'X': r'$\chi$',
    'atomic_pol': r'$\alpha$',
    'atomic_nvalences': r'$N_{\mathrm{v}}$',
    'atomic_ndval': r'$N_{d}$',
    'atomic_npval': r'$N_{p}$',
    'atomic_nsval': r'$N_{s}$',
    'atomic_radius': r'$r_{\mathrm{a}}$',
    'atomic_covalent Radius': r'$r_{\mathrm{c}}$',
    'atomic_volume': r'$V_{\mathrm{a}}$',
    'atomic_weight': r'$m_{\mathrm{a}}$',
}
C2DB_ORDER = ['density', 'atom_melting_point', 'atom_specific_heat_capacity', 'atomic_boiling_point',
              'atomic_fusion_heat', 'atomic_thermal_cond', 'atomic_ip', 'atomic_ea', 'atomic_hfomo',
              'atomic_lfumo', 'X', 'atomic_pol', 'atomic_nvalences', 'atomic_ndval', 'atomic_npval',
              'atomic_nsval', 'atomic_radius', 'atomic_covalent Radius', 'atomic_volume', 'atomic_weight']
# grouped display order (size, electronic, valence, thermal/bulk)
GROUP_ORDER = ['atomic_radius', 'atomic_covalent Radius', 'atomic_volume', 'atomic_weight', 'density',
               'atomic_ip', 'atomic_ea', 'atomic_hfomo', 'atomic_lfumo', 'X', 'atomic_pol',
               'atomic_nsval', 'atomic_npval', 'atomic_ndval', 'atomic_nvalences',
               'atom_melting_point', 'atomic_boiling_point', 'atomic_fusion_heat',
               'atom_specific_heat_capacity', 'atomic_thermal_cond']
# palette taken from the original submission figures (and Fig. 6 for B/Ga)
C_CGCNN, C_MEGNET = '#17374b', '#2e8b74'                     # original network colors (Figs. S3, S5)
C_CONV, C_NOGATE, C_MATERON, C_PCA, C_CTRL = '#9a9a9a', '#7f9bb3', '#17374b', '#b64a56', '#b6a5d2'  # original ablation
NAVY, TEAL, RUST, GOLD, PURPLE = '#17374b', '#2e8b74', '#c25e3a', '#d49a2e', '#8a6fb7'
C_B, C_GA = '#c0532f', '#2e7899'                              # sampled from the DFT figure (Fig. 6)
GREY, INK, LIGHT, PALE = '#9a9a9a', '#222222', '#c9d6d2', '#eef2f1'
SEP = '#d9dddd'
# legacy names used by the scripts
OCHRE, SAGE = GOLD, PURPLE
def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('teal_seq', ['#f7faf9', '#bfe0d6', TEAL, NAVY])

def style():
    plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
                         'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial',
                         'font.size': 7.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.edgecolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK, 'text.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'legend.handlelength': 1.2, 'legend.handletextpad': 0.4})

def label(ax, txt, x=-0.16, y=1.02):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

def lollipop(ax, vals, y, color, marker='o', dy=0.0, s=16):
    """Horizontal stems from zero with end markers."""
    for v, yy in zip(vals, y):
        ax.plot([0, v], [yy + dy, yy + dy], color=color, lw=0.9, alpha=0.55, solid_capstyle='round', zorder=2)
    ax.scatter(vals, np.asarray(y) + dy, s=s, marker=marker, color=color, zorder=3, lw=0)

def paired_bars(ax, a, b, y, ca=C_CGCNN, cb=C_MEGNET, h=0.36):
    """Two thin horizontal bars per row, from zero."""
    y = np.asarray(y)
    ax.barh(y - h / 2, a, height=h, color=ca, lw=0, zorder=2)
    ax.barh(y + h / 2, b, height=h, color=cb, lw=0, zorder=2)

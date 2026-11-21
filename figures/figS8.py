"""Fig. S8 redrawn in the unified style from the retained case files (no inference).
Data: data/evidence/sources/original_2d/results/... (paths of the evidence package).
Rows: HOIP hoip_0275 [(EA)2PbCl4] and JDFT2D mb-jdft2d-541 (Nb3SBr7). Columns: first coordination shell,
structure, within-backbone node scores (quantile-normalized), matched-node forward response ratios."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from pymatgen.core import Structure
from matplotlib.colors import LinearSegmentedColormap
from symbols import style, plt, C_CGCNN, C_MEGNET, INK, GREY
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts in PDF output
import ballstick as bs
import raster3d as r3

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'; OUT = HERE / 'output'; OUT.mkdir(exist_ok=True)
EV = DATA / 'evidence/sources/original_2d/results'
style()
CASES = {'hoip': ('hoip_bandgap', 'hoip_0275', 'case_studies/hoip_bandgap/cif/hoip_0275.cif',
                  'case_studies/hoip_bandgap/node_tables/hoip_0275_node_explanations.csv'),
         'jdft': ('jdft2d_exfoliation', 'mb-jdft2d-541', 'case_studies/jdft2d_exfoliation/cif/mb-jdft2d-541.cif',
                  'case_studies/jdft2d_exfoliation/node_tables/mb-jdft2d-541_node_explanations.csv')}
ROLE_NAME = {'metal_center': 'Pb center', 'organic_inorganic_interface': 'Interface', 'organic_spacer': 'Organic spacer',
             'halogen_bridge': 'Bridging Cl', 'interior_transition_metal': 'Internal Nb', 'surface_anion': 'Surface anion'}
SCORE_CMAP = LinearSegmentedColormap.from_list('score', ['#e3e3e3', '#f0a479', '#a8361a'])

def ratios(dataset, sample):
    out = {}
    for bb in ('cgcnn', 'megnet'):
        d = pd.read_csv(EV / f'interpretability/{dataset}/{bb}/role_matched_forward/role_matched_forward_sample_effects.csv.gz')
        d = d[d.sample_id == sample]
        out[bb] = dict(zip(d.role, d.actual_to_matched_random_ratio))
    return out

def periodic_sites(st, shift=(0, 0, 0)):
    """Cell sites plus half-bond stubs to periodic images."""
    sites = [(s.specie.symbol, s.coords + np.asarray(shift)) for s in st]
    pairs, stubs = [], []
    for i, nbrs in enumerate(st.get_all_neighbors(3.6)):
        ei = st[i].specie.symbol
        for n in nbrs:
            ej = n.specie.symbol
            if n.nn_distance > 1.18 * (bs.COVALENT[ei] + bs.COVALENT[ej]): continue
            if tuple(np.round(n.image).astype(int)) == (0, 0, 0):
                if n.index > i: pairs.append((i, n.index))
            else:
                stubs.append((i, n.coords + np.asarray(shift)))
    return sites, pairs, stubs

def layered_view(st, inorganic=('Pb', 'Cl')):
    """Whole organic molecules in one lateral cell, with the inorganic layer shown at both ends of the stack."""
    f = st.frac_coords.copy()
    pbz = f[[i for i, s in enumerate(st) if s.specie.symbol == 'Pb'], 2].mean()
    f[:, 2] = (f[:, 2] - pbz + 0.06) % 1.0
    el = [s.specie.symbol for s in st]
    L = st.lattice.matrix
    cand = []
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for i in range(len(st)):
                    g = f[i] + (ia, ib, dz)
                    if not -0.12 <= g[2] <= 1.24: continue
                    if el[i] not in inorganic and not 0 <= g[2] < 1: continue      # spacer molecules once
                    cand.append((el[i], g @ L, i, g))
    sites = [(e, p) for e, p, _, _ in cand]
    pairs = bs.bonds(sites)
    # connected components of the organic part
    org = [k for k, (e, *_ ) in enumerate(cand) if e not in inorganic]
    parent = {k: k for k in org}
    def find(k):
        while parent[k] != k: parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for i, j in pairs:
        if i in parent and j in parent: parent[find(i)] = find(j)
    comps = {}
    for k in org: comps.setdefault(find(k), []).append(k)
    keep = set()
    for members in comps.values():
        c = np.mean([cand[k][3] for k in members], axis=0)
        if 0 <= c[0] < 1 and 0 <= c[1] < 1: keep.update(members)
    metals = [k for k, (e, _, _, g) in enumerate(cand) if e == inorganic[0] and 0 <= g[0] < 1 and 0 <= g[1] < 1 and 0 <= g[2] < 1.2]
    keep.update(metals)                                   # complete PbCl6 octahedra around the metals of one lateral cell
    keep.update(k for k, (e, p, _, _) in enumerate(cand) if e == inorganic[1]
                and min(np.linalg.norm(p - cand[m][1]) for m in metals) < 3.2)
    keep = sorted(keep); idx = {k: n for n, k in enumerate(keep)}
    sites = [sites[k] for k in keep]; orig = [cand[k][2] for k in keep]
    pairs = [(idx[i], idx[j]) for i, j in pairs if i in idx and j in idx]
    return sites, pairs, orig

def shell(st, center, elements, rmax):
    c = st[center]
    nb = [n for n in st.get_neighbors(c, rmax) if n.specie.symbol in elements]
    sites = [(c.specie.symbol, c.coords)] + [(n.specie.symbol, n.coords) for n in nb]
    pairs = [(0, k) for k in range(1, len(sites))]
    hull = ConvexHull(np.array([p for _, p in sites[1:]]))
    faces = [np.array([sites[1 + v][1] for v in simplex]) for simplex in hull.simplices]
    return sites, pairs, faces, [n.nn_distance for n in nb], [n.specie.symbol for n in nb]

def quantile(v):
    r = pd.Series(v).rank(method='average').to_numpy()
    return (r - 1) / max(len(v) - 1, 1)

def ratio_panel(ax, roles, rr, xlim):
    y = np.arange(len(roles))
    for k, (bb, c, fill) in enumerate((('cgcnn', C_CGCNN, True), ('megnet', C_MEGNET, False))):
        v = [rr[bb][r] for r in roles]; yy = y + (k - 0.5) * 0.28
        for vv, y0 in zip(v, yy):
            ax.plot([1, vv], [y0, y0], color=c, lw=0.8, alpha=0.5, zorder=2)
            ax.text(vv * (1.32 if vv >= 1 else 1 / 1.32), y0, f'{vv:.2f}', va='center', ha='left' if vv >= 1 else 'right',
                    fontsize=6.5, color=c)
        ax.scatter(v, yy, s=22, zorder=3, color=c if fill else 'white', edgecolor=c, lw=1.0,
                   label=('CGCNN', 'MEGNet')[k])
    ax.axvline(1, color=GREY, lw=0.6, ls='--', zorder=1)
    ax.set_xscale('log'); ax.set_xlim(*xlim)
    ax.set_yticks(y, [ROLE_NAME[r] for r in roles]); ax.set_ylim(len(roles) - 0.5, -0.6)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Response / matched random')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [t for t in ((0.1, 1, 10) if xlim[1] > 20 else (0.5, 1, 2, 5, 10)) if xlim[0] <= t <= xlim[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())

ATOM_SCALE, BOND_R = 1.0, 0.13

def render(ax, sites, pairs, view, colors=None, polys=(), px=70, atom_scale=ATOM_SCALE, bond_r=BOND_R, poly_alpha=0.35):
    """Per-pixel shaded rendering placed in data coordinates (Å) so that vector labels can be added on top."""
    sc = r3.Scene(*view)
    col = colors or [bs.COLOR[e] for e, _ in sites]
    for (e, p), c in zip(sites, col): sc.sphere(p, bs.RADIUS[e] * atom_scale, c)
    for i, j in pairs: sc.bond(sites[i][1], sites[j][1], bond_r, col[i], col[j])
    for verts, c in polys: sc.polyhedron(verts, c, poly_alpha)
    img, (x0, x1, y0, y1) = sc.render(px_per_A=px)
    ax.imshow(img, extent=(x0, x1, y0, y1), interpolation='lanczos', zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.axis('off')
    return np.array([sc.project(p) for _, p in sites])

def octahedra(sites, center, ligands, rmax, color):
    out = []
    for e, p in sites:
        if e != center: continue
        nb = [q for f, q in sites if f in ligands and np.linalg.norm(q - p) < rmax]
        if len(nb) == 6: out.append((nb, color))
    return out

def element_key(fig, rect, elements):
    key = fig.add_axes(rect); key.axis('off'); key.set_xlim(0, 1.9 * len(elements)); key.set_ylim(0, 1); key.set_aspect('equal')
    for n, e in enumerate(elements):
        sc = r3.Scene((0, 0, 1), (0, 1, 0)); sc.sphere((0, 0, 0), 0.5, bs.COLOR[e])
        img, _ = sc.render(px_per_A=80, margin=0.05)
        key.imshow(img, extent=(0.12 + n * 1.9, 0.52 + n * 1.9, 0.3, 0.7), zorder=2)
        key.text(0.62 + n * 1.9, 0.5, e, va='center', fontsize=6.5)

def title(ax, s, c):
    ax.text(0.5, 1.0, s, transform=ax.transAxes, ha='center', va='bottom', fontsize=7.5, color=c)

fig = plt.figure(figsize=(7.0, 5.0))
PB_GREY, NB_TEAL = '#8d93a3', '#79c2ab'

def side_label(ax, s, c):
    ax.text(-0.01, 0.5, s, transform=ax.transAxes, ha='right', va='center', fontsize=7.5, color=c)

# ----------------------------------------------------------------- HOIP (EA)2PbCl4: layers run horizontally
ds, sid, cif, node = CASES['hoip']
st = Structure.from_file(EV / cif)
nt = pd.read_csv(EV / node); assert len(nt) == len(st) and (nt.element.values == [s.specie.symbol for s in st]).all()
pb = [i for i, s in enumerate(st) if s.specie.symbol == 'Pb'][0]
sites, pairs, faces, dist, _ = shell(st, pb, {'Cl'}, 3.2)
ax = fig.add_axes([0.0, 0.64, 0.12, 0.27])
P = render(ax, sites, pairs, ((0.9, -0.45, 0.35), (0, 0, 1)), polys=[([q for _, q in sites[1:]], PB_GREY)], px=160,
           atom_scale=0.9, bond_r=0.16)
ax.text(*P[0, :2], 'Pb', ha='center', va='center', fontsize=7, color='white', zorder=80)
ax.text(0.5, -0.05, f'Pb–Cl {min(dist):.3f}–{max(dist):.3f} Å', transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
fig.text(0.0, 0.965, '(a)', fontsize=9, fontweight='bold', va='bottom')
sites, pairs, orig = layered_view(st)
view = ((1.0, 0.22, 0.06), (0, 1, 0))                     # stacking axis c runs horizontally
rows = ((None, None, 0.80), ('CGCNN', nt.node_score_cgcnn, 0.645), ('MEGNet', nt.node_score_megnet, 0.49))
for k, (name, score, y) in enumerate(rows):
    ax = fig.add_axes([0.2, y, 0.47, 0.15])
    col = None if score is None else [SCORE_CMAP(quantile(score)[i]) for i in orig]
    polys = octahedra(sites, 'Pb', {'Cl'}, 3.2, PB_GREY) if score is None else ()
    P = render(ax, sites, pairs, view, colors=col, polys=polys, px=60)
    if name: side_label(ax, name, (C_CGCNN, C_MEGNET)[k - 1])
    if k == 0:
        x = P[:, 0]; inorg = [n for n, (e, _) in enumerate(sites) if e in ('Pb', 'Cl')]
        xs = sorted(P[inorg, 0])
        for txt, xx in (('Pb–Cl', np.mean(xs[:len(xs) // 2])), ('Spacer', np.median(x)), ('Pb–Cl', np.mean(xs[len(xs) // 2:]))):
            ax.text(xx, ax.get_ylim()[1], txt, ha='center', va='bottom', fontsize=6.5)
fig.text(0.155, 0.965, '(b)', fontsize=9, fontweight='bold', va='bottom')
fig.text(0.155, 0.8, '(c)', fontsize=9, fontweight='bold', va='bottom')
ax = fig.add_axes([0.81, 0.56, 0.18, 0.36])
rr = ratios(ds, sid)
ratio_panel(ax, ['metal_center', 'organic_inorganic_interface', 'organic_spacer', 'halogen_bridge'], rr, (0.05, 30))
ax.legend(loc='lower right', fontsize=6.5, handletextpad=0.2, borderaxespad=0.1, frameon=False)
fig.text(0.69, 0.965, '(d)', fontsize=9, fontweight='bold', va='bottom')
element_key(fig, [0.2, 0.455, 0.3, 0.03], ['Pb', 'Cl', 'N', 'S', 'C', 'H'])

# ----------------------------------------------------------------- JDFT2D Nb3SBr7: top view of the layer
ds, sid, cif, node = CASES['jdft']
st = Structure.from_file(EV / cif)
nt = pd.read_csv(EV / node); assert len(nt) == len(st) and (nt.element.values == [s.specie.symbol for s in st]).all()
sites, pairs, faces, dist, els = shell(st, 0, {'S', 'Br'}, 3.0)
ax = fig.add_axes([0.0, 0.14, 0.12, 0.25])
P = render(ax, sites, pairs, ((0.9, -0.5, 0.25), (0, 0, 1)), polys=[([q for _, q in sites[1:]], NB_TEAL)], px=160,
           atom_scale=0.9, bond_r=0.16)
ax.text(*P[0, :2], 'Nb', ha='center', va='center', fontsize=7, color=INK, zorder=80)
dS = [d for d, e in zip(dist, els) if e == 'S']; dB = [d for d, e in zip(dist, els) if e == 'Br']
ax.text(0.5, -0.05, f'Nb–S {dS[0]:.2f} Å\nNb–Br {min(dB):.2f}–{max(dB):.2f} Å', transform=ax.transAxes, ha='center', va='top',
        fontsize=6.5, linespacing=1.25)
fig.text(0.0, 0.37, '(e)', fontsize=9, fontweight='bold', va='bottom')
a, b = st.lattice.matrix[:2]
sup = [(s.specie.symbol, s.coords + ia * a + ib * b, i) for ia in range(2) for ib in range(2) for i, s in enumerate(st)]
ssites = [(e, p) for e, p, _ in sup]; orig = [i for _, _, i in sup]
spairs = bs.bonds(ssites)
view = ((0.0, -0.25, 1.0), (0, 1, 0))
for k, (name, score) in enumerate(((None, None), ('CGCNN', nt.node_score_cgcnn), ('MEGNet', nt.node_score_megnet))):
    ax = fig.add_axes([0.155 + 0.175 * k, 0.1, 0.165, 0.25])
    col = None if score is None else [SCORE_CMAP(quantile(score)[i]) for i in orig]
    polys = octahedra(ssites, 'Nb', {'S', 'Br'}, 3.0, NB_TEAL) if score is None else ()
    render(ax, ssites, spairs, view, colors=col, polys=polys, px=80)
    if name: title(ax, name, (C_CGCNN, C_MEGNET)[k - 1])
fig.text(0.155, 0.37, '(f)', fontsize=9, fontweight='bold', va='bottom')
fig.text(0.33, 0.37, '(g)', fontsize=9, fontweight='bold', va='bottom')
ax = fig.add_axes([0.81, 0.16, 0.18, 0.2])
rr = ratios(ds, sid)
ratio_panel(ax, ['interior_transition_metal', 'surface_anion'], rr, (0.2, 10))
fig.text(0.69, 0.37, '(h)', fontsize=9, fontweight='bold', va='bottom')
element_key(fig, [0.155, 0.03, 0.15, 0.03], ['Nb', 'S', 'Br'])

cax = fig.add_axes([0.43, 0.04, 0.17, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SCORE_CMAP), cax=cax, orientation='horizontal', ticks=[0, 1])
cb.ax.set_xticklabels(['Low', 'High'], fontsize=6.5); cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=0, pad=1)
cax.text(-0.04, 0.5, 'Node score', transform=cax.transAxes, ha='right', va='center', fontsize=6.5)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Figure_S8.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.03)

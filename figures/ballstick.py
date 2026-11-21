"""Minimal ball-and-stick renderer in matplotlib (orthographic projection, depth-sorted, shaded spheres).
Used for the structural panels of Fig. S3 and the re-typeset DFT figure, so that all structures share one style."""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

COLOR = {'H': '#e4e4e4', 'C': '#8a5a3c', 'N': '#7d8fd3', 'S': '#d9b84a', 'Cl': '#3f9f7a', 'Br': '#a8653f',
         'Pb': '#5d6170', 'Nb': '#79c2ab', 'Al': '#96aabd', 'B': '#e39a74', 'Ga': '#6f9fbe'}
RADIUS = {'H': 0.22, 'C': 0.33, 'N': 0.33, 'S': 0.42, 'Cl': 0.42, 'Br': 0.48, 'Pb': 0.62, 'Nb': 0.55,
          'Al': 0.44, 'B': 0.38, 'Ga': 0.46}
COVALENT = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'Pb': 1.46, 'Nb': 1.64,
            'Al': 1.21, 'B': 0.84, 'Ga': 1.22}

def view_matrix(direction, up=(0, 0, 1)):
    """Rows: screen x, screen y, depth (towards the viewer)."""
    w = np.asarray(direction, float); w /= np.linalg.norm(w)
    u = np.cross(up, w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.vstack([u, v, w])

def mix(c, other, f):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return tuple(a * (1 - f) + b * f)

def sphere(ax, xy, r, color, z, scale=1.0, edge=0.35):
    """Shaded sphere: darker rim, highlight towards the upper left."""
    ax.add_patch(Circle(xy, r * scale, fc=mix(color, 'black', 0.18), ec=mix(color, 'black', 0.45), lw=edge, zorder=z))
    n = 7
    for k in range(1, n + 1):
        f = k / n
        c = (xy[0] - 0.32 * r * scale * f, xy[1] + 0.32 * r * scale * f)
        ax.add_patch(Circle(c, r * scale * (1 - 0.82 * f), fc=mix(color, 'white', 0.62 * f ** 1.6), ec='none', zorder=z + 1e-4 * k))

def bonds(sites, cutoff_scale=1.18, lattice_images=None):
    """sites: list of (element, cart). Returns pairs (i, j, vector) for bonds inside the given set."""
    out = []
    for i, (ei, pi) in enumerate(sites):
        for j, (ej, pj) in enumerate(sites):
            if j <= i: continue
            d = np.linalg.norm(pj - pi)
            if 0.4 < d < cutoff_scale * (COVALENT[ei] + COVALENT[ej]):
                out.append((i, j))
    return out

def draw(ax, sites, view, pairs=None, colors=None, scale=1.0, bond_w=1.6, stubs=(), rings=(), ring_color='#222222',
         faces=(), face_color='#9aa0a6'):
    """sites: list of (element, cartesian xyz); colors: optional per-site override; stubs: (i, end_xyz) half bonds to images;
    rings: site indices to circle; faces: list of xyz polygons drawn translucent."""
    M = view_matrix(*view)                             # view = (direction, up)
    P = np.array([M @ p for _, p in sites])
    zmin, zmax = P[:, 2].min(), P[:, 2].max()
    zn = lambda z: 2 + 50 * (z - zmin) / (zmax - zmin + 1e-9)
    col = colors or [COLOR[e] for e, _ in sites]
    pairs = bonds(sites) if pairs is None else pairs
    for poly in faces:
        Q = np.array([M @ p for p in poly])
        ax.add_patch(Polygon(Q[:, :2], closed=True, fc=face_color, ec=mix(face_color, 'black', 0.3), lw=0.3, alpha=0.18,
                             zorder=zn(Q[:, 2].mean()) - 0.5))
    def half(a, b, c, z):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=mix(c, 'black', 0.35), lw=bond_w + 0.6, solid_capstyle='butt', zorder=z - 0.02)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=mix(c, 'white', 0.15), lw=bond_w, solid_capstyle='butt', zorder=z - 0.01)
    for i, j in pairs:
        m = (P[i] + P[j]) / 2
        half(P[i], m, col[i], zn((P[i][2] + m[2]) / 2) - 0.3)
        half(m, P[j], col[j], zn((P[j][2] + m[2]) / 2) - 0.3)
    for i, end in stubs:
        E = M @ end; m = (P[i] + E) / 2
        half(P[i], m, col[i], zn((P[i][2] + m[2]) / 2) - 0.3)
    for k, (e, _) in enumerate(sites):
        sphere(ax, P[k, :2], RADIUS[e], col[k], zn(P[k, 2]), scale)
    for k in rings:
        ax.add_patch(Circle(P[k, :2], RADIUS[sites[k][0]] * scale * 1.32, fc='none', ec=ring_color, lw=0.7, zorder=60))
    ax.set_aspect('equal'); ax.axis('off')
    return P

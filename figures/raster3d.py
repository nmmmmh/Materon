"""Small per-pixel renderer for crystal structures (orthographic, z-buffer, Phong shading, depth outlines,
translucent coordination polyhedra). Pure numpy/PIL; returns an RGBA image plus the projection so that labels can be
added as Arial vector text in matplotlib."""
import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb

LIGHT = np.array([-0.45, 0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0, 0, 1.0]); HALF /= np.linalg.norm(HALF)

def view_matrix(direction, up=(0, 0, 1)):
    w = np.asarray(direction, float); w /= np.linalg.norm(w)
    u = np.cross(up, w); u /= np.linalg.norm(u)
    return np.vstack([u, np.cross(w, u), w])

class Scene:
    def __init__(self, view, up=(0, 0, 1)):
        self.M = view_matrix(view, up)
        self.spheres, self.cyls, self.tris = [], [], []

    def sphere(self, p, r, color):
        self.spheres.append((self.M @ np.asarray(p, float), r, np.array(to_rgb(color))))

    def bond(self, p, q, r, c1, c2=None):
        """Cylinder split at the midpoint into the colors of its two atoms."""
        a, b = self.M @ np.asarray(p, float), self.M @ np.asarray(q, float)
        m = (a + b) / 2
        self.cyls.append((a, m, r, np.array(to_rgb(c1))))
        self.cyls.append((m, b, r, np.array(to_rgb(c2 or c1))))

    def polyhedron(self, verts, color, alpha=0.35):
        from scipy.spatial import ConvexHull
        V = np.array([self.M @ np.asarray(v, float) for v in verts])
        for s in ConvexHull(V).simplices:
            self.tris.append((V[s], np.array(to_rgb(color)), alpha))

    def project(self, p):
        return (self.M @ np.asarray(p, float))[:2]

    def render(self, px_per_A=90, ss=2, margin=0.6, outline=0.9, ambient=0.38, diffuse=0.62, spec=0.35, shininess=40):
        pts = [s[0] for s in self.spheres]
        rad = max([s[1] for s in self.spheres] + [0])
        P = np.array(pts)
        x0, y0 = P[:, :2].min(0) - rad - margin
        x1, y1 = P[:, :2].max(0) + rad + margin
        k = px_per_A * ss
        W, H = int((x1 - x0) * k) + 1, int((y1 - y0) * k) + 1
        X = x0 + (np.arange(W) + 0.5) / k
        Y = y1 - (np.arange(H) + 0.5) / k
        zbuf = np.full((H, W), -np.inf); col = np.zeros((H, W, 3)); nrm = np.zeros((H, W, 3)); base = np.zeros((H, W, 3))
        def box(cx, cy, r):
            i0 = max(int((cx - r - x0) * k), 0); i1 = min(int((cx + r - x0) * k) + 2, W)
            j0 = max(int((y1 - cy - r) * k), 0); j1 = min(int((y1 - cy + r) * k) + 2, H)
            return i0, i1, j0, j1
        for c, r, rgb in self.spheres:
            i0, i1, j0, j1 = box(c[0], c[1], r)
            if i0 >= i1 or j0 >= j1: continue
            dx = X[i0:i1][None, :] - c[0]; dy = Y[j0:j1][:, None] - c[1]
            d2 = dx ** 2 + dy ** 2; m = d2 < r * r
            dz = np.sqrt(np.clip(r * r - d2, 0, None)); z = c[2] + dz
            sub = zbuf[j0:j1, i0:i1]; upd = m & (z > sub)
            sub[upd] = z[upd]
            n = np.stack([dx + 0 * dy, dy + 0 * dx, dz], -1) / r
            nrm[j0:j1, i0:i1][upd] = n[upd]; base[j0:j1, i0:i1][upd] = rgb
        for a, b, r, rgb in self.cyls:
            D = b - a; L = np.linalg.norm(D)
            if L < 1e-6: continue
            D = D / L
            i0, i1, j0, j1 = box((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, L / 2 + r)
            if i0 >= i1 or j0 >= j1: continue
            px = X[i0:i1][None, :] + 0 * Y[j0:j1][:, None]; py = Y[j0:j1][:, None] + 0 * X[i0:i1][None, :]
            w = np.stack([px - a[0], py - a[1], -a[2] + 0 * px], -1)          # ray origin at z = 0, direction +z
            e = np.array([0, 0, 1.0])
            ep = e - e @ D * D
            wp = w - (w @ D)[..., None] * D
            A = ep @ ep
            if A < 1e-9: continue
            B = 2 * (wp @ ep); C = (wp * wp).sum(-1) - r * r
            disc = B * B - 4 * A * C; m = disc >= 0
            t = (-B + np.sqrt(np.clip(disc, 0, None))) / (2 * A)
             # hit point relative to a
            h = w + t[..., None] * e
            s = h @ D; m &= (s >= 0) & (s <= L)
            z = a[2] + h[..., 2]
            sub = zbuf[j0:j1, i0:i1]; upd = m & (z > sub)
            sub[upd] = z[upd]
            n = h - s[..., None] * D; n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
            nrm[j0:j1, i0:i1][upd] = n[upd]; base[j0:j1, i0:i1][upd] = rgb
        hit = np.isfinite(zbuf)
        lam = np.clip(nrm @ LIGHT, 0, None); sp = np.clip(nrm @ HALF, 0, None) ** shininess
        col = base * (ambient + diffuse * lam)[..., None] + spec * sp[..., None]
        # depth outlines: darken where depth jumps (silhouettes and occlusion edges)
        zf = np.where(hit, zbuf, zbuf[hit].min() - 5 if hit.any() else 0)
        g = np.zeros_like(zf)
        for sy, sx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            sh = np.roll(np.roll(zf, sy, 0), sx, 1); g = np.maximum(g, np.abs(zf - sh))
        edge = (g > 0.35)
        if outline:
            col[edge & hit] *= (1 - 0.55 * outline)
        alpha = hit.astype(float)
        # translucent polyhedra, sorted back to front and composited over the opaque scene
        for tri, rgb, al in sorted(self.tris, key=lambda t: t[0][:, 2].mean()):
            (ax_, ay_, az_), (bx_, by_, bz_), (cx_, cy_, cz_) = tri
            i0, i1, j0, j1 = box((ax_ + bx_ + cx_) / 3, (ay_ + by_ + cy_) / 3,
                                 max(np.hypot(*(tri[q, :2] - tri[:, :2].mean(0))) for q in range(3)))
            if i0 >= i1 or j0 >= j1: continue
            px = X[i0:i1][None, :] + 0 * Y[j0:j1][:, None]; py = Y[j0:j1][:, None] + 0 * X[i0:i1][None, :]
            den = (by_ - cy_) * (ax_ - cx_) + (cx_ - bx_) * (ay_ - cy_)
            if abs(den) < 1e-9: continue
            l1 = ((by_ - cy_) * (px - cx_) + (cx_ - bx_) * (py - cy_)) / den
            l2 = ((cy_ - ay_) * (px - cx_) + (ax_ - cx_) * (py - cy_)) / den
            l3 = 1 - l1 - l2
            m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            z = l1 * az_ + l2 * bz_ + l3 * cz_
            vis = m & (z > zbuf[j0:j1, i0:i1])
            nface = np.cross(tri[1] - tri[0], tri[2] - tri[0]); nface /= np.linalg.norm(nface) + 1e-12
            if nface[2] < 0: nface = -nface
            shade = rgb * (0.65 + 0.35 * max(nface @ LIGHT, 0))
            blk = col[j0:j1, i0:i1]; aa = alpha[j0:j1, i0:i1]
            cov = aa[vis][:, None]
            blk[vis] = np.where(cov > 0, blk[vis] * (1 - al) + shade * al, shade)
            aa[vis] = aa[vis] + al * (1 - aa[vis])
        rgba = np.concatenate([np.clip(col, 0, 1), alpha[..., None]], -1)
        img = Image.fromarray((rgba * 255).round().astype('uint8'), 'RGBA')
        if ss > 1: img = img.resize((W // ss, H // ss), Image.LANCZOS)
        return img, (x0, x1, y0, y1)

"""Plane-integrated charge-density difference of a metal/MoS2 heterostructure:
dlam(z) = lam_het - lam_metal - lam_MoS2 (e per Angstrom along z, integrated over the cell area),
with each layer computed alone in the heterostructure cell and geometry (jobs_cdd).
Q(z) = integral of dlam from the bottom of the cell; Q at the middle of the van der Waals gap is the charge that
moves across the interface (positive: electrons gained below the gap plane, i.e. on the metal side if the metal is below)."""
import os, sys, json
import numpy as np
R = os.environ.get('MATERON_INTERFACE_ROOT', '.')   # holds jobs/ and jobs_cdd/


def read_chgcar(path):
    with open(path) as f:
        lines = [next(f) for _ in range(7)]
        scale = float(lines[1]); latt = np.array([[float(x) for x in l.split()] for l in lines[2:5]]) * scale
        species = lines[5].split(); counts = [int(x) for x in lines[6].split()]
        mode = next(f)
        if mode.strip()[0] in 'Ss':
            mode = next(f)
        pos = np.array([[float(x) for x in next(f).split()[:3]] for _ in range(sum(counts))])
        line = next(f)
        while not line.strip():
            line = next(f)
        ng = [int(x) for x in line.split()]
        n = ng[0] * ng[1] * ng[2]
        vals = []
        while len(vals) < n:
            vals.extend(float(x) for x in next(f).split())
    rho = np.array(vals[:n]).reshape(ng[2], ng[1], ng[0])  # Fortran order: x fastest
    sp = [s for s, c in zip(species, counts) for _ in range(c)]
    return latt, sp, pos, rho


def lam(path):
    latt, sp, pos, rho = read_chgcar(path)
    c = latt[2, 2]
    # CHGCAR holds rho*V; plane integral per unit length = mean over the plane / c
    return latt, sp, pos, rho.mean(axis=(1, 2)) / c


out = {}
for name in sys.argv[1:]:
    latt, sp, pos, lh = lam(f'{R}/jobs/{name}/static/CHGCAR')
    _, _, _, lm = lam(f'{R}/jobs_cdd/{name}_metal/static/CHGCAR')
    _, _, _, ls = lam(f'{R}/jobs_cdd/{name}_MoS2/static/CHGCAR')
    c = latt[2, 2]; nz = len(lh); z = np.arange(nz) * c / nz; dz = c / nz
    d = lh - lm - ls
    Q = np.cumsum(d) * dz
    zat = pos[:, 2] * c
    zS = np.array([zat[i] for i, e in enumerate(sp) if e == 'S'])
    zTe = np.array([zat[i] for i, e in enumerate(sp) if e == 'Te'])
    if zTe.mean() < zS.mean():
        g0, g1 = zTe.max(), zS.min()
    else:
        g0, g1 = zS.max(), zTe.min()
    zmid = 0.5 * (g0 + g1)
    Qmid = float(np.interp(zmid, z, Q))
    out[name] = {'z': z.tolist(), 'dlam': d.tolist(), 'Q': Q.tolist(), 'gap': [float(g0), float(g1)],
                 'Q_mid_e_per_cell': Qmid, 'Q_mid_e_per_MoS2': Qmid / 7, 'area_A2': float(latt[0, 0] * latt[1, 1]),
                 'total_check': float(Q[-1]), 'metal_below': bool(zTe.mean() < zS.mean())}
    print(name, 'gap', round(g0, 3), round(g1, 3), 'Q_mid', round(Qmid, 4), 'e/cell', round(Qmid / 7, 5), 'e/MoS2',
          'total', round(float(Q[-1]), 5), 'metal_below', zTe.mean() < zS.mean(), flush=True)
json.dump(out, open(f'{R}/figdata/cdd.json', 'w'))

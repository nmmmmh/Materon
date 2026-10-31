"""Band edges of a semiconductor slab relative to the vacuum level of Eq. (S9):
EA = E_vac - CBM, IE = E_vac - VBM (monolayer MoS2 reference, Sec. S14.1).
Edges from EIGENVAL (both spins, occupation threshold 0.5); vacuum from the LOCPOT two-face mean.

Usage: python band_edges.py <static dir>
"""
import sys
from pathlib import Path

import numpy as np

from materon.dft.workfunction import locpot, vacuum_level


def band_edges(d):
    d = Path(d)
    L = (d / 'EIGENVAL').read_text().splitlines()
    nel, nk, nb = (int(v) for v in L[5].split())
    occ, unocc = [], []
    k = 7
    for _ in range(nk):
        k += 1
        for b in range(nb):
            e = [float(x) for x in L[k + b].split()[1:]]
            if len(e) == 4:      # spin polarized: e_up e_dn occ_up occ_dn
                for en, oc in ((e[0], e[2]), (e[1], e[3])):
                    (occ if oc > 0.5 else unocc).append(en)
            else:
                (occ if e[1] > 0.5 else unocc).append(e[0])
        k += nb + 1
    lat, prof = locpot(d / 'LOCPOT')
    vac = vacuum_level(prof)
    vbm, cbm = max(occ), min(unocc)
    return {'a_A': float(np.linalg.norm(lat[0])), 'gap_eV': cbm - vbm, 'IE_eV': vac - vbm, 'EA_eV': vac - cbm,
            'midgap_eV': vac - (vbm + cbm) / 2}


if __name__ == '__main__':
    r = band_edges(sys.argv[1])
    print(f"a = {r['a_A']:.4f} A  gap {r['gap_eV']:.3f} eV  IE {r['IE_eV']:.3f} eV  EA {r['EA_eV']:.3f} eV  "
          f"midgap {r['midgap_eV']:.3f} eV")

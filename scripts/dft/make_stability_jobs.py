"""VASP/phonopy inputs for the stability of MoTaTe2 (Sec. S14.2).

(1) Mo2Te2 in the Ta2Te2 prototype (mixing energy) and the competing monolayers 2H-MoTe2,
    1T-TaTe2 and 1H-TaTe2 (convex hull), all with in-plane lattice relaxation then static24;
    MAGMOM = <natoms>*0.1.
(2) Harmonic phonons: phonopy 4x4x1 supercell, 0.01 A displacements of the in-plane-relaxed
    unit cells (pass their static24/POSCAR); disp-000 is the undisplaced supercell whose
    residual forces are subtracted. Non-spin-polarized INCAR, 6x6x1 mesh.

Usage: python make_stability_jobs.py --out <dir> --potcar-root <dir> \
         --phonon MoTaTe2=<MoTaTe2__relaxlat/static24/POSCAR> --phonon Ta2Te2=<Ta2Te2__relaxlat/static24/POSCAR>
"""
import argparse
from pathlib import Path

import numpy as np

from materon.dft import phonons, vasp

C = vasp.CELL_C


def poscar(comment, lat, species, counts, frac):
    s = f'{comment}\n1.0\n' + ''.join(' '.join(f'{x:.10f}' for x in v) + '\n' for v in lat)
    return (s + ' '.join(species) + '\n' + ' '.join(map(str, counts)) + '\nDirect\n'
            + ''.join(' '.join(f'{x:.10f}' for x in q) + '\n' for q in frac))


def slab_job(root, name, text, species, natom, potcar_root):
    for stage, tail, mesh in (('relax16', vasp.RELAX_INPLANE, 16), ('static24', vasp.STATIC, 24)):
        incar = vasp.set_tag(vasp.SLAB_INCAR, 'MAGMOM', f'{natom}*0.1') + tail
        vasp.write_job(Path(root) / name / stage, text, species, incar, mesh, potcar_root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--potcar-root', required=True)
    p.add_argument('--phonon', action='append', default=[], help='NAME=unit-cell POSCAR')
    a = p.parse_args()
    root = Path(a.out)
    h, sq = 1.80 / C, 3.937
    names = ['Mo2Te2__relaxlat']
    slab_job(root, 'Mo2Te2__relaxlat',
             poscar('Mo2Te2 (Ta2Te2 prototype)', [[sq, 0, 0], [0, sq, 0], [0, 0, C]], ['Mo', 'Te'], [2, 2],
                    [[0, 0, .5], [.5, .5, .5], [0, .5, .5 - 0.0844], [.5, 0, .5 + 0.0844]]),
             ['Mo', 'Te'], 4, a.potcar_root)
    for name, lat_a, m, kind in (('MoTe2_2H', 3.55, 'Mo', 'H'), ('TaTe2_1T', 3.70, 'Ta', 'T'), ('TaTe2_1H', 3.65, 'Ta', 'H')):
        lat = [[lat_a, 0, 0], [-lat_a / 2, lat_a * np.sqrt(3) / 2, 0], [0, 0, C]]
        te = ([[1 / 3, 2 / 3, .5 + h], [1 / 3, 2 / 3, .5 - h]] if kind == 'H'
              else [[1 / 3, 2 / 3, .5 + h], [2 / 3, 1 / 3, .5 - h]])
        slab_job(root, name + '__relaxlat', poscar(name, lat, [m, 'Te'], [1, 2], [[0, 0, .5]] + te),
                 [m, 'Te'], 3, a.potcar_root)
        names.append(name + '__relaxlat')
    (root / 'SYSTEMS.txt').write_text('\n'.join(names) + '\n')

    incar = vasp.phonon_incar()
    dirs = []
    for item in a.phonon:
        name, src = item.split('=', 1)
        ph, cells = phonons.generate(src, root / f'PH_{name}')
        species = list(dict.fromkeys(cells[0].symbols))
        for k in range(len(cells)):
            d = root / f'PH_{name}' / f'disp-{k:03d}'
            (d / 'INCAR').write_text(incar)
            (d / 'KPOINTS').write_text(vasp.kpoints(phonons.MESH))
            (d / 'POTCAR').write_text(vasp.potcar(species, a.potcar_root))
            dirs.append(str(d))
        print(name, 'supercell atoms', len(cells[0]), 'displacements', len(cells) - 1, 'species', species)
    if dirs:
        (root / 'PH_LIST.txt').write_text('\n'.join(dirs) + '\n')


if __name__ == '__main__':
    main()

"""Finite-displacement inputs for exact frequencies of MoTaTe2 at q = (k/n, k/n) (Sec. S14.2).

The slanted supercell (1, -1) x (m, n - m), m = n // 2, of the 4-atom cell has determinant n and
makes q = (1/n, 1/n) commensurate (n = 5 is reported). INCAR and POTCAR are those of the 4x4x1
phonon runs; KPOINTS is the Gamma mesh of k-spacing 0.0672 A^-1 (density of the 6x6x1 mesh).

Usage: python make_phonon_q_jobs.py --ph-dir <PH_MoTaTe2> --out <dir> --n 5
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from materon.dft import phonons, vasp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ph-dir', required=True, help='4x4x1 phonon directory with phonopy_disp.yaml and disp-001')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, nargs='+', default=[5])
    a = p.parse_args()
    import phonopy
    from phonopy.interface.vasp import write_vasp

    ref = Path(a.ph_dir)
    base = phonopy.load(str(ref / 'phonopy_disp.yaml'), produce_fc=False, log_level=0)
    dirs = []
    for n in a.n:
        ph = phonopy.Phonopy(base.unitcell, supercell_matrix=phonons.slanted_supercell_matrix(n), primitive_matrix=None)
        sl = ph.supercell.cell[:2, :2] / base.unitcell.cell[0, 0]
        assert all(abs(((np.ones(2) / n) @ v + 1e-9) % 1) < 1e-6 for v in sl), sl
        ph.generate_displacements(distance=phonons.DISPLACEMENT_A)
        d = Path(a.out) / f'PHQ_MoTaTe2_{n}'
        d.mkdir(parents=True, exist_ok=True)
        ph.save(filename=str(d / 'phonopy_disp.yaml'))
        cells = [ph.supercell] + list(ph.supercells_with_displacements)
        for k, sc in enumerate(cells):
            dd = d / f'disp-{k:03d}'
            dd.mkdir(exist_ok=True)
            write_vasp(str(dd / 'POSCAR'), sc)
            (dd / 'INCAR').write_text(vasp.phonon_incar())
            (dd / 'KPOINTS').write_text(vasp.kpoints(phonons.kspacing_mesh(dd / 'POSCAR')))
            shutil.copy(ref / 'disp-001' / 'POTCAR', dd / 'POTCAR')
            dirs.append(str(dd))
        print(n, 'natoms', len(ph.supercell), 'ndisp', len(cells) - 1)
    (Path(a.out) / 'PHQ_LIST.txt').write_text('\n'.join(dirs) + '\n')


if __name__ == '__main__':
    main()

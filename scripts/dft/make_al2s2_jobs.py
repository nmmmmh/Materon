"""VASP inputs for the Al2S2 review calculations (Secs. S11.3, Table S32).

Jobs (relax16 = 16x16x1 relaxation, static24 = 24x24x1 static from the relaxed CONTCAR):
  A_<X>_fixed24   S->X (X = O, Se) at the unrelaxed parent geometry, static
  A_<X>_relax16   S->X ionic relaxation at the parent in-plane lattice
  A_<X>_static24  static on the relaxed geometry
  L_<C>_relax16   in-plane lattice + ionic relaxation (LATTICE_CONSTRAINTS = T T F), C in parent, B, Ga, O, Se
  L_<C>_static24  static on the lattice-relaxed geometry
  L_<C>_relax16b/static24b  damped rerun (IBRION = 1, POTIM = 0.1) used for B, Ga and O
  P_<s>/relax16, P_<s>/static24  parent under isotropic in-plane strain factor s (fixed lattice)
JOB.json names the relaxation whose CONTCAR seeds each static run.

Usage: python make_al2s2_jobs.py --out <dir> --potcar-root <PAW_PBE dir> [--strain 0.90,0.94,0.97,1.00,1.03]
"""
import argparse
from pathlib import Path

from materon.dft import vasp


def job(root, name, edit, incar, mesh, potcar_root, after=None, scale=1.0):
    text, order, idx = vasp.al2s2_poscar(name, edit, scale)
    vasp.write_job(Path(root) / name, text, order, vasp.BASE_INCAR + incar, mesh, potcar_root,
                   {'edit': edit, 'species_order': order, 'poscar_to_original_site': idx,
                    'start_from_contcar_of': after, 'mesh': mesh, 'in_plane_scale': scale})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--potcar-root', required=True)
    p.add_argument('--strain', default='0.90,0.94,0.97,1.00,1.03')
    a = p.parse_args()
    root, pot = Path(a.out), a.potcar_root
    for X in ('O', 'Se'):
        job(root, f'A_{X}_fixed24', X, vasp.STATIC, 24, pot)
        job(root, f'A_{X}_relax16', X, vasp.RELAX, 16, pot)
        job(root, f'A_{X}_static24', X, vasp.STATIC, 24, pot, after=f'A_{X}_relax16')
    for comp, edit in (('parent', None), ('B', 'B'), ('Ga', 'Ga'), ('O', 'O'), ('Se', 'Se')):
        job(root, f'L_{comp}_relax16', edit, vasp.RELAX_INPLANE_AL2S2, 16, pot)
        job(root, f'L_{comp}_static24', edit, vasp.STATIC, 24, pot, after=f'L_{comp}_relax16')
    for comp in ('B', 'Ga', 'O'):
        job(root, f'L_{comp}_relax16b', comp, vasp.RELAX_INPLANE_AL2S2_DAMPED, 16, pot)
        job(root, f'L_{comp}_static24b', comp, vasp.STATIC, 24, pot, after=f'L_{comp}_relax16b')
    for s in (float(x) for x in a.strain.split(',') if x):
        name = f'P_{s:.2f}'
        job(root, f'{name}/relax16', None, vasp.RELAX, 16, pot, scale=s)
        job(root, f'{name}/static24', None, vasp.STATIC, 24, pot, after=f'{name}/relax16', scale=s)
    print('jobs written to', root)


if __name__ == '__main__':
    main()

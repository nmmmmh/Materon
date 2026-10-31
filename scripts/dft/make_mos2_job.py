"""VASP inputs for the monolayer 2H-MoS2 reference (band edges, Sec. S14.1).

relax16: in-plane lattice relaxation on an 18x18x1 mesh (Gaussian smearing 0.01 eV);
static24: 24x24x1 static from the relaxed CONTCAR. POTCARs Mo_pv and S.

Usage: python make_mos2_job.py --out <dir> --potcar-root <dir>
"""
import argparse
from pathlib import Path

from materon.dft import vasp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--potcar-root', required=True)
    a = p.parse_args()
    root = Path(a.out)
    vasp.write_job(root / 'relax16', vasp.MOS2_POSCAR, ['Mo', 'S'], vasp.MOS2_INCAR + vasp.MOS2_RELAX, 18, a.potcar_root)
    vasp.write_job(root / 'static24', vasp.MOS2_POSCAR, ['Mo', 'S'], vasp.MOS2_INCAR + vasp.STATIC, 24, a.potcar_root)
    print('MoS2 jobs written to', root)


if __name__ == '__main__':
    main()

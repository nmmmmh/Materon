"""Site-resolved electronic descriptors of static VASP jobs, aligned to the vacuum level of Eq. (S9):
core electrostatic potential of every ion, PAW-sphere populations (s, p, d, total; LORBIT = 11),
d DOS at E_F, and s/p/d band centres of occupied states (E_F - 10 eV ... E_F) and of all states
(E_F - 10 eV ... E_F + 5 eV). PAW-sphere populations depend on RWIGS and are not Bader charges.

Usage: python site_analysis.py [--key-parent] <job dir> [<job dir> ...] > SITES.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from materon.dft.dos import band_centre, site_projected
from materon.dft.workfunction import contcar_species, locpot, vacuum_level


def outcar_sites(path):
    """E_F, core potentials, PAW-sphere populations and RWIGS from an OUTCAR."""
    t = Path(path).read_text()
    ef = float(re.findall(r'E-fermi :\s+(-?\d+\.\d+)', t)[-1])
    blk = t.split('average (electrostatic) potential at core')[-1].split('(the norm of the test charge')[1]
    blk = blk.split('\n', 1)[1].split('\n \n')[0]
    core = [float(v) for _, v in re.findall(r'(\d+)\s*(-\d+\.\d+)', blk)]
    pops = []
    for line in t.split('\n total charge\n')[-1].splitlines()[3:]:
        if line.startswith('---') or not line.strip():
            break
        pops.append([float(v) for v in line.split()[1:]])
    rw = re.findall(r'RWIGS\s*=\s*([\d.]+);\s*RWIGS\s*=\s*([\d.]+)', t)
    return ef, core, pops, [float(b) for _, b in rw]


def shifted(c, vac):
    return c - vac if c is not None else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--key-parent', action='store_true', help='key results by the parent directory name')
    p.add_argument('jobs', nargs='+')
    a = p.parse_args()
    res = {}
    for d in map(Path, a.jobs):
        _, prof = locpot(d / 'LOCPOT')
        vac = vacuum_level(prof)
        ef, core, pops, rw = outcar_sites(d / 'OUTCAR')
        elem = contcar_species(d / 'CONTCAR')
        dos = site_projected(d / 'DOSCAR', len(elem))
        sites = []
        for i, el in enumerate(elem):
            x = dos[i]
            rec = {'element': el, 'core_potential_minus_vacuum_eV': core[i] - vac,
                   'paw_s': pops[i][0], 'paw_p': pops[i][1], 'paw_d': pops[i][2] if len(pops[i]) > 3 else None,
                   'paw_tot': pops[i][-1], 'd_dos_at_EF': float(np.interp(ef, x['e'], x['d']))}
            for l in ('s', 'p', 'd'):
                rec[f'{l}_centre_occ_minus_vac_eV'] = shifted(band_centre(x['e'], x[l], ef - 10, ef), vac)
                rec[f'{l}_centre_all_minus_vac_eV'] = shifted(band_centre(x['e'], x[l], ef - 10, ef + 5), vac)
            sites.append(rec)
        key = d.parent.name if a.key_parent else d.name
        res[key] = {'vacuum_eV': vac, 'E_F_eV': ef, 'phi_eV': vac - ef, 'rwigs_A': rw, 'elements': elem, 'sites': sites}
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

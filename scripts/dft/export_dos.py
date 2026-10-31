"""Export DOS curves for the figure panels (spins summed, LORBIT = 11 projections).

  fermi   E - E_F, total DOS, integrated DOS, n(E_F) and optionally the d DOS of the first site of
          one element:   label=<static dir>[:<element>]   (mechanism figure, Fig. 4)
  vacuum  E - E_vac, total DOS, E_F - E_vac, Phi and the d DOS summed over all Ta and Mo sites:
          label=<static dir>   (Fig. 6; vacuum level from Eq. S9)

Usage: python export_dos.py fermi <out json> Ta2Te2=<dir> W_in_Ta2Te2=<dir>:W ...
       python export_dos.py vacuum <out json> Ta2Te2=<dir> MoTaTe2=<dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from materon.dft.dos import read_total_dos, site_d_dos
from materon.dft.workfunction import contcar_species, job


def fermi_record(spec):
    d, _, el = spec.partition(':')
    d = Path(d)
    e, tot, idos, ef = read_total_dos(d / 'DOSCAR')
    rec = {'E_minus_EF': (e - ef).tolist(), 'total': tot.tolist(), 'idos': idos.tolist(),
           'n_at_EF': float(np.interp(ef, e, idos))}
    if el:
        rec['site_d'] = site_d_dos(d / 'DOSCAR', contcar_species(d / 'CONTCAR').index(el)).tolist()
        rec['site'] = el
    return rec


def vacuum_record(d, metals=('Ta', 'Mo')):
    d = Path(d)
    j = job(d)
    ef = j['E_F_eV']
    vac = j['phi_eV'] + ef
    e, tot, _, _ = read_total_dos(d / 'DOSCAR')
    sym = contcar_species(d / 'CONTCAR')
    pd = {}
    for i, s in enumerate(sym):
        pd[s] = pd.get(s, 0) + site_d_dos(d / 'DOSCAR', i)
    return {'E_minus_vac': (e - vac).tolist(), 'total': tot.tolist(), 'EF_minus_vac': ef - vac, 'phi_eV': j['phi_eV'],
            'd_proj': {k: v.tolist() for k, v in pd.items() if k in metals}}


def main():
    mode, out, specs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    res = {}
    for arg in specs:
        label, spec = arg.split('=', 1)
        res[label] = fermi_record(spec) if mode == 'fermi' else vacuum_record(spec)
    out.write_text(json.dumps(res))
    print(list(res))


if __name__ == '__main__':
    main()

"""Figure data for one interface static: plane-averaged potential (relative to E_F) and
layer-resolved DOS (metal layer vs MoS2 layer), written to <name>_fig.json."""
import os, sys, json, re
import numpy as np
from pymatgen.io.vasp import Vasprun
ROOT = os.environ.get('MATERON_INTERFACE_ROOT', '.')   # holds jobs/ and figdata/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import planar, structure, efermi

for name in sys.argv[1:]:
    d = f'{ROOT}/jobs/{name}/static'
    sp, z, c = structure(d); ef = efermi(d)
    v = planar(d); zz = np.arange(len(v)) * c / len(v)
    vr = Vasprun(f'{d}/vasprun.xml', parse_potcar_file=False, parse_eigen=False)
    dos = vr.complete_dos
    zS = np.array([z[i] for i, e in enumerate(sp) if e == 'S'])
    mos2 = [i for i in range(len(sp)) if (sp[i] == 'S') or (sp[i] == 'Mo' and z[i] > zS.min() - 0.5)]
    metal = [i for i in range(len(sp)) if i not in mos2]
    E = dos.energies - ef
    def layer(idx):
        tot = np.zeros_like(E)
        for i in idx:
            for orb, dd in dos.get_site_spd_dos(dos.structure[i]).items():
                tot += sum(dd.densities.values())
        return tot
    out = {'name': name, 'E_F': ef, 'z': zz.tolist(), 'V_minus_EF': (v - ef).tolist(),
           'atoms_z': z.tolist(), 'species': sp, 'E': E.tolist(),
           'dos_MoS2': layer(mos2).tolist(), 'dos_metal': layer(metal).tolist(), 'dos_total': dos.get_densities().tolist()}
    json.dump(out, open(f'{ROOT}/figdata/{name}_fig.json', 'w'))
    print('saved', name, len(mos2), len(metal))

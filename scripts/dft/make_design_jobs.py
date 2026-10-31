"""VASP inputs for a locked design round and its elemental references (Secs. III E, S14).

For every design in DESIGN*_PREREGISTERED.json:
  <name>__design/{relax16,static24}    parent in-plane lattice (ionic relaxation, then static)
  <name>__relaxlat/{relax16,static24}  in-plane lattice relaxation (ISIF = 3, LATTICE_CONSTRAINTS = T T F)
                                       from the same starting POSCAR
Optionally the parent itself (--parent-poscar, e.g. the contrast-set Ta2Te2 relax16/POSCAR) as
<parent>__relaxlat, and the elemental bulk references REF_<El> (--references, ISIF = 3 relax then static).
Round 1 used references Ta,W,Nb,Re,Os,Hf,Te; round 2 added Mo,I.

Usage: python make_design_jobs.py --lock DESIGN_PREREGISTERED.json --out <dir> --potcar-root <dir>
       [--references Ta,W,Nb,Re,Os,Hf,Te] [--parent-poscar <POSCAR> --parent-name Ta2Te2]
"""
import argparse
import json
from pathlib import Path

from pymatgen.core import Element

from materon.dft import vasp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--potcar-root', required=True)
    p.add_argument('--references', default='')
    p.add_argument('--parent-poscar')
    p.add_argument('--parent-name', default='Ta2Te2')
    a = p.parse_args()
    root = Path(a.out)
    names = []
    for r in json.loads(Path(a.lock).read_text())['designs']:
        syms = [Element.from_Z(k).symbol for k in r['z']]
        for s, e in r['substitutions'].items():
            syms[int(s)] = e
        text, order = vasp.design_poscar(r['name'], r['lattice'], r['coords'], syms)
        for suffix, inplane in (('__design', False), ('__relaxlat', True)):
            vasp.write_slab_system(root, r['name'] + suffix, text, order, a.potcar_root, inplane=inplane)
            names.append(r['name'] + suffix)
    if a.parent_poscar:
        text = Path(a.parent_poscar).read_text()
        species = text.splitlines()[5].split()
        vasp.write_slab_system(root, a.parent_name + '__relaxlat', text, species, a.potcar_root, inplane=True)
        names.append(a.parent_name + '__relaxlat')
    structures, meshes = vasp.bulk_references()
    for el in (x for x in a.references.split(',') if x):
        vasp.write_bulk_reference(root, el, structures[el], meshes[el], a.potcar_root)
        names.append(f'REF_{el}')
    (root / 'SYSTEMS.txt').write_text('\n'.join(names) + '\n')
    print(names)


if __name__ == '__main__':
    main()

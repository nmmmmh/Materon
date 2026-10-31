"""VASP inputs for a locked substitution set: the contrast set (parents and candidates) or a
screening round (candidates only; parents are reused from the contrast set).

Each system <name>__<C2DB id suffix> gets relax16 (ionic relaxation at the parent in-plane
lattice) and static24. Geometry: parent C2DB coordinates in a 24 A cell, substituted species
on the locked site.

Usage:
  python make_screen_jobs.py --lock CONTRAST_PREREGISTERED_PREDICTIONS.json --set contrast --out <dir> --potcar-root <dir>
  python make_screen_jobs.py --lock ROUND1_PREREGISTERED.json --set screen --contrast-root <contrast dir> --out <dir> --potcar-root <dir>
"""
import argparse
import json
from pathlib import Path

from pymatgen.core import Element

from materon.dft import vasp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', required=True)
    p.add_argument('--set', choices=['contrast', 'screen'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--potcar-root', required=True)
    p.add_argument('--contrast-root', help='screen rounds: directory holding the parent static24 jobs')
    a = p.parse_args()
    root = Path(a.out)
    names = []
    for c in json.loads(Path(a.lock).read_text())['candidates']:
        tag = c['parent_id'].split('-')[1]
        parent_syms = [Element.from_Z(k).symbol for k in c['z']]
        cand_syms = list(parent_syms)
        cand_syms[c['substituted_site']] = c['substituent']
        systems = [('candidate', c['candidate'], cand_syms)]
        if a.set == 'contrast':
            systems.insert(0, ('parent', c['parent'], parent_syms))
        else:
            parent_dir = Path(a.contrast_root) / f"{c['parent']}__{tag}"
            if not (parent_dir / 'static24' / 'LOCPOT').exists():
                raise FileNotFoundError(f'parent job missing: {parent_dir}')
        for role, label, syms in systems:
            name = f'{label}__{tag}'
            text, order, idx = vasp.slab_poscar(name, c['lattice'], c['coords'], syms)
            vasp.write_slab_system(root, name, text, order, a.potcar_root,
                                   {'role': role, 'parent_id': c['parent_id'], 'species_order': order,
                                    'poscar_to_original_site': idx})
            names.append(name)
    (root / 'SYSTEMS.txt').write_text('\n'.join(names) + '\n')
    print(len(names), 'systems')


if __name__ == '__main__':
    main()

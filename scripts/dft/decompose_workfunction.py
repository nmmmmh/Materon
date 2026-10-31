"""Decompose the work function of every <root>/<system>/static24 job.

  --method slab   Phi = D - F with the slab-mean potential (locked hypothesis H2, main and +1 A windows)
  --method core   Phi = (V_vac - V_X) - (E_F - V_X) with the chalcogen core potential (Eq. S10, exploratory)

Usage: python decompose_workfunction.py --method slab --out PHI_DECOMPOSE.json <root> [<root> ...]
"""
import argparse
import json
from pathlib import Path

from materon.dft.mechanism import core_decomposition, slab_mean_decomposition


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['slab', 'core'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('roots', nargs='+')
    a = p.parse_args()
    fn = slab_mean_decomposition if a.method == 'slab' else core_decomposition
    res = {}
    for root in map(Path, a.roots):
        for d in sorted(root.glob('*/static24')):
            if d.parent.name.startswith('_') or not (d / 'LOCPOT').exists():
                continue
            try:
                rec = fn(d)
                if a.method == 'slab':
                    rec = {'root': root.name, **rec}
            except Exception as ex:
                rec = {'root': root.name, 'error': repr(ex)} if a.method == 'slab' else {'error': repr(ex)}
            res[d.parent.name] = rec
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f'{k:30s}', v.get('error') or f"phi {v['phi_eV']:.3f}")


if __name__ == '__main__':
    main()

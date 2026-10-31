"""Energetic stability of MoTaTe2 (Sec. S14.2, Table S37): mixing energy against Ta2Te2 and Mo2Te2
in the same prototype, and the distance above the lower convex hull spanned by the elemental
ground states and the competing monolayers (all PBE, same settings, monolayers in-plane relaxed).

Usage: python stability_energetics.py --out STABILITY_ENERGETICS.json \
         --ref Mo=<REF_Mo/static24> --ref Ta=<...> --ref Te=<...> \
         --phase MoTaTe2=<dir> --phase Ta2Te2=<dir> --phase Mo2Te2=<dir> \
         --phase MoTe2-2H=<dir> --phase TaTe2-1T=<dir> --phase TaTe2-1H=<dir>
"""
import argparse
import json
from pathlib import Path

from materon.dft.stability import chemical_potentials, hull_distance, phase_rows

ELEMENTS = ['Mo', 'Ta', 'Te']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--ref', action='append', required=True)
    p.add_argument('--phase', action='append', required=True)
    p.add_argument('--targets', default='MoTaTe2,Ta2Te2')
    a = p.parse_args()
    mu = chemical_potentials(dict(x.split('=', 1) for x in a.ref))
    phases = {}
    for item in a.phase:
        name, d = item.split('=', 1)
        outcar = Path(d) / 'OUTCAR'
        if not outcar.exists() or 'General timing' not in outcar.read_text():
            print('missing', name)
            continue
        phases[name] = Path(d)
    rows = phase_rows(phases, mu, ELEMENTS)
    res = {'mu_eV_atom': mu, 'phases': rows}
    if {'MoTaTe2', 'Ta2Te2', 'Mo2Te2'} <= rows.keys():
        res['E_mix_eV_atom'] = rows['MoTaTe2']['E_eV_atom'] - 0.5 * (rows['Ta2Te2']['E_eV_atom'] + rows['Mo2Te2']['E_eV_atom'])
    for target in a.targets.split(','):
        res[target] = hull_distance(rows, target)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, r in rows.items():
        print(f"{k:12s} Ef {r['Ef_eV_atom']:+.3f}")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ('phases', 'mu_eV_atom')})


if __name__ == '__main__':
    main()

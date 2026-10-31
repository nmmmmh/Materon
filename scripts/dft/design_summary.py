"""Design summary (Table S34): work function at the parent lattice and after in-plane relaxation,
face values and their difference, lattice constants, formation energy per atom relative to the
elemental references (same PBE protocol), total DOS at E_F (metallicity) and magnetization.

Systems: NAME=<parent-lattice static24>[,<in-plane-relaxed static24>]
References: EL=<REF_EL/static24>   (round 1: Ta W Nb Re Os Hf Te; round 2: Ta Te Mo I)

Usage: python design_summary.py --out DESIGN_SUMMARY.json \
         --ref Ta=<dir> --ref Te=<dir> ... --system Ta2Te2=<contrast static24>,<relaxlat static24> --system MoTaTe2=<...>,<...>
"""
import argparse
import json
from pathlib import Path

from materon.dft.dos import dos_at_fermi
from materon.dft.stability import chemical_potentials, formation_energy
from materon.dft.workfunction import job


def describe(d, mu, tag):
    j = job(d)
    return {f'phi_{tag}_eV': j['phi_eV'], f'phi_faces_{tag}_eV': [j['phi_left_eV'], j['phi_right_eV']],
            f'face_diff_{tag}_eV': abs(j['phi_left_eV'] - j['phi_right_eV']), f'a_{tag}_A': j['a_A'], f'b_{tag}_A': j['b_A'],
            f'Ef_{tag}_eV_atom': formation_energy(d, mu), f'dos_EF_{tag}': dos_at_fermi(Path(d) / 'DOSCAR'),
            f'mag_{tag}': j['magnetization_muB']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--ref', action='append', required=True, help='EL=<static24 dir>')
    p.add_argument('--system', action='append', required=True, help='NAME=<clamped dir>[,<relaxed dir>]')
    a = p.parse_args()
    mu = chemical_potentials(dict(x.split('=', 1) for x in a.ref))
    rows = {}
    for item in a.system:
        name, dirs = item.split('=', 1)
        dirs = dirs.split(',')
        rec = describe(Path(dirs[0]), mu, 'clamped')
        if len(dirs) > 1 and (Path(dirs[1]) / 'LOCPOT').exists():
            rec.update(describe(Path(dirs[1]), mu, 'relaxed'))
        rows[name] = rec
    Path(a.out).write_text(json.dumps({'mu_eV_atom': mu, 'rows': rows}, indent=1))
    for k, v in rows.items():
        s = f"{k:8s} Phi clamped {v['phi_clamped_eV']:.3f} Ef {v['Ef_clamped_eV_atom']:+.3f} DOS(EF) {v['dos_EF_clamped']:.2f}"
        if 'phi_relaxed_eV' in v:
            s += (f" | relaxed Phi {v['phi_relaxed_eV']:.3f} a {v['a_relaxed_A']:.3f}/{v['b_relaxed_A']:.3f}"
                  f" Ef {v['Ef_relaxed_eV_atom']:+.3f} DOS(EF) {v['dos_EF_relaxed']:.2f}")
        print(s)


if __name__ == '__main__':
    main()

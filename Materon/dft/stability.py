"""Formation, mixing and hull energies (Secs. III E, S14.2, S14.3). Energies in eV per atom.

    E_f   = [E(slab) - sum_s n_s mu_s] / N,  mu_s = E(bulk element s) per atom
    E_mix = E(MoTaTe2) - [E(Ta2Te2) + E(Mo2Te2)] / 2          (same prototype, in-plane relaxed)
    E_hull(x) = min sum_k w_k E_f,k  s.t.  sum_k w_k x_k = x, w_k >= 0  (scipy linprog)
Total energies are the last OUTCAR 'free energy TOTEN' of the static runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from materon.dft.workfunction import contcar_composition, total_energy


def energy_composition(d):
    d = Path(d)
    return total_energy(d / 'OUTCAR'), contcar_composition(d / 'CONTCAR')


def chemical_potentials(reference_dirs):
    """{element: E per atom} from {element: static job dir of the elemental reference}."""
    mu = {}
    for el, d in reference_dirs.items():
        e, c = energy_composition(d)
        mu[el] = e / c[el]
    return mu


def formation_energy(d, mu):
    e, comp = energy_composition(d)
    return (e - sum(k * mu[s] for s, k in comp.items())) / sum(comp.values())


def phase_rows(phases, mu, elements):
    """Per-phase energy, composition fractions and formation energy; elements added as bulk phases."""
    rows = {}
    for k, d in phases.items():
        e, c = energy_composition(d)
        n = sum(c.values())
        rows[k] = {'E_eV_atom': e / n, 'x': [c.get(s, 0) / n for s in elements],
                   'Ef_eV_atom': (e - sum(c[s] * mu[s] for s in c)) / n}
    for s in elements:
        rows[s + ' (bulk)'] = {'E_eV_atom': mu[s], 'x': [float(s == t) for t in elements], 'Ef_eV_atom': 0.0}
    return rows


def hull_distance(rows, target):
    """Lower-hull energy at the target composition from all other phases, and the distance above it."""
    from scipy.optimize import linprog

    names = [k for k in rows if k != target]
    comp = [rows[k] for k in names]
    A = np.array([r['x'] for r in comp]).T
    b = np.array(rows[target]['x'])
    lp = linprog([r['Ef_eV_atom'] for r in comp], A_eq=A, b_eq=b, bounds=[(0, None)] * len(comp))
    return {'hull_eV_atom': lp.fun, 'decomposition': {n: float(w) for n, w in zip(names, lp.x) if w > 1e-6},
            'E_above_hull_eV_atom': rows[target]['Ef_eV_atom'] - lp.fun}

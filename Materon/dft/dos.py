"""DOSCAR readers, rigid-band Fermi shifts and effective doping (Sec. S12.2).

Total DOS columns: E, DOS(up), DOS(down), IDOS(up), IDOS(down) for ISPIN = 2 (5 columns),
or E, DOS, IDOS for ISPIN = 1. Spins are summed. Energies in eV, DOS in states/eV/cell,
integrated DOS in electrons per cell.

Rigid band: with n(E) the integrated total DOS of the parent and n0 = n(E_F),
    fermi_shift(dN)      = E[n0 + dN] - E_F                 (dN electrons added per cell)
    effective_doping(dE) = n(E_F + dE) - n(E_F)            (Delta N_eff for an observed shift dE)
Both use linear interpolation on the DOSCAR grid; E_F is the DOSCAR header value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _lines(path):
    path = Path(path)
    return (path / 'DOSCAR' if path.is_dir() else path).read_text().splitlines()


def read_total_dos(path):
    """(energies, total DOS, integrated DOS, E_F) from a DOSCAR file or a job directory."""
    L = _lines(path)
    n = int(L[5].split()[2])
    ef = float(L[5].split()[3])
    a = np.array([[float(v) for v in L[6 + i].split()] for i in range(n)])
    if a.shape[1] == 5:
        dos, idos = a[:, 1] + a[:, 2], a[:, 3] + a[:, 4]
    else:
        dos, idos = a[:, 1], a[:, 2]
    return a[:, 0], dos, idos, ef


def fermi_shift(doscar_path, dn):
    """Rigid-band shift of E_F (eV) when dn electrons per cell are added to the parent."""
    e, _, idos, ef = read_total_dos(doscar_path)
    n0 = np.interp(ef, e, idos)
    return float(np.interp(n0 + dn, idos, e) - ef)


def effective_doping(doscar_path, shift):
    """Delta N_eff = n(E_F + shift) - n(E_F) in the parent integrated DOS (electrons per cell)."""
    e, _, idos, ef = read_total_dos(doscar_path)
    return float(np.interp(ef + shift, e, idos) - np.interp(ef, e, idos))


def dos_at_fermi(doscar_path):
    """Total DOS interpolated at E_F (metallicity criterion of the design locks)."""
    e, dos, _, ef = read_total_dos(doscar_path)
    return float(np.interp(ef, e, dos))


def dos_window_mean(doscar_path, half_width=0.1):
    """Mean total DOS within E_F +- half_width."""
    e, dos, _, ef = read_total_dos(doscar_path)
    m = (e > ef - half_width) & (e < ef + half_width)
    return float(dos[m].mean())


def site_projected(doscar_path, nions):
    """Per-site s, p, d DOS (spins summed) from LORBIT = 11 blocks: list of dicts with e, s, p, d."""
    L = _lines(doscar_path)
    nedos = int(L[5].split()[2])
    out = []
    k = 6 + nedos
    for _ in range(nions):
        k += 1
        rows = np.array([[float(v) for v in L[k + i].split()] for i in range(nedos)])
        k += nedos
        e, d = rows[:, 0], rows[:, 1:]
        if d.shape[1] == 18:      # spin-polarized: s+, s-, py+, py-, pz+, pz-, px+, px-, d (10 columns)
            s, p, dd = d[:, 0] + d[:, 1], d[:, 2:8].sum(1), d[:, 8:18].sum(1)
        elif d.shape[1] == 9:
            s, p, dd = d[:, 0], d[:, 1:4].sum(1), d[:, 4:9].sum(1)
        else:
            raise ValueError(d.shape)
        out.append({'e': e, 's': s, 'p': p, 'd': dd})
    return out


def site_d_dos(doscar_path, site_index):
    """d-projected DOS (spins summed) of one site (0-based POSCAR order)."""
    L = _lines(doscar_path)
    n = int(L[5].split()[2])
    k = 6 + n + site_index * (n + 1) + 1
    b = np.array([[float(v) for v in L[k + j].split()] for j in range(n)])
    return b[:, 9:19].sum(1) if b.shape[1] >= 19 else b[:, 5:10].sum(1)


def band_centre(e, w, lo, hi):
    """First moment of w(E) over [lo, hi] (eV), None if the weight vanishes."""
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    m = (e >= lo) & (e <= hi)
    norm = trapz(w[m], e[m])
    return float(trapz(w[m] * e[m], e[m]) / norm) if norm > 1e-8 else None

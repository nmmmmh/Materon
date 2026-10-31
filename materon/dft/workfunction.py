"""Work function and run diagnostics of a VASP slab job (Eq. S9).

    Phi_face = <V(z)>_window - E_F,     Phi = (Phi_left + Phi_right) / 2

V(z) is the plane-averaged ionic + Hartree potential (LOCPOT written with LVHAR = .TRUE.),
averaged over the fractional-height windows 0.1 <= z/c < 0.3 (left face) and
0.7 <= z/c < 0.9 (right face) of the 24 A cell; E_F is the last OUTCAR Fermi energy.
Energies are in eV.

Two grid conventions are kept because they were used for different reported numbers:
``grid='centre'`` (z_k = (k + 0.5)/N, review-stage jobs: screens, mechanism, designs) and
``grid='edge'`` (z_k = k/N, archived Al2S2 B/Ga package, Table S31).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

WINDOWS = ((0.1, 0.3), (0.7, 0.9))
CHALCOGENS = ('S', 'Se', 'Te')


def locpot(path):
    """Read a LOCPOT: (3x3 lattice in A, plane-averaged potential along c in eV).

    Only the first data block (the total potential) is read.
    """
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(v) for v in lines[k].split()] for k in (2, 3, 4)]) * scale
    counts = [int(v) for v in lines[6].split()]
    k = 7
    if lines[k].strip()[0] in 'sS':
        k += 1
    k += 1 + sum(counts)
    while not lines[k].strip():
        k += 1
    nx, ny, nz = (int(v) for v in lines[k].split())
    k += 1
    vals = []
    need = nx * ny * nz
    while len(vals) < need:
        vals.extend(float(v) for v in lines[k].split())
        k += 1
    v = np.array(vals[:need]).reshape((nz, ny, nx))
    return lat, v.mean(axis=(1, 2))


def grid(n, convention='centre'):
    return (np.arange(n) + 0.5) / n if convention == 'centre' else np.arange(n, dtype=float) / n


def face_windows(profile, convention='centre', windows=WINDOWS):
    """Values of V(z) in the left and right vacuum windows."""
    z = grid(len(profile), convention)
    return [profile[(z >= lo) & (z < hi)] for lo, hi in windows]


def vacuum_level(profile, convention='centre'):
    """Two-face vacuum mean of Eq. (S9)."""
    return float(np.mean([w.mean() for w in face_windows(profile, convention)]))


def fermi(outcar):
    """Last 'E-fermi' value of an OUTCAR (eV)."""
    e = re.findall(r'E-fermi :\s+(-?\d+\.\d+)', Path(outcar).read_text())
    return float(e[-1])


def fermi_any(outcar):
    """Archived-package convention: last 'Fermi energy:' line, else last 'E-fermi'."""
    text = Path(outcar).read_text(encoding='latin-1', errors='replace')
    values = re.findall(r'Fermi energy:\s*([-+0-9.Ee]+)', text) or re.findall(r'E-fermi\s*:\s*([-+0-9.Ee]+)', text)
    if not values:
        raise ValueError('OUTCAR Fermi energy is missing')
    return float(values[-1])


def occupations(eigenval):
    """Number of partially occupied states (0.02 < f < 0.98) in a spin-polarized EIGENVAL."""
    L = Path(eigenval).read_text().splitlines()
    nel, nk, nb = (int(v) for v in L[5].split())
    occ = []
    k = 7
    for _ in range(nk):
        k += 1
        for _ in range(nb):
            parts = L[k].split()
            k += 1
            if len(parts) >= 5:
                occ.extend(float(v) for v in parts[3:5])
        k += 1
    occ = np.array(occ)
    return int(((occ > 0.02) & (occ < 0.98)).sum())


def oszicar_magnetization(oszicar):
    mag = re.findall(r'mag=\s*(-?\d+\.\d+)', Path(oszicar).read_text())
    return float(mag[-1]) if mag else None


def total_energy(outcar):
    """Last 'free energy TOTEN' of an OUTCAR (eV per cell)."""
    e = re.findall(r'free  energy   TOTEN  =\s+(-?\d+\.\d+)', Path(outcar).read_text())
    return float(e[-1]) if e else None


def contcar_composition(contcar):
    L = Path(contcar).read_text().splitlines()
    comp = {}
    for s, n in zip(L[5].split(), map(int, L[6].split())):
        comp[s] = comp.get(s, 0) + n
    return comp


def contcar_species(contcar):
    """Per-site element symbols in CONTCAR order."""
    L = Path(contcar).read_text().splitlines()
    return [s for s, n in zip(L[5].split(), map(int, L[6].split())) for _ in range(n)]


def job(d):
    """Work function (Eq. S9, centre grid) and diagnostics of one job directory.

    Flags used downstream: two-face difference |phi_left - phi_right| > 0.1 eV or
    |magnetization| > 0.1 muB (locked mechanism file).
    """
    d = Path(d)
    lat, prof = locpot(d / 'LOCPOT')
    nz = len(prof)
    z = (np.arange(nz) + 0.5) / nz
    ef = fermi(d / 'OUTCAR')
    out = {'job': d.name, 'a_A': float(np.linalg.norm(lat[0])), 'b_A': float(np.linalg.norm(lat[1])), 'E_F_eV': ef}
    for side, (lo, hi) in (('left', WINDOWS[0]), ('right', WINDOWS[1])):
        w = prof[(z >= lo) & (z < hi)]
        out[f'phi_{side}_eV'] = float(w.mean() - ef)
        out[f'plateau_range_{side}_eV'] = float(w.max() - w.min())
    out['phi_eV'] = (out['phi_left_eV'] + out['phi_right_eV']) / 2
    osz = (d / 'OSZICAR').read_text()
    out['magnetization_muB'] = oszicar_magnetization(d / 'OSZICAR')
    out['ionic_steps'] = len(re.findall(r' F=', osz))
    oc = (d / 'OUTCAR').read_text()
    out['converged_electronic'] = 'aborting loop because EDIFF is reached' in oc
    forces = re.findall(r'TOTAL-FORCE \(eV/Angst\)\n -+\n((?:.*\n)+?) -+', oc)
    if forces:
        f = np.array([[float(v) for v in l.split()[3:6]] for l in forces[-1].strip().splitlines()])
        out['max_force_eV_A'] = float(np.linalg.norm(f, axis=1).max())
    out['partially_occupied_states'] = occupations(d / 'EIGENVAL')
    out['E_total_eV'] = total_energy(d / 'OUTCAR')
    pos = (d / 'CONTCAR').read_text().splitlines()
    out['z_coords_frac'] = [float(l.split()[2]) for l in pos[8:8 + 4]]
    return out


def is_flagged(parent, candidate, face_tol=0.1, mag_tol=0.1):
    """Locked quality flag of a pair of ``job`` records."""
    for r in (parent, candidate):
        if abs(r['phi_left_eV'] - r['phi_right_eV']) > face_tol:
            return True
        if r.get('magnetization_muB') is not None and abs(r['magnetization_muB']) > mag_tol:
            return True
    return False


# ---------------------------------------------------------------------------------------
# Archived Al2S2 B/Ga package (Sec. S11.2, Tables S31 and S33, Fig. S9b): edge grid,
# 'Fermi energy' convention, PAW-sphere populations and an occupation-threshold metal check.
# ---------------------------------------------------------------------------------------

def _floats(fields):
    values = []
    for field in fields:
        try:
            values.append(float(field.replace('D', 'E').replace('d', 'e')))
        except ValueError:
            return None
    return values


def projector_table(outcar, n_sites):
    """Last complete OUTCAR 'total charge' table (LORBIT = 11 PAW-sphere populations), POSCAR order."""
    lines = Path(outcar).read_text(encoding='latin-1', errors='replace').splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().lower() == 'total charge']
    for start in reversed(starts):
        columns, rows = None, []
        for line in lines[start + 1:start + n_sites + 12]:
            fields = line.split()
            if '# ' in line and 'ion' in [f.lower() for f in fields]:
                columns = fields[3:]
                continue
            if fields and fields[0].isdigit():
                values = _floats(fields[1:])
                if int(fields[0]) == len(rows) + 1 and values:
                    rows.append(values)
                elif rows:
                    break
                if len(rows) == n_sites:
                    break
        if len(rows) == n_sites and len({len(r) for r in rows}) == 1:
            if not columns or len(columns) != len(rows[0]):
                columns = ['column_%d' % i for i in range(1, len(rows[0]) + 1)]
            return columns, rows
    raise ValueError('OUTCAR total-charge projector table is missing/incomplete')


def outcar_total_magnetization(outcar):
    lines = Path(outcar).read_text(encoding='latin-1', errors='replace').splitlines()
    starts = [i for i, line in enumerate(lines) if 'magnetization (x)' in line.lower()]
    for start in reversed(starts):
        for line in lines[start + 1:start + 80]:
            fields = line.split()
            if fields and fields[0].lower() == 'tot':
                values = _floats(fields[1:])
                if values:
                    return float(values[-1])
    raise ValueError('OUTCAR total magnetization table is missing')


def metal_diagnostic(vasprun):
    """Occupation-threshold (0.5) gap and band crossings from a pymatgen Vasprun (diagnostic only)."""
    occupied, empty, crossings = [], [], []
    for values in vasprun.eigenvalues.values():
        array = np.asarray(values, dtype=float)
        energy, occ = array[..., 0], array[..., 1]
        occupied.extend(energy[occ > 0.5].tolist())
        empty.extend(energy[occ <= 0.5].tolist())
        crossings.append(bool(np.any((np.min(occ, axis=0) < 0.5) & (np.max(occ, axis=0) > 0.5))))
    gap = float(min(empty) - max(occupied))
    return {'sampled_gap_eV': gap, 'band_crosses_Fermi_by_occupations': any(crossings),
            'sampled_metallic_or_ambiguous': bool(any(crossings) or gap <= 0.02)}


def archived_job(d):
    """Work function and diagnostics of one archived Al2S2 job (edge grid, Table S31)."""
    from pymatgen.io.vasp.outputs import Vasprun

    d = Path(d)
    vr = Vasprun(str(d / 'vasprun.xml'), parse_dos=False, parse_projected_eigen=False)
    ef = fermi_any(d / 'OUTCAR')
    _, prof = locpot(d / 'LOCPOT')
    faces = {}
    for name, w in zip(('left', 'right'), face_windows(prof, 'edge')):
        faces[name] = {'mean_eV': float(w.mean()), 'range_eV': float(w.max() - w.min()),
                       'work_function_eV': float(w.mean() - ef)}
    columns, rows = projector_table(d / 'OUTCAR', len(vr.final_structure))
    forces = np.asarray(vr.ionic_steps[-1]['forces'], dtype=float)
    return {
        'job': d.name, 'electronic_converged': bool(vr.converged_electronic), 'fermi_level_eV': ef,
        'total_energy_eV': float(vr.final_energy), 'total_magnetization_muB': outcar_total_magnetization(d / 'OUTCAR'),
        'max_force_eV_A': float(np.linalg.norm(forces, axis=1).max()), 'metal_diagnostic': metal_diagnostic(vr),
        'faces': faces,
        'mean_work_function_eV': (faces['left']['work_function_eV'] + faces['right']['work_function_eV']) / 2,
        'site_projectors': [{'element': s.specie.symbol, 'columns': columns, 'values_e': r}
                            for s, r in zip(vr.final_structure, rows)],
    }

"""Harmonic phonons by finite displacements with phonopy (Sec. S14.2; phonopy 4.8.4).

Protocol: in-plane-relaxed unit cell, 4 x 4 x 1 supercell, displacement 0.01 A,
non-spin-polarized static runs on a 6 x 6 x 1 mesh (``vasp.phonon_incar``), forces of the
undisplaced supercell (``disp-000``) subtracted from every displaced run, symmetrized force
constants. Frequencies in THz (imaginary modes are negative).

Exact frequencies at q = (k/n, k/n) use a slanted supercell spanned by the primitive vectors
(1, -1) and (m, n - m), m = n // 2 (n = 5: (1, -1) and (2, 3)), which makes q commensurate.
Its k mesh is the Gamma mesh with n_i = ceil(|b_i| / 0.0672) (b_i including 2 pi, A^-1).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DISPLACEMENT_A = 0.01
SUPERCELL = (4, 4, 1)
MESH = (6, 6, 1)
KSPACING = 0.0672  # A^-1, matches the 6 x 6 x 1 density of the 4 x 4 x 1 supercell (2 pi / (6 x 15.58 A))
PATH_SQUARE = (('G', (0, 0, 0)), ('X', (0.5, 0, 0)), ('M', (0.5, 0.5, 0)), ('G', (0, 0, 0)))


def vasprun_forces(d):
    v = ET.parse(Path(d) / 'vasprun.xml').getroot().findall(".//calculation/varray[@name='forces']")[-1]
    return np.array([[float(x) for x in r.text.split()] for r in v])


def generate(unitcell_poscar, out_dir, supercell_matrix=np.diag(SUPERCELL)):
    """Write phonopy_disp.yaml and disp-000 (undisplaced) ... disp-NNN POSCARs; returns the cells."""
    import phonopy
    from phonopy.interface.vasp import read_vasp, write_vasp

    unit = read_vasp(str(unitcell_poscar))
    ph = phonopy.Phonopy(unit, supercell_matrix=supercell_matrix, primitive_matrix=None)
    ph.generate_displacements(distance=DISPLACEMENT_A)
    d0 = Path(out_dir)
    d0.mkdir(parents=True, exist_ok=True)
    ph.save(filename=str(d0 / 'phonopy_disp.yaml'))
    cells = [ph.supercell] + list(ph.supercells_with_displacements)
    for k, cell in enumerate(cells):
        d = d0 / f'disp-{k:03d}'
        d.mkdir(exist_ok=True)
        write_vasp(str(d / 'POSCAR'), cell)
    return ph, cells


def slanted_supercell_matrix(n):
    """Supercell (columns are new vectors) commensurate with q = (1/n, 1/n): (1, -1) and (m, n - m)."""
    m = n // 2
    return np.array([[1, -1, 0], [m, n - m, 0], [0, 0, 1]]).T


def kspacing_mesh(poscar):
    """Gamma mesh n_i = max(1, ceil(|b_i| / KSPACING)) for the in-plane reciprocal vectors."""
    L = Path(poscar).read_text().splitlines()
    lat = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)])
    b = 2 * np.pi * np.linalg.inv(lat).T
    n = [max(1, int(np.ceil(np.linalg.norm(b[i]) / KSPACING - 1e-6))) for i in (0, 1)]
    return (n[0], n[1], 1)


def load(d0, subtract_residual=True):
    """phonopy object with force constants from the finite-displacement runs in ``d0``."""
    import phonopy

    d0 = Path(d0)
    ph = phonopy.load(str(d0 / 'phonopy_disp.yaml'), produce_fc=False, log_level=0)
    nd = len(ph.displacements)
    f0 = vasprun_forces(d0 / 'disp-000') if subtract_residual else 0.0
    ph.forces = np.array([vasprun_forces(d0 / f'disp-{k:03d}') - f0 for k in range(1, nd + 1)])
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    return ph


def dispersion(ph, npts=61, mesh=(24, 24, 1), sigma=0.1):
    """Bands along Gamma-X-M-Gamma, total DOS and extreme frequencies (phonon_post)."""
    pts = [np.array(q) for _, q in PATH_SQUARE]
    qpts = [[list(a + (b - a) * t) for t in np.linspace(0, 1, npts)] for a, b in zip(pts[:-1], pts[1:])]
    ph.run_band_structure(qpts, with_eigenvectors=True, is_band_connection=True)
    bs = ph.get_band_structure_dict()
    ph.run_mesh(list(mesh))
    ph.run_total_dos(sigma=sigma)
    dos = ph.get_total_dos_dict()
    fm = ph.get_mesh_dict()['frequencies']
    return {'distances': [list(map(float, x)) for x in bs['distances']],
            'frequencies_THz': [x.tolist() for x in bs['frequencies']], 'labels': ['Γ', 'X', 'M', 'Γ'],
            'dos_freq_THz': dos['frequency_points'].tolist(), 'dos': dos['total_dos'].tolist(),
            'min_freq_mesh_THz': float(fm.min()), 'min_freq_path_THz': float(min(x.min() for x in bs['frequencies'])),
            'max_freq_THz': float(fm.max()), 'n_displacements': len(ph.displacements)}


def commensurate_frequencies(ph, n=4):
    """Frequencies at the n x n wave vectors commensurate with an n x n x 1 supercell."""
    q = [[i / n, j / n, 0] for i in range(n) for j in range(n)]
    ph.run_qpoints(q)
    return q, ph.qpoints.frequencies


def frequencies_at(ph, qpoints):
    ph.run_qpoints([list(q) for q in qpoints])
    return ph.qpoints.frequencies


def mode_weights(ph, q=(0.2, 0.2, 0), band=0):
    """Polarization (x, y, z) and per-atom weights |e|^2 of one mode at q."""
    ph.run_qpoints([list(q)], with_eigenvectors=True)
    f = ph.qpoints.frequencies[0]
    e = ph.qpoints.eigenvectors[0][:, band].reshape(-1, 3)
    w = np.abs(e) ** 2
    return {'frequencies_THz': f.tolist(), 'xyz_weights': w.sum(0).tolist(),
            'atom_weights': [(s, float(w[i].sum())) for i, s in enumerate(ph.primitive.symbols)]}

"""VASP input writers for the slab, reference and phonon calculations (PBE/PAW, VASP 6.5.1).

Settings reproduce the INCAR/KPOINTS files of the calculations reported in the paper
(Methods VI D, Table S30). Every slab has a 24 A perpendicular cell, a dipole correction
along z and spin polarization with 0.1 muB initial moments. A slab system is computed in
two stages: an ionic relaxation (16 x 16 x 1) and a static run (24 x 24 x 1) that starts
from the relaxed CONTCAR (copy ``relax16/CONTCAR`` to ``static24/POSCAR`` before running
the static job). The VASP command used was ``mpirun -np 1 vasp_gpu`` (one GPU per job).

POTCAR files are licensed and are not distributed; pass the root of a PAW_PBE library
(``<root>/<symbol>/POTCAR``) with ``potcar_root``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CELL_C = 24.0  # perpendicular cell length (A)

# Base slab INCAR (first-round Al2S2 review jobs, Table S30: ISYM = 0).
BASE_INCAR = """SYSTEM = Materon review-stage DFT
PREC = Accurate
ENCUT = 520
ISPIN = 2
MAGMOM = 0.1 0.1 0.1 0.1
ISYM = 0
EDIFF = 1E-7
ISMEAR = 1
SIGMA = 0.05
NELM = 160
LASPH = .TRUE.
LREAL = .FALSE.
LVHAR = .TRUE.
LVACPOTAV = .TRUE.
LDIPOL = .TRUE.
IDIPOL = 3
DIPOL = 0.5 0.5 0.5
LORBIT = 11
LWAVE = .FALSE.
LCHARG = .TRUE.
"""

# Ionic relaxation at fixed (parent) in-plane lattice.
RELAX = "IBRION = 2\nNSW = 80\nISIF = 2\nEDIFFG = -0.02\nPOTIM = 0.3\n"
STATIC = "IBRION = -1\nNSW = 0\n"
# In-plane lattice relaxation of the Al2S2 compositions (Table S32); the B, Ga and O
# compositions were rerun with the damped variant after the first run did not converge.
RELAX_INPLANE_AL2S2 = ("IBRION = 2\nNSW = 120\nISIF = 3\nLATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE.\n"
                       "EDIFFG = -0.01\nPOTIM = 0.3\n")
RELAX_INPLANE_AL2S2_DAMPED = ("IBRION = 1\nNSW = 150\nISIF = 3\nLATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE.\n"
                              "EDIFFG = -0.01\nPOTIM = 0.1\n")
# In-plane lattice relaxation of designs, parents and competing monolayers (Table S34, Sec. S14.2).
RELAX_INPLANE = ("IBRION = 1\nNSW = 120\nISIF = 3\nEDIFFG = -0.02\nPOTIM = 0.2\n"
                 "LATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE.\n")
# Elemental bulk references (Sec. S14.3): no dipole correction, ISIF = 3.
BULK_RELAX = "IBRION = 2\nNSW = 60\nISIF = 3\nEDIFFG = -0.01\nPOTIM = 0.3\n"

# Monolayer MoS2 reference (semiconductor smearing, no initial moments).
MOS2_INCAR = """SYSTEM = Materon review-stage DFT
PREC = Accurate
ENCUT = 520
ISPIN = 2
ISYM = 2
EDIFF = 1E-7
NELM = 160
LASPH = .TRUE.
LREAL = .FALSE.
LVHAR = .TRUE.
LVACPOTAV = .TRUE.
LDIPOL = .TRUE.
IDIPOL = 3
DIPOL = 0.5 0.5 0.5
LORBIT = 11
LWAVE = .FALSE.
LCHARG = .TRUE.
ISMEAR = 0
SIGMA = 0.01
"""
MOS2_RELAX = ("IBRION = 1\nNSW = 80\nISIF = 3\nEDIFFG = -0.01\nPOTIM = 0.2\n"
              "LATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE.\n")
MOS2_POSCAR = """MoS2 monolayer 2H
1.0
3.1840 0.0 0.0
-1.5920 2.757425 0.0
0.0 0.0 24.0
Mo S
1 2
Direct
0.0 0.0 0.5
0.333333333 0.666666667 0.565
0.333333333 0.666666667 0.435
"""

# PAW_PBE datasets used for each element.
POTCAR_SYMBOLS = {
    'Ag': 'Ag', 'Al': 'Al', 'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Bi': 'Bi_d', 'Ca': 'Ca_sv', 'Cd': 'Cd',
    'Cs': 'Cs_sv', 'Cu': 'Cu_pv', 'Ga': 'Ga_d', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'I': 'I', 'Ir': 'Ir', 'K': 'K_sv',
    'Mg': 'Mg', 'Mo': 'Mo_pv', 'Na': 'Na_pv', 'Nb': 'Nb_pv', 'O': 'O', 'Os': 'Os_pv', 'Pb': 'Pb_d', 'Pd': 'Pd',
    'Pt': 'Pt', 'Rb': 'Rb_sv', 'Re': 'Re_pv', 'Rh': 'Rh_pv', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv',
    'Se': 'Se', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 'Te': 'Te', 'Ti': 'Ti_pv', 'W': 'W', 'Y': 'Y_sv',
    'Zn': 'Zn', 'Zr': 'Zr_sv',
}


def set_tag(incar, tag, value):
    """Replace the line of ``tag`` in an INCAR string (the line must exist)."""
    lines = incar.splitlines()
    if not any(l.strip().startswith(tag) for l in lines):
        raise KeyError(tag)
    return '\n'.join(f'{tag} = {value}' if l.strip().startswith(tag) else l for l in lines) + '\n'


def drop_tags(incar, tags):
    return '\n'.join(l for l in incar.splitlines() if not l.strip().startswith(tuple(tags))) + '\n'


# Screens, contrast set and designs ran with ISYM = 2 (validated against the ISYM = 0 parent).
SLAB_INCAR = set_tag(BASE_INCAR, 'ISYM', 2)
BULK_INCAR = drop_tags(SLAB_INCAR, ('IDIPOL', 'LDIPOL', 'DIPOL', 'ISYM', 'MAGMOM'))


def phonon_incar(static_incar=SLAB_INCAR + STATIC):
    """Finite-displacement INCAR: non-spin-polarized, EDIFF 1E-8, no dipole (Sec. S14.2)."""
    ph = drop_tags(static_incar, ('ISPIN', 'MAGMOM', 'EDIFF', 'LVHAR', 'LVACPOTAV', 'LDIPOL', 'IDIPOL',
                                  'DIPOL', 'LORBIT', 'LCHARG', 'ISYM', 'NELM')).rstrip('\n')
    return ph + '\nISPIN = 1\nEDIFF = 1E-8\nISYM = 0\nNELM = 200\nADDGRID = .TRUE.\nLCHARG = .FALSE.\nNCORE = 1\n'


def kpoints(mesh):
    """Gamma-centred mesh; ``mesh`` is an int (n x n x 1) or a 3-tuple."""
    mesh = (mesh, mesh, 1) if isinstance(mesh, int) else tuple(mesh)
    return 'Gamma mesh\n0\nGamma\n' + ' '.join(map(str, mesh)) + '\n0 0 0\n'


def potcar(species, potcar_root):
    root = Path(potcar_root)
    return ''.join((root / POTCAR_SYMBOLS[s] / 'POTCAR').read_text() for s in species)


def write_job(directory, poscar_text, species, incar, mesh, potcar_root, meta=None):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'POSCAR').write_text(poscar_text)
    (d / 'INCAR').write_text(incar)
    (d / 'KPOINTS').write_text(kpoints(mesh))
    (d / 'POTCAR').write_text(potcar(species, potcar_root))
    if meta is not None:
        (d / 'JOB.json').write_text(json.dumps(meta, indent=1))
    return d


def slab_poscar(name, lattice, coords, symbols, c=CELL_C):
    """POSCAR of a C2DB slab in a c = 24 A cell, centred at c/2, species in order of first appearance.

    ``lattice``/``coords`` are the C2DB lattice and Cartesian coordinates; the in-plane vectors are
    kept and the perpendicular vector replaced. Returns (text, species order, POSCAR->original site).
    """
    lattice = np.asarray(lattice)
    coords = np.asarray(coords)
    a, b = lattice[0].copy(), lattice[1].copy()
    a[2] = b[2] = 0.0
    zc = coords[:, 2] - coords[:, 2].mean() + c / 2
    inv = np.linalg.inv(np.array([a[:2], b[:2]]).T)
    frac = np.column_stack([(inv @ coords[:, :2].T).T % 1.0, zc / c])
    order = []
    for s in symbols:
        if s not in order:
            order.append(s)
    idx = [k for s in order for k, t in enumerate(symbols) if t == s]
    lines = [name, '1.0', ' '.join(f'{v:.10f}' for v in a), ' '.join(f'{v:.10f}' for v in b), f'0.0 0.0 {c:.10f}',
             ' '.join(order), ' '.join(str(symbols.count(s)) for s in order), 'Direct']
    lines += [' '.join(f'{v:.10f}' for v in frac[k]) for k in idx]
    return '\n'.join(lines) + '\n', order, idx


def design_poscar(name, lattice, coords, symbols, c=CELL_C):
    """Design slabs: same geometry as ``slab_poscar`` but written by pymatgen with sites sorted by Z."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp import Poscar

    lattice = np.asarray(lattice)
    xyz = np.asarray(coords)
    a, b = lattice[0].copy(), lattice[1].copy()
    a[2] = b[2] = 0.0
    zc = xyz[:, 2] - xyz[:, 2].mean() + c / 2
    inv = np.linalg.inv(np.array([a[:2], b[:2]]).T)
    frac = np.column_stack([(inv @ xyz[:, :2].T).T % 1.0, zc / c])
    st = Structure(Lattice([a, b, [0, 0, c]]), symbols, frac).get_sorted_structure(key=lambda x: x.specie.Z)
    order = [str(e) for e in dict.fromkeys(str(x.specie) for x in st)]
    return Poscar(st, comment=name).__str__(), order


def write_slab_system(root, name, poscar_text, species, potcar_root, meta=None, inplane=False):
    """relax16 + static24 of one slab; ``inplane`` selects the in-plane lattice relaxation."""
    relax = RELAX_INPLANE if inplane else RELAX
    write_job(Path(root) / name / 'relax16', poscar_text, species, SLAB_INCAR + relax, 16, potcar_root, meta)
    write_job(Path(root) / name / 'static24', poscar_text, species, SLAB_INCAR + STATIC, 24, potcar_root, meta)


def write_bulk_reference(root, element, structure, mesh, potcar_root):
    """Elemental ground state (Sec. S14.3): ISIF = 3 relaxation then static, same mesh."""
    from pymatgen.io.vasp import Poscar

    text = Poscar(structure, comment=f'{element} bulk').__str__()
    write_job(Path(root) / f'REF_{element}' / 'relax16', text, [element], BULK_INCAR + BULK_RELAX, mesh, potcar_root)
    write_job(Path(root) / f'REF_{element}' / 'static24', text, [element], BULK_INCAR + STATIC, mesh, potcar_root)


def bulk_references():
    """Starting structures and meshes of the elemental references (bcc, hcp, trigonal Te, Cmce I)."""
    from pymatgen.core import Lattice, Structure

    bcc = lambda a, e: Structure(Lattice.cubic(a), [e] * 2, [[0, 0, 0], [.5, .5, .5]])
    hcp = lambda a, c, e: Structure(Lattice.hexagonal(a, c), [e] * 2, [[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]])
    refs = {
        'Ta': bcc(3.306, 'Ta'), 'W': bcc(3.165, 'W'), 'Nb': bcc(3.300, 'Nb'), 'Mo': bcc(3.147, 'Mo'),
        'Re': hcp(2.761, 4.456, 'Re'), 'Os': hcp(2.734, 4.317, 'Os'), 'Hf': hcp(3.195, 5.051, 'Hf'),
        'Te': Structure(Lattice.hexagonal(4.457, 5.929), ['Te'] * 3,
                        [[.2636, 0, 1 / 3], [0, .2636, 2 / 3], [-.2636, -.2636, 0]]),
        'I': Structure.from_spacegroup('Cmce', Lattice.orthorhombic(7.18, 4.71, 9.81), ['I'],
                                       [[0, 0.1543, 0.1174]]).get_primitive_structure(),
    }
    meshes = {'Ta': (16, 16, 16), 'W': (16, 16, 16), 'Nb': (16, 16, 16), 'Mo': (16, 16, 16),
              'Re': (18, 18, 12), 'Os': (18, 18, 12), 'Hf': (18, 18, 12), 'Te': (12, 12, 8), 'I': (6, 6, 6)}
    return refs, meshes


# First-round Al2S2 review jobs (Sec. S11.3): archived parent, site order 0-3 (Al0, Al1, S, S).
AL2S2_A = 4.9509439468383789
AL2S2_PARENT = [('Al', (0.0, 0.0, 0.5004166666666663)), ('Al', (0.5, 0.5, 0.4995833333333337)),
                ('S', (0.5, 0.0, 0.5004166666666663)), ('S', (0.0, 0.5, 0.4995833333333337))]
AL2S2_EDITS = {'B': (0, 'B'), 'Ga': (0, 'Ga'), 'O': (2, 'O'), 'Se': (2, 'Se')}


def al2s2_poscar(name, edit=None, scale=1.0):
    """Al2S2 parent or a single-site edit; ``scale`` multiplies the in-plane lattice (strain series)."""
    sites = [list(x) for x in AL2S2_PARENT]
    if edit is not None:
        k, element = AL2S2_EDITS[edit]
        sites[k][0] = element
    order = []
    for e, _ in sites:
        if e not in order:
            order.append(e)
    idx = [k for e in order for k, (f, _) in enumerate(sites) if f == e]
    a = AL2S2_A * scale
    lines = [name, '1.0', f'{a:.16f} 0.0 0.0', f'0.0 {a:.16f} 0.0', f'0.0 0.0 {CELL_C:.16f}',
             ' '.join(order), ' '.join(str(sum(1 for f, _ in sites if f == e)) for e in order), 'Direct']
    lines += [' '.join(f'{v:.16f}' for v in sites[k][1]) for k in idx]
    return '\n'.join(lines) + '\n', order, idx

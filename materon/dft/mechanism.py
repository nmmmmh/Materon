"""Band-filling mechanism analysis of single-site substitutions (Secs. III C, S12; Table S36).

Pair quantities compare a candidate q with its parent p at the same in-plane lattice
(static24 jobs). Energies in eV.

* Slab-mean decomposition (locked hypothesis H2):
      Phi = D - F,  D = V_vac - Vbar,  F = E_F - Vbar,
  Vbar the mean plane-averaged potential between the outermost atomic planes (main
  window) or over those planes extended by 1 A on each side (sensitivity window).
* Core-referenced decomposition (Eq. S10):
      Phi = (V_vac - V_X) - (E_F - V_X),
  V_X the mean OUTCAR core potential of the chalcogen atoms (or of the unsubstituted host
  metal, ``M`` reference). V_vac is the two-face vacuum level of Eq. (S9) on the same zero.
* Effective doping: Delta N_eff = n[E_F + Delta(E_F - V_ref)] - n(E_F) in the parent
  integrated total DOS (electrons per cell).
* Valence count N = group number for groups 1-12, group - 10 for groups 13-18;
  Delta N = N(substituent) - N(host). Early hosts: groups 3-6.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from materon.dft.dos import read_total_dos
from materon.dft.workfunction import CHALCOGENS, contcar_species, fermi, locpot, oszicar_magnetization, vacuum_level

EARLY_HOST_GROUPS = (3, 4, 5, 6)


def valence_count(symbol):
    from pymatgen.core import Element

    g = Element(symbol).group
    return g if g <= 12 else g - 10


def delta_valence(substituent, host):
    return valence_count(substituent) - valence_count(host)


def is_early_host(symbol):
    from pymatgen.core import Element

    return Element(symbol).group in EARLY_HOST_GROUPS


def mech_class(host, sub, anion):
    """Mechanism class of an early-host substitution (written before the round-2 DFT, Sec. S12.3).

    'gap': Ta2Te2 host; 'ionic': s/p substituent (groups 1, 2, 13-15); 'filled': filled-shell d
    substituent (group >= 9); 'local': other transition metals.
    """
    from pymatgen.core import Element

    g = Element(sub).group
    if host == 'Ta' and anion == 'Te':
        return 'gap'
    if g in (1, 2, 13, 14, 15):
        return 'ionic'
    if g >= 9:
        return 'filled'
    return 'local'


def predicted_sign(cls, dn):
    """Predicted sign of dPhi: s/p and filled-d raise Phi; local-d and Ta2Te2 opposite to dN."""
    if cls in ('ionic', 'filled'):
        return 1
    return int(-np.sign(dn))


def core_potentials(outcar, n_sites):
    """OUTCAR 'average (electrostatic) potential at core' values (eV, VASP zero), POSCAR order."""
    t = Path(outcar).read_text()
    blk = t.split('average (electrostatic) potential at core')[-1].split('the norm of the test charge is')[1]
    blk = blk.split('E-fermi')[0].split('\n', 1)[1]
    return [float(v) for v in re.findall(r'\d+\s+(-?\d+\.\d+)', blk)][:n_sites]


def slab_heights(contcar):
    """Element symbols and Cartesian z (A, wrapped to [0, c)) from a CONTCAR."""
    L = Path(contcar).read_text().splitlines()
    c = float(L[4].split()[2]) * float(L[1].split()[0])
    sym = contcar_species(contcar)
    z = np.array([float(l.split()[2]) for l in L[8:8 + len(sym)]]) % 1.0 * c
    return sym, z


def slab_mean_decomposition(d):
    """Dipole D and filling F terms (main and +1 A windows) of one static job."""
    d = Path(d)
    lat, prof = locpot(d / 'LOCPOT')
    ef = fermi(d / 'OUTCAR')
    nz = len(prof)
    z = (np.arange(nz) + 0.5) / nz
    c = np.linalg.norm(lat[2])
    pos = (d / 'CONTCAR').read_text().splitlines()
    counts = [int(v) for v in pos[6].split()]
    zs = np.array([float(l.split()[2]) for l in pos[8:8 + sum(counts)]]) % 1.0
    vvac = vacuum_level(prof)
    rec = {'phi_eV': float(vvac - ef), 'thickness_A': float((zs.max() - zs.min()) * c)}
    for tag, pad in (('', 0.0), ('_pad1', 1.0 / c)):
        vbar = prof[(z >= zs.min() - pad) & (z <= zs.max() + pad)].mean()
        rec['dipole' + tag + '_eV'] = float(vvac - vbar)
        rec['filling' + tag + '_eV'] = float(ef - vbar)
    return rec


def core_decomposition(d):
    """Core-referenced terms of Eq. (S10) for one static job."""
    d = Path(d)
    sym = contcar_species(d / 'CONTCAR')
    vals = core_potentials(d / 'OUTCAR', len(sym))
    _, prof = locpot(d / 'LOCPOT')
    ef = fermi(d / 'OUTCAR')
    vvac = vacuum_level(prof)
    vx = np.mean([v for s, v in zip(sym, vals) if s in CHALCOGENS])
    return {'phi_eV': float(vvac - ef), 'core_X_eV': float(vx), 'vac_minus_coreX_eV': float(vvac - vx),
            'EF_minus_coreX_eV': float(ef - vx),
            'core_by_species_eV': {s: float(np.mean([v for t, v in zip(sym, vals) if t == s])) for s in set(sym)}}


def _state(d, host):
    sym, z = slab_heights(d / 'CONTCAR')
    cp = core_potentials(d / 'OUTCAR', len(sym))
    _, prof = locpot(d / 'LOCPOT')
    X = [i for i, s in enumerate(sym) if s in CHALCOGENS]
    M = [i for i, s in enumerate(sym) if s == host]
    return dict(sym=sym, z=z, ef=fermi(d / 'OUTCAR'), vac=vacuum_level(prof),
                coreX=np.mean([cp[i] for i in X]), coreM=np.mean([cp[i] for i in M]),
                thick=float(z.max() - z.min()), mag=oszicar_magnetization(d / 'OSZICAR') or 0.0)


def pair_descriptors(parent_dir, candidate_dir, host, sub):
    """dPhi, level shifts, E_F-versus-core shifts and Delta N_eff (X and M references) of one pair."""
    P, Q = Path(parent_dir), Path(candidate_dir)
    p, q = _state(P, host), _state(Q, host)
    rec = {'dPhi': (q['vac'] - q['ef']) - (p['vac'] - p['ef'])}
    e, _, idos, efd = read_total_dos(P / 'DOSCAR')
    for ref in ('X', 'M'):
        lev = (q['core' + ref] - q['vac']) - (p['core' + ref] - p['vac'])    # core level shift vs vacuum
        fill = (q['ef'] - q['core' + ref]) - (p['ef'] - p['core' + ref])      # E_F shift vs core
        rec[f'level_shift_{ref}_eV'] = float(lev)
        rec[f'EF_vs_core_{ref}_eV'] = float(fill)
        rec[f'dN_eff_{ref}'] = float(np.interp(efd + fill, e, idos) - np.interp(efd, e, idos))
    rec['dThick_A'] = q['thick'] - p['thick']
    hz = [q['z'][i] for i, s in enumerate(q['sym']) if s == host]
    sz = [q['z'][i] for i, s in enumerate(q['sym']) if s == sub]
    rec['sub_offset_A'] = float(sz[0] - np.mean(hz)) if sz and hz else None
    rec['mag_p'], rec['mag_q'] = p['mag'], q['mag']
    return rec


# Locked hypotheses (written and hashed before the round-1 DFT results were read).
MECHANISM_LOCK = {
    'read_before_lock': ['contrast-set DFT work-function changes (16 pairs)',
                         'model gate-rule summary (C2DB-trained ensemble L400)'],
    'not_read_before_lock': ['any potential decomposition (dipole/filling terms)', 'round-1 application DFT results',
                             'round-2 and design DFT'],
    'definitions': {
        'decomposition': 'Phi = D - F with D = V_vac - Vbar, F = E_F - Vbar; V(z) plane-averaged ionic+Hartree potential, V_vac the two-face '
                         'vacuum mean of Eq. (S8), Vbar the mean of V(z) between the outermost atomic planes (main) and over those planes '
                         'extended by 1 A on each side (sensitivity). Static24 jobs, parent in-plane lattice (clamped).',
        'pair change': 'dX = X(candidate) - X(parent) for X in Phi, D, F.',
        'valence count': 'N = group number for groups 1-12, group - 10 for groups 13-18; dN = N(substituent) - N(host) for one M-site substitution.',
        'early host': 'host metal of groups 3-6 (Sc, Y, Ti, Zr, Hf, Nb, Ta, Mo, W).',
        'flagged pairs': 'two-face difference > 0.1 eV or |magnetization| > 0.1 muB in either system: reported separately, excluded from tests.'},
    'hypotheses': {
        'H1_sign_rule': 'In early hosts, single M-site substitutions with dN != 0 change Phi with sign opposite to dN (electron addition lowers Phi). '
                        'Formulated from the contrast set (8/8); tested only on pairs computed after this lock (round 1, round 2, design). '
                        'Supported if the sign agreement exceeds 50% with a one-sided binomial p < 0.05.',
        'H2_filling_term': 'In early hosts, dF has the sign of dN (electron addition raises E_F relative to the slab-mean potential) in a majority '
                           'of pairs, and -dF tracks dPhi more closely than dD does: Spearman(dPhi, -dF) > Spearman(dPhi, dD). Tested on all '
                           'early-host pairs (contrast, round 1, round 2, design); reported for both Vbar windows.',
        'H3_model_variable': 'The C2DB-trained ensemble identifies valence count as the controlling variable but with the inter-compound sign: '
                             'within early hosts its predicted change correlates positively with dN (Spearman > 0 in all five members). '
                             'Reported on the same pairs as H2.'},
    'not_claimed': 'Late hosts (groups 8-12) and the p-block host Sb2Se2 are reported descriptively; no sign rule is asserted for them.',
}

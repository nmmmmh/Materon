"""Interface analysis from static runs: work functions on both vacuum sides, interface dipole step,
MoS2 conduction-band minimum aligned through Mo core potentials, and the n-type Schottky barrier.
EA and gap of isolated MoS2 come from the paper protocol (primitive cell, PBE): EA 4.253 eV, gap 1.666 eV."""
import os, re, sys, json
import numpy as np
ROOT = os.path.join(os.environ.get('MATERON_INTERFACE_ROOT', '.'), 'jobs')   # directory holding jobs/<system>/{relax,static}
EA, GAP = 4.253, 1.666

def efermi(d):
    # A static stopped with LABORT prints no E-fermi line in OUTCAR; vasprun.xml (= DOSCAR header) then gives it.
    m = re.findall(r'E-fermi :\s+(-?\d+\.\d+)', open(f'{d}/OUTCAR').read())
    if m:
        return float(m[-1])
    return float(re.findall(r'name="efermi">\s*(-?\d+\.\d+)', open(f'{d}/vasprun.xml').read())[-1])

def cores(d):
    txt = open(f'{d}/OUTCAR').read()
    if 'average (electrostatic) potential at core' not in txt:
        # LABORT skipped the core printout; the non-SCF run reads the same CHGCAR and prints the same core potentials.
        txt = open(d.replace('/jobs/', '/nscfK/').rsplit('/static', 1)[0] + '/OUTCAR').read()
    blk = txt.rsplit('average (electrostatic) potential at core', 1)[1]
    blk = blk.split('\n\n\n')[0] if '\n\n\n' in blk else blk[:20000]
    lines = blk.splitlines()[3:]
    vals = {}
    for ln in lines:
        nums = re.findall(r'(\d+)\s+(-?\d+\.\d+)', ln)
        if not nums and vals: break
        for i, v in nums: vals[int(i)] = float(v)
    return vals

def structure(d):
    L = open(f'{d}/CONTCAR').read().splitlines()
    c = float(L[4].split()[2]); els = L[5].split(); cnt = list(map(int, L[6].split()))
    sp = [e for e, n in zip(els, cnt) for _ in range(n)]
    z = np.array([float(l.split()[2]) for l in L[8:8 + len(sp)]]) * c
    return sp, z, c

def planar(d):
    with open(f'{d}/LOCPOT') as f:
        L = [next(f) for _ in range(7)]
        n = sum(map(int, L[6].split())); [next(f) for _ in range(n + 1)]
        while True:
            line = next(f)
            if line.strip(): break
        nx, ny, nz = map(int, line.split())
        data = np.fromstring(f.read(), sep=' ', count=nx * ny * nz)
    return data.reshape(nz, ny, nx).mean(axis=(1, 2))

def vacuum(d, sp, z, c):
    v = planar(d); nz = len(v); zz = np.arange(nz) * c / nz
    lo, hi = z.min(), z.max()
    def win(a, b):
        m = (zz >= a % c) & (zz < b % c) if a % c < b % c else (zz >= a % c) | (zz < b % c)
        return v[m].mean(), v[m].std()
    return win(lo - 7, lo - 4), win(hi + 4, hi + 7)   # below (metal side), above (MoS2 side)

def analyze(name):
    d = f'{ROOT}/{name}/static'
    sp, z, c = structure(d); ef = efermi(d); cp = cores(d)
    (vb, sb), (vt, st) = vacuum(d, sp, z, c)
    out = {'name': name, 'E_F': ef, 'Phi_bottom': vb - ef, 'Phi_top': vt - ef, 'vac_noise': max(sb, st)}
    mo = [i for i, e in enumerate(sp) if e == 'Mo']
    if 'S' in sp:
        zS = np.array([z[i] for i, e in enumerate(sp) if e == 'S'])
        mos2_mo = [i for i in mo if z[i] > zS.min() - 0.5]
        out['V_core_Mo_MoS2'] = float(np.mean([cp[i + 1] for i in mos2_mo]))
    out['E_relax'] = float(re.findall(r'energy  without entropy=\s+(-?\d+\.\d+)', open(f'{ROOT}/{name}/relax/OUTCAR').read())[-1])
    return out

if __name__ == '__main__':
    ref = analyze('MoS2_iso')
    # isolated MoS2: CBM relative to its core level, with the CBM fixed at -EA from its own vacuum
    cbm_minus_core = (ref['Phi_top'] + ref['E_F'] - EA) - ref['V_core_Mo_MoS2']
    res = {'MoS2_iso': ref}
    for name in sys.argv[1:]:
        r = analyze(name)
        if name.endswith('_iso'):
            r['Phi'] = 0.5 * (r['Phi_bottom'] + r['Phi_top']); r['SchottkyMott_n'] = r['Phi'] - EA
        else:
            cbm = r['V_core_Mo_MoS2'] + cbm_minus_core
            r['Phi_B_n_core'] = cbm - r['E_F']                  # Fermi level to MoS2 CBM, core-aligned
            r['Phi_B_n_vac'] = r['Phi_top'] - EA                 # same, MoS2-side vacuum as reference
            r['dipole_step'] = r['Phi_top'] - r['Phi_bottom']    # vacuum-level step across the bilayer
        res[name] = r
    print(json.dumps(res, indent=1))

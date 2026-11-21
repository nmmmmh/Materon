"""Split relaxed heterostructures into their two layers at fixed geometry for charge-density-difference statics
(quantity 3 of ANALYSIS_PLAN.md). Usage: python setup_cdd.py <hetero_dir> [...]"""
import sys, os, shutil, re
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('MATERON_INTERFACE_ROOT', '.'))
J = ROOT / 'jobs'; OUT = ROOT / 'jobs_cdd'; OUT.mkdir(exist_ok=True)

def read_poscar(f):
    l = Path(f).read_text().splitlines()
    sc = float(l[1]); lat = np.array([[float(x) for x in l[i].split()] for i in (2, 3, 4)]) * sc
    el = l[5].split(); n = list(map(int, l[6].split())); i = 7
    sel = l[i].strip()[0] in 'Ss'
    if sel: i += 1
    cart = l[i].strip()[0] in 'CcKk'; i += 1
    X = np.array([[float(x) for x in l[j].split()[:3]] for j in range(i, i + sum(n))])
    if cart: X = X @ np.linalg.inv(lat)
    sp = sum([[e] * k for e, k in zip(el, n)], [])
    return lat, el, sp, X % 1.0

def potcar_blocks(f):
    txt = Path(f).read_text(); parts = re.split(r'(?<=End of Dataset\n)', txt); parts = [p for p in parts if p.strip()]
    out = {}
    for p in parts:
        t = re.search(r'TITEL\s*=\s*PAW_PBE\s+(\S+)', p).group(1); out[t.split('_')[0]] = p
    return out

for h in sys.argv[1:]:
    d = J / h
    lat, el, sp, X = read_poscar(d / 'relax' / 'CONTCAR')
    z = X[:, 2]; o = np.sort(z); gaps = np.diff(np.r_[o, o[0] + 1]); k = np.argsort(-gaps)[:2]
    # two largest gaps: vacuum and the van der Waals gap; cut at the vdW gap (the smaller of the two)
    k = sorted(k, key=lambda i: gaps[i])[0]; zc = (o[k] + gaps[k] / 2) % 1.0
    low = z < zc if o[k] + gaps[k] / 2 < 1 else z >= 0
    pots = potcar_blocks(d / 'POTCAR')
    sets = {'A': np.where(low)[0], 'B': np.where(~low)[0]}
    for tag, idx in sets.items():
        species = [e for e in el if any(sp[i] == e for i in idx)]
        name = 'MoS2' if set(sp[i] for i in idx) <= {'Mo', 'S'} else 'metal'
        od = OUT / f'{h}_{name}'; od.mkdir(exist_ok=True)
        lines = [f'{h} {name} layer at heterostructure geometry', '1.0'] + ['  '.join('%.10f' % v for v in r) for r in lat]
        lines += ['  '.join(species), '  '.join(str(sum(sp[i] == e for i in idx)) for e in species), 'Direct']
        for e in species:
            lines += ['  '.join('%.10f' % v for v in X[i]) for i in idx if sp[i] == e]
        (od / 'POSCAR').write_text('\n'.join(lines) + '\n')
        (od / 'POTCAR').write_text(''.join(pots[e] for e in species))
        inc = (d / 'INCAR.static').read_text()
        inc = re.sub(r'^SYSTEM = .*', f'SYSTEM = {h} {name} cdd', inc, flags=re.M)
        inc = re.sub(r'^KPAR = .*', 'KPAR = 7', inc, flags=re.M); inc = re.sub(r'^NCORE = .*\n', '', inc, flags=re.M)
        (od / 'INCAR.static').write_text(inc); shutil.copy(d / 'KPOINTS.static', od / 'KPOINTS.static')
        shutil.copy(d / 'run.sh', od / 'run.sh')
        (od / 'relax').mkdir(exist_ok=True); shutil.copy(od / 'POSCAR', od / 'relax' / 'CONTCAR'); (od / 'relax' / 'DONE').touch()
        print(od.name, species, [sum(sp[i] == e for i in idx) for e in species], 'cut z=%.3f' % zc)

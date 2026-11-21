"""MoS2 band edges at K from the non-SCF Gamma/K run: states whose PROCAR weight lies mostly on the MoS2 layer.
E_F comes from the self-consistent static run. Prints CBM/VBM at K relative to E_F and to the vacuum level."""
import os, sys, re, json
import numpy as np
ROOT = os.environ.get('MATERON_INTERFACE_ROOT', '.')   # holds jobs/ and nscfK/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv, names = sys.argv[:1], sys.argv[1:]
from analyze import structure, efermi, vacuum

def procar(path, nions):
    txt = open(path).read()
    out = []
    for kb in re.split(r'\n k-point\s+', txt)[1:]:
        kidx = int(kb.split()[0])
        for bb in re.split(r'\nband\s+', kb)[1:]:
            e = float(re.search(r'energy\s+(-?\d+\.\d+)', bb).group(1))
            rows = [l.split() for l in bb.splitlines() if l.strip()[:1].isdigit()]
            w = np.array([float(r[-1]) for r in rows[:nions]])
            out.append((kidx, e, w))
    return out

import subprocess
core_est = {}
try:
    core_est = {k: v['Phi_B_n_core'] for k, v in json.loads(subprocess.run([sys.executable, f'{HERE}/analyze.py'] + [x for x in names if not x.endswith('_iso')], capture_output=True, text=True, cwd=HERE).stdout).items() if 'Phi_B_n_core' in v}
except Exception:
    pass
res = {}
for name in names:
    s = f'{ROOT}/jobs/{name}/static'; n = f'{ROOT}/nscfK/{name}'
    sp, z, c = structure(s); ef = efermi(s)
    (vb, _), (vt, _) = vacuum(s, sp, z, c)
    zS = np.array([z[i] for i, e in enumerate(sp) if e == 'S'])
    mos2 = np.array([(sp[i] == 'S') or (sp[i] == 'Mo' and z[i] > zS.min() - 0.5) for i in range(len(sp))])
    st = procar(f'{n}/PROCAR', len(sp))
    K = [(e - ef, w[mos2].sum() / max(w.sum(), 1e-9)) for k, e, w in st if k in (2, 3)]
    # CBM: lowest strongly MoS2-weighted state above (core-aligned estimate - 0.3 eV); the estimate only brackets the search
    est = core_est.get(name)
    if est is None:
        m = np.array(sorted(e for e, f in K if f > 0.6)); gaps = np.diff(m)
        i = int(np.argmax(np.where((m[:-1] > -3) & (m[1:] < 3), gaps, 0))); cbm = m[i + 1]
    else:
        cbm = min(e for e, f in K if f > 0.7 and e > est - 0.3)
    res[name] = {'E_F': ef, 'CBM_K_minus_EF': cbm, 'core_estimate': est,
                 'CBM_vs_vac_top': cbm - (vt - ef), 'Phi_top': vt - ef, 'Phi_bottom': vb - ef}
print(json.dumps(res, indent=1))

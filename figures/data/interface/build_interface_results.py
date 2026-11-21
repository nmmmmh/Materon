"""Collect the interface results (interface/figdata) into interface_results.json.
Inputs here: K_results.json (K-point CBM, two lowest-energy registries), core_results_all.json (Mo core-level and
MoS2-side vacuum alignment, all registries and isolated layers), <job>_fig.json (plane-averaged potential), cdd.json
(plane-integrated charge-density difference). Barriers are n-type, E_CBM(MoS2) - E_F, in eV.
Tunnel barrier: maximum of V - E_F between the outermost Te and S planes; width: length over which V > E_F there."""
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
K = json.loads((H / 'K_results.json').read_text())
C = json.loads((H / 'core_results_all.json').read_text())
Q = json.loads((H / 'cdd.json').read_text())
out = {'isolated': {k: {'Phi': C[k]['Phi']} for k in ('Ta2Te2_iso', 'MoTaTe2_iso', 'MoS2_iso')}}
out['isolated']['MoS2_iso']['EA_this_cell'] = -K['MoS2_iso']['CBM_vs_vac_top']
dphi = C['Ta2Te2_iso']['Phi'] - C['MoTaTe2_iso']['Phi']
out['isolated']['dPhi_same_cell'] = dphi
reg = {}
for job, v in C.items():
    if '_MoS2_s' not in job:
        continue
    metal = job.split('_')[0]
    reg.setdefault(metal, {})[job] = {'E_relax': v['E_relax'], 'Phi_B_n_core': v['Phi_B_n_core'],
                                      'Phi_B_n_vac': v['Phi_B_n_vac'], 'dipole_step': v['dipole_step'],
                                      'Phi_metal_side': v['Phi_bottom'], 'Phi_MoS2_side': v['Phi_top']}
for metal, d in reg.items():
    e0 = min(x['E_relax'] for x in d.values())
    for job, x in d.items():
        x['dE_meV'] = 1000 * (x['E_relax'] - e0)
out['registries'] = reg
low = {'Ta2Te2': 'Ta2Te2_MoS2_s1', 'MoTaTe2': 'MoTaTe2_MoS2_s0'}
for metal, job in low.items():
    x = dict(reg[metal][job]); x['job'] = job
    x['Phi_B_n_K'] = K[job]['CBM_K_minus_EF']
    r = json.loads((H / f'{job}_fig.json').read_text())
    z = np.array(r['z']); v = np.array(r['V_minus_EF']); zat = np.array(r['atoms_z']); sp = r['species']
    zTe = max(zat[i] for i, e in enumerate(sp) if e == 'Te'); zS = min(zat[i] for i, e in enumerate(sp) if e == 'S')
    m = (z > zTe) & (z < zS)
    x['tunnel_height'] = float(v[m].max()); x['tunnel_width'] = float((v[m] > 0).sum() * (z[1] - z[0]))
    x['Te_S_gap'] = float(zS - zTe)
    x['Q_gap_e_per_cell'] = Q[job]['Q_mid_e_per_cell']; x['Q_gap_e_per_MoS2'] = Q[job]['Q_mid_e_per_cell'] / sp.count('S') * 2
    x['Q_gap_e_per_cm2'] = Q[job]['Q_mid_e_per_cell'] / (Q[job]['area_A2'] * 1e-16)
    out[metal] = x
drop = out['Ta2Te2']['Phi_B_n_K'] - out['MoTaTe2']['Phi_B_n_K']
out['summary'] = {'barrier_drop_K': drop, 'S_K': drop / dphi,
                  'S_core': (out['Ta2Te2']['Phi_B_n_core'] - out['MoTaTe2']['Phi_B_n_core']) / dphi,
                  'S_vac': (out['Ta2Te2']['Phi_B_n_vac'] - out['MoTaTe2']['Phi_B_n_vac']) / dphi}
(H / 'interface_results.json').write_text(json.dumps(out, indent=1))
for k in ('Ta2Te2', 'MoTaTe2'):
    print(k, {a: round(b, 3) for a, b in out[k].items() if isinstance(b, float)})
print('dPhi', round(dphi, 3), {a: round(b, 3) for a, b in out['summary'].items()})
for metal, d in reg.items():
    for job, x in sorted(d.items()):
        print(job, {a: round(b, 3) for a, b in x.items()})

"""Phonon post-processing of the finite-displacement runs (Sec. S14.2, Fig. 6e,f).

  dispersion  <out json> <PH root> NAME [NAME ...]  bands along Gamma-X-M-Gamma, DOS, minimum frequencies
              (reads <PH root>/PH_<NAME>)
  commensurate <PH dir>  frequencies at the 16 wave vectors commensurate with the 4x4x1 supercell, and
              the effect of not subtracting the residual forces at q = (0.1..0.2, 0.1..0.2)
  qpoints <out json> <PHQ root> n [n ...]  exact frequencies at q = (k/n, k/n) from the slanted supercells
              (reads <PHQ root>/PHQ_MoTaTe2_<n>)
  mode <PHQ dir>  polarization and per-atom weights of the lowest mode at q = (1/5, 1/5)
Frequencies in THz; imaginary modes are negative.
"""
import json
import sys
from pathlib import Path

import numpy as np

from materon.dft import phonons


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'dispersion':
        out, root, names = Path(args[0]), Path(args[1]), args[2:] or ['MoTaTe2', 'Ta2Te2']
        res = {}
        for name in names:
            d0 = root / f'PH_{name}'
            if not d0.exists():
                continue
            res[name] = phonons.dispersion(phonons.load(d0))
            print(name, 'min mesh', round(res[name]['min_freq_mesh_THz'], 3), 'THz; min path',
                  round(res[name]['min_freq_path_THz'], 3), 'max', round(res[name]['max_freq_THz'], 2))
        out.write_text(json.dumps(res))
    elif cmd == 'commensurate':
        d0 = Path(args[0])
        ph = phonons.load(d0)
        q, fr = phonons.commensurate_frequencies(ph, 4)
        for qq, f in zip(q, fr):
            print(qq[:2], f[:4].round(3).tolist())
        print('min commensurate (excluding Gamma acoustic)', min(f.min() for qq, f in zip(q, fr) if qq != [0, 0, 0]).round(3))
        raw = phonons.load(d0, subtract_residual=False)
        for qq in ([0.1, 0.1, 0], [0.15, 0.15, 0], [0.2, 0.2, 0]):
            print('q', qq[:2], 'sub', phonons.frequencies_at(ph, [qq])[0][:2].round(3).tolist(),
                  'raw', phonons.frequencies_at(raw, [qq])[0][:2].round(3).tolist())
        print('max |residual force| disp-000', np.abs(phonons.vasprun_forces(d0 / 'disp-000')).max().round(4))
    elif cmd == 'qpoints':
        out, root = Path(args[0]), Path(args[1])
        res = {}
        for n in map(int, args[2:]):
            ph = phonons.load(root / f'PHQ_MoTaTe2_{n}')
            q = [[k / n, k / n, 0] for k in range(1, n)]
            for qq, f in zip(q, phonons.frequencies_at(ph, q)):
                res[f'{qq[0]:.4f}'] = f.tolist()
                print(n, [round(x, 4) for x in qq[:2]], 'lowest', f[:3].round(3).tolist())
        out.write_text(json.dumps(res))
    elif cmd == 'mode':
        r = phonons.mode_weights(phonons.load(Path(args[0])), (0.2, 0.2, 0))
        print('freq', np.round(r['frequencies_THz'][:3], 3), 'weights x,y,z', np.round(r['xyz_weights'], 3),
              'per atom', [(s, round(w, 3)) for s, w in r['atom_weights']])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()

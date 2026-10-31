"""Work functions (Eq. S9) and run diagnostics of finished VASP jobs.

Layouts:
  flat    <root>/<job>/            (Al2S2 review jobs; static/fixed jobs need LOCPOT)
  nested  <root>/<system>/<stage>/ (screens, designs, stability; keys '<system>/<stage>')
  archived  explicit job directories of the archived Al2S2 B/Ga package (edge grid, PAW
            populations; Tables S31 and S33, Fig. S9b)
Directories whose name starts with '_' are skipped.

Usage: python extract_workfunctions.py --layout nested --root <dir> --out RESULTS.json
       python extract_workfunctions.py --layout archived --jobs <job dir> [<job dir> ...] --out RESULTS.json
"""
import argparse
import json
from pathlib import Path

from materon.dft.workfunction import archived_job, job


def candidates(root, layout):
    root = Path(root)
    dirs = sorted(root.glob('*/*')) if layout == 'nested' else sorted(root.iterdir())
    for d in dirs:
        if not d.is_dir() or d.name.startswith('_') or d.parent.name.startswith('_'):
            continue
        if not (d / 'OUTCAR').exists():
            continue
        if ('static' in d.name or 'fixed' in d.name) and not (d / 'LOCPOT').exists():
            continue
        yield (f'{d.parent.name}/{d.name}' if layout == 'nested' else d.name), d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', choices=['flat', 'nested', 'archived'], default='nested')
    p.add_argument('--root')
    p.add_argument('--jobs', nargs='*', default=[])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    res = {}
    if a.layout == 'archived':
        for d in map(Path, a.jobs):
            res[d.name] = archived_job(d)
            print(d.name, round(res[d.name]['mean_work_function_eV'], 6))
    else:
        for key, d in candidates(a.root, a.layout):
            try:
                res[key] = job(d)
            except Exception as ex:
                res[key] = {'error': repr(ex)}
            v = res[key]
            print(key, v.get('error') or (round(v['phi_eV'], 4), v['magnetization_muB'], v['ionic_steps'],
                                          v['converged_electronic']))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

"""Coverage of O and S chemistry in the C2DB training partition (diagnosis of the S->O miss,
Sec. III D): counts and work-function statistics of O-, S-, mixed O/S- and Al-containing training
structures, O co-elements, O-containing cells with <= 4 atoms, and same-cation S/O analogues.
Reads the prepared C2DB directory (GRAPHS.pkl.gz, SPLIT.json, *_LABELS.json); no model.

Usage: python c2db_anion_coverage.py <prepared C2DB dir> <out json>
"""
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    st, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    class Reader(pickle.Unpickler):     # NumPy 2 pickles read under NumPy 1
        def find_class(self, module, name):
            if module.startswith('numpy._core') and not hasattr(np, '_core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)

    with gzip.open(st / 'GRAPHS.pkl.gz', 'rb') as f:
        graphs = Reader(f).load()
    split = json.loads((st / 'SPLIT.json').read_text())
    lab = {}
    for part in ('train', 'validation', 'test'):
        lab.update(json.loads((st / f'{part.upper()}_LABELS.json').read_text()))
    part_of = {i: p for p in ('train', 'validation', 'test') for i in split['ids'][p]}

    def els(i):
        return set(int(z) for z in graphs[i]['z'])

    out = {}
    tr = split['ids']['train']
    groups = {'all': lambda e: True, 'O': lambda e: 8 in e, 'S': lambda e: 16 in e, 'O and S': lambda e: 8 in e and 16 in e,
              'S, no O': lambda e: 16 in e and 8 not in e, 'O, no S': lambda e: 8 in e and 16 not in e,
              'Al': lambda e: 13 in e, 'Al and O': lambda e: 13 in e and 8 in e, 'Al and S': lambda e: 13 in e and 16 in e}
    for name, fn in groups.items():
        v = [lab[i] for i in tr if fn(els(i))]
        out[name] = {'n_train': len(v), 'mean_phi': float(np.mean(v)) if v else None,
                     'median_phi': float(np.median(v)) if v else None}
        print(f'{name:10s} n={len(v):5d}')
    co = Counter()
    for i in tr:
        e = els(i)
        if 8 in e:
            for z in e - {8}:
                co[z] += 1
    out['O_coelements_top'] = co.most_common(15)
    out['O_small_cells'] = [(i, sorted(els(i)), lab[i], part_of[i]) for i in lab if 8 in els(i) and len(graphs[i]['z']) <= 4]

    def key(i, anion):
        return tuple(sorted(els(i) - {anion}))

    s_by, o_by = {}, {}
    for i in lab:
        e = els(i)
        if 16 in e and 8 not in e:
            s_by.setdefault(key(i, 16), []).append(i)
        if 8 in e and 16 not in e:
            o_by.setdefault(key(i, 8), []).append(i)
    pairs = [{'cations': k, 'phi_S': float(np.mean([lab[i] for i in s_by[k]])), 'phi_O': float(np.mean([lab[i] for i in o_by[k]])),
              'n_S': len(s_by[k]), 'n_O': len(o_by[k])} for k in set(s_by) & set(o_by)]
    d = [p['phi_O'] - p['phi_S'] for p in pairs]
    out['S_vs_O_same_cations'] = {'n_pairs': len(pairs), 'mean_phiO_minus_phiS': float(np.mean(d)) if d else None,
                                  'n_O_higher': int(sum(x > 0 for x in d)), 'pairs': pairs}
    out_path.write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()

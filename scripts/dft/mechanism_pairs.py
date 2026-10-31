"""Per-pair mechanism descriptors (Sec. S12.2, Table S36): dPhi, chalcogen- and host-core level
shifts, E_F shifts relative to the cores, effective doping Delta N_eff (rigid-band inversion in the
parent DOS), slab-thickness change, substituent height offset and magnetizations.

Input: JSON list of [parent static dir, candidate static dir, host element, substituent element].
Usage: python mechanism_pairs.py <pairs json> <out json>
"""
import json
import sys
from pathlib import Path

from materon.dft.mechanism import pair_descriptors


def main():
    pairs = json.loads(Path(sys.argv[1]).read_text())
    res = []
    for pd_, qd_, host, sub in pairs:
        rec = {'parent': Path(pd_).parent.name, 'candidate': Path(qd_).parent.name, 'host': host, 'sub': sub}
        try:
            rec.update(pair_descriptors(pd_, qd_, host, sub))
        except Exception as ex:
            rec['error'] = repr(ex)
        res.append(rec)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
    for r in res:
        print(r['candidate'][:12], r.get('error') or ' '.join(
            f'{k} {v:+.2f}' for k, v in r.items() if isinstance(v, float)))


if __name__ == '__main__':
    main()

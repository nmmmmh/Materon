"""Write and hash the band-filling mechanism hypotheses before any potential decomposition or
round-1 result is read (Sec. S12.1). Writes MECHANISM_PREREGISTERED.json and its SHA-256.

Usage: python lock_mechanism.py <out dir>
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

from materon.dft.mechanism import MECHANISM_LOCK


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    lock = {'locked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **MECHANISM_LOCK}
    text = json.dumps(lock, indent=1)
    (out / 'MECHANISM_PREREGISTERED.json').write_text(text)
    (out / 'MECHANISM_PREREGISTERED.sha256').write_text(hashlib.sha256(text.encode()).hexdigest() + '\n')
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()

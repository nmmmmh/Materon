"""Regenerate every figure of the Letter and the Supporting Information into ./output (plotting only).

Main text: fig1_jpcl.py ... fig4_jpcl.py and toc_jpcl.py. Supporting Information: the scripts listed in SI_MAP,
whose outputs are copied to the SI figure numbers after plotting (README.md)."""
import os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
MAIN = ['fig1_jpcl.py', 'fig2_jpcl.py', 'fig3_jpcl.py', 'fig4_jpcl.py', 'toc_jpcl.py']
SI = ['figS1_S4_S12.py', 'fig2.py', 'figS11.py', 'figS3.py', 'figS6_S7.py', 'figS14.py', 'figS10.py', 'fig5.py',
      'fig6.py', 'figS11_jpcl.py', 'figS8.py', 'figS2_S9.py', 'figS13.py']
# SI figure number <- file written by the plotting script (Figure S9 uses panels a and b of Figure_5)
SI_MAP = {'S1': 'Figure_S1', 'S2': 'Figure_2', 'S3': 'Figure_S11', 'S4': 'Figure_S3', 'S5': 'Figure_S12',
          'S6': 'Figure_S6', 'S7': 'Figure_S14', 'S8': 'Figure_S10', 'S9': 'Figure_5', 'S10': 'Figure_6',
          'S11': 'Figure_SI_11', 'A1': 'Figure_S8', 'A2': 'Figure_S2', 'A3': 'Figure_S7', 'A4': 'Figure_S9',
          'A5': 'Figure_S13'}
env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
for s in MAIN + SI:
    t = time.time()
    r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, env=env, capture_output=True, text=True, encoding='utf-8')
    out = (r.stdout + r.stderr).strip()
    print(f'{s:<18} {"ok" if r.returncode == 0 else "FAILED"}  {time.time() - t:5.1f} s' + (f'\n    {out}' if out else ''))
    if r.returncode:
        sys.exit(r.returncode)
for number, stem in SI_MAP.items():
    for ext in ('png', 'pdf'):
        src = OUT / f'{stem}.{ext}'
        if src.exists() and stem != f'Figure_SI_{number}':
            shutil.copyfile(src, OUT / f'Figure_SI_{number}.{ext}')
print('figures written to', OUT)

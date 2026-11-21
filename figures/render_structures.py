"""Optional: re-render the three OVITO structure images used as figure inputs (data/structures/structure_*.png).
The figure scripts read the stored PNGs, so OVITO is not needed to redraw the figures. Ray-traced output can differ
slightly between OVITO versions and machines; the stored PNGs are the ones used in the paper.

Usage: python render_structures.py [OUTDIR]          (default OUTDIR: data/structures)
The OVITO interpreter is taken from the environment variable OVITO_PYTHON (default: the current interpreter)."""
import os, sys, subprocess
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
S = HERE / 'data' / 'structures'
OUTDIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else S
OUTDIR.mkdir(parents=True, exist_ok=True)
OVITO = os.environ.get('OVITO_PYTHON', sys.executable)
# (POSCAR, output, nx, ny, environment, crop to the non-transparent bounding box)
JOBS = [
    ('Al2S2_parent.vasp', 'structure_Al2S2.png', 2, 2,           # Fig. S13(a); OV_MARK also writes structure_Al2S2.json
     {'OV_EDGE': '1', 'OV_DIR': '-0.35,0.55,-1.0', 'OV_SIZE': '1400,1200', 'OV_MARK': '0,0,0;1,0,0'}, False),
    ('Ta2Te2_3x3_W.vasp', 'structure_Ta2Te2_W.png', 1, 1, {}, True),               # Fig. 5(d) inset
    ('MoTaTe2_relaxed.vasp', 'structure_MoTaTe2_cell.png', 4, 4,                    # Fig. 6(a)
     {'OV_CELL': '1', 'OV_DIR': '0.45,1.0,-0.35', 'OV_SIZE': '1600,1500'}, True),
]
for poscar, png, nx, ny, env, crop in JOBS:
    out = OUTDIR / png
    subprocess.run([OVITO, str(HERE / 'render_structure_ovito.py'), str(S / poscar), str(out), str(nx), str(ny)],
                   env={**os.environ, **env}, check=True)
    if crop:
        im = Image.open(out); im.crop(im.getbbox()).save(out)
    print('rendered', out)

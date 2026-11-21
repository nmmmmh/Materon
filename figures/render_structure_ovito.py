"""Ray-traced structure image (OVITO Tachyon renderer, ambient occlusion) of a relaxed monolayer, for figure panels.
A finite nx x ny cluster is built from the POSCAR so that no bonds cross periodic images; metal-Te bonds only, half-colored
by atom; transparent background. Run with the OVITO environment:
python render_structure_ovito.py <POSCAR> <out.png> [nx ny]"""
import os, sys, tempfile
import numpy as np
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier
from ovito.vis import Viewport, TachyonRenderer, BondsVis

src, out = sys.argv[1], sys.argv[2]
nx, ny = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (3, 3)
COL = {'Mo': (0.18, 0.55, 0.45), 'Ta': (0.55, 0.60, 0.67), 'Te': (0.83, 0.65, 0.24), 'W': (0.09, 0.22, 0.29),
       'Al': (0.55, 0.60, 0.67), 'S': (0.83, 0.65, 0.24)}
RAD = {'Mo': 0.62, 'Ta': 0.62, 'W': 0.62, 'Te': 0.55, 'Al': 0.55, 'S': 0.48}

L = open(src).read().splitlines(); sc = float(L[1])
lat = np.array([[float(v) for v in L[i].split()] for i in (2, 3, 4)]) * sc
els = [e for e, n in zip(L[5].split(), map(int, L[6].split())) for _ in range(n)]
frac = np.array([[float(v) for v in L[8 + i].split()[:3]] for i in range(len(els))])
xyz = frac @ lat
CELL = os.environ.get('OV_CELL') == '1'   # draw the nx x ny x 1 computational cell (with its vacuum) and boundary atoms
EDGE = CELL or os.environ.get('OV_EDGE') == '1'   # boundary atoms without the cell box
MARK = [tuple(map(int, t.split(','))) for t in os.environ.get('OV_MARK', '').split(';') if t]   # (site, i, j) to locate in the image
atoms, where = [], {}
for i in range(nx + EDGE):
    for j in range(ny + EDGE):
        for k, (e, fr) in enumerate(zip(els, frac)):
            u, v = fr[0] % 1 + i, fr[1] % 1 + j
            if u <= nx + 1e-6 and v <= ny + 1e-6 and (EDGE or (i < nx and j < ny)):
                where[(k, i, j)] = len(atoms); atoms.append((e, u * lat[0] + v * lat[1] + (fr[2] % 1) * lat[2]))
tmp = os.path.join(tempfile.gettempdir(), 'ovito_cluster.xyz')
box = ' '.join(f'{x:.6f}' for x in np.concatenate([nx * lat[0], ny * lat[1], lat[2]]))
with open(tmp, 'w') as f:
    f.write(f'{len(atoms)}\nLattice="{box}" pbc="F F F" Properties=species:S:1:pos:R:3\n')
    for e, p in atoms: f.write(f'{e} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n')

pipe = import_file(tmp)
bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
for m in ('Mo', 'Ta', 'W'):
    bonds.set_pairwise_cutoff(m, 'Te', 3.1)
bonds.set_pairwise_cutoff('Al', 'S', 2.8)
bonds.vis.width = 0.22
bonds.vis.coloring_mode = BondsVis.ColoringMode.ByParticle
pipe.modifiers.append(bonds)
types = pipe.source.data.particles_.particle_types_
for k in range(len(types.types)):
    t = types.make_mutable(types.types[k])
    if t.name in COL:
        t.color = COL[t.name]; t.radius = RAD[t.name]
cv = pipe.source.data.cell_.vis
cv.enabled = CELL
cv.rendering_color = (0.25, 0.25, 0.25); cv.line_width = float(os.environ.get('OV_LW', '0.05'))
pipe.add_to_scene()
ortho = os.environ.get('OV_ORTHO') == '1'
vp = Viewport(type=Viewport.Type.Ortho if ortho else Viewport.Type.Perspective)
vp.camera_dir = tuple(float(v) for v in os.environ.get('OV_DIR', '0.35,0.8,-1.0').split(','))
if not ortho: vp.fov = 0.32
size = tuple(int(v) for v in os.environ.get('OV_SIZE', '1600,1200').split(','))
vp.zoom_all(size=size)
REN = TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.75, shadows=False,
                      antialiasing_samples=24, direct_light_intensity=0.95)
vp.render_image(filename=out, size=size, alpha=True, renderer=REN)
print('saved', out)
if MARK:   # locate marked atoms: re-render with each one recoloured and take the centroid of the changed pixels
    import json
    from PySide6.QtGui import QImage
    from ovito.modifiers import ExpressionSelectionModifier, AssignColorModifier

    def grab():
        q = vp.render_image(size=size, alpha=True, renderer=REN).convertToFormat(QImage.Format.Format_RGBA8888)
        return np.array(q.constBits(), dtype=np.uint8).reshape(q.height(), q.bytesPerLine())[:, :4 * q.width()].reshape(q.height(), q.width(), 4).astype(int)
    base = grab(); pos = {}
    for m in MARK:
        pipe.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIndex == {where[m]}'))
        pipe.modifiers.append(AssignColorModifier(color=(1, 0, 1)))
        img = grab(); pipe.modifiers.pop(); pipe.modifiers.pop()
        yy, xx = np.nonzero(np.abs(img[..., :3] - base[..., :3]).sum(-1) > 120)
        pos['%d,%d,%d' % m] = [float(xx.mean()), float(yy.mean()), float((xx.max() - xx.min() + yy.max() - yy.min()) / 4)]
    Path = __import__('pathlib').Path; Path(out).with_suffix('.json').write_text(json.dumps({'size': size, 'marks': pos}))
    print(pos)

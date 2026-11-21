"""MoTaTe2/MoS2 and Ta2Te2/MoS2 interfaces (74 atoms) plus isolated layers in the same cell.
x: long axis = 7 a(MoS2) = 4*sqrt2 a(metal); y: short axis = sqrt3 a(MoS2) = sqrt2 a(metal). MoS2 unstrained."""
import numpy as np, os, sys
OUT = sys.argv[1]
aM = 3.182; dSS = 3.13                       # MoS2 lattice and S-S height (PBE)
LX, LY, C = 7 * aM, np.sqrt(3) * aM, 36.0

def metal_layer(a, top, bot, mtop):
    # square cell: M1 (0,0), M2 (.5,.5) in plane; X at (0,.5) below, (.5,0) above
    base = [(mtop[0], 0, 0, 0.0), (mtop[1], .5, .5, 0.0), ('Te', 0, .5, -bot), ('Te', .5, 0, top)]
    e1 = np.array([1, 1]) / np.sqrt(2); e2 = np.array([1, -1]) / np.sqrt(2)
    Lx0, Ly0 = 4 * np.sqrt(2) * a, np.sqrt(2) * a
    atoms = []
    for i in range(-8, 9):
        for j in range(-8, 9):
            for el, fx, fy, z in base:
                r = np.array([(i + fx) * a, (j + fy) * a])
                x, y = r @ e1, r @ e2
                u, v = (x / Lx0) % 1.0, (y / Ly0) % 1.0
                atoms.append((el, round(u, 6) % 1.0, round(v, 6) % 1.0, z))
    uniq = {}
    for el, u, v, z in atoms:
        uniq[(el, round(u, 4) % 1.0, round(v, 4) % 1.0, round(z, 4))] = (el, u, v, z)
    at = list(uniq.values())
    assert len(at) == 32, len(at)
    return at, Lx0, Ly0

def mos2_layer(shift_y):
    at = []
    for i in range(7):
        for (el, fx, fy, z) in (('Mo', 0, 0, 0), ('Mo', .5, .5, 0), ('S', .5, 1 / 6, -dSS / 2), ('S', .5, 1 / 6, dSS / 2),
                                ('S', 0, 2 / 3, -dSS / 2), ('S', 0, 2 / 3, dSS / 2)):
            at.append((el, (i + fx) / 7, (fy + shift_y) % 1.0, z))
    return at

def write(path, atoms, comment):
    order = ['Mo', 'Ta', 'Te', 'S']
    els = [e for e in order if any(a[0] == e for a in atoms)]
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'POSCAR'), 'w') as f:
        f.write(comment + '\n1.0\n')
        f.write(f'{LX:.10f} 0 0\n0 {LY:.10f} 0\n0 0 {C:.4f}\n')
        f.write(' '.join(els) + '\n' + ' '.join(str(sum(a[0] == e for a in atoms)) for e in els) + '\nDirect\n')
        for e in els:
            for a in atoms:
                if a[0] == e: f.write(f'{a[1]:.8f} {a[2]:.8f} {a[3] / C:.8f}\n')
    return els

# relaxed monolayer geometries: MoTaTe2 (a 3.8954, Te heights +/-2.0266 A); Ta2Te2 (a 3.9373, same heights as start)
systems = {'MoTaTe2': (3.8953922, 2.02653, ('Mo', 'Ta')), 'Ta2Te2': (3.937287, 2.02653, ('Ta', 'Ta'))}
zM, gap = 10.0, 3.40                          # metal plane height; Te(top)-S(bottom) vertical gap
info = []
for name, (a, h, ms) in systems.items():
    lay, Lx0, Ly0 = metal_layer(a, h, h, ms)
    sx, sy = LX / Lx0 - 1, LY / Ly0 - 1
    info.append(f'{name}: strain x {sx*100:+.2f}% y {sy*100:+.2f}%')
    met = [(e, u, v, zM + z) for e, u, v, z in lay]
    zMo = zM + h + gap + dSS / 2
    els = write(f'{OUT}/{name}_iso', met, f'{name} strained to MoS2 7x1 rect cell')
    for k, s in enumerate((0.0, 0.25, 0.5)):
        mos = [(e, u, v, zMo + z) for e, u, v, z in mos2_layer(s)]
        write(f'{OUT}/{name}_MoS2_s{k}', met + mos, f'{name}/MoS2 shift_y {s}')
write(f'{OUT}/MoS2_iso', [(e, u, v, 17.0 + z) for e, u, v, z in mos2_layer(0.0)], 'MoS2 7x1 rect')
print('\n'.join(info)); print('cell', LX, LY, C)

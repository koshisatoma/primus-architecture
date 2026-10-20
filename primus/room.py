import random

_rng = random.Random(7)

FURNITURE = [
    ('table',  'box', (0.9, 0.5, 0.03), (0.0, 0.9, 0.75)),
    ('chair1', 'box', (0.22, 0.22, 0.02), (0.6, 0.9, 0.45)),
    ('chair2', 'box', (0.22, 0.22, 0.02), (-0.6, 0.9, 0.45)),
    ('shelf',  'box', (0.8, 0.15, 0.02), (1.6, 0.2, 1.2)),
    ('counter','box', (1.2, 0.3, 0.04), (-1.5, 0.1, 0.9)),
    ('sofa',   'box', (0.9, 0.4, 0.2), (0.0, -1.8, 0.3)),
]

ITEMS = ['cup', 'bowl', 'plate', 'mug', 'bottle', 'can', 'apple', 'orange',
         'book', 'box', 'pan', 'spoon', 'knife', 'fork', 'glass', 'jar',
         'lemon', 'onion', 'pot', 'kettle', 'remote', 'phone', 'pen', 'towel']

SHAPES = {'cup': ('cyl', 0.04, 0.09), 'mug': ('cyl', 0.04, 0.10),
          'bowl': ('sph', 0.06, 0.05), 'plate': ('cyl', 0.11, 0.015),
          'bottle': ('cyl', 0.035, 0.22), 'can': ('cyl', 0.033, 0.12),
          'apple': ('sph', 0.04, 0.04), 'orange': ('sph', 0.04, 0.04),
          'lemon': ('sph', 0.03, 0.03), 'onion': ('sph', 0.04, 0.04),
          'book': ('box', 0.11, 0.03), 'box': ('box', 0.09, 0.09),
          'pan': ('cyl', 0.12, 0.05), 'glass': ('cyl', 0.035, 0.12),
          'jar': ('cyl', 0.05, 0.14), 'pot': ('cyl', 0.10, 0.12),
          'kettle': ('cyl', 0.08, 0.16), 'remote': ('box', 0.09, 0.02),
          'phone': ('box', 0.07, 0.01), 'pen': ('cyl', 0.005, 0.14),
          'spoon': ('box', 0.02, 0.01), 'knife': ('box', 0.02, 0.01),
          'fork': ('box', 0.02, 0.01), 'towel': ('box', 0.15, 0.02)}

SURFACES = [(0.0, 0.9, 0.78, 0.8, 0.4), (-1.5, 0.1, 0.94, 1.1, 0.25),
            (1.6, 0.2, 1.22, 0.7, 0.12)]


def room_xml():
    x = []
    for name, kind, size, pos in FURNITURE:
        unique_name = f'room_{name}'
        x.append(f'<body name="{unique_name}" pos="{pos[0]} {pos[1]} {pos[2]}">'
                 f'<geom type="{kind}" size="{" ".join(map(str, size))}" '
                 f'rgba="0.4 0.3 0.25 1" mass="5"/></body>')
    n = 0
    for i, item in enumerate(ITEMS):
        for copy in range(2 if i < 8 else 1):
            kind, r, h = SHAPES[item]
            sx, sy, sz, hx, hy = _rng.choice(SURFACES)
            px = sx + _rng.uniform(-hx * 0.8, hx * 0.8)
            py = sy + _rng.uniform(-hy * 0.8, hy * 0.8)
            pz = sz + h
            nm = f"room_{item}{copy}"
            free = kind != 'box' or _rng.random() < 0.5
            if kind == 'cyl':
                g = f'<geom type="cylinder" size="{r} {h}" mass="0.2"/>'
            elif kind == 'sph':
                g = f'<geom type="sphere" size="{r}" mass="0.15"/>'
            else:
                g = f'<geom type="box" size="{r} {r * 0.7} {h}" mass="0.2"/>'
            if free:
                x.append(f'<body name="{nm}" pos="{px:.2f} {py:.2f} {pz:.2f}">{g}<joint type="free" /></body>')
            else:
                x.append(f'<body name="{nm}" pos="{px:.2f} {py:.2f} {pz:.2f}">{g}</body>')
            n += 1
    return '\n'.join(x), n

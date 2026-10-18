from __future__ import annotations
import numpy as np

def tps(source, target, vertices):
    source, target, vertices = map(lambda a: np.asarray(a, float), (source, target, vertices))
    scale = np.ptp(source, axis=0).max()
    a = source / scale
    p = np.column_stack((np.ones(len(a)), a))
    k = np.linalg.norm(a[:, None] - a[None], axis=2)
    system = np.block([[k + np.eye(len(a)) * 1e-09, p], [p.T, np.zeros((4, 4))]])
    solution = np.linalg.solve(system, np.vstack((target, np.zeros((4, 3)))))
    result = []
    for v in np.array_split(vertices / scale, max(1, len(vertices) // 2000)):
        basis = np.column_stack((np.linalg.norm(v[:, None] - a[None], axis=2), np.ones(len(v)), v))
        result.append(basis @ solution)
    return np.concatenate(result)

def segment_projection(vertices, a, b):
    v, a, b = map(lambda x: np.asarray(x, float), (vertices, a, b))
    d = b - a
    t = np.clip((v - a) @ d / max(float(d @ d), 1e-15), 0, 1)
    q = a + t[:, None] * d
    return (q, np.linalg.norm(v - q, axis=1), t)

def fit_hip_from_leg(knee, ankle, hip_z):
    k, a = (np.asarray(knee, float), np.asarray(ankle, float))
    slope = (k[0] - a[0]) / max(k[2] - a[2], 1e-09)
    x = k[0] + np.clip(slope, -0.15, 0.15) * (hip_z - k[2]) * 0.5
    return float(x)

def validate_input(data):
    h = data.get('height_cm')
    if isinstance(h, bool) or not isinstance(h, (int, float)) or (not np.isfinite(h)) or (h <= 0):
        raise ValueError('Mandatory finite positive height_cm')
    for name in ('source', 'mpfb_base', 'out'):
        if not data.get(name):
            raise ValueError('Missing ' + name)
        if 'private' in str(data[name]).replace('\\', '/').lower().split('/'):
            raise ValueError('Private input/output requires separate review')
    if data.get('character_id') and str(data['out']).replace('\\', '/').rstrip('/').split('/')[-1] != data['character_id']:
        raise ValueError('Output leaf must match character_id')
    return data

def voxel_surface(cells, pitch):
    cells = set(map(tuple, cells))
    corners = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    candidates = {tuple((c[k] - o[k] for k in range(3))) for c in cells for o in corners}
    vertices = []
    triangles = []
    lookup = {}
    for c in sorted(candidates):
        grid = corners + np.array(c)
        inside = np.array([tuple(q) in cells for q in grid])
        if inside.all() or not inside.any():
            continue
        for tet in tets:
            pos = [i for i in tet if inside[i]]
            neg = [i for i in tet if not inside[i]]
            if not pos or not neg:
                continue
            if len(pos) == 1:
                edges = [(pos[0], j) for j in neg]
            elif len(neg) == 1:
                edges = [(i, neg[0]) for i in pos]
            else:
                edges = [(pos[0], neg[0]), (pos[0], neg[1]), (pos[1], neg[1]), (pos[1], neg[0])]
            ids = []
            for i, j in edges:
                key = tuple(sorted((tuple(grid[i]), tuple(grid[j]))))
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append((grid[i] + grid[j] + 1) * pitch / 2)
                ids.append(lookup[key])
            normal = grid[neg].mean(0) - grid[pos].mean(0)
            for face in ([ids[0], ids[1], ids[2]],) if len(ids) == 3 else ([ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]):
                a, b, d = (vertices[i] for i in face)
                if np.dot(np.cross(b - a, d - a), normal) < 0:
                    face = [face[0], face[2], face[1]]
                triangles.append(face)
    return (np.array(vertices), triangles)

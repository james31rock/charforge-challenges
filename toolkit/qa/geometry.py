import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def tree(vertices, triangles):
    if not len(triangles):
        return None
    return BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)

def closed_topology(v, t):
    _, inv = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    q = inv[t]
    e = np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))

class VolumeSurface:

    def __init__(self, v, t):
        _, ids, inv = np.unique(np.round(v, 6), axis=0, return_index=True, return_inverse=True)
        self.ids = ids
        tri = inv[t]
        tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])]
        _, unique = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
        tri = tri[np.sort(unique)]
        directed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        unordered = np.sort(directed, axis=1)
        _, first, counts = np.unique(unordered, axis=0, return_index=True, return_counts=True)
        self.manifold = not np.any(counts > 2)
        boundary = directed[first[counts == 1]]
        outgoing = {}
        incoming = {}
        for a, b in boundary:
            outgoing.setdefault(int(a), []).append(int(b))
            incoming.setdefault(int(b), []).append(int(a))
        self.loops = []
        extra = []
        pending = set(outgoing)
        self.manifold &= all((len(x) == 1 for x in outgoing.values())) and all((len(x) == 1 for x in incoming.values())) and (set(outgoing) == set(incoming))
        if self.manifold:
            while pending:
                start = min(pending)
                loop = []
                cur = start
                while cur not in loop:
                    loop.append(cur)
                    pending.discard(cur)
                    cur = outgoing[cur][0]
                if cur != start or len(loop) < 3:
                    self.manifold = False
                    break
                centre = len(ids) + len(self.loops)
                self.loops.append(loop)
                extra.extend(((b, a, centre) for a, b in zip(loop, loop[1:] + loop[:1])))
        self.tri = np.vstack([tri, np.asarray(extra, dtype=int).reshape(-1, 3)])
        self.capped = bool(self.loops)
        self.valid = bool(self.manifold and len(tri))

    def vertices(self, posed):
        v = posed[self.ids]
        return np.vstack([v, [v[loop].mean(0) for loop in self.loops]]) if self.loops else v

    def bvh(self, posed):
        return tree(self.vertices(posed), self.tri) if self.valid else None

def penetration(points, surface, max_distance, closed=True):
    if surface is None or not len(points):
        return (None, None)
    best = 0.0
    witness = None
    directions = [Vector((1, 0.173, 0.367)).normalized(), Vector((0.213, 1, 0.417)).normalized(), Vector((0.371, 0.197, 1)).normalized()]
    for index, p in enumerate(points):
        origin = Vector(p)
        hit, normal, face, distance = surface.find_nearest(origin, max_distance)
        if hit is None or distance < 1e-07:
            continue
        if not closed and normal.dot(origin - hit) >= -1e-07:
            continue
        votes = 0
        for direction in directions:
            q = origin.copy()
            hits = 0
            for _ in range(64):
                loc, _, _, _ = surface.ray_cast(q, direction)
                if loc is None:
                    break
                hits += 1
                q = loc + direction * 1e-06
            votes += hits % 2
            if direction == directions[1] and votes == 0:
                break
        if votes >= 2 and distance > best:
            best = float(distance)
            witness = dict(probe=int(index), point=list(p), surface=list(hit), votes=votes)
    return (best, witness)

def edge_strain(rest, pose, triangles):
    if not len(triangles):
        return None
    e = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    a = np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1)
    b = np.linalg.norm(pose[e[:, 0]] - pose[e[:, 1]], axis=1)
    valid = a > 1e-06
    return float(np.quantile(b[valid] / a[valid], 0.99)) if valid.any() else None

def surface_crossing(segments, surface, tolerance):
    best = 0.0
    witness = None
    if surface is None:
        return (best, witness)
    for index, (a, b) in enumerate(segments):
        delta = Vector(b - a)
        length = delta.length
        if length <= 2 * tolerance:
            continue
        direction = delta / length
        hit, normal, face, distance = surface.ray_cast(Vector(a) + direction * tolerance, direction, length - 2 * tolerance)
        if hit is None or abs(normal.dot(direction)) < 1e-05:
            continue
        extent = min(distance + tolerance, length - distance - tolerance)
        if extent > max(best, tolerance):
            best = float(extent)
            witness = dict(edge=int(index), point=list(hit), extent_m=best)
    return (best, witness)

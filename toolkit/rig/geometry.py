import numpy as np
import heapq
from toolkit.rig.math_core import vec, unit, perpendicular
from toolkit.rig.silhouette import branch_distance

def connected_components(n, triangles):
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for row in triangles:
        a = root(row[0])
        for j in row[1:]:
            b = root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
                a = min(a, b)
    labels = np.array([root(i) for i in range(n)])
    return [np.flatnonzero(labels == g) for g in np.unique(labels)]

def hull(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    hi = []
    for p in pts:
        while len(lo) > 1 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) > 1 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])

def section_loops(vertices, triangles, point, normal):
    p = vec(point)
    n = unit(normal)
    tri = vertices[triangles]
    dots = (tri - p) @ n
    sel = (dots.min(axis=1) <= 0) & (dots.max(axis=1) > 0)
    tri = tri[sel]
    dots = dots[sel]
    hits = {}
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        ok = (dots[:, a] * dots[:, b] <= 0) & (abs(dots[:, a] - dots[:, b]) > 1e-12)
        t = dots[ok, a] / (dots[ok, a] - dots[ok, b])
        for i, q in zip(np.flatnonzero(ok), tri[ok, a] + t[:, None] * (tri[ok, b] - tri[ok, a])):
            hits.setdefault(int(i), []).append(q)
    eps = max(float(np.ptp(vertices, axis=0).max()) * 1e-06, 1e-10)
    coords = []
    lookup = {}
    edges = []
    for points in hits.values():
        indices = []
        for q in points:
            key = tuple(np.round(q / eps).astype(np.int64))
            if key not in lookup:
                lookup[key] = len(coords)
                coords.append(q)
            indices.append(lookup[key])
        if len(set(indices)) >= 2:
            edges.append([indices[0], indices[-1], indices[-1]])
    if not coords:
        return []
    q = np.array(coords)
    groups = connected_components(len(q), np.array(edges, dtype=int))
    u = perpendicular(n, [1, 0, 0])
    v = np.cross(n, u)
    results = []
    for idx in groups:
        if len(idx) < 6:
            continue
        cloud = q[idx]
        xy = np.column_stack(((cloud - p) @ u, (cloud - p) @ v))
        poly = hull(xy)
        if len(poly) < 3:
            continue
        nxt = np.roll(poly, -1, axis=0)
        cross = poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1]
        area = cross.sum() / 2
        if abs(area) < 1e-12:
            continue
        c = ((poly + nxt) * cross[:, None]).sum(axis=0) / (6 * area)
        centre = p + c[0] * u + c[1] * v
        edge = nxt - poly
        inside = np.all(edge[:, 0] * -poly[:, 1] - edge[:, 1] * -poly[:, 0] >= -eps ** 2)
        results.append(dict(centre=centre, radius=float(np.sqrt(abs(area) / np.pi)), area=float(abs(area)), outline=p + poly[:, 0, None] * u + poly[:, 1, None] * v, widths=np.ptp(poly, axis=0), support_points=len(cloud), contains_seed=bool(inside), distance=float(np.linalg.norm(centre - p)), extent=float(np.linalg.norm(cloud - p, axis=1).max()), method='connected-triangle-section-convex-centroid'))
    return results

def section(vertices, triangles, point, normal, radius):
    loops = [r for r in section_loops(vertices, triangles, point, normal) if r['distance'] < radius and r['radius'] < radius]
    enclosed = [r for r in loops if r['contains_seed']]
    return min(enclosed, key=lambda r: r['area']) if enclosed else min(loops, key=lambda r: r['distance']) if loops else None

def arm_section_tracks(vertices, triangles, height, crotch, neck, min_lateral, silhouette=None):
    h = float(height)
    current = {'left': [], 'right': []}
    runs = {'left': [], 'right': []}
    for z in np.linspace(crotch + 0.03 * h, neck - 0.015 * h, 60):
        loops = section_loops(vertices, triangles, [0, np.median(vertices[:, 1]), z], [0, 0, 1])
        for side, sign in [('left', 1), ('right', -1)]:
            choices = [r for r in loops if not r['contains_seed'] and sign * r['centre'][0] > min_lateral and (0.004 * h < r['radius'] < 0.1 * h)]
            supported = [r for r in choices if (branch_distance(silhouette, z / h, r['centre'][0] / h, 'arm') or 0) < 0.04]
            if supported:
                choices = supported
            previous = current[side][-1] if current[side] else None
            close = [r for r in choices if previous is None or np.linalg.norm(r['centre'][:2] - previous[:2]) < 0.04 * h]
            if previous is not None and (not close):
                runs[side].append(current[side])
                current[side] = []
                previous = None
                close = choices
            if close:
                chosen = min(close, key=lambda r: np.linalg.norm(r['centre'][:2] - previous[:2])) if previous is not None else max(close, key=lambda r: sign * r['centre'][0])
                current[side].append(chosen['centre'])
    result = {}
    for side in current:
        runs[side].append(current[side])
        candidates = [np.array(r) for r in runs[side] if len(r) >= 5 and r[-1][2] - r[0][2] > 0.055 * h]
        if candidates:
            result[side] = max(candidates, key=lambda r: r[-1, 2] - r[0, 2])
    return result

def depth_interval(vertices, triangles, point):
    p = vec(point)
    tri = vertices[triangles]
    xz = tri[:, :, [0, 2]]
    a = xz[:, 1] - xz[:, 0]
    b = xz[:, 2] - xz[:, 0]
    q = p[[0, 2]] - xz[:, 0]
    den = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    valid = abs(den) > 1e-12
    u = np.zeros(len(tri))
    v = np.zeros(len(tri))
    u[valid] = (q[valid, 0] * b[valid, 1] - q[valid, 1] * b[valid, 0]) / den[valid]
    v[valid] = (a[valid, 0] * q[valid, 1] - a[valid, 1] * q[valid, 0]) / den[valid]
    ok = valid & (u >= -1e-09) & (v >= -1e-09) & (u + v <= 1 + 1e-09)
    hits = np.sort(tri[ok, 0, 1] + u[ok] * (tri[ok, 1, 1] - tri[ok, 0, 1]) + v[ok] * (tri[ok, 2, 1] - tri[ok, 0, 1]))
    eps = max(float(np.ptp(vertices, axis=0).max()) * 1e-06, 1e-10)
    hits = hits[np.r_[True, np.diff(hits) > eps]] if len(hits) else hits
    if len(hits) < 2 or len(hits) % 2:
        return None
    pairs = hits.reshape(-1, 2)
    pairs = pairs[np.diff(pairs, axis=1)[:, 0] > eps]
    if not len(pairs):
        return None
    pair = min(pairs, key=lambda x: abs(float(np.mean(x)) - p[1]))
    centre = p.copy()
    centre[1] = np.mean(pair)
    return dict(centre=centre, interval=pair.tolist(), half_thickness=float((pair[1] - pair[0]) / 2))

def arm_region_masks(vertices, triangles, height, points):
    v = np.asarray(vertices)
    h = float(height)
    masks = {s: np.ones(len(v), bool) for s in ['left', 'right']}
    seeds = {s: np.zeros(len(v), bool) for s in masks}
    body_seed = np.zeros(len(v), bool)
    girth = {s: [] for s in masks}
    lo = min((points[s + 'Hand'][2] for s in masks)) - 0.05 * h
    hi = max((points[s + 'Shoulder'][2] for s in masks)) - 0.035 * h
    if hi <= lo:
        return (masks, dict(constrained_vertices=0, sections=0))
    bounds = np.linspace(lo, hi, 55)
    sections = 0
    changed = np.zeros(len(v), bool)

    def inside_loop(loop, indices):
        poly = loop['outline'][:, [0, 1]]
        edge = np.roll(poly, -1, axis=0) - poly
        delta = v[indices, None, :2] - poly[None, :, :]
        cross = edge[None, :, 0] * delta[:, :, 1] - edge[None, :, 1] * delta[:, :, 0]
        margin = 0.006 * h * np.linalg.norm(edge, axis=1)
        return np.all(cross >= -margin, axis=1) | np.all(cross <= margin, axis=1)
    for z0, z1 in zip(bounds[:-1], bounds[1:]):
        indices = np.flatnonzero((v[:, 2] >= z0) & (v[:, 2] < z1))
        if not len(indices):
            continue
        loops = section_loops(v, triangles, [0, np.median(v[:, 1]), (z0 + z1) / 2], [0, 0, 1])
        central = [r for r in loops if r['contains_seed'] and r['radius'] > 0.035 * h]
        hip_height = min((points[s + 'Hip'][2] for s in masks))
        leg_pair = all((any((sign * r['centre'][0] > 0 and abs(r['centre'][0]) < 1.25 * abs(points[side + 'Hip'][0]) for r in loops)) for side, sign in [('left', 1), ('right', -1)]))
        if not central and (not (leg_pair and z1 < hip_height + 0.05 * h)):
            continue
        body_loops = central or [r for r in loops if abs(r['centre'][0]) < 1.25 * max((abs(points[s + 'Hip'][0]) for s in masks))]
        separated = 0
        for side, sign in [('left', 1), ('right', -1)]:
            hand = np.asarray(points[side + 'Hand'])
            shoulder = np.asarray(points[side + 'Shoulder'])
            t = np.clip(((z0 + z1) / 2 - hand[2]) / (shoulder[2] - hand[2]), 0, 1)
            expected = abs(hand[0] * (1 - t) + shoulder[0] * t)
            lateral = max(abs(points[side + 'Hip'][0]) * 1.15, 0.09 * h, expected - 0.045 * h)
            arms = [r for r in loops if not r['contains_seed'] and sign * r['centre'][0] > lateral and (0.005 * h < r['radius'] < 0.11 * h)]
            if not arms:
                continue
            loop = max(arms, key=lambda r: sign * r['centre'][0])
            inside = inside_loop(loop, indices)
            girth[side].append(loop['radius'])
            seeds[side][indices[inside]] = True
            changed[indices] = True
            sections += 1
            separated += 1
        for loop in body_loops:
            inside = inside_loop(loop, indices)
            if separated < 2:
                inside &= abs(v[indices, 0]) < 0.65 * min((abs(points[s + 'Hip'][0]) for s in masks))
            body_seed[indices[inside]] = True
    edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
    adjacency = [[] for _ in v]
    for (a, b), length in zip(edges, np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)):
        adjacency[a].append((int(b), float(length)))
        adjacency[b].append((int(a), float(length)))

    def distances(seed):
        dist = np.full(len(v), np.inf)
        dist[seed] = 0
        queue = [(0.0, int(i)) for i in np.flatnonzero(seed)]
        heapq.heapify(queue)
        while queue:
            cost, i = heapq.heappop(queue)
            if cost != dist[i]:
                continue
            for j, length in adjacency[i]:
                candidate = cost + length
                if candidate < dist[j]:
                    dist[j] = candidate
                    heapq.heappush(queue, (candidate, j))
        return dist
    core_width = 0.5 * min((abs(points[s + 'Hip'][0]) for s in masks))
    core_seed = abs(v[:, 0]) < core_width
    if 'neck' in points:
        core_seed |= v[:, 2] >= points['neck'][2]
    body_seed |= core_seed
    body_seed &= ~(seeds['left'] | seeds['right'])
    erode_body = np.zeros(len(v), bool)
    for side in seeds:
        if not seeds[side].any():
            continue
        width = 3 * float(np.median(girth[side]))
        arm_distance = distances(seeds[side])
        erode_body |= arm_distance < width
    body_seed = body_seed & ~erode_body | core_seed
    body_seed &= ~(seeds['left'] | seeds['right'])
    body_distance = distances(body_seed)
    ownership = {}
    for side in masks:
        if not seeds[side].any():
            continue
        arm_distance = distances(seeds[side])
        observed = np.isfinite(body_distance) | np.isfinite(arm_distance)
        field = np.zeros(len(v))
        both = np.isfinite(body_distance) & np.isfinite(arm_distance)
        field[np.isfinite(arm_distance) & ~np.isfinite(body_distance)] = 1
        u = body_distance[both] / np.maximum(body_distance[both] + arm_distance[both], 1e-12 * h)
        field[both] = u * u * (3 - 2 * u)
        fixed = body_seed | seeds[side] | seeds['right' if side == 'left' else 'left']
        field[fixed] = seeds[side][fixed].astype(float)
        src = np.r_[edges[:, 0], edges[:, 1]]
        dst = np.r_[edges[:, 1], edges[:, 0]]
        degree = np.bincount(src, minlength=len(v))
        free = observed & ~fixed & (degree > 0)
        for _ in range(400):
            avg = np.bincount(src, weights=field[dst], minlength=len(v)) / np.maximum(degree, 1)
            field[free] = 0.5 * (field[free] + avg[free])
        masks[side][observed] = field[observed] > 0
        masks[side] &= v[:, 0] * (1 if side == 'left' else -1) >= -0.008 * h
        field[~masks[side]] = 0
        ownership[side] = dict(mass=field, observed=observed)
    report = dict(constrained_vertices=int(sum(((~m).sum() for m in masks.values()))), sections=sections, arm_seed_vertices={s: int(seed.sum()) for s, seed in seeds.items()}, body_seed_vertices=int(body_seed.sum()), arm_radius_m={s: float(np.median(r)) if r else None for s, r in girth.items()}, method='cross-section-Dirichlet-branches-harmonic-unobserved-bridge')
    masks['ownership'] = ownership
    return (masks, report)

def interior_arm_seed(vertices, triangles, seed, sign, radius):
    seed = np.array(seed, copy=True)
    if depth_interval(vertices, triangles, seed):
        return (seed, False)
    loops = section_loops(vertices, triangles, [0, 0, seed[2]], [0, 0, 1])
    if not loops:
        return (seed, False)
    loop = min(loops, key=lambda r: np.linalg.norm(r['centre'][:2] - seed[:2]))
    xs = sign * loop['outline'][:, 0]
    inset = min(radius * 0.5, (xs.max() - xs.min()) * 0.25)
    seed[0] = sign * np.clip(sign * seed[0], xs.min() + inset, xs.max() - inset)
    depth = depth_interval(vertices, triangles, seed)
    if depth:
        seed = depth['centre']
    return (seed, depth is not None)

def detect(vertices, triangles, height, observations=None, silhouette=None):
    v = np.asarray(vertices, float)
    t = np.asarray(triangles, int)
    h = float(height)
    if len(v) < 20 or h <= 0 or (not np.isfinite(v).all()):
        raise ValueError('Invalid body geometry')
    obs = {k: vec(p) for k, p in (observations or {}).items()}
    components = [v[idx] for idx in connected_components(len(v), t) if len(idx) >= 24]
    legs = {}
    arms = {}
    for s, sign in [('left', 1), ('right', -1)]:
        for c in components:
            span = np.ptp(c, axis=0)
            lo = c.min(axis=0)
            hi = c.max(axis=0)
            sided = np.mean(sign * c[:, 0] > -0.005 * h) > 0.98
            if not sided:
                continue
            if span[2] > 0.28 * h and hi[2] < 0.7 * h and (lo[2] < 0.2 * h) and (max(abs(lo[0]), abs(hi[0])) < 0.2 * h):
                if s not in legs or span[2] > np.ptp(legs[s][:, 2]):
                    legs[s] = c
            if span[0] > 0.2 * h and span[0] > 3 * span[2] and (lo[2] > 0.5 * h) and (hi[2] < 0.9 * h):
                if s not in arms or span[0] > np.ptp(arms[s][:, 0]):
                    arms[s] = c
    result = {}
    details = {}
    warnings = []

    def put(name, p, normal=(0, 0, 1), radius=None, refine=True):
        supplied = name in obs
        seed = obs.get(name, vec(p))
        ring = section(v, t, seed, normal, radius or 0.1 * h) if refine else None
        value = seed if supplied or ring is None else ring['centre']
        result[name] = value
        details[name] = dict(source='s3-observation' if supplied else 'geometry-refined-prior' if ring else 'proportion-prior', confidence=0.95 if supplied else 0.55 if ring else 0.2, radius_m=ring['radius'] if ring else None, section_offset_m=float(np.linalg.norm(seed - ring['centre'])) if ring else None, support_points=ring['support_points'] if ring else 0)
        return value
    mid = v[(abs(v[:, 0]) < 0.018 * h) & (v[:, 2] > 0.2 * h) & (v[:, 2] < 0.66 * h)]
    cz = float(mid[:, 2].min()) if len(mid) > 4 else 0.45 * h
    if len(legs) == 2:
        cz = float(np.mean([c[:, 2].max() for c in legs.values()]))
    else:
        tracks = None
        last_z = None
        for z in np.linspace(0.14 * h, 0.64 * h, 65):
            loops = section_loops(v, t, [0, np.median(v[:, 1]), z], [0, 0, 1])
            pair = []
            for i, sign in enumerate([1, -1]):
                choices = [r for r in loops if 0.02 * h < sign * r['centre'][0] < 0.16 * h and 0.012 * h < r['radius'] < 0.11 * h]
                supported = [r for r in choices if (branch_distance(silhouette, z / h, r['centre'][0] / h, 'leg') or 0) < 0.03]
                if supported:
                    choices = supported
                if tracks is not None:
                    choices = [r for r in choices if np.linalg.norm(r['centre'][:2] - tracks[i][:2]) < 0.035 * h]
                if not choices:
                    break
                pair.append(min(choices, key=lambda r: np.linalg.norm(r['centre'][:2] - (tracks[i][:2] if tracks is not None else np.array([sign * 0.075 * h, 0])))))
            if len(pair) != 2:
                if tracks is not None:
                    break
                continue
            tracks = [r['centre'] for r in pair]
            last_z = z
        if last_z is not None and last_z > 0.25 * h:
            cz = last_z
    cz = float(np.clip(cz, 0.25 * h, 0.58 * h))
    put('crotch', [0, np.median(v[:, 1]), cz], refine=False)
    cz = result['crotch'][2]
    candidates = []
    for z in np.linspace(max(cz + 0.15 * h, 0.53 * h), 0.85 * h, 45):
        r = section(v, t, [0, np.median(v[:, 1]), z], [0, 0, 1], 0.18 * h)
        if r and r['widths'][0] > 0.025 * h:
            candidates.append((r['area'], z, r))
    nr = min(candidates, key=lambda row: row[0]) if candidates else None
    nz = nr[1] if nr else 0.75 * h
    put('neck', [0, np.median(v[:, 1]), nz], radius=0.14 * h)
    nz = result['neck'][2]
    put('chin', [0, -0.06 * h, nz + 0.04 * h], refine=False)
    put('head', [0, result['neck'][1], nz + 0.09 * h], refine=False)
    put('head_top', [0, result['neck'][1], h], refine=False)
    hipz = cz + (0.025 * h if len(legs) == 2 else 0.06 * h)
    hip_ring = section(v, t, [0, np.median(v[:, 1]), hipz], [0, 0, 1], 0.22 * h)
    width = hip_ring['widths'][0] if hip_ring else 0.22 * h
    if len(legs) != 2:
        hipz = cz + 0.4 * width
    hipx = float(np.clip(width * 0.26, 0.045 * h, 0.12 * h))
    torsos = [section(v, t, [0, np.median(v[:, 1]), z], [0, 0, 1], 0.25 * h) for z in np.linspace(cz + 0.22 * (nz - cz), cz + 0.65 * (nz - cz), 15)]
    torsos = [r for r in torsos if r and r['contains_seed'] and (r['widths'][0] > 0.09 * h)]
    torso = min(torsos, key=lambda r: r['widths'][0]) if torsos else None
    halfwidth = torso['widths'][0] / 2 if torso else 0.12 * h
    armtracks = arm_section_tracks(v, t, h, cz, nz, max(halfwidth * 0.9, hipx + 0.025 * h), silhouette)
    for side, sign in [('left', 1), ('right', -1)]:
        hip = put(side + 'Hip', [sign * hipx, hip_ring['centre'][1] if hip_ring else np.median(v[:, 1]), hipz], radius=hipx * 0.95)
        low = v[(sign * v[:, 0] > 0.015 * h) & (v[:, 2] < 0.13 * h)]
        if len(low) < 6:
            raise ValueError('No separated foot geometry; need S3 measurements or a neutral pose')
        x = float(np.median(low[:, 0]))
        back = float(np.quantile(low[:, 1], 0.95))
        front = float(np.quantile(low[:, 1], 0.03))
        ankle = put(side + 'Ankle', [x, back - 0.22 * (back - front), 0.065 * h], radius=0.065 * h)
        put(side + 'Heel', [x, back, 0.025 * h], refine=False)
        put(side + 'Toe', [x, front, 0.025 * h], refine=False)
        kneez = ankle[2] + 0.52 * (hip[2] - ankle[2])
        put(side + 'Knee', [hip[0] * 0.6 + ankle[0] * 0.4, hip[1], kneez], radius=min(hipx * 0.9, 0.09 * h))
        arm = v[(sign * v[:, 0] > halfwidth * 0.9) & (v[:, 2] > cz - 0.1 * h) & (v[:, 2] < nz + 0.02 * h)]
        if len(arm) < 12:
            raise ValueError('Arm not separable: provide A/T pose and S3 landmarks')
        sh = np.array([sign * halfwidth * 0.82, np.median(arm[:, 1]), nz - 0.045 * h])
        radial = np.linalg.norm(arm - sh, axis=1)
        far = arm[radial > np.quantile(radial, 0.97)]
        tip = np.median(far, axis=0)
        component_arm = arms.get(side)
        track = armtracks.get(side)
        if component_arm is not None:
            axis_values = sign * component_arm[:, 0]
            proximal = component_arm[axis_values < np.quantile(axis_values, 0.06)]
            distal = component_arm[axis_values > np.quantile(axis_values, 0.94)]
            root = np.median(proximal, axis=0)
            end = np.median(distal, axis=0)
            sh = root + 0.06 * (end - root)
        elif track is not None:
            tip = track[0]
            proximal = track[-1]
            axis = unit(tip - proximal)
            extension = float(np.clip((nz - 0.045 * h - proximal[2]) / axis[2], -0.6 * (nz - cz), 0.02 * h))
            sh = proximal + axis * extension
            sh[0] = sign * max(sign * sh[0], halfwidth * 0.82)
        direction = unit(tip - sh)
        tracked = component_arm is None and track is not None

        def arm_joint(name, seed, radius):
            if not tracked:
                return put(name, seed, direction, radius)
            seed = np.array(seed, copy=True)
            observed = track[0, 2] <= seed[2] <= track[-1, 2]
            if observed:
                seed[:2] = [np.interp(seed[2], track[:, 2], track[:, axis]) for axis in [0, 1]]
                return put(name, seed, (0, 0, 1), radius)
            if not name.endswith('Shoulder') and len(track) >= 4:
                tail = track[-min(6, len(track)):]
                z = tail[:, 2]
                slope = np.array([np.polyfit(z, tail[:, axis], 1)[0] for axis in [0, 1]])
                seed[:2] = tail[-1, :2] + slope * (seed[2] - tail[-1, 2])
            proximal_section = section(v, t, track[-1], [0, 0, 1], 0.12 * h)
            measured_radius = proximal_section['radius'] if proximal_section else radius
            seed, projected = interior_arm_seed(v, t, seed, sign, measured_radius)
            depth = depth_interval(v, t, seed)
            if depth:
                seed = depth['centre']
            value = put(name, seed, refine=False)
            if name not in obs:
                details[name]['source'] = 'arm-track-interior-projection' if projected else 'arm-track-extrapolation'
                details[name]['confidence'] = 0.35 if depth else 0.2
                details[name]['radius_m'] = proximal_section['radius'] if proximal_section else None
                details[name]['radius_source'] = 'proximal-visible-arm-section'
                details[name]['depth_interval_m'] = depth['interval'] if depth else None
            return value
        shoulder = arm_joint(side + 'Shoulder', sh, 0.065 * h)
        length = np.linalg.norm(tip - shoulder)
        wrist_seed = end if component_arm is not None else shoulder + (tip - shoulder) * 0.82
        wrist = arm_joint(side + 'Wrist', wrist_seed, 0.055 * h)
        arm_joint(side + 'Elbow', shoulder + (wrist - shoulder) * 0.52, 0.065 * h)
        put(side + 'Hand', tip, refine=False)
        for name in ['Shoulder', 'Elbow', 'Wrist', 'Hand']:
            details[side + name]['arm_seed_method'] = 'disconnected-skin-tube' if component_arm is not None else 'separate-section-track' if track is not None else 'unvalidated-radial-fallback'
            details[side + name]['arm_track_samples'] = len(track) if track is not None else 0
        put(side + 'Eye', [sign * 0.035 * h, -0.06 * h, nz + 0.13 * h], refine=False)
    for name, p in obs.items():
        if name not in result:
            put(name, p, refine=False)
    for name, entry in details.items():
        if silhouette:
            entry['s0_visible_branch_evidence'] = True
        if entry['confidence'] < 0.8:
            warnings.append(name + ': anatomical location not validated by S0/S3')
    return (result, details, warnings)

def skeleton(points, details):
    p = {k: vec(v) for k, v in points.items()}
    bones = []
    hips = (p['leftHip'] + p['rightHip']) / 2
    neck = p['neck']
    spine = hips + (neck - hips) * 0.25
    chest = hips + (neck - hips) * 0.55
    upper = hips + (neck - hips) * 0.8

    def add(name, start, end, parent=None, role='core'):
        if np.linalg.norm(end - start) < 1e-06:
            raise ValueError('Zero-length ' + name)
        bones.append(dict(name=name, head=start.tolist(), tail=end.tolist(), parent=parent, role=role))
    add('hips', hips, spine)
    add('spine', spine, chest, 'hips')
    add('chest', chest, upper, 'spine')
    add('upperChest', upper, neck, 'chest')
    add('neck', neck, p['head'], 'upperChest')
    add('head', p['head'], p['head_top'], 'neck')
    for s in ['left', 'right']:
        add(s + 'Shoulder', upper, p[s + 'Shoulder'], 'upperChest')
        add(s + 'UpperArm', p[s + 'Shoulder'], p[s + 'Elbow'], s + 'Shoulder', 'arm')
        add(s + 'LowerArm', p[s + 'Elbow'], p[s + 'Wrist'], s + 'UpperArm', 'arm')
        add(s + 'Hand', p[s + 'Wrist'], p[s + 'Hand'], s + 'LowerArm', 'arm')
        add(s + 'UpperLeg', p[s + 'Hip'], p[s + 'Knee'], 'hips', 'leg')
        add(s + 'LowerLeg', p[s + 'Knee'], p[s + 'Ankle'], s + 'UpperLeg', 'leg')
        add(s + 'Foot', p[s + 'Ankle'], p[s + 'Toe'], s + 'LowerLeg', 'leg')
        toe = p[s + 'Toe']
        heel = p[s + 'Heel']
        add(s + 'Toes', toe, toe + (toe - heel) * 0.18, s + 'Foot', 'leg')
    return bones

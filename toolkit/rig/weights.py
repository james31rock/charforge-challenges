import numpy as np

def diffuse(vertices, edges, bones, height, iterations=300, part='body', arm_regions=None):
    v = np.asarray(vertices, float)
    e = np.asarray(edges, int).reshape(-1, 2)
    names = [b['name'] for b in bones]
    n = len(v)
    k = len(bones)
    distance = np.empty((n, k))
    along = np.empty((n, k))
    for j, b in enumerate(bones):
        a = np.array(b['head'])
        d = np.array(b['tail']) - a
        u = np.clip((v - a) @ d / (d @ d), 0, 1)
        along[:, j] = u
        distance[:, j] = np.linalg.norm(v - a - u[:, None] * d, axis=1)
    allowed = np.ones((n, k), bool)
    legtops = [b['head'][2] for b in bones if b['name'].endswith('UpperLeg')]
    hipz = min(legtops)
    hips = [np.array(b['head']) for b in bones if b['name'].endswith('UpperLeg')]
    if len(hips) == 2 and 'hips' in names:
        a, d = (hips[0], hips[1] - hips[0])
        u = np.clip((v - a) @ d / (d @ d), 0, 1)
        distance[:, names.index('hips')] = np.minimum(distance[:, names.index('hips')], np.linalg.norm(v - a - u[:, None] * d, axis=1))
    armcols = [j for j, b in enumerate(bones) if b['role'] == 'arm']
    legcols = [j for j, b in enumerate(bones) if b['role'] == 'leg']
    near_arm = distance[:, armcols].min(axis=1) if armcols else np.full(n, np.inf)
    near_leg = distance[:, legcols].min(axis=1) if legcols else np.full(n, np.inf)
    for j, b in enumerate(bones):
        name = b['name']
        role = b['role']
        if name.startswith('left'):
            allowed[:, j] &= v[:, 0] >= -0.008 * height
        if name.startswith('right'):
            allowed[:, j] &= v[:, 0] <= 0.008 * height
        if role == 'arm':
            allowed[:, j] &= (v[:, 2] > hipz - 0.03 * height) | (near_arm < 0.9 * near_leg)
        if role == 'arm' and arm_regions:
            side = 'left' if name.startswith('left') else 'right'
            field = arm_regions.get('ownership', {}).get(side)
            if field is not None:
                allowed[:, j] |= field['mass'] > 0
            allowed[:, j] &= arm_regions[side]
        if role == 'leg':
            allowed[:, j] &= v[:, 2] < hipz + 0.16 * height
    if part in ('hair', 'rigid_head'):
        w = np.zeros((n, k))
        w[:, names.index('head')] = 1
        return (w, dict(method='declared-rigid-head', iterations=0, unweighted=0, max_influences=1))
    nearest = np.min(np.where(allowed, distance, np.inf), axis=1)
    sigma = 0.028 * height
    source = np.exp(-((distance - nearest[:, None]) / sigma) ** 2) * allowed
    u = np.clip((hipz + 0.08 * height - v[:, 2]) / (0.12 * height), 0, 1)
    source[:, legcols] *= (u * u * (3 - 2 * u))[:, None]
    source /= source.sum(axis=1, keepdims=True)

    def project_branches(w, seed=False):
        for side, field in (arm_regions or {}).get('ownership', {}).items():
            cols = np.array([b['role'] == 'arm' and b['name'].startswith(side) for b in bones])
            if not cols.any():
                continue
            rows = np.flatnonzero(field['observed'])
            target = field['mass'][rows]
            for group, mass in [(cols, target), (~cols, 1 - target)]:
                ids = np.flatnonzero(group)
                block = w[np.ix_(rows, ids)]
                total = block.sum(axis=1)
                if seed:
                    distances = distance[np.ix_(rows, ids)]
                    permit = allowed[np.ix_(rows, ids)]
                    nearest_group = np.min(np.where(permit, distances, np.inf), axis=1)
                    nearest_group[~np.isfinite(nearest_group)] = 0
                    block = np.exp(-((distances - nearest_group[:, None]) / sigma) ** 2) * permit
                    for column, j in enumerate(ids):
                        if j in legcols:
                            block[:, column] *= u[rows] * u[rows] * (3 - 2 * u[rows])
                    total = block.sum(axis=1)
                empty = total < 1e-25
                if empty.any():
                    choice = np.argmin(np.where(allowed[np.ix_(rows[empty], ids)], distance[np.ix_(rows[empty], ids)], np.inf), axis=1)
                    block[empty] = 0
                    block[np.flatnonzero(empty), choice] = 1
                    total[empty] = 1
                w[np.ix_(rows, ids)] = block / total[:, None] * mass[:, None]
        return w
    source = project_branches(project_branches(source, seed=True))
    if len(e):
        src = np.r_[e[:, 0], e[:, 1]]
        dst = np.r_[e[:, 1], e[:, 0]]
        degree = np.bincount(src, minlength=n).astype(float)
        edge_length = np.median(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1))
        w = source.copy()
        alpha = float(np.clip((edge_length / (0.035 * height)) ** 2, 0.002, 0.2))
        for iteration in range(iterations):
            sums = np.empty_like(w)
            for j in range(k):
                sums[:, j] = np.bincount(src, weights=w[dst, j], minlength=n)
            out = (sums + alpha * degree[:, None] * source) / np.maximum((1 + alpha) * degree[:, None], 1)
            out[degree == 0] = source[degree == 0]
            out *= allowed
            out = project_branches(out)
            out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-30)
            error = float(abs(out - w).max())
            w = out
            if error < 1e-07:
                break
    else:
        w = source
        iteration = 0
        error = 0.0
    order = np.argsort(-w, axis=1, kind='stable')
    keep = np.zeros_like(w, bool)
    np.put_along_axis(keep, order[:, :8], True, axis=1)
    w *= keep
    w[w < 1e-05] = 0
    w /= w.sum(axis=1, keepdims=True)
    return (w, dict(method='screened-mesh-graph-heat', iterations=iteration + 1, residual=error, unweighted=int((w.sum(axis=1) == 0).sum()), max_influences=int((w > 0).sum(axis=1).max()), forbidden_weight=float(w[~allowed].sum()), normalization_error=float(abs(w.sum(axis=1) - 1).max())))

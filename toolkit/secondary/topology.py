import numpy as np

def from_roots_tips(roots, tips, segments, topology, lateral_links):
    roots = np.asarray(roots, float)
    tips = np.asarray(tips, float)
    if roots.shape != tips.shape or roots.ndim != 2 or roots.shape[1] != 3:
        raise ValueError('Matching Nx3 roots/tips required')
    if segments < 1:
        raise ValueError('At least one segment')
    if not np.isfinite(roots).all() or not np.isfinite(tips).all():
        raise ValueError('Nonfinite topology')
    points = []
    parents = []
    pinned = []
    lateral = []
    n = segments + 1
    for c, (root, tip) in enumerate(zip(roots, tips)):
        if np.linalg.norm(tip - root) < 1e-06:
            raise ValueError('Zero-length feature')
        for j in range(n):
            points.append(root + (tip - root) * j / segments)
            parents.append(c * n + j - 1 if j else -1)
            pinned.append(j == 0)
    if lateral_links:
        pairs = [(c, c + 1) for c in range(len(roots) - 1)]
        if topology == 'ring' and len(roots) > 2:
            pairs.append((len(roots) - 1, 0))
        for a, b in pairs:
            for j in range(1, n):
                lateral.append((a * n + j, b * n + j))
            for j in range(segments):
                lateral.extend([(a * n + j, b * n + j + 1), (b * n + j, a * n + j + 1)])
    return dict(rest=np.asarray(points), parents=parents, pinned=pinned, lateral=lateral)

import numpy as np
from .math_core import two_bone

def circular_phase_error(a, b):
    ca = np.fft.rfft(np.asarray(a) - np.mean(a))[1]
    cb = np.fft.rfft(np.asarray(b) - np.mean(b))[1]
    if min(abs(ca), abs(cb)) < 1e-08:
        raise ValueError('Undefined first-harmonic phase')
    return float(np.angle(ca / cb))

def hip_curves(offsets, root, targets, lengths, bow, trunk_pitch):
    hips = np.asarray(offsets).copy()
    hips[:, :, 2] += np.asarray(root)[:, None]
    lat = hips[:, 0] - hips[:, 1]
    lat[:, 2] = 0
    lat /= np.linalg.norm(lat, axis=1)[:, None]
    fwd = np.cross(lat, [0, 0, 1])
    result = np.empty((len(root), 2))
    for i in range(len(root)):
        for s in range(2):
            h = hips[i, s]
            a, b = lengths[s]
            delta = targets[i, s] - h
            d = np.linalg.norm(delta)
            axis = delta / d
            along = (a * a - b * b + d * d) / (2 * d)
            r = np.sqrt(max(0, a * a - along * along))
            offset = float(bow[i, s] * r * np.sqrt(max(0, 1 - axis[0] ** 2)))
            k = two_bone(h, targets[i, s], a, b, [0, -1, 0], offset).knee
            v = k - h
            result[i, s] = np.arctan2(v @ fwd[i], -v[2]) - trunk_pitch[i]
    return result

def swing_floor(reference, side, phase, target_leg):
    u = (phase - (0 if side == 'left' else 0.5)) % 1
    if u < 0.6:
        return 0.0005
    rest = reference['rest']
    L = np.mean([sum((np.linalg.norm(np.array(rest[s + a]) - rest[s + b]) for a, b in [('UpperLeg', 'LowerLeg'), ('LowerLeg', 'Foot')])) for s in ['left', 'right']])
    pairs = [(0.6, 0.0), (1.0, 0.0)]
    for row in reference['samples']:
        q = (row['phase'] - (0 if side == 'left' else 0.5)) % 1
        if 0.6 < q < 1:
            pairs.append((q, max(0.0, (row['feet'][side]['min_z_m'] - reference.get('floor_m', reference.get('scene_floor_m', 0.0))) / L)))
    pairs = sorted(pairs)
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    sec = np.diff(y) / np.diff(x)
    m = np.zeros(len(x))
    for i in range(1, len(x) - 1):
        if sec[i - 1] * sec[i] > 0:
            m[i] = 2 / (1 / sec[i - 1] + 1 / sec[i])
    j = min(len(x) - 2, int(np.searchsorted(x, u, side='right') - 1))
    dx = x[j + 1] - x[j]
    t = (u - x[j]) / dx
    z = (2 * t ** 3 - 3 * t * t + 1) * y[j] + (t ** 3 - 2 * t * t + t) * dx * m[j] + (-2 * t ** 3 + 3 * t * t) * y[j + 1] + (t ** 3 - t * t) * dx * m[j + 1]
    return 0.0005 + max(0.0, float(z)) * target_leg

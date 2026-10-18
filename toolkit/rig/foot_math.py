import numpy as np

def smoothstep(x):
    u = np.clip(x, 0, 1)
    return u * u * (3 - 2 * u)

def shoe_region(vertices, ankle, other, height):
    v = np.asarray(vertices)
    a = np.asarray(ankle)
    b = np.asarray(other)
    return (np.linalg.norm(v - a, axis=1) < height * 0.2) & (np.linalg.norm(v - a, axis=1) < np.linalg.norm(v - b, axis=1)) & (v[:, 2] < a[2] + 0.015 * height)

def shoe_features(vertices, ankle, toe):
    v = np.asarray(vertices)
    a = np.asarray(ankle)
    forward = np.asarray(toe) - a
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    length = v @ forward
    lo, hi = np.quantile(length, [0.01, 0.99])
    floor = float(np.quantile(v[:, 2], 0.01))
    ball = a.copy()
    ball += forward * (lo + 0.76 * (hi - lo) - ball @ forward)
    ball[2] = floor
    heel = ball - forward * 0.76 * (hi - lo)
    return dict(forward=forward, length=hi - lo, back=lo, front=hi, floor=floor, ball=ball, heel=heel)

def bind_shoes(vertices, weights, bones, height):
    names = [b['name'] for b in bones]
    v = np.asarray(vertices)
    reports = {}
    for side, other in [('left', 'right'), ('right', 'left')]:
        foot = next((b for b in bones if b['name'] == side + 'Foot'))
        opposite = next((b for b in bones if b['name'] == other + 'Foot'))
        ankle = np.array(foot['head'])
        mask = shoe_region(v, ankle, opposite['head'], height)
        ids = np.flatnonzero(mask)
        if len(ids) < 8:
            continue
        f = shoe_features(v[ids], ankle, foot['tail'])
        u = (v[ids] @ f['forward'] - f['back']) / f['length']
        toes = smoothstep((u - 0.68) / 0.16)
        strength = smoothstep((ankle[2] + 0.015 * height - v[ids, 2]) / (0.03 * height))
        weights[ids] *= 1 - strength[:, None]
        weights[ids, names.index(side + 'Foot')] += strength * (1 - toes)
        weights[ids, names.index(side + 'Toes')] += strength * toes
        reports[side] = dict(vertices=len(ids), ball_m=f['ball'].tolist(), shoe_length_m=float(f['length']), policy='continuous ankle cuff; shoe heel to measured 76-percent ball pivot; forefoot toe blend')
    return (weights, reports)

def pivoted_toe(ankle_pose, ankle_rest, toe_rest, ball, foot_delta, toe_delta):
    return np.asarray(ankle_pose) + foot_delta @ (np.asarray(ball) - ankle_rest) + toe_delta @ (np.asarray(toe_rest) - ball)

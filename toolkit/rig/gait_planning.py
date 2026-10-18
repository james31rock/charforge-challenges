import numpy as np

def fourier_basis(n, harmonics=3):
    phase = np.arange(n) / n
    return np.column_stack([np.ones(n)] + [f(2 * np.pi * k * phase) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)])

def constrained_quadratic(H, g, A, b, initial):
    x = np.asarray(initial, float).copy()
    active = []
    if np.max(A @ x - b) > 1e-08:
        raise ValueError('QP initial point is infeasible')
    for iteration in range(500):
        grad = H @ x - g
        W = A[active]
        K = np.block([[H, W.T], [W, np.zeros((len(active), len(active)))]])
        sol = np.linalg.lstsq(K, np.r_[-grad, np.zeros(len(active))], rcond=1e-11)[0]
        p = sol[:len(x)]
        if np.linalg.norm(p) < 1e-09:
            multipliers = sol[len(x):]
            if not len(active) or multipliers.min() >= -1e-08:
                return (x, iteration + 1)
            del active[int(np.argmin(multipliers))]
            continue
        rates = A @ p
        valid = np.flatnonzero(rates > 1e-10)
        ratios = (b[valid] - A[valid] @ x) / rates[valid]
        k = int(np.argmin(ratios)) if len(valid) else None
        alpha = min(1.0, max(0.0, float(ratios[k]))) if k is not None else 1.0
        x += alpha * p
        if alpha < 1 - 1e-10 and int(valid[k]) not in active:
            active.append(int(valid[k]))
    raise RuntimeError('Periodic gait QP did not converge')

def plan_root(source_z, hips_offset, ankles, lengths, source_flex, stance, leg):
    n = len(source_z)
    B = fourier_basis(n)
    desired = []
    weights = []
    bounds = []
    for side in range(2):
        a, b = lengths[side]
        hip = hips_offset[:, side]
        goal = ankles[:, side]
        r2 = np.sum((goal[:, :2] - hip[:, :2]) ** 2, axis=1)
        distance2 = a * a + b * b + 2 * a * b * np.cos(source_flex[:, side])
        desired.append(goal[:, 2] + np.sqrt(np.maximum(0, distance2 - r2)) - hip[:, 2])
        minimum_flex = float(np.min(source_flex[:, side]))
        maximum_distance = np.sqrt(a * a + b * b + 2 * a * b * np.cos(minimum_flex))
        reach = (maximum_distance * (1 - 0.0005)) ** 2 - r2
        if np.any(reach <= 0):
            raise ValueError('Horizontal foot goal outside leg reach')
        bounds.append(goal[:, 2] + np.sqrt(reach) - hip[:, 2])
        weights.append(np.where(stance[:, side], 1.0, 0.12))
    desired = np.array(desired).T / leg
    weights = np.array(weights).T
    z0 = np.asarray(source_z) / leg
    z0 -= z0.mean()
    mean = np.sum(desired * weights) / weights.sum()
    target = z0 + mean
    sample_w = 1 + weights.sum(1)
    H = B.T @ (sample_w[:, None] * B)
    g = B.T @ (target + (desired * weights).sum(1))
    H += np.diag([1e-09] + [0.02 * k ** 6 for k in range(1, 4) for _ in range(2)])
    upper = np.min(bounds, axis=0) / leg
    pairs = np.array([B[i] - B[j] for i in range(n) for j in range(i)])
    span = float(np.ptp(z0))
    A = np.vstack([B, pairs, -pairs])
    limits = np.r_[upper, np.full(2 * len(pairs), span)]
    initial = np.zeros(B.shape[1])
    initial[0] = upper.min() - 0.01
    coeff, iterations = constrained_quadratic(H, g, A, limits, initial)
    z = B @ coeff * leg
    return (z, dict(coefficients_z_over_leg=coeff.tolist(), harmonics=3, iterations=iterations, source_range_over_leg=span, planned_range_over_leg=float(np.ptp(z) / leg), maximum_reach_bound_violation_m=float(np.max(z - upper * leg)), desired_support_root_m=(desired * leg).tolist(), upper_root_m=(upper * leg).tolist(), policy=dict(source_weight=1.0, stance_knee_weight=1.0, swing_knee_weight=0.12, phase_jerk_weight=0.02, reach_reserve=0.0005, minimum_flexion='minimum observed source flexion per side')))

def clipped_x_extents(triangles, lo, hi):
    t = np.asarray(triangles)
    candidates = t[np.all(t.max(1)[:, 1:] >= lo, axis=1) & np.all(t.min(1)[:, 1:] <= hi, axis=1)]
    values = []
    for triangle in candidates:
        poly = list(triangle)
        for axis, bound, sign in [(1, lo[0], 1), (1, hi[0], -1), (2, lo[1], 1), (2, hi[1], -1)]:
            result = []
            if not poly:
                break
            for a, b in zip(poly, poly[1:] + poly[:1]):
                da = (a[axis] - bound) * sign
                db = (b[axis] - bound) * sign
                if da >= 0:
                    result.append(a)
                if (da >= 0) != (db >= 0):
                    result.append(a + (b - a) * da / (da - db))
            poly = result
        values.extend((p[0] for p in poly))
    return (min(values), max(values)) if values else None

def swing_clearance_offsets(required, leg, stance_fraction=0.6):
    needed = np.maximum(0, np.asarray(required, float))
    phase = np.arange(len(needed)) / len(needed)
    u = (phase[:, None] - np.array([0.0, 0.5])) % 1
    v = np.clip((u - stance_fraction) / (1 - stance_fraction), 0, 1)
    bump = (4 * v * (1 - v)) ** 3
    total = bump.sum(1)
    base = float(needed[total < 1e-08].max(initial=0))
    amplitude = float(np.max(np.divide(np.maximum(0, needed - base), total, out=np.zeros_like(total), where=total >= 1e-08)))
    offsets = base / 2 + bump * amplitude
    return (offsets, dict(stance_extra_separation_m=base, swing_extra_amplitude_m=amplitude, swing_extra_amplitude_over_leg=amplitude / leg, stance_fraction=stance_fraction, required_separation_m=needed.tolist(), swing_envelope='[4*v*(1-v)]^3; v=(phase-stance)/(1-stance)', bound_violation_m=float(np.max(needed - offsets.sum(1)))))

def shoe_spacing(shoes, ankles, centres, source_relative_x, leg, triangles=None, swing_only=False):
    desired = centres[:, None, 0] + source_relative_x
    required = 0.0
    witness = None
    margin = 0.01 * leg
    needed = np.zeros(len(shoes))
    for i, clouds in enumerate(shoes):
        shifted = [p + np.array([desired[i, s] - ankles[i, s, 0], 0, 0]) for s, p in enumerate(clouds)]
        lo = [p.min(0) for p in shifted]
        hi = [p.max(0) for p in shifted]
        if np.all(hi[0][1:] + margin >= lo[1][1:]) and np.all(hi[1][1:] + margin >= lo[0][1:]):
            supports = [(lo[s][0], hi[s][0]) for s in range(2)]
            if triangles is not None:
                lower = np.maximum(lo[0][1:], lo[1][1:]) - margin / 2
                upper = np.minimum(hi[0][1:], hi[1][1:]) + margin / 2
                supports = [clipped_x_extents(triangles[i][s] + np.array([desired[i, s] - ankles[i, s, 0], 0, 0]), lower, upper) for s in range(2)]
                if any((x is None for x in supports)):
                    continue
            gap = supports[0][0] - supports[1][1]
            extra = max(0.0, margin - gap)
            needed[i] = extra
            if extra > required:
                required = extra
                witness = i + 1
    detail = {}
    if swing_only:
        offsets, detail = swing_clearance_offsets(needed, leg)
        desired += offsets * np.array([1.0, -1.0])
    else:
        desired += np.array([required / 2, -required / 2])
    return (desired, dict(extra_separation_m=required, extra_over_leg=required / leg, margin_m=margin, witness_frame=witness, **detail, method='Constant cycle offset from native triangles clipped to the Y/Z overlap prism; no hip-width multiplier' if triangles is not None else 'Constant cycle offset from 3D-overlapping native shoe support extents; no hip-width multiplier'))

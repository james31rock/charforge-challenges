import numpy as np
from .core import measure, unit, dot

def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180

def excursion(a):
    return float(np.ptp(np.degrees(np.unwrap(np.radians(a)))))

def phase_difference(a, b):
    ca = np.fft.rfft(np.asarray(a) - np.mean(a))[1]
    cb = np.fft.rfft(np.asarray(b) - np.mean(b))[1]
    if min(abs(ca), abs(cb)) < 1e-06:
        return None
    return float(abs(wrap(np.degrees(np.angle(ca / cb)))))

def measure_full(d):
    if 'rest_rotations' in d:
        for row in [d['rest_rotations']] + [r.get('rotations', {}) for r in d['samples']]:
            for name, rotation in row.items():
                R = np.asarray(rotation, float)
                if R.shape != (3, 3) or not np.isfinite(R).all() or (not np.allclose(R @ R.T, np.eye(3), atol=0.0001)) or (not np.isclose(np.linalg.det(R), 1, atol=0.0001)):
                    raise ValueError('Invalid orientation ' + name)
    base = measure(d)
    L = base['summary']['leg_length_m']
    H = d.get('height_m')
    N = len(d['samples'])
    rest = {k: np.array(v) for k, v in d['rest'].items()}
    p = {k: np.array([r['joints'][k] for r in d['samples']]) for k in rest}
    lat = rest['leftUpperLeg'] - rest['rightUpperLeg']
    lat[2] = 0
    lat = unit(lat)
    fw = np.cross(lat, [0, 0, 1])
    up = np.array([0, 0, 1])
    c = dict(base['curves'])
    m = {}
    notes = []
    phase = np.arange(N) / N

    def add(key, value):
        m[key] = float(value) if value is not None else None
    add('pelvis_bounce_L', base['summary']['pelvis']['range'])
    add('cadence_steps_min', 120 / d['cycle_seconds'])
    centre = (p['leftUpperLeg'] + p['rightUpperLeg']) / 2
    lateral = centre @ lat
    c['lateral_shift_L'] = (lateral - lateral.mean()) / L
    add('lateral_shift_L', np.ptp(lateral) / L)
    if H:
        add('step_width_height', base['summary']['contact_step_width_L'] * L / H)
    add('step_width_L', base['summary']['contact_step_width_L'])
    stride = d.get('stride_vector_m')
    add('step_length_height', abs(np.asarray(stride) @ fw) / 2 / H if stride is not None and H else None)
    add('step_length_L', abs(np.asarray(stride) @ fw) / 2 / L if stride is not None else None)
    masks = []
    for side, off in [('left', 0), ('right', 0.5)]:
        ph = (phase - d.get('phase_origin', 0) - off) % 1
        if all((side in r.get('contacts', {}) for r in d['samples'])):
            mask = np.array([r['contacts'][side] for r in d['samples']], bool)
            method = d.get('contact_method', 'supplied contact labels')
        elif d.get('declared_stance_fraction') is not None:
            mask = ph < d['declared_stance_fraction']
            method = 'declared phase schedule; not measured force support'
        else:
            mask = None
            method = 'unavailable'
        if mask is not None:
            masks.append(mask)
        declared = d.get('declared_stance_fraction') if not all((side in r.get('contacts', {}) for r in d['samples'])) else None
        add(side + '_stance_fraction', declared if declared is not None else np.mean(mask) if mask is not None else None)
        if mask is not None:
            c[side + '_contact'] = mask.astype(float)
        knee = np.asarray(c[side + '_knee_deg'])
        ankle = np.asarray(c[side + '_ankle_deg'])
        for key, values, select, fn in [('loading_knee_peak_deg', knee, ph < 0.2, np.max), ('midstance_knee_min_deg', knee, (ph >= 0.2) & (ph < 0.5), np.min), ('swing_knee_peak_deg', knee, ph >= 0.6, np.max), ('knee_min_deg', knee, np.ones(N, bool), np.min), ('ankle_dorsiflexion_deg', ankle, np.ones(N, bool), np.max), ('ankle_plantarflexion_deg', ankle, np.ones(N, bool), np.min)]:
            add(side + '_' + key, fn(values[select]))
        if side + 'UpperArm' in p:
            arm = p[side + 'LowerArm'] - p[side + 'UpperArm']
            trunk = p['chest'] - p['hips']
            flex = np.degrees(np.arctan2(arm @ fw, -arm[:, 2])) - np.degrees(np.arctan2(trunk @ fw, trunk[:, 2]))
            abd = np.degrees(np.arctan2(arm @ lat * (1 if side == 'left' else -1), -arm[:, 2]))
            c[side + '_arm_flex_deg'] = flex
            c[side + '_arm_abduction_deg'] = abd
            add(side + '_arm_swing_deg', excursion(flex))
            add(side + '_arm_abduction_mean_deg', np.mean(abd))
            add(side + '_arm_leg_phase_deg', phase_difference(flex, c[side + '_hip_deg']))
        markers = [r.get('feet', {}).get(side, {}) for r in d['samples']]
        if all(('heel' in r and 'toe' in r for r in markers)):
            v = np.array([np.array(r['toe']) - r['heel'] for r in markers])
            roll = np.degrees(np.arctan2(v[:, 2], v @ fw))
            c[side + '_sole_roll_deg'] = roll
            add(side + '_sole_roll_rom_deg', excursion(roll))
            for name, select in [('heel_strike_roll_deg', ph < 0.09), ('push_off_roll_deg', (ph >= 0.45) & (ph <= 0.625))]:
                add(side + '_' + name, np.max(roll[select]) if name.startswith('heel') else np.min(roll[select]))
        else:
            notes.append(side + ' material heel/toe unavailable: bone foot pitch is not native sole roll')
    add('double_support_fraction', max(0, 2 * d['declared_stance_fraction'] - 1) if d.get('declared_stance_fraction') is not None else np.mean(masks[0] & masks[1]) if len(masks) == 2 else None)
    native = {s: np.array([r.get('feet', {}).get(s, {}).get('min_z_m', np.nan) for r in d['samples']]) for s in ['left', 'right']}
    if all((np.isfinite(v).all() for v in native.values())):
        floor = d.get('scene_floor_m', 0)
        near = {s: v - floor <= 0.005 * L for s, v in native.items()}
        for s in near:
            add(s + '_geometric_stance_fraction', np.mean(near[s]))
            c[s + '_near_floor'] = near[s].astype(float)
        add('geometric_double_support_fraction', np.mean(near['left'] & near['right']))
        add('geometric_double_support_seconds', m['geometric_double_support_fraction'] * d['cycle_seconds'])
        notes.append('Geometric support = native sole within 0.005 L of scene floor; may include sliding/scuffing, not force support; 24-sample resolution')
        for s in ['left', 'right']:
            ph = (phase - d.get('phase_origin', 0) - (0 if s == 'left' else 0.5)) % 1
            add(s + '_late_swing_min_clearance_L', np.min((native[s][ph >= 0.85] - floor) / L))
    for segment in ('hips', 'chest', 'head'):
        if 'rest_rotations' not in d or not all((segment in r.get('rotations', {}) for r in d['samples'])):
            continue
        rb = np.array(d['rest_rotations'][segment])
        rs = np.array([r['rotations'][segment] for r in d['samples']])
        delta = rs @ rb.T
        vf = delta @ fw
        vz = delta @ up
        angles = {'yaw': np.degrees(np.arctan2(vf @ lat, vf @ fw)), 'pitch': np.degrees(np.arctan2(vf[:, 2], np.linalg.norm(vf[:, :2], axis=1))), 'roll': np.degrees(np.arctan2(vz @ lat, vz[:, 2]))}
        for axis, a in angles.items():
            c[segment + '_' + axis + '_deg'] = a
            add(segment + '_' + axis + '_rom_deg', excursion(a))
    shoulder = p.get('leftUpperArm', None)
    if shoulder is not None:
        v = shoulder - p['rightUpperArm']
        yaw = np.degrees(np.arctan2(-v @ fw, v @ lat))
        c['shoulder_yaw_deg'] = yaw
        add('shoulder_yaw_rom_deg', excursion(yaw))
    if 'hips_yaw_deg' in c and 'chest_yaw_deg' in c:
        rel = wrap(c['chest_yaw_deg'] - c['hips_yaw_deg'])
        c['thorax_pelvis_relative_yaw_deg'] = rel
        add('thorax_pelvis_relative_yaw_rom_deg', excursion(rel))
        add('thorax_pelvis_phase_deg', phase_difference(c['chest_yaw_deg'], c['hips_yaw_deg']))
        if 'shoulder_yaw_deg' in c:
            add('shoulder_pelvis_phase_deg', phase_difference(c['shoulder_yaw_deg'], c['hips_yaw_deg']))
    headlat = p['head'] @ lat
    headz = p['head'][:, 2]
    add('head_lateral_shift_L', np.ptp(headlat) / L)
    add('head_vertical_bounce_L', np.ptp(headz) / L)
    add('head_pelvis_vertical_gain', np.ptp(headz) / max(np.ptp(p['hips'][:, 2]), 1e-10))
    for key in list(m):
        if key.startswith('left_') and 'right_' + key[5:] in m:
            a, b = (m[key], m['right_' + key[5:]])
            add(key[5:], (a + b) / 2 if a is not None and b is not None else None)
    return {'schema_version': 1, 'kind': 'charforge-naturalness', 'label': d['label'], 'metrics': m, 'curves': {k: np.asarray(v).tolist() for k, v in c.items()}, 'metadata': {'leg_m': L, 'height_m': H, 'height_method': d.get('height_method', 'unspecified'), 'cycle_seconds': d['cycle_seconds'], 'contact_method': method, 'angle_method': 'bind-calibrated projected axes, not clinical marker-model Euler angles', 'notes': notes}}
TOLERANCE = {'deg': 5.0, 'fraction': 0.05, 'height': 0.02, '_L': 0.03, 'steps_min': 10.0, 'gain': 0.15}

def rank(references, tests):
    rows = []
    for key in sorted(set.union(*(set(x['metrics']) for x in references))):
        if key.startswith(('left_', 'right_')):
            continue
        available = [x for x in references if x['metrics'].get(key) is not None]
        if key.endswith('_height'):
            available = [x for x in available if 'proxy' not in x['metadata'].get('height_method', '')]
        vals = [x['metrics'][key] for x in available]
        if not vals:
            continue
        lo, hi = (min(vals), max(vals))
        floor = next((v for suffix, v in TOLERANCE.items() if key.endswith(suffix)), 0.03)
        width = max(hi - lo, floor)
        scores = {}
        for t in tests:
            value = t['metrics'].get(key)
            if value is not None:
                scores[t['label']] = {'value': value, 'outside_band_units': max(lo - value, 0, value - hi) / width}
        rows.append({'metric': key, 'control_band': [lo, hi], 'controls': [x['label'] for x in available], 'normalizer': width, 'normalizer_floor': floor, 'tests': scores, 'max_excess': max([v['outside_band_units'] for v in scores.values()] or [0]), 'band_kind': 'CMU plus approved/native rig control envelope; NOT population normal'})
    return sorted(rows, key=lambda r: r['max_excess'], reverse=True)

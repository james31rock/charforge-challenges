from __future__ import annotations
import math
import numpy as np
DEFAULTS = dict(palm_plane_deg=15.0, palm_contact_height=0.006, hand_ground_height=0.001, hand_skate_height_per_s=0.08, wrist_extension_min_deg=60.0, wrist_extension_max_deg=90.0, wrist_extension_tolerance_deg=5.0, finger_flat_deg=20.0, rest_wrist_deg=10.0, rest_finger_deg=20.0)
SUPPORT_IDS = ('C7.palm_plane', 'C7.palm_contact', 'C7.wrist_extension', 'C7.hand_skate', 'C7.hand_ground_penetration', 'C7.support_fingers')
REST_IDS = ('C7.rest_wrist', 'C7.rest_fingers')

def unit(v):
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-10:
        raise ValueError('C7 needs finite nondegenerate anatomical vectors')
    return v / np.linalg.norm(v)

def anatomy(elbow, wrist, hand_tip, index_base, little_base, fingers, side):
    forward = unit(np.asarray(hand_tip) - wrist)
    normal = unit(np.cross(forward, np.asarray(little_base) - index_base)) * (-1 if side == 'L' else 1)
    forearm = unit(np.asarray(wrist) - elbow)
    extension = math.degrees(math.atan2(float(forearm @ normal), float(forearm @ forward)))
    neutral = math.degrees(math.acos(float(np.clip(forearm @ forward, -1, 1))))
    bends = []
    for chain in fingers:
        angles = [math.degrees(math.atan2(float(unit(v) @ normal), float(unit(v) @ forward))) for v in chain]
        delta = np.degrees(np.arctan2(np.sin(np.radians(np.diff([0.0] + angles))), np.cos(np.radians(np.diff([0.0] + angles)))))
        bends.extend(abs(delta).tolist())
        bends.extend(abs(np.asarray(angles)).tolist())
    if len(fingers) != 4 or any((len(c) != 3 for c in fingers)):
        raise ValueError('C7 requires three phalanges on each of four fingers')
    return dict(forward=forward.tolist(), palm_normal=normal.tolist(), wrist_extension_deg=extension, wrist_neutral_deg=neutral, finger_bend_deg=float(max(bends)))

def surface_metrics(patches, palm_normal, vertices, floor):
    p = np.asarray(patches, float)
    if p.shape != (4, 3) or not np.isfinite(p).all():
        raise ValueError('Four finite palm skin patches required')
    _, singular, vh = np.linalg.svd(p - p.mean(0))
    if singular[1] < 1e-08:
        raise ValueError('Degenerate palm patch plane')
    n = vh[-1]
    if n @ palm_normal < 0:
        n = -n
    return dict(palm_points=p.tolist(), palm_plane_deg=math.degrees(math.acos(float(np.clip(-n[2], -1, 1)))), palm_gap_m=float(np.max(abs(p[:, 2] - floor))), ground_depth_m=max(0.0, float(floor - np.asarray(vertices)[:, 2].min())))

def required(trace):
    source = trace.get('source') or {}
    clip = str(source.get('clip', '')).lower()
    if trace.get('hand_support_required'):
        return True
    plan = trace.get('hand_support', {})
    if plan.get('bound'):
        return any((v.get('intervals') for v in plan.get('hands', {}).values()))
    return not (clip == 'rest' and (not source.get('cache_sha256')))

def validate(trace):
    plan = trace.get('hand_support')
    if plan:
        if plan.get('schema') != 'qa-hand-support/1':
            raise ValueError('Unknown hand-support schema')
        if plan.get('fps') != trace['fps']:
            raise ValueError('Hand-support fps mismatch')
        if set(plan.get('hands', {})) != {'L', 'R'}:
            raise ValueError('Both source hand labels required')
        total = plan.get('frame_count')
        if not isinstance(total, int) or total < 3:
            raise ValueError('Source hand frame count required')
        for row in plan['hands'].values():
            band = row.get('wrist_extension_range_deg')
            if band is not None:
                if not isinstance(band, (list, tuple)) or len(band) != 2 or any((not isinstance(v, (int, float)) or not math.isfinite(v) for v in band)) or (not 0 < band[0] < band[1] < 180):
                    raise ValueError('Ordered finite source wrist-extension band required')
                if not row.get('wrist_profile_reason'):
                    raise ValueError('Source wrist band requires a reviewed profile reason')
            last = 0
            for pair in row.get('intervals', []):
                if len(pair) != 2 or any((type(i) != int for i in pair)):
                    raise ValueError('Integer half-open source intervals required')
                lo, hi = pair
                if not 0 <= last <= lo < hi <= total:
                    raise ValueError('Invalid source hand interval')
                last = hi
        if plan.get('bound') and (not plan.get('source_sha256') or not plan.get('cache_sha256')):
            raise ValueError('Bound hand plan needs source and cache digests')
        actual = trace.get('source', {}).get('cache_sha256')
        if plan.get('bound') and actual != plan['cache_sha256']:
            raise ValueError('Hand plan belongs to another target cache')
    for row in list(trace.get('rest_hands', {}).values()) + [x for f in trace['frames'] for x in f.get('hands', {}).values()]:
        for key in ('wrist_extension_deg', 'wrist_neutral_deg', 'finger_bend_deg', 'palm_plane_deg', 'palm_gap_m', 'ground_depth_m'):
            if key in row and (not isinstance(row[key], (int, float)) or not math.isfinite(row[key])):
                raise ValueError('Finite C7 measurement required: ' + key)
            if key in row and key != 'wrist_extension_deg' and (row[key] < 0):
                raise ValueError('Nonnegative C7 measurement required: ' + key)
        if 'palm_points' in row:
            p = np.asarray(row['palm_points'])
            if p.shape != (4, 3) or not np.isfinite(p).all():
                raise ValueError('Four finite material palm patches required')

def evaluate_hands(trace, cfg):
    validate(trace)
    frames = trace['frames']
    h = trace['height_m']
    fps = trace['fps']
    plan = trace.get('hand_support', {})
    bound = bool(plan.get('bound'))
    requested = required(trace)
    all_ids = {f['frame'] for f in frames}
    checks = []
    samples = {k: [] for k in SUPPORT_IDS}
    expected = {s: {k + 1 for lo, hi in plan.get('hands', {}).get(s, {}).get('intervals', []) for k in range(lo, hi)} for s in 'LR'}
    bands = {s: plan.get('hands', {}).get(s, {}).get('wrist_extension_range_deg', [cfg['wrist_extension_min_deg'], cfg['wrist_extension_max_deg']]) for s in 'LR'}
    complete = bound and any(expected.values()) and all((v <= all_ids for v in expected.values()))
    for side in 'LR':
        previous = None
        for f in frames:
            if f['frame'] not in expected[side]:
                previous = None
                continue
            row = f.get('hands', {}).get(side, {})
            frame = f['frame']

            def record(cid, key, fn=lambda v: v):
                samples[cid].append((frame, fn(row[key]) if key in row else None, side))
            record('C7.palm_plane', 'palm_plane_deg')
            record('C7.palm_contact', 'palm_gap_m', lambda v: v / h)
            record('C7.hand_ground_penetration', 'ground_depth_m', lambda v: v / h)
            record('C7.wrist_extension', 'wrist_extension_deg', lambda v: max(0, bands[side][0] - v, v - bands[side][1]))
            record('C7.support_fingers', 'finger_bend_deg')
            if previous is not None:
                a, b = (previous.get('palm_points'), row.get('palm_points'))
                speed = float(np.max(np.linalg.norm(np.asarray(b) - a, axis=1)) * fps / h) if a is not None and b is not None else None
                samples['C7.hand_skate'].append((frame, speed, side))
            previous = row

    def add(cid, values, limit, unit_name, detail, available=True):
        valid = [(i, v, s) for i, v, s in values if v is not None]
        bad = [(i, v, s) for i, v, s in valid if v > limit]
        checks.append(dict(id=cid, status='FAIL' if bad else 'PASS' if available and values and (len(values) == len(valid)) else 'UNAVAILABLE', value=max((v for _, v, _ in valid), default=None), threshold=limit, unit=unit_name, frames=sorted({i for i, _, _ in bad if i > 0}), detail=detail, witnesses=[dict(frame=i, side=s, value=v) for i, v, s in bad], applicable=cid in REST_IDS or requested))
    for cid, limit, unit_name, detail in [('C7.palm_plane', cfg['palm_plane_deg'], 'degrees', 'Palm-facing fitted skin plane to downward floor normal; dorsal support fails.'), ('C7.palm_contact', cfg['palm_contact_height'], 'height', 'Maximum absolute height of four fixed palm skin patches during source support.'), ('C7.wrist_extension', cfg['wrist_extension_tolerance_deg'], 'degrees outside band', 'Signed anatomical extension outside reviewed source band or configured fallback; flexion is negative.'), ('C7.hand_skate', cfg['hand_skate_height_per_s'], 'height/s', 'Maximum 3D speed of fixed palm material patches within source episodes, including floating hands.'), ('C7.hand_ground_penetration', cfg['hand_ground_height'], 'height', 'Minimum evaluated hand/finger skin height below fixed ground.'), ('C7.support_fingers', cfg['finger_flat_deg'], 'degrees', 'Maximum sagittal phalanx deviation or joint bend, four fingers; spread allowed, thumb excluded.')]:
        add(cid, samples[cid] if bound else [], limit, unit_name, detail, complete)
    for cid, key, limit in [('C7.rest_wrist', 'wrist_neutral_deg', cfg['rest_wrist_deg']), ('C7.rest_fingers', 'finger_bend_deg', cfg['rest_finger_deg'])]:
        values = [(0, trace.get('rest_hands', {}).get(s, {}).get(key), s) for s in 'LR']
        add(cid, values, limit, 'degrees', 'Evaluated neutral rig, animation disabled; open neutral asset profile. Rest witnesses use frame 0.')
    return (checks, dict(required=requested, source_bound=bound, complete_support_coverage=complete, expected_frames={s: sorted(v) for s, v in expected.items()}, wrist_extension_bands_deg=bands, thumb_coverage=False, source=plan))

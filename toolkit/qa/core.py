from __future__ import annotations
import math, hashlib
from pathlib import Path
from typing import Any, TypedDict
import numpy as np
from .hands import DEFAULTS as HAND_DEFAULTS, evaluate_hands, validate as validate_hands
EVALUATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('hands.py').read_bytes()).hexdigest()

class Check(TypedDict):
    id: str
    status: str
    value: float | None
    threshold: float | None
    unit: str
    frames: list[int]
    detail: str
DEFAULTS = {**HAND_DEFAULTS, 'penetration_height': 0.0005, 'ground_penetration_height': 0.001, 'contact_height': 0.006, 'skate_contact_height': 0.0015, 'sole_angle_deg': 12.0, 'tiptoe_fraction': 0.35, 'skate_height_per_s': 0.08, 'float_height': 0.015, 'step_width_hips': 1.65, 'knee_width_hips': 1.65, 'step_width_leg': 0.35, 'section_retained': 0.55, 'volume_retained': 0.5, 'webbing_strain': 2.5, 'pop_height': 0.025, 'knee_deg': 165.0, 'elbow_deg': 170.0, 'hair_accel_height_per_s2': 0.25, 'hair_response_ratio': 0.01, 'support_margin_height': -0.04}

def validate_trace(t: dict) -> None:
    if t.get('schema') != 'qa-trace/1':
        raise ValueError('Expected qa-trace/1')
    if t.get('units') != 'metres-z-up':
        raise ValueError('Expected metres-z-up')
    if t.get('intent') not in ('walk', 'run', 'jump', 'static', 'rom', 'other'):
        raise ValueError('Known clip intent required')
    if not isinstance(t.get('floor_m', 0), (int, float)) or not math.isfinite(t.get('floor_m', 0)):
        raise ValueError('Finite fixed floor required')
    for key in ('height_m', 'hip_width_m', 'leg_length_m', 'fps'):
        if not isinstance(t.get(key), (int, float)) or not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError('Positive finite ' + key + ' required')
    if len(t.get('frames', [])) < 3:
        raise ValueError('At least three frames required')
    last = -math.inf
    for f in t['frames']:
        if not isinstance(f.get('frame'), int) or f['frame'] <= last:
            raise ValueError('Frame numbers must strictly increase')
        last = f['frame']
        for p in f.get('points', {}).values():
            if np.asarray(p).shape != (3,) or not np.isfinite(p).all():
                raise ValueError('Landmarks must be finite xyz vectors')
        for foot in f.get('feet', {}).values():
            for key in ('heel', 'toe'):
                if np.asarray(foot.get(key)).shape != (3,) or not np.isfinite(foot[key]).all():
                    raise ValueError('Foot probes must be finite xyz vectors')
            if not isinstance(foot.get('min_z'), (int, float)) or not math.isfinite(foot['min_z']):
                raise ValueError('Finite min_z required')
            for p in foot.get('support', []):
                if np.asarray(p).shape != (3,) or not np.isfinite(p).all():
                    raise ValueError('Finite support xyz required')
        for key in ('hand_depth_m', 'hand_crossing_extent_m', 'hair_depth_m', 'section_retained', 'volume_retained', 'webbing_strain'):
            v = f.get(key)
            if v is not None and (not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0):
                raise ValueError('Nonnegative finite ' + key + ' or null required')
    if any((b['frame'] - a['frame'] != 1 for a, b in zip(t['frames'], t['frames'][1:]))):
        raise ValueError('Full-rate consecutive frames required')
    if t.get('scope', 'all') not in ('all', 'c7'):
        raise ValueError('Known QA scope required')
    validate_hands(t)

def episodes(mask):
    a = np.asarray(mask, bool)
    ends = np.diff(np.r_[False, a, False].astype(int))
    return list(zip(np.flatnonzero(ends == 1), np.flatnonzero(ends == -1)))

def bend(a, b, c):
    u = np.asarray(a) - b
    v = np.asarray(c) - b
    d = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    return 180 - np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1) / np.maximum(d, 1e-12), -1, 1)))

def foot_phases(feet, floor, tolerance, angle):
    heel = np.array([f['heel'] for f in feet])
    toe = np.array([f['toe'] for f in feet])
    delta = toe - heel
    pitch = np.degrees(np.arctan2(abs(delta[:, 2]), np.linalg.norm(delta[:, :2], axis=1)))
    hc = abs(heel[:, 2] - floor) <= tolerance
    tc = abs(toe[:, 2] - floor) <= tolerance
    return (np.where(hc & tc & (pitch <= angle), 'flat', np.where(hc & ~tc, 'heel', np.where(tc & ~hc, 'toe', 'air'))), pitch)

def support_margin(point, supports):
    pts = sorted(set((tuple(map(float, p[:2])) for p in supports)))
    if len(pts) < 3:
        return None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    hi = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    hull = np.asarray(lo[:-1] + hi[:-1])
    q = np.asarray(point[:2])
    d = []
    inside = True
    if len(hull) < 3:
        return None
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        e = b - a
        t = np.clip(np.dot(q - a, e) / np.dot(e, e), 0, 1)
        d.append(np.linalg.norm(q - a - t * e))
        inside &= cross(a, b, q) >= -1e-12
    return float(min(d)) * (1 if inside else -1)

def evaluate(trace: dict, thresholds: dict | None=None) -> dict[str, Any]:
    validate_trace(trace)
    if thresholds:
        if set(thresholds) - set(DEFAULTS):
            raise ValueError('Unknown threshold key')
        if any((not isinstance(v, (int, float)) or not math.isfinite(v) for v in thresholds.values())):
            raise ValueError('Finite numeric thresholds required')
        if any((v <= 0 for k, v in thresholds.items() if k != 'support_margin_height')):
            raise ValueError('Positive thresholds required')
    cfg = {**DEFAULTS, **(thresholds or {})}
    fs = trace['frames']
    n = len(fs)
    if cfg['wrist_extension_min_deg'] >= cfg['wrist_extension_max_deg']:
        raise ValueError('Ordered wrist extension band required')
    h = trace['height_m']
    hip = trace['hip_width_m']
    fps = trace['fps']
    ids = [f['frame'] for f in fs]
    checks = []
    perframe = [dict(frame=i, flags=[], phases={}) for i in ids]

    def add(cid, values, limit, unit, detail, lower=False, available=True):
        if values is None or not len(values):
            checks.append(Check(id=cid, status='UNAVAILABLE', value=None, threshold=limit, unit=unit, frames=[], detail=detail))
            return
        valid = [(int(i), float(v)) for i, v in values if v is not None and math.isfinite(v)]
        bad = [i for i, v in valid if (v < limit if lower else v > limit)]
        status = 'FAIL' if bad else 'PASS' if available and len(valid) == len(values) else 'UNAVAILABLE'
        for i in bad:
            if i in ids:
                perframe[ids.index(i)]['flags'].append(cid)
        checks.append(Check(id=cid, status=status, value=(min if lower else max)((v for _, v in valid)) if valid else None, threshold=limit, unit=unit, frames=sorted(set(bad)), detail=detail))

    def series(field):
        return [(f['frame'], f.get(field)) for f in fs]
    coverage = trace.get('coverage', {})
    hands = [(f['frame'], max(f.get('hand_depth_m') or 0.0, f.get('hand_crossing_extent_m') or 0.0) if f.get('hand_depth_m') is not None or f.get('hand_crossing_extent_m') else None) for f in fs]
    add('C1.hand_penetration', hands, cfg['penetration_height'] * h, 'm', 'Volume depth or confirmed transverse hand-edge crossing extent. Per-frame fields distinguish the two; crossing is not signed depth. Open coverage cannot pass.', available=coverage.get('hand_volume', False))
    add('C2.hair_penetration', series('hair_depth_m'), cfg['penetration_height'] * h, 'm', 'Hair below root exclusion against body/arms.', available=coverage.get('hair_volume', False))
    hair = np.array([f.get('hair_relative_tips', [f.get('hair_relative_tip', [np.nan] * 3)]) for f in fs], float)
    heads = np.array([f.get('points', {}).get('head', [np.nan] * 3) for f in fs], float)
    if np.isfinite(hair).all() and np.isfinite(heads).all():
        accel = np.linalg.norm(np.diff(heads, n=2, axis=0) * fps ** 2, axis=1)
        excitation = float(np.sqrt(np.mean(accel ** 2))) / h
        amplitude = float(np.sqrt(np.mean(np.sum(np.ptp(hair, axis=0) ** 2, axis=-1)))) / h
        response = amplitude / max(excitation, 1e-09)
        excited = excitation >= cfg['hair_accel_height_per_s2']
        add('C2.hair_response', [(ids[int(np.argmax(accel)) + 1], response)] if excited else None, cfg['hair_response_ratio'], 's²', 'Relative tip range / RMS head acceleration. Provisional expressive-hair band; insufficient excitation is unavailable.', lower=True, available=coverage.get('hair_tips', False))
    else:
        add('C2.hair_response', None, cfg['hair_response_ratio'], 's²', 'Mapped moving hair tips required', lower=True)
    landing = [hands[i] for i, f in enumerate(fs) if f.get('landing')]
    add('C6.landing_hands', landing, cfg['penetration_height'] * h, 'm', 'C1 restricted to declared landing windows', available=coverage.get('hand_volume', False))
    tip = []
    skate = []
    penetration = []
    floating = []
    flat_diagnostics = {}
    support_by_frame = [[] for _ in fs]
    for side in ('L', 'R'):
        feet = [f.get('feet', {}).get(side) for f in fs]
        if any((x is None for x in feet)):
            continue
        phases, pitch = foot_phases(feet, trace.get('floor_m', 0), cfg['contact_height'] * h, cfg['sole_angle_deg'])
        intent = np.array([bool(f.get('stance', {}).get(side, phases[k] != 'air')) for k, f in enumerate(fs)])
        for k, phase in enumerate(phases):
            perframe[k]['phases'][side] = str(phase)
        middle = np.zeros(n, bool)
        persistent = np.zeros(n, bool)
        for lo, hi in episodes(intent):
            trim = max(1, int((hi - lo) * 0.2))
            if hi - lo < max(4, math.ceil(fps * 0.2)):
                continue
            middle[lo + trim:max(lo + trim, hi - trim)] = True
            flat_runs = episodes(phases[lo:hi] == 'flat')
            has_flat = any((b - a >= max(2, math.ceil(fps / 12)) for a, b in flat_runs))
            if not has_flat:
                persistent[lo:hi] = True
        use = middle & intent
        elevated = (pitch > cfg['sole_angle_deg']) & persistent & intent
        fraction = float(np.mean(elevated[use])) if use.any() else None
        if trace.get('intent') == 'walk' and fraction is not None:
            tip.extend(((ids[k], fraction if elevated[k] else 0.0) for k in np.flatnonzero(use)))
        flat_diagnostics[side] = dict(phase_counts={p: int(np.count_nonzero(phases == p)) for p in ('heel', 'flat', 'toe', 'air')}, midstance_samples=int(use.sum()), tiptoe_fraction=fraction, max_pitch_deg=float(pitch.max()))
        for k, ft in enumerate(feet):
            penetration.append((ids[k], max(0, trace.get('floor_m', 0) - ft['min_z'])))
            if intent[k]:
                floating.append((ids[k], max(0, ft['min_z'] - trace.get('floor_m', 0))))
            if phases[k] != 'air':
                support_by_frame[k].extend(ft.get('support', []))
            if k and intent[k] and intent[k - 1] and trace.get('travel_observable', False):
                candidates = []
                support = np.asarray(ft.get('support', []))
                previous = np.asarray(feet[k - 1].get('support', []))
                if len(support) and support.shape == previous.shape:
                    planted = (abs(support[:, 2] - trace.get('floor_m', 0)) < cfg['skate_contact_height'] * h) & (abs(previous[:, 2] - trace.get('floor_m', 0)) < cfg['skate_contact_height'] * h)
                    if planted.any():
                        candidates.append(float(np.median(np.linalg.norm((support - previous)[planted, :2], axis=1) * fps / h)))
                else:
                    for key in ('heel', 'toe'):
                        if max(abs(ft[key][2] - trace.get('floor_m', 0)), abs(feet[k - 1][key][2] - trace.get('floor_m', 0))) < cfg['skate_contact_height'] * h:
                            v = (np.array(ft[key]) - feet[k - 1][key]) * fps
                            candidates.append(float(np.linalg.norm(v[:2])) / h)
                if candidates:
                    skate.append((ids[k], max(candidates)))
    add('C3.tiptoe', tip, cfg['tiptoe_fraction'], 'fraction', 'Persistent tilted middle stance without a sustained foot-flat interval; normal terminal heel rise allowed. Walk intent only.')
    add('C3.skate', skate, cfg['skate_height_per_s'], 'height/s', 'Median speed of fixed sole material points within 0.0015 stature of the floor in both frames; max across contact frames.')
    add('C3.ground_penetration', penetration, cfg['ground_penetration_height'] * h, 'm', 'Minimum evaluated foot surface below floor; numerical/contact envelope from reference and VRoid full walks.')
    add('C3.float', floating, cfg['float_height'] * h, 'm', 'Stance sole gap; requires declared or inferred stance.', available=coverage.get('stance', False))
    widths = []
    knees = []
    angles = []
    points = {}
    for key in fs[0].get('points', {}):
        if all((key in f.get('points', {}) for f in fs)):
            points[key] = np.array([f['points'][key] for f in fs])
    needed = ['Lhip', 'Rhip', 'Lknee', 'Rknee', 'Lankle', 'Rankle']
    if all((k in points for k in needed)):
        lat = points['Lhip'] - points['Rhip']
        lat[:, 2] = 0
        lat /= np.maximum(np.linalg.norm(lat, axis=1)[:, None], 1e-09)
        widths = list(zip(ids, abs(np.sum((points['Lankle'] - points['Rankle']) * lat, axis=1)) / hip))
        knees = list(zip(ids, abs(np.sum((points['Lknee'] - points['Rknee']) * lat, axis=1)) / hip))
        for s in 'LR':
            for j, a, b, c, lim in [('knee', 'hip', 'knee', 'ankle', cfg['knee_deg']), ('elbow', 'shoulder', 'elbow', 'wrist', cfg['elbow_deg'])]:
                if all((s + x in points for x in (a, b, c))):
                    angles.extend(((i, float(v - lim)) for i, v in zip(ids, bend(points[s + a], points[s + b], points[s + c]))))
    add('C4.step_width', widths if trace.get('intent') == 'walk' else None, cfg['step_width_hips'], 'hip widths', 'Lateral ankle separation in pelvis frame; footstep diagnostic, not clinical norm.')
    add('C4.knee_separation', knees if trace.get('intent') == 'walk' else None, cfg['knee_width_hips'], 'hip widths', 'Lateral knee separation in pelvis frame.')
    add('C4.width_leg_length', [(i, v * hip / trace['leg_length_m']) for i, v in widths] if trace.get('intent') == 'walk' else None, cfg['step_width_leg'], 'leg lengths', 'Companion width metric: an over-wide fitted pelvis can conceal splay in hip-normalized ratios.')
    for cid, field, limit, lower, detail in [('C5.section', 'section_retained', cfg['section_retained'], True, 'Smallest local elbow/knee contour area / bind area.'), ('C5.volume', 'volume_retained', cfg['volume_retained'], True, 'Rest-defined material section volume / bind volume.'), ('C5.webbing', 'webbing_strain', cfg['webbing_strain'], False, 'Maximum p99 arm/sleeve triangle-edge stretch; a deformation indicator, not silhouette proof.')]:
        add(cid, series(field), limit, 'ratio', detail, lower=lower, available=coverage.get(field, False))
    pops = []
    jitter = {}
    if 'hips' in points:
        for key, p in points.items():
            if key == 'hips':
                continue
            residual = np.linalg.norm(np.diff(p - points['hips'], n=2, axis=0), axis=1) / h
            median = float(np.median(residual))
            mad = float(np.median(abs(residual - median)))
            adaptive = median + 6 * 1.4826 * mad if len(residual) >= 8 else 0.0
            limit = max(cfg['pop_height'] * (24 / fps) ** 2, adaptive)
            jitter[key] = dict(raw_peak=float(residual.max()), limit=limit, unit='height/frame²')
            pops.extend(zip(ids[1:-1], residual / limit))
    add('J1.pop', pops, 1.0, 'local threshold multiples', 'Root-relative second difference > max(24-fps absolute floor, median + 6 scaled MAD). Detects isolated pops; sustained rough motion can escape.')
    add('J2.joint_limits', angles, 0.0, 'degrees beyond limit', 'Unsigned knee/elbow bend; signed twist needs explicit local axes.')
    balance = []
    if trace.get('intent') == 'static':
        for k, f in enumerate(fs):
            if f.get('com') is not None:
                m = support_margin(f['com'], support_by_frame[k])
                balance.append((ids[k], m / h if m is not None else None))
    add('B1.static_support', balance, cfg['support_margin_height'], 'height', 'Static COM/support polygon only; walking requires dynamic balance.', lower=True, available=coverage.get('mass_model', False))
    hand_checks, hand_diagnostics = evaluate_hands(trace, cfg)
    checks.extend(hand_checks)
    for c in hand_checks:
        for f in perframe:
            if f['frame'] in c['frames']:
                f['flags'].append(c['id'])
    if trace.get('scope') == 'c7':
        checks = hand_checks
        for f in perframe:
            f['flags'] = [x for x in f['flags'] if x.startswith('C7.')]
    applicable = [c for c in checks if c.get('applicable', True) and (not (c['id'] == 'C6.landing_hands' and (not any((f.get('landing') for f in fs))) or (c['id'] == 'B1.static_support' and trace.get('intent') != 'static') or (c['id'].startswith('C4.') and trace.get('intent') != 'walk') or (c['id'] == 'C3.tiptoe' and trace.get('intent') != 'walk')))]
    failed = [c['id'] for c in applicable if c['status'] == 'FAIL']
    missing = [c['id'] for c in applicable if c['status'] == 'UNAVAILABLE']
    calibrated = bool(trace.get('calibration', {}).get('validated'))
    return dict(schema='qa-report/1', evaluator_sha256=EVALUATOR_SHA256, status='FAIL' if failed else 'BLOCKED' if missing or not calibrated else 'PASS', presentation_allowed=not failed and (not missing) and calibrated and (trace.get('scope', 'all') == 'all'), failed=failed, unavailable=missing, calibration=trace.get('calibration', dict(validated=False, reason='Thresholds are provisional engineering bands')), source=trace.get('source'), fps=fps, frame_count=n, thresholds=cfg, checks=checks, frames=perframe, foot_phases=flat_diagnostics, jitter=jitter, coverage=coverage, hand_support=hand_diagnostics, scope=trace.get('scope', 'all'), limitations=['No finite mesh sampler proves absence of all intersections.', 'C5 webbing strain does not prove every silhouette is acceptable.', 'Dynamic balance and signed joint twist require explicit rig/mass models.'])

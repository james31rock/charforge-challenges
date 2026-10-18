import json, hashlib
from pathlib import Path
import numpy as np
SIDES = ('left', 'right')
REQUIRED = ('hips', 'chest', 'head') + tuple((s + p for s in SIDES for p in ('UpperLeg', 'LowerLeg', 'Foot', 'Toes')))

def load(path):
    d = json.loads(Path(path).read_text(encoding='utf8'))
    if d.get('kind') != 'charforge-gait-trace':
        raise ValueError('Expected a trace or an input descriptor; videos need a matching 3D sidecar')
    if d.get('schema_version') != 1 or not np.isfinite(d.get('cycle_seconds', 0)) or d['cycle_seconds'] <= 0:
        raise ValueError('Invalid duration/version')
    rows = d['samples']
    ph = np.array([r['phase'] for r in rows])
    origin = d.get('phase_origin', 0)
    if len(rows) < 8 or not np.isfinite(ph).all() or ph[0] < 0 or (ph[-1] >= 1) or (np.diff(ph) <= 0).any() or (not 0 <= origin < 1):
        raise ValueError('Need unique increasing phases in [0,1)')
    if not np.isclose(ph[0], 0, atol=1e-08) or not np.allclose(np.diff(ph), 1 / len(ph), atol=1e-06):
        raise ValueError('Uniform complete cycle starting at phase zero required; resample before comparing')
    for row in [d['rest']] + [r['joints'] for r in rows]:
        for n in REQUIRED:
            a = np.asarray(row[n], float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError('Nonfinite/missing joint ' + n)
    return d

def unit(x):
    a = np.asarray(x, float)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if (n < 1e-10).any():
        raise ValueError('Degenerate segment or hip width')
    return a / n

def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))

def dot(a, b):
    return np.sum(a * b, axis=-1)

def smoothness(y, T, periodic=True):
    y = np.asarray(y, float)
    n = len(y)
    dy = np.roll(y, -1) - y
    d3 = np.roll(y, -3) - 3 * np.roll(y, -2) + 3 * np.roll(y, -1) - y
    if not periodic:
        dy = np.diff(y)
        d3 = np.diff(y, 3)
    spectrum = np.fft.rfft(y - y.mean())
    power = abs(spectrum) ** 2
    return {'range': float(np.ptp(y)), 'max_step': float(abs(dy).max()), 'peak_phase_jerk': float(abs(d3).max() * n ** 3), 'peak_time_jerk': float(abs(d3).max() * (n / T) ** 3), 'high_harmonic_power_fraction': float(power[4:].sum() / max(power[1:].sum(), 1e-30)) if periodic else None, 'derivative_boundary': 'cyclic' if periodic else 'interior only; raw capture not loop-closed', 'end_to_start_delta': float(y[0] - y[-1])}

def measure(d):
    p = {n: np.array([r['joints'][n] for r in d['samples']]) for n in d['samples'][0]['joints']}
    rest = {n: np.array(v) for n, v in d['rest'].items()}
    N = len(d['samples'])
    T = d['cycle_seconds']
    z = np.array([0.0, 0.0, 1.0])
    leg = float(np.mean([sum((np.linalg.norm(rest[s + a] - rest[s + b]) for a, b in [('UpperLeg', 'LowerLeg'), ('LowerLeg', 'Foot')])) for s in SIDES]))
    if leg <= 1e-08:
        raise ValueError('Zero leg length')
    hipvec = p['leftUpperLeg'] - p['rightUpperLeg']
    lat = hipvec.copy()
    lat[:, 2] = 0
    lat = unit(lat)
    fwd = np.cross(lat, z)
    base_lat = rest['leftUpperLeg'] - rest['rightUpperLeg']
    base_lat[2] = 0
    base_lat = unit(base_lat)
    base_fwd = np.cross(base_lat, z)
    width = float(np.linalg.norm(rest['leftUpperLeg'] - rest['rightUpperLeg']))
    trunk = unit(p['chest'] - p['hips'])
    periodic = d.get('periodic', d.get('provenance', {}).get('adapter') != 'bvh')
    curves = {'phase': [r['phase'] for r in d['samples']], 'pelvis_height_L': (p['hips'][:, 2] / leg).tolist(), 'hip_centre_height_L': ((p['leftUpperLeg'][:, 2] + p['rightUpperLeg'][:, 2]) / (2 * leg)).tolist(), 'step_width_H': (abs(dot(p['leftFoot'] - p['rightFoot'], lat)) / width).tolist(), 'step_width_L': (abs(dot(p['leftFoot'] - p['rightFoot'], lat)) / leg).tolist(), 'knee_separation_H': (abs(dot(p['leftLowerLeg'] - p['rightLowerLeg'], lat)) / width).tolist()}
    summary = {'leg_length_m': leg, 'hip_width_m': width, 'hip_width_L': width / leg, 'pelvis': smoothness(p['hips'][:, 2] / leg, T, periodic), 'physical_pelvis_range_mm': float(np.ptp(p['hips'][:, 2]) * 1000), 'physical_pelvis_peak_jerk_m_s3': smoothness(p['hips'][:, 2], T, periodic)['peak_time_jerk'], 'step_width_H_max': max(curves['step_width_H']), 'knee_separation_H_max': max(curves['knee_separation_H']), 'rest_knee_separation_H': float(abs(dot(rest['leftLowerLeg'] - rest['rightLowerLeg'], base_lat)) / width), 'rest_foot_separation_H': float(abs(dot(rest['leftFoot'] - rest['rightFoot'], base_lat)) / width), 'true_COM': 'unavailable: character-specific segment masses/inertias not supplied'}
    phases = np.array(curves['phase'])
    origin = d.get('phase_origin', 0)
    left_ic = int(np.argmin(abs((phases - origin + 0.5) % 1 - 0.5)))
    right_ic = int(np.argmin(abs((phases - origin - 0.5 + 0.5) % 1 - 0.5)))
    contact_width = float(abs((p['leftFoot'][left_ic] - p['rightFoot'][right_ic]) @ base_lat))
    summary.update(contact_step_width_L=contact_width / leg, contact_step_width_H=contact_width / width, contact_step_width_method='ankle centres at declared/proxy left 0% and right 50% initial contacts; one step pair', progression_axis=base_fwd.tolist(), progression_axis_method='fixed rest-hip transverse axis crossed with world up', width_curve_method='instantaneous rotating-pelvis ankle separation; distinct from consecutive-contact step width')
    segments = d.get('leg_mass_model') or [{'a': 'hips', 'b': 'head', 'mass': 0.678, 'fraction': 0.55}, *[{'a': s + a, 'b': s + b, 'mass': m, 'fraction': f} for s in SIDES for a, b, m, f in [('UpperLeg', 'LowerLeg', 0.1, 0.433), ('LowerLeg', 'Foot', 0.0465, 0.433), ('Foot', 'Toes', 0.0145, 0.5)]]]
    masses = sum((s['mass'] for s in segments))
    if masses <= 0 or any((s['mass'] < 0 or not 0 <= s['fraction'] <= 1 for s in segments)):
        raise ValueError('Invalid COM mass model')
    com = sum((s['mass'] * (p[s['a']] * (1 - s['fraction']) + p[s['b']] * s['fraction']) for s in segments)) / masses
    curves['COM_proxy_height_L'] = (com[:, 2] / leg).tolist()
    summary['COM_proxy'] = smoothness(com[:, 2] / leg, T, periodic)
    summary['COM_proxy_model'] = 'supplied segment model' if d.get('leg_mass_model') else 'engineering proxy: HAT .678; each thigh .10, shank .0465, foot .0145; NOT validated cartoon mass distribution'
    for s, sign in [('left', 1), ('right', -1)]:
        h, k, a, to = [p[s + n] for n in ('UpperLeg', 'LowerLeg', 'Foot', 'Toes')]
        thigh = k - h
        shank = a - k
        foot = to - a
        axis = unit(a - h)
        bow = k - h - axis * dot(k - h, axis)[:, None]
        r = np.linalg.norm(bow, axis=1)
        knee = angle(thigh, shank)
        hip = np.degrees(np.arctan2(dot(thigh, fwd), -thigh[:, 2])) - np.degrees(np.arctan2(dot(trunk, fwd), trunk[:, 2]))

        def ankle_geom(sh, ft, fw):
            return np.degrees(np.arctan2(ft[..., 2], dot(ft, fw))) - np.degrees(np.arctan2(dot(sh, fw), -sh[..., 2]))
        ankle = ankle_geom(shank, foot, fwd) - ankle_geom(rest[s + 'Foot'] - rest[s + 'LowerLeg'], rest[s + 'Toes'] - rest[s + 'Foot'], base_fwd)
        heading = np.degrees(np.arctan2(sign * dot(foot, base_lat), dot(foot, base_fwd)))
        heading_valid = np.linalg.norm(foot[:, :2], axis=1) / np.linalg.norm(foot, axis=1) > 0.25
        knee_heading = np.degrees(np.arctan2(sign * dot(bow, base_lat), dot(bow, base_fwd)))
        knee_pelvis_heading = np.degrees(np.arctan2(sign * dot(bow, lat), dot(bow, fwd)))
        valid = (r / leg > 0.005) & (knee > 3.0)
        curves[s + '_hip_deg'] = hip.tolist()
        curves[s + '_knee_deg'] = knee.tolist()
        curves[s + '_ankle_deg'] = ankle.tolist()
        curves[s + '_foot_progression_deg'] = [float(v) if ok else None for v, ok in zip(heading, heading_valid)]
        curves[s + '_knee_progression_deg'] = [float(v) if ok else None for v, ok in zip(knee_heading, valid)]
        curves[s + '_knee_plane_pelvis_deg'] = [float(v) if ok else None for v, ok in zip(knee_pelvis_heading, valid)]
        curves[s + '_knee_bow_L'] = (sign * dot(bow, lat) / leg).tolist()
        curves[s + '_knee_minus_ankle_L'] = (sign * dot(k - a, lat) / leg).tolist()
        sole = [row.get('feet', {}).get(s, {}).get('min_z_m') for row in d['samples']]
        floor = d.get('floor_m', 0)
        if all((v is not None for v in sole)):
            clearance = (np.array(sole) - floor) / leg
            curves[s + '_sole_clearance_L'] = clearance.tolist()
            summary[s + '_native_floor'] = {'min_L': float(clearance.min()), 'max_L': float(clearance.max()), 'near_floor_fraction': float(np.mean(clearance < 0.01)), 'method': 'rest-selected native foot vertices; near-floor fraction is NOT stance'}
        phase = np.mod(np.asarray(curves['phase']) - d.get('phase_origin', 0) - (0 if s == 'left' else 0.5), 1)
        stance = d.get('declared_stance_fraction')
        if stance is not None:
            if not 0 < float(stance) < 1:
                raise ValueError('Invalid declared stance fraction')
            mask = phase < float(stance)
            method = 'declared source schedule, not force-plate observation'
        else:
            rel = dot(a - p['hips'], fwd)
            speed = np.roll(rel, -1) - np.roll(rel, 1)
            mask = (speed < 0) & (a[:, 2] < np.quantile(a[:, 2], 0.65))
            method = 'ankle position/velocity proxy, not ground truth'
        fraction = float(stance) if stance is not None else float(mask.mean())
        curves[s + '_stance'] = mask.tolist()
        summary[s + '_stance'] = {'fraction': fraction, 'sampled_fraction': float(mask.mean()), 'swing_fraction': 1 - fraction, 'ratio': fraction / max(1 - fraction, 1e-09), 'method': method}
        summary[s + '_angles'] = {'hip_range': [float(hip.min()), float(hip.max())], 'knee_range': [float(knee.min()), float(knee.max())], 'ankle_range': [float(ankle.min()), float(ankle.max())], 'knee_heading_valid_frames': int(valid.sum()), 'knee_heading_range': [float(knee_heading[valid].min()), float(knee_heading[valid].max())] if valid.any() else None, 'foot_progression_range': [float(heading[heading_valid].min()), float(heading[heading_valid].max())] if heading_valid.any() else None, 'stance_foot_progression_range': [float(heading[mask & heading_valid].min()), float(heading[mask & heading_valid].max())] if (mask & heading_valid).any() else None, 'knee_bow_L_max': max(curves[s + '_knee_bow_L'])}
    return {'label': d['label'], 'summary': summary, 'curves': curves, 'phase_origin': d.get('phase_origin', 0), 'phase_method': d.get('phase_method')}

def phase_values(values, origin, phases):
    a = np.array([np.nan if x is None else x for x in values], float)
    n = len(a)
    return np.interp(np.mod(phases + origin, 1), np.arange(n) / n, a, period=1)

def compare(ref, test):
    r, t = (measure(ref), measure(test))
    phase = np.arange(101) / 100
    diff = {}
    for k in r['curves'].keys() & t['curves'].keys():
        if k == 'phase' or k.endswith('_stance'):
            continue
        a = phase_values(r['curves'][k], r['phase_origin'], phase)
        b = phase_values(t['curves'][k], t['phase_origin'], phase)
        if 'height_L' in k:
            a -= np.nanmean(a)
            b -= np.nanmean(b)
        valid = np.isfinite(a) & np.isfinite(b)
        for data in [ref, test]:
            if not data.get('periodic', data.get('provenance', {}).get('adapter') != 'bvh'):
                native = np.mod(phase + data.get('phase_origin', 0), 1)
                valid &= native <= data['samples'][-1]['phase']
        diff[k] = {'rmse': float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2))) if valid.any() else None, 'coverage': float(valid.mean())}
    return {'schema_version': 1, 'kind': 'charforge-gait-comparison', 'status': 'diagnostic-not-acceptance', 'reference': r, 'test': t, 'phase_matched_error': diff, 'limits': ['Adult clinical context is advisory, not a chibi/child acceptance threshold.', 'COM estimate is model-dependent, not measured physical COM.', 'Joint-centre geometry is not an anatomical joint coordinate system.', 'Knee heading is undefined near extension.', 'Stance is declared or a kinematic proxy, not measured force contact.']}

from dataclasses import dataclass
import math
import numpy as np

def vec(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected a finite 3-vector')
    return a

def unit(value):
    a = vec(value)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError('Degenerate direction')
    return a / n

def frame(up, forward):
    z = unit(up)
    f = vec(forward) - z * np.dot(forward, z)
    f = unit(f)
    return np.column_stack((np.cross(z, f), -f, z))

def perpendicular(axis, hint):
    a = unit(axis)
    h = vec(hint) - a * np.dot(hint, a)
    if np.linalg.norm(h) < 1e-08:
        basis = np.eye(3)[np.argmin(abs(a))]
        h = basis - a * np.dot(basis, a)
    return unit(h)

@dataclass(frozen=True)
class IKResult:
    knee: np.ndarray
    ankle: np.ndarray
    reach_error: float
    reachable: bool

def two_bone(hip, ankle, upper, lower, forward, lateral_offset=0.0, lateral=(1, 0, 0)):
    h, target = (vec(hip), vec(ankle))
    if min(upper, lower) <= 0 or not np.isfinite([upper, lower, lateral_offset]).all():
        raise ValueError('Invalid limb dimensions')
    delta = target - h
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        raise ValueError('Coincident hip and ankle')
    axis = delta / length
    eps = (upper + lower) * 1e-10
    d = float(np.clip(length, abs(upper - lower) + eps, upper + lower - eps))
    end = h + axis * d
    along = (upper ** 2 - lower ** 2 + d ** 2) / (2 * d)
    radius = math.sqrt(max(0.0, upper ** 2 - along ** 2))
    f = perpendicular(axis, forward)
    side = unit(np.cross(f, axis))
    lat = unit(lateral)
    a, b = (np.dot(f, lat), np.dot(side, lat))
    amplitude = math.hypot(a, b)
    theta = 0.0
    if radius > eps and amplitude > 1e-09:
        phi = math.atan2(a, b)
        t = math.asin(float(np.clip(lateral_offset / (radius * amplitude), -1.0, 1.0)))
        candidates = [t - phi, math.pi - t - phi]
        theta = max(candidates, key=math.cos)
    knee = h + axis * along + radius * (f * math.cos(theta) + side * math.sin(theta))
    error = abs(d - length)
    return IKResult(knee, end, error, error <= (upper + lower) * 1e-08)

def knee_plane(hip, knee, ankle, forward, lateral, attenuation=0.25):
    h, k, a = map(vec, (hip, knee, ankle))
    axis = unit(a - h)
    bow = k - h - axis * np.dot(k - h, axis)
    return two_bone(h, a, np.linalg.norm(k - h), np.linalg.norm(a - k), forward, np.dot(bow, unit(lateral)) * attenuation, lateral)

def cyclic_lowpass(values, samples_per_cycle, harmonics=1):
    a = np.asarray(values, dtype=float)
    if samples_per_cycle < 3 or len(a) % samples_per_cycle or harmonics < 0 or (not np.isfinite(a).all()):
        raise ValueError('Require complete, finite cycles and a nonnegative harmonic count')
    cycle = a.reshape((-1, samples_per_cycle) + a.shape[1:]).mean(axis=0)
    spectrum = np.fft.rfft(cycle, axis=0)
    spectrum[harmonics + 1:] = 0
    cycle = np.fft.irfft(spectrum, n=samples_per_cycle, axis=0)
    return np.concatenate([cycle] * (len(a) // samples_per_cycle), axis=0)

def shoulder_counterrotation(pelvis_yaw, leg_length, hip_width, stride_length):
    if not np.isfinite([leg_length, hip_width, stride_length]).all() or leg_length <= 0 or min(hip_width, stride_length) < 0:
        raise ValueError('Invalid proportions')
    p = np.unwrap(np.asarray(pelvis_yaw, dtype=float))
    if p.size < 2 or not np.isfinite(p).all():
        raise ValueError('Need a finite yaw sequence')
    gain = float(np.clip(0.5 * (1 + hip_width / leg_length) / (1 + 0.25 * stride_length / leg_length), 0.3, 0.8))
    return -gain * (p - p.mean())

def arm_swing(phase, leg_length, arm_length, stride_length, hip_width):
    if not np.isfinite([phase, leg_length, arm_length, stride_length, hip_width]).all() or min(leg_length, arm_length) <= 0 or min(stride_length, hip_width) < 0:
        raise ValueError('Invalid proportions')
    swing = math.atan2(0.5 * stride_length, arm_length)
    clearance = math.atan2(0.25 * hip_width, arm_length)
    return dict(swing=swing * math.sin(phase), elbow=0.18 + 0.25 * (1 - math.cos(phase - 0.35)), clearance=clearance)

def lock_contact(hip, ankle, contact_local, foot_rotation, anchor, upper, lower, forward):
    rotation = np.asarray(foot_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-06) or np.linalg.det(rotation) < 0.99:
        raise ValueError('Foot rotation must be orthonormal')
    vec(ankle)
    target = vec(anchor) - rotation @ vec(contact_local)
    result = two_bone(hip, target, upper, lower, forward)
    residual = np.linalg.norm(result.ankle + rotation @ vec(contact_local) - vec(anchor))
    return (result, float(residual))

def corrective_profile(radius, proximal_length, distal_length):
    if not np.isfinite([radius, proximal_length, distal_length]).all() or min(radius, proximal_length, distal_length) <= 0:
        raise ValueError('Nonpositive girth or segment length')
    slenderness = radius / min(proximal_length, distal_length)
    return dict(start_angle=float(np.clip(math.atan2(1.0, 4 * slenderness), 0.35, 1.05)), support=min(2.5 * radius, 0.4 * min(proximal_length, distal_length)), max_radial_scale=float(np.clip(1 + slenderness, 1.03, 1.25)))

from dataclasses import dataclass
import json, math
from pathlib import Path
import numpy as np
PRESETS = json.loads(Path(__file__).with_name('presets.json').read_text())

def resolve_preset(name, measurements, length_m=None):
    h = float(measurements['height_cm']) / 100
    mass = float(measurements['mass_proxy_kg'])
    if not (math.isfinite(h) and 0.1 <= h <= 10 and math.isfinite(mass) and (mass > 0)):
        raise ValueError('Explicit finite height_cm and positive mass_proxy_kg required')
    p = dict(PRESETS['presets'][name])
    length = float(length_m if length_m is not None else h * p['length_height_ratio'])
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Positive measured feature length required')
    omega = 2 * math.pi * p['frequency_hz'] * math.sqrt(PRESETS['reference_length_m'] / length)
    feature_mass = mass * p['mass_fraction']
    node_mass = feature_mass / (p['chains'] * p['segments'])
    p.update(name=name, length_m=length, feature_mass_kg=feature_mass, particle_mass_kg=node_mass, stiffness_n_m=node_mass * omega ** 2, damping_n_s_m=2 * p['damping_ratio'] * node_mass * omega, omega=omega, radius_m=length * p['radius_length_ratio'], gravity_m_s2=9.81 * p['gravity_scale'], settle_limit_s=6 / (p['damping_ratio'] * omega) + 0.4)
    return p

@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    radius: float

    def clearance(self, points, radius=0):
        ab = np.asarray(self.b) - self.a
        t = np.clip((points - self.a) @ ab / max(float(ab @ ab), 1e-20), 0, 1)
        delta = points - (self.a + t[..., None] * ab)
        return (np.linalg.norm(delta, axis=-1) - self.radius - radius, delta)

class Lattice:

    def __init__(self, rest, parents, pinned, preset, lateral=(), iterations=8):
        self.rest = np.asarray(rest, dtype=float)
        self.parents = np.asarray(parents, dtype=int)
        self.pin = np.asarray(pinned, dtype=bool)
        if self.rest.ndim != 2 or self.rest.shape[1] != 3 or (not np.isfinite(self.rest).all()):
            raise ValueError('Finite Nx3 points required')
        if len(self.parents) != len(self.rest) or len(self.pin) != len(self.rest):
            raise ValueError('Topology length mismatch')
        if any((p >= i or p < -1 for i, p in enumerate(self.parents))):
            raise ValueError('Parents must precede children')
        if any((p < 0 and (not self.pin[i]) for i, p in enumerate(self.parents))):
            raise ValueError('Roots must be pinned')
        self.edges = [(p, i) for i, p in enumerate(self.parents) if p >= 0] + list(lateral)
        self.lengths = np.array([np.linalg.norm(self.rest[j] - self.rest[i]) for i, j in self.edges])
        if np.any(self.lengths < 1e-07):
            raise ValueError('Zero-length edge')
        self.preset = preset
        self.iterations = iterations
        self.pos = self.rest.copy()
        self.vel = np.zeros_like(self.pos)
        self.resets = 0
        self.previous_target = self.rest.copy()
        count = preset.get('segment_collision_samples', 0)
        self.virtual_i = np.repeat([i for i, j in self.edges], count)
        self.virtual_j = np.repeat([j for i, j in self.edges], count)
        self.virtual_u = np.tile(np.linspace(0, 1, count + 2)[1:-1], len(self.edges))[:, None]

    def step(self, target, dt, colliders=()):
        if not 0 < dt <= 1 / 60:
            raise ValueError('Fixed step must be <= 1/60 second')
        target = np.asarray(target, dtype=float)
        if target.shape != self.rest.shape or not np.isfinite(target).all():
            raise ValueError('Bad target sample')
        p = self.preset
        old = self.pos.copy()
        desired = target.copy()
        for i, j in enumerate(self.parents):
            if j >= 0:
                desired[i] = self.pos[j] + target[i] - target[j]
        acceleration = p['omega'] ** 2 * (desired - self.pos)
        acceleration[:, 2] -= p['gravity_m_s2']
        drive_velocity = (target - self.previous_target) / dt
        self.vel = drive_velocity + (self.vel + dt * acceleration - drive_velocity) * math.exp(-2 * p['damping_ratio'] * p['omega'] * dt)
        self.previous_target = target.copy()
        self.pos += dt * self.vel
        self.pos[self.pin] = target[self.pin]
        for _ in range(self.iterations):
            for (i, j), length in zip(self.edges, self.lengths):
                delta = self.pos[j] - self.pos[i]
                d = np.linalg.norm(delta)
                if d < 1e-12:
                    delta = self.rest[j] - self.rest[i]
                    d = np.linalg.norm(delta)
                wi, wj = (float(not self.pin[i]), float(not self.pin[j]))
                if wi + wj:
                    correction = delta * ((d - length) / d / (wi + wj))
                    self.pos[i] += wi * correction
                    self.pos[j] -= wj * correction
            angle = math.radians(p['max_angle_deg'])
            for i, j in enumerate(self.parents):
                if j < 0 or self.pin[i]:
                    continue
                base = target[i] - target[j]
                base /= np.linalg.norm(base)
                delta = self.pos[i] - self.pos[j]
                length = np.linalg.norm(delta)
                if length < 1e-12:
                    continue
                u = delta / length
                dot = float(np.clip(u @ base, -1, 1))
                if dot < math.cos(angle):
                    tangent = u - base * dot
                    if np.linalg.norm(tangent) < 1e-10:
                        tangent = np.cross(base, [1, 0, 0] if abs(base[0]) < 0.9 else [0, 1, 0])
                    tangent /= np.linalg.norm(tangent)
                    self.pos[i] = self.pos[j] + length * (base * math.cos(angle) + tangent * math.sin(angle))
            for collider in colliders:
                margin = collider.radius + p['radius_m']
                if np.any(self.pos.max(axis=0) < np.minimum(collider.a, collider.b) - margin) or np.any(self.pos.min(axis=0) > np.maximum(collider.a, collider.b) + margin):
                    continue
                depth, delta = collider.clearance(self.pos, p['radius_m'])
                mask = (depth < 0) & ~self.pin
                norms = np.linalg.norm(delta, axis=1)
                normals = delta / np.maximum(norms[:, None], 1e-12)
                normals[norms < 1e-12] = [1, 0, 0]
                self.pos[mask] -= depth[mask, None] * normals[mask]
                if len(self.virtual_u):
                    i, j, u = (self.virtual_i, self.virtual_j, self.virtual_u)
                    points = (1 - u) * self.pos[i] + u * self.pos[j]
                    depths, deltas = collider.clearance(points, p['radius_m'])
                    wi, wj = ((~self.pin[i])[:, None], (~self.pin[j])[:, None])
                    denom = wi * (1 - u) ** 2 + wj * u * u
                    active = (depths < 0) & (denom[:, 0] > 0)
                    if active.any():
                        norms = np.linalg.norm(deltas, axis=1)
                        normals = deltas / np.maximum(norms[:, None], 1e-12)
                        if (norms < 1e-12).any():
                            fallback = np.cross(self.pos[j] - self.pos[i], [0.0, 0, 1])
                            vertical = np.linalg.norm(fallback, axis=1) < 1e-12
                            fallback[vertical] = np.cross((self.pos[j] - self.pos[i])[vertical], [0.0, 1, 0])
                            fallback /= np.maximum(np.linalg.norm(fallback, axis=1)[:, None], 1e-12)
                            normals[norms < 1e-12] = fallback[norms < 1e-12]
                        correction = -np.minimum(depths, 0)[:, None] * normals / np.maximum(denom, 1e-12)
                        shifts = np.zeros_like(self.pos)
                        counts = np.zeros((len(self.pos), 1))
                        np.add.at(shifts, i, wi * (1 - u) * correction)
                        np.add.at(shifts, j, wj * u * correction)
                        np.add.at(counts, i, wi * active[:, None])
                        np.add.at(counts, j, wj * active[:, None])
                        self.pos += shifts / np.maximum(counts, 1)
            self.pos[self.pin] = target[self.pin]
        self.vel = (self.pos - old) / dt
        if not np.isfinite(self.pos).all():
            raise FloatingPointError('Simulation exploded')
        return self.pos.copy()

def assess(positions, targets, fps, preset, parents, colliders_by_frame=None, stop_frame=None):
    x = np.asarray(positions)
    target = np.asarray(targets)
    length = preset['length_m']
    dynamic = np.array([p >= 0 for p in parents])
    relative = x - target
    amp = np.linalg.norm(relative[:, dynamic], axis=2)
    acceleration = np.diff(relative[:, dynamic], n=2, axis=0) * fps ** 2
    max_acc = float(np.max(np.linalg.norm(acceleration, axis=2))) if len(acceleration) else 0
    acc_bound = length * preset['omega'] ** 2 * 2.5
    edge_errors = []
    for j, i in enumerate(parents):
        if i >= 0:
            ref = np.linalg.norm(target[:, j] - target[:, i], axis=1)
            edge_errors.extend((np.abs(np.linalg.norm(x[:, j] - x[:, i], axis=1) - ref) / np.maximum(ref, 1e-09)).tolist())
    penetration = 0.0
    segment_penetration = 0.0
    if colliders_by_frame is not None:
        for f, cols in enumerate(colliders_by_frame):
            for c in cols:
                penetration = max(penetration, float(np.maximum(-c.clearance(x[f, dynamic], preset['radius_m'])[0], 0).max(initial=0)))
            count = preset.get('segment_collision_samples', 0)
            for j, i in enumerate(parents):
                if i < 0 or not count:
                    continue
                u = np.linspace(0, 1, count + 2)[1:-1, None]
                points = x[f, i] * (1 - u) + x[f, j] * u
                for c in cols:
                    segment_penetration = max(segment_penetration, float(np.maximum(-c.clearance(points, preset['radius_m'])[0], 0).max(initial=0)))
    settle = None
    if stop_frame is not None and stop_frame < len(x) - 3:
        equilibrium = x[-1] - target[-1]
        err = np.max(np.linalg.norm(relative - equilibrium, axis=2), axis=1)
        velocity = np.r_[np.max(np.linalg.norm(np.diff(relative, axis=0), axis=2), axis=1) * fps, 0]
        stable = (err < length * 0.02) & (velocity < length * 0.1)
        for f in range(stop_frame, len(x) - max(2, int(0.25 * fps))):
            if stable[f:].all():
                settle = (f - stop_frame) / fps
                break
    return dict(finite=bool(np.isfinite(x).all()), max_amplitude_over_length=float(amp.max(initial=0) / length), rms_amplitude_over_length=float(np.sqrt(np.mean(amp ** 2)) / length), max_relative_acceleration_m_s2=max_acc, acceleration_bound_m_s2=acc_bound, jitter_pass=max_acc <= acc_bound, max_edge_relative_error=max(edge_errors, default=0), constraint_pass=max(edge_errors, default=0) <= 0.03, proxy_penetration_m=penetration, proxy_collision_tested=colliders_by_frame is not None, sampled_segment_penetration_m=segment_penetration, proxy_collision_pass=penetration <= 1e-05 if colliders_by_frame is not None else None, settle_seconds=settle, settle_limit_s=preset['settle_limit_s'], settle_pass=settle is not None and settle <= preset['settle_limit_s'] if stop_frame is not None else None, explosion_pass=bool(np.isfinite(x).all() and amp.max(initial=0) < 2.1 * length), garment_surface_gate='not-tested-by-particle-proxies')

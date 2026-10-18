import hashlib
from pathlib import Path
import numpy as np

def intervals(mask, minimum=1):
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]

def envelope(alpha):
    mask = np.asarray(alpha) > 127
    yy, xx = np.nonzero(mask)
    if len(xx) < 100:
        raise ValueError('Empty silhouette')
    x0, x1 = (int(xx.min()), int(xx.max() + 1))
    y0, y1 = (int(yy.min()), int(yy.max() + 1))
    height = y1 - y0
    centre = (x0 + x1) / 2
    rows = []
    for y in np.linspace(y1 - 1, y0, 192).astype(int):
        runs = intervals(mask[y], max(2, int(0.006 * height)))
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= max(1, int(0.003 * height)):
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        rows.append(dict(z=(y1 - y - 0.5) / height, intervals=[[(a - centre) / height, (b - centre) / height] for a, b in merged]))
    gaps = []
    for r in rows:
        if not 0.1 < r['z'] < 0.65:
            continue
        left = [p for p in r['intervals'] if p[1] < -0.003]
        right = [p for p in r['intervals'] if p[0] > 0.003]
        if left and right and (not any((p[0] <= 0 <= p[1] for p in r['intervals']))):
            gaps.append(r['z'])
    arms = [r['z'] for r in rows if 0.28 < r['z'] < 0.85 and len(r['intervals']) >= 3 and any((a < 0 < b for a, b in r['intervals']))]
    return dict(schema_version=1, coordinate_system='front silhouette; horizontal bbox centre and floor; units/full visible height', confidence=0.4, visible_leg_gap_top=max(gaps) if gaps else None, visible_arm_separation_z=[min(arms), max(arms)] if arms else None, rows=rows, limitation='Clothing silhouette; constrains visible branches only, never supplies anatomical joint lengths or centres')

def from_intake(intake_path, front):
    from PIL import Image
    path = (Path(intake_path).parent / front['cutout']).resolve()
    if 'private' in {p.lower() for p in path.parts}:
        raise ValueError('Private silhouette refused')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != front['cutout_sha256']:
        raise ValueError('S0 cutout changed')
    result = envelope(np.array(Image.open(path).convert('RGBA'))[:, :, 3])
    result['cutout_sha256'] = digest
    return result

def branch_distance(evidence, z, x, kind):
    if not evidence or not evidence.get('rows'):
        return None
    row = min(evidence['rows'], key=lambda r: abs(r['z'] - z))
    spans = row['intervals']
    sign = 1 if x >= 0 else -1
    if kind == 'arm' and (len(spans) < 3 or not any((a < 0 < b for a, b in spans))):
        return None
    candidates = [(a, b) for a, b in spans if (a > 0 if sign > 0 else b < 0)]
    if not candidates:
        return None
    a, b = max(candidates, key=lambda p: sign * (p[0] + p[1])) if kind == 'arm' else min(candidates, key=lambda p: abs(p[0] + p[1]))
    return max(a - x, 0, x - b)

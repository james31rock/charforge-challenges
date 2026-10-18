import numpy as np
import trimesh
from PIL import Image,ImageDraw

def topology(mesh, merge=True):
    m = mesh.copy()
    if merge: m.merge_vertices()
    counts = np.bincount(m.edges_unique_inverse)
    return {'vertices': len(m.vertices), 'triangles': len(m.faces),
            'boundary_edges': int(np.sum(counts == 1)), 'nonmanifold_edges': int(np.sum(counts > 2)),
            'watertight': bool(m.is_watertight), 'winding_consistent': bool(m.is_winding_consistent),
            'components': len(m.split(only_watertight=False)), 'volume': float(m.volume)}

def mask(meshes, bounds, view='front', resolution=768):
    lo, hi = np.asarray(bounds)
    axis = 0 if view == 'front' else 1
    span = max(hi[2] - lo[2], hi[axis] - lo[axis]); margin = .07 * span
    scale = (resolution - 1) / (span + 2 * margin)
    image = Image.new('1', (resolution, resolution), 0); draw = ImageDraw.Draw(image)
    mid = (hi[axis] + lo[axis]) / 2
    for m in meshes:
        xy = np.column_stack([(m.vertices[:, axis] - mid) * scale + resolution / 2,
                              ((hi[2] + lo[2])/2 + span/2 + margin - m.vertices[:, 2]) * scale])
        for face in m.faces:
            draw.polygon([tuple(p) for p in xy[face]], fill=1)
    return np.asarray(image, dtype=bool)

def iou(reference, candidate):
    union = np.count_nonzero(reference | candidate)
    if not union: raise ValueError('Empty silhouette is not a valid match')
    return float(np.count_nonzero(reference & candidate) / union)

def vertical_coverage(meshes, height, bins=256):
    """Conservative surface-height occupancy; neither gaps nor coverage certify anatomy."""
    if not np.isfinite(height) or height <= 0: raise ValueError('Invalid diagnostic height')
    changes = np.zeros(bins + 1, dtype=int)
    for mesh in meshes:
        z = np.asarray(mesh.vertices)[mesh.faces, 2]
        low, high = z.min(1), z.max(1)
        use = (high >= 0) & (low <= height)
        start = np.clip(np.floor(low[use] / height * bins).astype(int), 0, bins - 1)
        end = np.clip(np.floor(high[use] / height * bins).astype(int), 0, bins - 1) + 1
        np.add.at(changes, start, 1); np.add.at(changes, end, -1)
    occupied = np.cumsum(changes)[:-1] > 0
    transitions = np.diff(np.r_[False, ~occupied, False].astype(int))
    intervals = np.c_[np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)] / bins
    return {'surface_height_coverage_fraction': float(occupied.mean()),
            'uncovered_height_intervals_fraction': intervals.tolist(), 'bins': bins,
            'interpretation': 'A gap is missing surface evidence. Full vertical coverage does not prove anatomical completeness.'}

def render(mesh, colors=None, size=(360, 460), view='front', bounds=None):
    """CPU triangle preview; orthographic painter sorting, no GPU dependency."""
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    lo, hi = np.asarray(bounds if bounds is not None else mesh.bounds)
    axis, depth, sign = (0, 1, -1) if view == 'front' else (1, 0, 1)
    span = max(hi[2] - lo[2], (hi[axis] - lo[axis]) * size[1] / size[0])
    scale = (size[1] - 36) / max(span, 1e-8)
    xy = np.column_stack([(v[:, axis] - (hi[axis] + lo[axis]) / 2) * scale + size[0] / 2,
                          ((hi[2] + lo[2]) / 2 - v[:, 2]) * scale + size[1] / 2])
    if colors is None: colors = np.tile([110, 165, 190], (len(f), 1))
    normals = mesh.face_normals
    lamp = np.array([-.4, -.7, .6]); lamp /= np.linalg.norm(lamp)
    shade = .75 + .25 * np.abs(normals @ lamp)
    rgb = np.clip(np.asarray(colors) * shade[:, None], 0, 255).astype(np.uint8)
    image = Image.new('RGB', size, (245, 246, 248)); draw = ImageDraw.Draw(image)
    for idx in np.argsort(v[f].mean(1)[:, depth] * sign):
        draw.polygon([tuple(p) for p in xy[f[idx]]], fill=tuple(rgb[idx]))
    return image

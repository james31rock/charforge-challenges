import argparse, json, math
from pathlib import Path
import numpy as np
import cv2
from toolkit.face.layers import read, save, write

def sample(im, x, y):
    return cv2.remap(im, np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def load_mesh(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'mesh.json').read_text())
    m = meta['meshes'][0]
    data = dict(np.load(folder / (m['name'] + '.npz')))
    data['texture'] = read(folder / m['materials'][0]['images'][0]['image'])
    data['meta'] = meta
    return data

def framing(character, mesh):
    character = Path(character)
    intake = json.loads((character / 's0/intake.json').read_text())
    ref = next((i for i in intake['images'] if i['view']['label'] == 'front'))
    crop = json.loads((character / 'face-grids/manifest.json').read_text())['portrait_crop']
    bbox = ref['foreground_bbox']
    height = float(np.ptp(mesh['vertices'][:, 2]))
    span = height * (crop[3] - crop[1]) / (bbox[3] - bbox[1])
    return {'center_x': 0, 'center_z': float(mesh['vertices'][:, 2].max() - height * ((crop[1] + crop[3]) / 2 - bbox[1]) / (bbox[3] - bbox[1])), 'span': span}

def project(mesh, size=512, yaw=0, frame=None, forward=-1):
    v = mesh['vertices']
    n = mesh['normals']
    tri = mesh['triangles']
    uv = mesh['uv']
    a = math.radians(yaw)
    right = np.array([-math.cos(a) * forward, math.sin(a), 0])
    toward = np.array([math.sin(a), math.cos(a) * forward, 0])
    if frame is None:
        height = np.ptp(v[:, 2])
        frame = {'center_x': 0, 'center_z': float(v[:, 2].max() - 0.21 * height), 'span': float(0.49 * height)}
    span = frame['span']
    sx = (v @ right - frame['center_x']) / span * size + size / 2
    sy = (frame['center_z'] - v[:, 2]) / span * size + size / 2
    depth = v @ toward
    pos = np.stack((sx, sy), axis=1)
    zbuf = np.full((size, size), -np.inf, dtype=np.float32)
    world = np.zeros((size, size, 3), np.float32)
    normal = world.copy()
    tex = np.zeros((size, size, 2), np.float32)
    ids = np.full((size, size), -1, np.int32)
    for ti, indices in enumerate(tri):
        pts = pos[indices]
        mn = np.maximum(np.floor(pts.min(axis=0)).astype(int), 0)
        mx = np.minimum(np.ceil(pts.max(axis=0)).astype(int), size - 1)
        if (mn > mx).any():
            continue
        x0, y0 = mn
        x1, y1 = mx
        a0, b0, c0 = pts
        den = (b0[1] - c0[1]) * (a0[0] - c0[0]) + (c0[0] - b0[0]) * (a0[1] - c0[1])
        if abs(den) < 1e-08:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        xx = xx + 0.5
        yy = yy + 0.5
        b1 = ((b0[1] - c0[1]) * (xx - c0[0]) + (c0[0] - b0[0]) * (yy - c0[1])) / den
        b2 = ((c0[1] - a0[1]) * (xx - c0[0]) + (a0[0] - c0[0]) * (yy - c0[1])) / den
        b3 = 1 - b1 - b2
        weights = np.stack((b1, b2, b3), axis=-1)
        dep = weights @ depth[indices]
        patch = zbuf[y0:y1 + 1, x0:x1 + 1]
        ok = (weights.min(axis=-1) >= -1e-06) & (dep > patch)
        if not ok.any():
            continue
        patch[ok] = dep[ok]
        world[y0:y1 + 1, x0:x1 + 1][ok] = (weights @ v[indices])[ok]
        normal[y0:y1 + 1, x0:x1 + 1][ok] = (weights @ n[indices])[ok]
        tex[y0:y1 + 1, x0:x1 + 1][ok] = (weights @ uv[ti])[ok]
        ids[y0:y1 + 1, x0:x1 + 1][ok] = ti
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-08)
    texture = mesh['texture']
    color = sample(texture, tex[:, :, 0] * (texture.shape[1] - 1), (1 - tex[:, :, 1]) * (texture.shape[0] - 1))
    color[ids < 0] = [237, 238, 241]
    return {'color': color, 'world': world, 'normal': normal, 'ids': ids, 'uv': tex, 'frame': frame, 'yaw': yaw, 'forward': forward}
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--character')
    p.add_argument('--forward', type=int, default=-1)
    a = p.parse_args()
    f = Path(a.folder)
    mesh = load_mesh(f / 'inspect')
    r = project(mesh, forward=a.forward, frame=framing(a.character, mesh) if a.character else None)
    save(f / 'original-front.png', r['color'])
    np.savez_compressed(f / 'front-map.npz', **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
    write(f / 'camera.json', r['frame'])

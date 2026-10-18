import argparse, json, shutil, hashlib, datetime
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from toolkit.face.raster import load_mesh, sample
from toolkit.face.surface import load_groups
from toolkit.face.layers import save, read, write

def erase(source, inspect, skin, out, freeze):
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(freeze).is_file():
        raise RuntimeError('C8 failing baseline must be frozen first')
    if shutil.disk_usage(out).free < 40 * 1024 ** 3:
        raise RuntimeError('BLOCKED disk')
    b = json.loads((source / 'binding.json').read_text())
    g = load_groups(source, b)
    d = b['source_landmarks']['interocular_px']
    w, h = b['canvas']
    field = read(skin)
    union = np.maximum.reduce([p[:, :, 3] for key in ('brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth', 'blush-left', 'blush-right') for p in g[key].values()])
    radius = max(4, round(0.06 * d))
    outer = max(2, round(0.025 * d))
    core = cv2.dilate(np.uint8(union > 5), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    distance = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
    fade = np.clip(1 - distance / outer, 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    plate = read(source / 'feature-free.png')
    original_plate = plate.copy()
    plate = np.uint8(plate * (1 - fade[:, :, None]) + field * fade[:, :, None])
    yy, xx = np.indices((h, w))
    mx, my = b['source_landmarks']['mouth']
    ey = np.mean(np.array(b['source_landmarks']['eyes'])[:, 1])
    occupied = np.maximum.reduce([p[:, :, 3] for key in ('eye-left', 'eye-right', 'mouth') for p in g[key].values()])
    occupied = cv2.dilate(np.uint8(occupied > 0), np.ones((7, 7), np.uint8))
    nose = (abs(xx - mx) < d * 0.34) & (yy > ey + d * 0.08) & (yy < my - d * 0.12) & ~occupied.astype(bool)
    nw = np.clip(cv2.distanceTransform(np.uint8(nose), cv2.DIST_L2, 5) / max(4, d * 0.045), 0, 1)[:, :, None]
    plate = np.uint8(plate * (1 - nw) + original_plate * nw)
    base_fade = np.array(Image.open(source / 'feature-free-mask.png').convert('L')) / 255
    fade = np.maximum(fade, base_fade)
    field = plate
    save(out / 'feature-free.png', plate)
    save(out / 'erasure-projector.png', np.uint8(fade * 255))
    for name in ('track.json', 'layers.json', 'mouth-morph.json', 'revision-qa.json'):
        if (source / name).exists() and source.resolve() != out.resolve():
            shutil.copy2(source / name, out / name)
    for directory in ('atlases', 'layers'):
        if source.resolve() != out.resolve():
            shutil.copytree(source / directory, out / directory, dirs_exist_ok=True)
    base = g['base']
    spec = b['atlases']['base']
    atlas = np.array(Image.open(out / spec['path']).convert('RGBA'))
    for i, name in enumerate(spec['states']):
        atlas[i // spec['cols'] * h:(i // spec['cols'] + 1) * h, i % spec['cols'] * w:(i % spec['cols'] + 1) * w, :3] = plate
    save(out / spec['path'], atlas)
    layer_index = json.loads((out / 'layers.json').read_text())

    def update_patch(name, key, patch):
        entry = next((r for r in layer_index['entries'][name] if r['feature'] == key))
        py, px = np.where(patch[:, :, 3] > 0)
        box = (int(px.min()), int(py.min()), int(px.max()) + 1, int(py.max()) + 1) if len(px) else (0, 0, 1, 1)
        x0, y0, x1, y1 = box
        path = out / entry['path']
        save(path, patch[y0:y1, x0:x1])
        entry.update(offset=[x0, y0], size=[x1 - x0, y1 - y0], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for name, patch in g['base'].items():
        update_patch(name, 'base', np.dstack((plate, patch[:, :, 3])))
    brow_proof = []
    brow_support = np.zeros((h, w), np.uint8)
    for key in ('brow-left', 'brow-right'):
        spec = b['atlases'][key]
        atlas = np.array(Image.open(out / spec['path']).convert('RGBA'))
        for i, name in enumerate(spec['states']):
            patch = g[key][name].copy()
            gray = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_RGB2GRAY)
            ink = np.uint8((gray < 120) & (patch[:, :, 3] > 60))
            dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
            radius = int(np.floor(float(dist.max()) * 0.42))
            opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))) if radius else ink
            n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
            if n < 2:
                raise RuntimeError('No isolated brow ' + key + ' ' + name)
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            main = np.uint8(labels == label)
            support = cv2.GaussianBlur(main.astype(float), (0, 0), 0.45)
            patch[:, :, 3] = np.uint8(np.minimum(patch[:, :, 3], support * 255))
            brow_support = np.maximum(brow_support, patch[:, :, 3])
            atlas[i // spec['cols'] * h:(i // spec['cols'] + 1) * h, i % spec['cols'] * w:(i % spec['cols'] + 1) * w] = patch
            update_patch(name, key, patch)
            brow_proof.append({'feature': key, 'state': name, 'thickness_radius': float(dist.max()), 'opening_radius': radius, 'remaining_ink_components': 1})
        save(out / spec['path'], atlas)
    write(out / 'brow-isolation.json', {'method': 'dominant thick stroke; thin connected branch removal; one-pixel antialias only', 'states': brow_proof})
    layer_index['method'] = 'C8 native UV erasure and isolated dominant brow stroke'
    write(out / 'layers.json', layer_index)
    mesh = load_mesh(inspect)
    v = mesh['vertices']
    uv = mesh['uv']
    tri = mesh['triangles']
    texture = mesh['texture']
    th, tw = texture.shape[:2]
    original = texture.copy()
    alpha = np.zeros((th, tw), np.float32)
    paint = np.zeros_like(texture)
    cam = b['camera']
    m = np.array(b['front_to_source'])
    fx = (v[:, 0] - cam['center_x']) / cam['span'] * w + w / 2
    fy = (cam['center_z'] - v[:, 2]) / cam['span'] * h + h / 2
    sx = fx * m[0, 0] + fy * m[0, 1] + m[0, 2]
    sy = fx * m[1, 0] + fy * m[1, 1] + m[1, 2]
    coords = np.stack((sx, sy), axis=1)
    yy0, xx0 = np.where(fade > 0)
    low = np.array([xx0.min() - 2, yy0.min() - 2])
    high = np.array([xx0.max() + 2, yy0.max() + 2])
    touched = 0
    for ti, indices in enumerate(tri):
        local = coords[indices]
        if (v[indices, 1] >= 0).all() or (local.max(axis=0) < low).any() or (local.min(axis=0) > high).any():
            continue
        pts = np.stack((uv[ti, :, 0] * (tw - 1), (1 - uv[ti, :, 1]) * (th - 1)), axis=1)
        mn = np.maximum(np.floor(pts.min(axis=0)).astype(int) - 1, 0)
        mx = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + 1, [tw - 1, th - 1])
        x0, y0 = mn
        x1, y1 = mx
        a, bb, c = pts
        den = (bb[1] - c[1]) * (a[0] - c[0]) + (c[0] - bb[0]) * (a[1] - c[1])
        if abs(den) < 1e-08:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        q1 = ((bb[1] - c[1]) * (xx - c[0]) + (c[0] - bb[0]) * (yy - c[1])) / den
        q2 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        weights = np.stack((q1, q2, 1 - q1 - q2), axis=-1)
        projected = weights @ local
        support = sample(fade, projected[:, :, 0], projected[:, :, 1])
        inside = (weights.min(axis=-1) >= -0.015) & (weights @ v[indices, 1] < 0)
        support *= inside
        patch = alpha[y0:y1 + 1, x0:x1 + 1]
        ok = support > patch
        if not ok.any():
            continue
        patch[ok] = support[ok]
        paint[y0:y1 + 1, x0:x1 + 1][ok] = sample(field, projected[:, :, 0], projected[:, :, 1])[ok]
        touched += 1
    kernel = np.ones((5, 5), np.uint8)
    expanded = cv2.dilate(alpha, kernel)
    nearest = np.uint8(alpha > 0)
    _, labels = cv2.distanceTransformWithLabels(1 - nearest, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    known = paint[nearest > 0]
    gutter = (expanded > alpha) & (alpha == 0)
    if len(known):
        paint[gutter] = known[labels[gutter] - 1]
        alpha[gutter] = expanded[gutter]
    texture = np.uint8(np.clip(original * (1 - alpha[:, :, None]) + paint * alpha[:, :, None], 0, 255))
    save(out / 'static-uv.png', texture)
    save(out / 'static-uv-erasure.png', np.uint8(alpha * 255))
    b['static_uv'] = {'path': 'static-uv.png', 'source_image_count': 1, 'base_overlay_enabled': False, 'erasure_mask': 'static-uv-erasure.png'}
    save(out / 'brow-clear.png', cv2.dilate(np.uint8(brow_support > 0) * 255, np.ones((7, 7), np.uint8)))
    b['visibility_override'] = 'brow-clear.png'
    b['status'] = 'C8 review required'
    write(out / 'binding.json', b)
    write(out / 'static-erasure.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'baseline_freeze_sha256': hashlib.sha256(Path(freeze).read_bytes()).hexdigest(), 'source_atlas_sha256': hashlib.sha256(original.tobytes()).hexdigest(), 'clean_atlas_sha256': hashlib.sha256(texture.tobytes()).hexdigest(), 'changed_texels': int(np.any(texture != original, axis=-1).sum()), 'affected_triangles': touched, 'normal_visibility_used_for_erasure': False, 'source_geometry_unchanged': True, 'all_static_image_nodes_expected': 1, 'ao_and_line_maps': 'none in source material; installer rejects additional maps'})
    print(json.dumps({'out': str(out), 'changed_texels': int(np.any(texture != original, axis=-1).sum())}), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--inspect', required=True)
    p.add_argument('--skin', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--freeze', required=True)
    erase(**vars(p.parse_args()))

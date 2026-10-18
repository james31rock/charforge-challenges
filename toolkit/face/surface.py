import argparse, json, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
from toolkit.face.layers import read, save, write, landmarks, soft_box
from toolkit.face.raster import load_mesh, project, framing, sample

def full_patch(folder, row, shape):
    out = np.zeros((*shape[:2], 4), np.uint8)
    im = np.array(Image.open(folder / row['path']).convert('RGBA'))
    x, y = row['offset']
    h, w = im.shape[:2]
    out[y:y + h, x:x + w] = im
    return out

def prepare(folder, character):
    from toolkit.face.matte_v2 import reconstruct
    from toolkit.face.mouth_morph import build
    f = Path(folder)
    idx = json.loads((f / 'layers.json').read_text())
    lm = idx['landmarks']
    front = read(f / 'original-front.png')
    camera = json.loads((f / 'camera.json').read_text())
    dest = landmarks(front)
    src = np.float32([*lm['eyes'], lm['mouth']])
    dst = np.float32([*dest['eyes'], dest['mouth']])
    m = cv2.getAffineTransform(src, dst)
    inv = cv2.invertAffineTransform(m)
    groups, native, fade = reconstruct(f)
    h, w = read(f / 'base.png').shape[:2]
    if (f / 'track.json').exists():
        track = json.loads((f / 'track.json').read_text())
        if track['schema_version'] == 'charforge.face.track/2':
            build(f, groups, track)
    brow = np.zeros((h, w), np.uint8)
    for key in ('brow-left', 'brow-right'):
        for p in groups[key].values():
            brow = np.maximum(brow, p[:, :, 3])
    radius = max(3, round(lm['interocular_px'] * 0.045))
    brow = cv2.dilate(np.uint8(brow > 0), np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
    clean = read(f / 'feature-free.png')
    skin = read(f / 'skin-field.png')
    clean[brow > 0] = skin[brow > 0]
    fade = np.maximum(fade, brow)
    for name in groups['base']:
        groups['base'][name] = np.dstack((clean, np.uint8(fade * 255)))
    save(f / 'feature-free.png', clean)
    save(f / 'feature-free-mask.png', np.uint8(fade * 255))
    save(f / 'brow-clear.png', np.uint8(brow * 255))
    atlas_dir = f / 'atlases'
    atlas_dir.mkdir(exist_ok=True)
    meta = {}
    entries = {}
    from toolkit.face.layers import sha
    for key, states in groups.items():
        names = list(states)
        cols = 8 if key == 'mouth' else 4
        rows = math.ceil(len(names) / cols)
        if key == 'mouth':
            union = np.maximum.reduce([p[:, :, 3] for p in states.values()])
            py, px = np.where(union > 0)
            rect = [max(0, int(px.min()) - 3), max(0, int(py.min()) - 3), min(w, int(px.max()) + 4), min(h, int(py.max()) + 4)]
        else:
            rect = [0, 0, w, h]
        x0, y0, x1, y1 = rect
        cw, ch = (x1 - x0, y1 - y0)
        atlas = np.zeros((ch * rows, cw * cols, 4), np.uint8)
        for i, name in enumerate(names):
            p = states[name]
            atlas[i // cols * ch:(i // cols + 1) * ch, i % cols * cw:(i % cols + 1) * cw] = p[y0:y1, x0:x1]
            py, px = np.where(p[:, :, 3] > 0)
            if not len(px):
                box = [0, 0, 1, 1]
            else:
                box = [int(px.min()), int(py.min()), int(px.max()) + 1, int(py.max()) + 1]
            bx, by, ex, ey = box
            path = f / 'layers' / name / (key + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            save(path, p[by:ey, bx:ex])
            entries.setdefault(name, []).append({'feature': key, 'path': str(path.relative_to(f)).replace('\\', '/'), 'offset': [bx, by], 'size': [ex - bx, ey - by], 'alpha': 'straight', 'sha256': sha(path)})
        save(atlas_dir / (key + '.png'), atlas)
        meta[key] = {'states': names, 'cols': cols, 'rows': rows, 'path': 'atlases/' + key + '.png', 'source_rect': rect, 'tile_size': [cw, ch]}
    idx['entries'] = entries
    idx['method'] = 'temporal feature-free skin reconstruction plus isolated feature alpha'
    idx['schema_version'] = 'charforge.face.layers/2'
    write(f / 'layers.json', idx)
    binding = {'schema_version': 'charforge.face.binding/2', 'source_to_front': m.tolist(), 'front_to_source': inv.tolist(), 'source_landmarks': lm, 'mesh_front_landmarks': dest, 'camera': camera, 'canvas': [w, h], 'atlases': meta, 'front_direction': [0, -1, 0], 'method': 'rest-space face UV; invariant feature-free skin; feature-only alpha', 'native_closed_expressions': native, 'status': 'review_required'}
    binding['visibility_override'] = 'brow-clear.png'
    write(f / 'binding.json', binding)
    groups['_visibility_override'] = brow.astype(float)
    return (binding, groups)

def load_groups(f, binding):
    out = {}
    w, h = binding['canvas']
    for k, v in binding['atlases'].items():
        a = np.array(Image.open(f / v['path']).convert('RGBA'))
        out[k] = {}
        cw, ch = v.get('tile_size', [w, h])
        x0, y0, x1, y1 = v.get('source_rect', [0, 0, w, h])
        for i, name in enumerate(v['states']):
            p = np.zeros((h, w, 4), np.uint8)
            p[y0:y1, x0:x1] = a[i // v['cols'] * ch:(i // v['cols'] + 1) * ch, i % v['cols'] * cw:(i % v['cols'] + 1) * cw]
            out[k][name] = p
    if binding.get('visibility_override'):
        out['_visibility_override'] = read(f / binding['visibility_override'])[:, :, 0] / 255
    if binding.get('hair_protection'):
        out['_hair_protection'] = cv2.resize(read(f / binding['hair_protection'])[:, :, 0], (w, h), interpolation=cv2.INTER_AREA) / 255
    if binding.get('hair_protection'):
        out['_hair_binding'] = binding
        out['_static_texture'] = read(f / binding['static_uv']['path'])
        out['_native_hair_protection'] = read(f / binding['static_uv']['hair_protection'])[:, :, 0] / 255
    return out

def coordinates(r, b):
    world = r['world']
    camera = b['camera']
    w, h = b['canvas']
    fx = (world[:, :, 0] - camera['center_x']) / camera['span'] * w + w / 2
    fy = (camera['center_z'] - world[:, :, 2]) / camera['span'] * h + h / 2
    m = np.array(b['front_to_source'])
    return (fx * m[0, 0] + fy * m[0, 1] + m[0, 2], fx * m[1, 0] + fy * m[1, 1] + m[1, 2])

def render(r, b, groups, mouth='closed', expression='neutral', eyes='open', brows=None, shade=True, mouth_sample=None):
    sx, sy = coordinates(r, b)
    color = r['color'].astype(float).copy()
    original = color.copy()
    front = np.clip((-r['normal'][:, :, 1] - 0.08) / 0.18, 0, 1) * (r['world'][:, :, 1] < 0) * (r['ids'] >= 0)
    if '_visibility_override' in groups:
        front = np.maximum(front, sample(groups['_visibility_override'], sx, sy) * (r['world'][:, :, 1] < 0) * (r['ids'] >= 0))
    support = np.zeros(front.shape, float)
    for key in ('base', 'blush-left', 'blush-right', 'brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth'):
        name = 'expression-' + expression if key == 'base' else mouth_sample or 'mouth-' + mouth if key == 'mouth' else 'eyes-' + eyes if key.startswith('eye-') and eyes != 'open' and (expression not in b.get('native_closed_expressions', {}).get(key, [])) else 'expression-' + (brows if key.startswith('brow-') and brows else expression)
        p = sample(groups[key][name], sx, sy)
        a = p[:, :, 3] / 255 * front
        support = np.maximum(support, a)
        color = color * (1 - a[:, :, None]) + p[:, :, :3] * a[:, :, None]
    if shade:
        light = np.array([-0.35, -0.65, 0.7])
        light /= np.linalg.norm(light)
        lambert = np.clip(r['normal'] @ light, 0, 1)
        mult = 0.83 + 0.17 * lambert
        mult[r['ids'] < 0] = 1
        color *= mult[:, :, None]
    return (np.uint8(np.clip(color, 0, 255)), {'outside_support_max_byte_delta': float(np.max(np.abs(color - original)[(support == 0) & (r['ids'] >= 0)], initial=0)) if not shade else None, 'support_pixels': int((support > 0).sum())})

def grids(folder, character):
    f = Path(folder)
    b, groups = prepare(f, character)
    mesh = load_mesh(f / 'inspect')
    views = [project(mesh, yaw=a, frame=b['camera']) for a in (0, 35)]
    mouths = ['closed', 'mbp', 'small-a', 'wide-a', 'ee', 'o', 'fv', 'lth']
    sheet = Image.new('RGB', (8 * 256, 2 * 282), '#eceef1')
    d = ImageDraw.Draw(sheet)
    for j, r in enumerate(views):
        for i, mouth in enumerate(mouths):
            im, _ = render(r, b, groups, mouth=mouth)
            sheet.paste(Image.fromarray(im).resize((256, 256)), (i * 256, j * 282 + 26))
            d.text((i * 256 + 8, j * 282 + 7), f"{mouth} | {r['yaw']} deg", fill='black')
    sheet.save(f / 'mouth-grid.png')
    exps = ['neutral', 'happy', 'worried', 'surprised', 'sad', 'determined']
    sheet = Image.new('RGB', (6 * 256, 2 * 282), '#eceef1')
    d = ImageDraw.Draw(sheet)
    for j, r in enumerate(views):
        for i, e in enumerate(exps):
            im, _ = render(r, b, groups, expression=e)
            sheet.paste(Image.fromarray(im).resize((256, 256)), (i * 256, j * 282 + 26))
            d.text((i * 256 + 8, j * 282 + 7), f'{e}', fill='black')
    sheet.save(f / 'expression-grid.png')
    print(json.dumps({'character': Path(character).name, 'mesh_landmarks': b['mesh_front_landmarks'], 'source_to_front': b['source_to_front']}))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--character', required=True)
    a = p.parse_args()
    grids(a.folder, a.character)

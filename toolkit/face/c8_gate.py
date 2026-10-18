import argparse, json, hashlib, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from toolkit.face.raster import load_mesh, project, sample
from toolkit.face.surface import coordinates, load_groups
CONFIG = {'version': 'C8/7', 'ink_luminance': 120, 'alpha_core': 0.12, 'region_dilation_interocular': 0.045, 'raster_tolerance_px': 2, 'minimum_component_area_interocular_squared': 0.0005, 'minimum_stroke_width_interocular': 0.07, 'static_feature_pixels_allowed': 0, 'active_feature_regions_exclusive': True, 'eye_skin_colour_distance': 35, 'assign_active_blush_separately': True, 'eye_join_interocular': 0.01, 'occlusion_depth_tolerance_pixels': 1.5, 'occlusion_minimum_hidden_fraction': 0.5, 'missing_core_pixels_allowed': 4, 'oracle_edge_erosion_px': 1, 'occlusion_uses_source_identity_not_screen_width': True, 'camera_motion': 'smoothstep/1 or historical 2.5 degree steps'}
FEATURES = ['brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth']
CONFIG.update(version='C8/10', static_hair_excluded_only_by_original_source_labels=True, hair_may_split_one_stroke_into_multiple_visible_fragments=True, partial_occlusion_requires_one_observed_supported_fragment=True)
_MAPS = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return np.array(Image.open(p).convert('RGB'))

def ink(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) < CONFIG['ink_luminance']

def eye_evidence(im, skin):
    return np.linalg.norm(im.astype(float) - skin.astype(float), axis=-1) > CONFIG['eye_skin_colour_distance']

def components(mask, d, eye=False, minimum_width=True):
    mask = np.uint8(mask)
    if eye:
        radius = max(1, round(d * CONFIG['eye_join_interocular']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    n, lab, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [dict(x=int(x), y=int(y), w=int(w), h=int(h), area=int(area)) for x, y, w, h, area in stats[1:] if area >= max(4, d * d * CONFIG['minimum_component_area_interocular_squared']) and (not minimum_width or w >= max(3, d * CONFIG['minimum_stroke_width_interocular']))]

def regions(groups, b):
    d = b['source_landmarks']['interocular_px']
    radius = max(3, round(d * CONFIG['region_dilation_interocular']))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    out = {}
    for key in FEATURES:
        union = np.maximum.reduce([p[:, :, 3] for p in groups[key].values()])
        out[key] = cv2.dilate(np.uint8(union > 255 * CONFIG['alpha_core']), kernel).astype(float)
    return out

def mapping(b, mesh, cache, yaw):
    path = cache / f'depth-map-{yaw:.8f}.npz'
    if str(path) in _MAPS:
        return _MAPS[str(path)]
    if path.exists():
        cached = dict(np.load(path))
        if 'uv' in cached:
            _MAPS[str(path)] = cached
            return cached
    r = project(mesh, yaw=yaw, frame=b['camera'])
    sx, sy = coordinates(r, b)
    valid = (r['ids'] >= 0) & (r['world'][:, :, 1] < 0)
    a = math.radians(yaw)
    depth = r['world'][:, :, 0] * math.sin(a) - r['world'][:, :, 1] * math.cos(a)
    result = {'sx': sx, 'sy': sy, 'valid': valid, 'world': r['world'], 'depth': depth, 'occupied': r['ids'] >= 0, 'uv': r['uv'], 'original_color': r['color']}
    np.savez_compressed(path, **result)
    _MAPS[str(path)] = result
    return result

def state(key, row, b):
    if key == 'mouth':
        return row.get('mouth_sample', 'mouth-' + row['mouth'])
    if key.startswith('eye-') and row['eyes'] != 'open' and (row['expression'] not in b.get('native_closed_expressions', {}).get(key, [])):
        return 'eyes-' + row['eyes']
    return 'expression-' + row['brows' if key.startswith('brow-') else 'expression']

def masks(mapping_, reg):
    return {k: (sample(v, mapping_['sx'], mapping_['sy']) > 0.5) & mapping_['valid'] for k, v in reg.items()}

def protected_hair(m, groups):
    if '_hair_protection' not in groups:
        return np.zeros(m['valid'].shape, bool)
    if '_hair_binding' in groups and 'original_color' in m:
        if '_source_hair' not in m:
            from toolkit.face.hair_masks import original_masks
            _, hair = original_masks(m['original_color'], groups['_hair_binding'], groups, (m['sx'], m['sy']))
            m['_source_hair'] = cv2.dilate(np.uint8(hair), np.ones((5, 5), np.uint8)) > 0
        hair = m['_source_hair']
        if '_native_hair_protection' in groups and 'uv' in m:
            tex = groups['_native_hair_protection']
            native = sample(tex, m['uv'][:, :, 0] * (tex.shape[1] - 1), (1 - m['uv'][:, :, 1]) * (tex.shape[0] - 1)) > 0.1
            hair = hair | cv2.dilate(np.uint8(native), np.ones((3, 3), np.uint8)).astype(bool)
        return hair
    return cv2.dilate(np.uint8(sample(groups['_hair_protection'], m['sx'], m['sy']) > 0.1), np.ones((3, 3), np.uint8)) > 0

def active_masks(m, groups, row, b):
    out = {}
    tolerance = CONFIG['raster_tolerance_px']
    for key in FEATURES:
        if key == 'mouth':
            continue
        p = groups[key][state(key, row, b)]
        expected = sample(np.uint8(ink(p[:, :, :3]) & (p[:, :, 3] > 30)) * 255, m['sx'], m['sy']) > 20
        out[key] = cv2.dilate(np.uint8(expected), np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)) > 0
    for key in ('blush-left', 'blush-right'):
        p = groups[key]['expression-' + row['expression']]
        expected = sample(p[:, :, 3], m['sx'], m['sy']) > 30
        out[key] = cv2.dilate(np.uint8(expected), np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)) > 0
    return out

def exclusive(roi, key, expected):
    other = np.logical_or.reduce([value for name, value in expected.items() if name != key])
    return roi & ~other

def camera_yaw(time, smooth=False):
    u = max(0, min((time - 6.7) / 1.25, (9.6 - time) / 1.2, 1))
    return 35 * u * u * (3 - 2 * u) if smooth else round(35 * u / 2.5) * 2.5

def brow_oracle(key, row, b, groups, m, front, yaw):
    p = groups[key][state(key, row, b)]
    source = ink(p[:, :, :3]) & (p[:, :, 3] > 127)
    expected = (sample(np.uint8(source) * 255, m['sx'], m['sy']) > 127) & m['valid']
    front_mask = (sample(np.uint8(source) * 255, front['sx'], front['sy']) > 127) & front['valid']
    raw_front = front_mask.copy()
    hair_hidden = protected_hair(front, groups) & front_mask
    front_mask &= ~protected_hair(front, groups)
    expected &= ~protected_hair(m, groups)
    points = front['world'][front_mask]
    a = math.radians(yaw)
    cam = b['camera']
    size = 512
    if len(points):
        x = (points[:, 0] * math.cos(a) + points[:, 1] * math.sin(a) - cam['center_x']) / cam['span'] * size + size / 2 - 0.5
        y = (cam['center_z'] - points[:, 2]) / cam['span'] * size + size / 2 - 0.5
        depth = points[:, 0] * math.sin(a) - points[:, 1] * math.cos(a)
        surface_depth = sample(m['depth'], x.reshape(1, -1), y.reshape(1, -1)).ravel()
        hidden = surface_depth - depth > cam['span'] / size * CONFIG['occlusion_depth_tolerance_pixels']
        fraction = float(hidden.mean())
    else:
        fraction = 0
    d = b['source_landmarks']['interocular_px']
    proof = {'source_strokes': len(components(source, d)), 'unoccluded_front_strokes': len(components(raw_front, d)), 'front_strokes': len(components(front_mask, d)), 'front_pixels': int(front_mask.sum()), 'predicted_visible_pixels': int(expected.sum()), 'predicted_visible_strokes': len(components(expected, d)), 'depth_hidden_fraction': fraction, 'native_hair_hidden_fraction': float(hair_hidden.sum() / max(1, raw_front.sum()))}
    return (expected, proof)

def assess_brow(observed, expected, d, proof):
    kernel = np.ones((5, 5), np.uint8)
    cc = components(observed, d)
    extra = components(observed & ~cv2.dilate(np.uint8(expected), kernel).astype(bool), d)
    core = cv2.erode(np.uint8(expected), np.ones((3, 3), np.uint8)).astype(bool)
    missing = core & ~cv2.dilate(np.uint8(observed), kernel).astype(bool)
    support_matches = int(missing.sum()) <= CONFIG['missing_core_pixels_allowed']
    if expected.sum() >= 4 and (not observed.any()):
        support_matches = False
    one_front_source = proof.get('unoccluded_front_strokes', proof['front_strokes']) == 1
    occluded = proof['source_strokes'] == 1 and one_front_source and (proof['front_strokes'] >= 1 and proof['depth_hidden_fraction'] >= CONFIG['occlusion_minimum_hidden_fraction'] or proof.get('native_hair_hidden_fraction', 0) >= CONFIG['occlusion_minimum_hidden_fraction'])
    fragments = components(observed, d, minimum_width=False)
    fragment_extra = components(observed & ~cv2.dilate(np.uint8(expected), kernel).astype(bool), d, minimum_width=False)
    partial = proof['source_strokes'] == 1 and one_front_source and (len(fragments) == 1) and (not fragment_extra) and (expected.sum() >= 4) and (proof['depth_hidden_fraction'] > 0 or proof.get('native_hair_hidden_fraction', 0) > 0)
    occluded = occluded or partial
    passed = (len(cc) == 1 or (len(cc) == 0 and occluded)) and (not extra) and support_matches
    return {'connected_groups': len(cc), 'components': cc, 'visible_fragment_components': fragments, 'partial_occlusion_supported': bool(partial), 'extra_ink_groups': len(extra), 'extra_components': extra, 'missing_core_pixels': int(missing.sum()), 'occlusion_exception_used': len(cc) == 0 and occluded, 'occlusion_proof': proof, 'pass': bool(passed)}

def assess_features(crop, rs, m, groups, row, b, skin_plate, front, yaw):
    d = b['source_landmarks']['interocular_px']
    col = ink(crop)
    expected_masks = active_masks(m, groups, row, b)
    features = {}
    for key, roi in rs.items():
        if key == 'mouth':
            continue
        roi = exclusive(roi, key, expected_masks) & ~protected_hair(m, groups)
        if key.startswith('brow-'):
            expected, proof = brow_oracle(key, row, b, groups, m, front, yaw)
            features[key] = assess_brow(col & roi, expected & roi, d, proof)
        else:
            if '_static_texture' in groups and 'uv' in m:
                tex = groups['_static_texture']
                skin = sample(tex, m['uv'][:, :, 0] * (tex.shape[1] - 1), (1 - m['uv'][:, :, 1]) * (tex.shape[0] - 1))
            else:
                skin = sample(skin_plate, m['sx'], m['sy'])
            evidence = eye_evidence(crop, skin)
            cc = components(evidence & roi, d, True)
            extra = components(col & roi & ~expected_masks[key], d)
            features[key] = {'connected_groups': len(cc), 'components': cc, 'extra_ink_groups': len(extra), 'extra_components': extra, 'pass': len(cc) == 1 and (not extra)}
    return features

def zoom(im, roi, label, width=512):
    yy, xx = np.where(roi)
    if not len(xx):
        return Image.new('RGB', (width, 170), 'red')
    x0 = max(0, int(xx.min()) - 8)
    x1 = min(512, int(xx.max()) + 9)
    y0 = max(0, int(yy.min()) - 12)
    y1 = min(512, int(yy.max()) + 13)
    crop = Image.fromarray(im[y0:y1, x0:x1])
    scale = min(width / crop.width, 150 / crop.height)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)))
    out = Image.new('RGB', (width, 180), '#ededed')
    out.paste(crop, ((width - crop.width) // 2, 26))
    ImageDraw.Draw(out).text((8, 6), label, fill='black')
    return out

def run(folder, reference, inspect, out, static=None, video=None, track=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reference = Path(reference)
    folder = Path(folder)
    b = json.loads((reference / 'binding.json').read_text())
    groups = load_groups(reference, b)
    reg = regions(groups, b)
    d = b['source_landmarks']['interocular_px']
    mesh = load_mesh(inspect)
    cache = out.parent / ('geometry-' + folder.name)
    cache.mkdir(exist_ok=True)
    skin_plate = read(reference / 'feature-free.png')
    report = {'config': CONFIG, 'gate_sha256': sha(__file__), 'reference_binding_sha256': sha(reference / 'binding.json'), 'reference_layer_sha256': sha(reference / 'layers.json'), 'static': [], 'frames': [], 'accepted': False}
    current_b = json.loads((folder / 'binding.json').read_text()) if (folder / 'binding.json').exists() else b
    current_groups = load_groups(folder, current_b) if (folder / 'binding.json').exists() else groups
    report['active_binding_sha256'] = sha(folder / 'binding.json') if (folder / 'binding.json').exists() else None
    report['active_layer_sha256'] = sha(folder / 'layers.json') if (folder / 'layers.json').exists() else None
    front = mapping(b, mesh, cache, 0)
    if (folder / 'feature-free.png').exists():
        plate = read(folder / 'feature-free.png')
        col = ink(plate)
        if current_b.get('plate_hair_protection'):
            from toolkit.face.matte_v2 import skin_pixels
            col &= ~skin_pixels(plate)
            col &= ~cv2.dilate(np.uint8(read(folder / current_b['plate_hair_protection'])[:, :, 0] > 10), np.ones((5, 5), np.uint8)).astype(bool)
        report['plate_feature_colour_pixels'] = {key: int((col & (mask > 0.5)).sum()) for key, mask in reg.items()}
    if static:
        static = Path(static)
        proof = json.loads((static / 'static-sources.json').read_text())
        report['source_inventory'] = proof
        for row in proof['records']:
            m = mapping(b, mesh, cache, row['yaw'])
            rs = masks(m, reg)
            im = read(static / row['path'])
            col = ink(im) & ~protected_hair(m, current_groups)
            if '_hair_binding' in current_groups:
                from toolkit.face.matte_v2 import skin_pixels
                col &= ~skin_pixels(im)
            counts = {key: int((col & mask).sum()) for key, mask in rs.items()}
            report['static'].append({**row, 'feature_colour_pixels': counts, 'pass': sum(counts.values()) == 0})
            debug = im.copy()
            debug[col & np.logical_or.reduce(list(rs.values()))] = [255, 0, 255]
            Image.fromarray(debug).save(out / row['path'])
    if video:
        video = Path(video)
        track = json.loads(Path(track or folder / 'track.json').read_text())
        cap = cv2.VideoCapture(str(video))
        frame = 0
        changes = []
        previous = None
        zoom_rows = []
        failed = 0
        report['video_sha256'] = sha(video)
        report['video'] = str(video)
        smooth = False
        if (folder / 'render-frames.json').exists():
            receipt = json.loads((folder / 'render-frames.json').read_text())
            smooth = receipt.get('camera_motion_version') == 'smoothstep/1'
            if smooth:
                if receipt['blend_sha256'] != sha(folder / 'face.blend') or receipt['track'] != track:
                    raise ValueError('Stale render camera receipt')
                if any((abs(r['turn_degrees'] - camera_yaw(r['time_s'], True)) > 1e-08 for r in receipt['records'])):
                    raise ValueError('Camera curve differs from declared smooth motion')
                report['render_receipt_sha256'] = sha(folder / 'render-frames.json')
        report['camera_motion'] = 'smoothstep/1' if smooth else 'historical-2.5-degree-steps'
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            row = track['frames'][min(frame // track.get('exposure_frames', 1), len(track['frames']) - 1)]
            time = frame / track['fps']
            yaw = camera_yaw(time, smooth)
            record = {'frame': frame + 1, 'expression': row['expression'], 'views': []}
            changed = previous != row['expression']
            previous = row['expression']
            for side, angle in enumerate((yaw, 35)):
                crop = im[56:568, side * 512:(side + 1) * 512]
                m = mapping(b, mesh, cache, angle)
                rs = masks(m, reg)
                features = assess_features(crop, rs, m, current_groups, row, current_b, skin_plate, front, angle)
                passed = all((x['pass'] for x in features.values()))
                record['views'].append({'yaw': angle, 'features': features, 'pass': passed})
                if changed or frame in (78, 79, 80, 81, 82, 158, 159, 160, 161, 162):
                    zoom_rows.append(zoom(crop, rs['brow-left'] | rs['brow-right'], f"frame {frame + 1} {row['expression']} {angle:g} deg | " + ('PASS' if passed else 'FAIL')))
            record['pass'] = all((x['pass'] for x in record['views']))
            failed += not record['pass']
            report['frames'].append(record)
            frame += 1
        cap.release()
        report['decoded_frames'] = frame
        report['failed_frames'] = failed
        sheet = Image.new('RGB', (1024, 180 * math.ceil(len(zoom_rows) / 2)), 'white')
        for i, tile in enumerate(zoom_rows):
            sheet.paste(tile, (i % 2 * 512, i // 2 * 180))
        sheet.save(out / 'brow-motion-zoom.png')
    report['accepted'] = bool(report['static'] and report['frames']) and all((x['pass'] for x in report['static'])) and (not report.get('failed_frames', 1)) and (report['decoded_frames'] == 240) and (sum(report.get('plate_feature_colour_pixels', {'missing': 1}).values()) == 0)
    (out / 'c8.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'out': str(out), 'static_counts': [x['feature_colour_pixels'] for x in report['static']], 'failed_frames': report.get('failed_frames'), 'accepted': report['accepted']}))
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--inspect', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--static')
    p.add_argument('--video')
    p.add_argument('--track')
    a = p.parse_args()
    run(**vars(a))

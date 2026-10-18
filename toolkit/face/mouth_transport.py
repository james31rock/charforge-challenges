import cv2, numpy as np
from toolkit.face.mouth_morph import profile as envelope

def endpoint_art(profiles, weights, xx, yy, u, tt, bb):
    h, w = xx.shape
    rgb = np.zeros((h, w, 3), float)
    alpha = np.zeros((h, w), float)
    for i, weight in enumerate(weights):
        if weight < 1e-08:
            continue
        p = profiles[i]
        sx = p['x0'] + u * (p['x1'] - p['x0'])
        st = np.interp(u.ravel(), np.linspace(0, 1, 129), p['top']).reshape(h, w)
        sb = np.interp(u.ravel(), np.linspace(0, 1, 129), p['bottom']).reshape(h, w)
        sy = st + (yy - tt) / np.maximum(bb - tt, 1) * (sb - st) if p['open'] else (st + sb) / 2 + yy - (tt + bb) / 2
        source = p['body'].astype(np.float32) / 255
        source[:, :, :3] *= source[:, :, 3:4]
        warped = cv2.remap(source, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb += weight * warped[:, :, :3] * 255
        alpha += weight * warped[:, :, 3]
    rgb /= np.maximum(alpha[:, :, None], 1e-08)
    return np.dstack((rgb, alpha * 255))

def profile(p):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(p[:, :, 3] > 70), 8)
    label = 1 + np.argmax(stats[1:, 4])
    main = cv2.dilate(np.uint8(labels == label), np.ones((5, 5), np.uint8)) > 0
    body = p.copy()
    body[~main] = 0
    shadow = p.copy()
    shadow[main] = 0
    dark = (body[:, :, 3] > 100) & (cv2.cvtColor(body[:, :, :3], cv2.COLOR_RGB2GRAY) < 145)
    contours, _ = cv2.findContours(np.uint8(dark), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros(dark.shape, np.uint8)
    cv2.drawContours(solid, contours, -1, 255, -1)
    geometry = body.copy()
    geometry[:, :, 3] = solid
    r = envelope(geometry)
    r['body'] = body
    r['shadow'] = shadow
    ink = dark & (body[:, :, 1].astype(float) / np.maximum(body[:, :, 0], 1) > 0.43)
    dist = cv2.distanceTransform(np.uint8(ink), cv2.DIST_L2, 5)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 0.01) & (dist > 0)
    r['stroke'] = float(np.median(2 * dist[ridge] - 1))
    colors = p[:, :, :3][ink]
    colors = colors[np.argsort(colors.mean(1))]
    r['ink'] = np.median(colors[:max(1, len(colors) // 4)], axis=0)
    r['open'] = np.median(r['bottom'][32:97] - r['top'][32:97]) > max(13, r['stroke'] * 3)
    sy, sx = np.where(shadow[:, :, 3] > 70)
    r['shadow_center'] = [float(sx.mean()), float(sy.mean())] if len(sx) else [(r['x0'] + r['x1']) / 2, float(np.max(r['bottom']) + 12)]
    return r

def morph(patches, profiles, weights):
    active = [i for i, v in enumerate(weights) if v > 1e-08]
    if len(active) == 1:
        return patches[active[0]].copy()
    h, w = patches[0].shape[:2]
    yy, xx = np.indices((h, w))
    x0 = sum((weights[i] * profiles[i]['x0'] for i in active))
    x1 = sum((weights[i] * profiles[i]['x1'] for i in active))
    top = sum((weights[i] * profiles[i]['top'] for i in active))
    bottom = sum((weights[i] * profiles[i]['bottom'] for i in active))
    u = (xx - x0) / max(x1 - x0, 1)
    tt = np.interp(u.ravel(), np.linspace(0, 1, 129), top).reshape(h, w)
    bb = np.interp(u.ravel(), np.linspace(0, 1, 129), bottom).reshape(h, w)
    v = (yy - tt) / np.maximum(bb - tt, 1)
    x = np.linspace(x0, x1, 129)
    points = np.concatenate((np.stack((x, top), 1), np.stack((x[::-1], bottom[::-1]), 1)))
    mask = np.zeros((h * 4, w * 4), np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(points * 4))], 1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) / 4
    stroke = sum((weights[i] * profiles[i]['stroke'] for i in active))
    inside = np.clip(dist - stroke + 0.5, 0, 1)
    coverage = cv2.resize(mask.astype(float), (w, h), interpolation=cv2.INTER_AREA)
    inside = cv2.resize(inside, (w, h), interpolation=cv2.INTER_AREA)
    colour = np.zeros((h, w, 3), float)
    den = np.zeros((h, w), float)
    opening = [i for i in active if profiles[i]['open']]
    for i in opening:
        p = profiles[i]
        sx = p['x0'] + u * (p['x1'] - p['x0'])
        st = np.interp(u.ravel(), np.linspace(0, 1, 129), p['top']).reshape(h, w)
        sb = np.interp(u.ravel(), np.linspace(0, 1, 129), p['bottom']).reshape(h, w)
        sy = st + v * (sb - st)
        source = p['body'].astype(np.float32) / 255
        source[:, :, :3] *= source[:, :, 3:4]
        warped = cv2.remap(source, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        colour += weights[i] * warped[:, :, :3] * 255
        den += weights[i] * warped[:, :, 3]
    colour /= np.maximum(den[:, :, None], 1e-08)
    ink = sum((weights[i] * profiles[i]['ink'] for i in active))
    if not opening:
        inside[:] = 0
    rgb = ink[None, None, :] * (1 - inside[:, :, None]) + colour * inside[:, :, None]
    alpha = coverage
    endpoint = sum((weights[i] ** 4 for i in active)) if opening else 1.0
    near = endpoint_art(profiles, weights, xx, yy, u, tt, bb)
    na = near[:, :, 3] / 255
    premul = rgb * alpha[:, :, None] * (1 - endpoint) + near[:, :, :3] * na[:, :, None] * endpoint
    alpha = alpha * (1 - endpoint) + na * endpoint
    rgb = premul / np.maximum(alpha[:, :, None], 1e-08)
    target = sum((weights[i] * np.array(profiles[i]['shadow_center']) for i in active))
    srgb = np.zeros((h, w, 3), float)
    sa = np.zeros((h, w), float)
    for i in active:
        p = profiles[i]
        cx, cy = p['shadow_center']
        scale = (x1 - x0) / max(1, p['x1'] - p['x0'])
        matrix = np.float32([[scale, 0, target[0] - scale * cx], [0, 1, target[1] - cy]])
        source = p['shadow'].astype(np.float32) / 255
        source[:, :, :3] *= source[:, :, 3:4]
        s = cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_LINEAR)
        srgb += weights[i] * s[:, :, :3] * 255
        sa += weights[i] * s[:, :, 3]
    premul = rgb * alpha[:, :, None]
    premul = premul * (1 - sa[:, :, None]) + srgb
    alpha = alpha * (1 - sa) + sa
    rgb = premul / np.maximum(alpha[:, :, None], 1e-08)
    rgb[alpha < 0.001] = 0
    return np.dstack((np.uint8(np.clip(rgb, 0, 255)), np.uint8(np.clip(alpha * 255, 0, 255))))

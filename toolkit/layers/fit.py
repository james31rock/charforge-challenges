import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from toolkit.runtime import output_root
import argparse, json, hashlib, shutil, math
from collections import deque
import bpy, bmesh, numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from toolkit.layers.core import tps, segment_projection, fit_hip_from_leg, validate_input, voxel_surface
from toolkit.rig.blender_rig import arrays, create_armature, helpers, apply_weights
from toolkit.rig.geometry import skeleton, section, arm_region_masks

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2, allow_nan=False), encoding='utf8')

def base_obj(path):
    vertices = []
    faces = []
    groups = {}
    group = ''
    for line in Path(path).read_text().splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == 'v':
            vertices.append([float(x) for x in s[1:4]])
        elif s[0] == 'g':
            group = s[1]
        elif s[0] == 'f':
            face = [int(x.split('/')[0]) - 1 for x in s[1:]]
            groups.setdefault(group, set()).update(face)
            if group == 'body':
                faces.append(face)
    v = np.asarray(vertices)[:, [0, 2, 1]] * np.array([0.1, -0.1, 0.1])
    anchors = {name: v[sorted(ids)].mean(0) for name, ids in groups.items() if name.startswith('joint-')}
    ix = np.unique(np.concatenate(faces))
    remap = {old: i for i, old in enumerate(ix)}
    return (v[ix], [[remap[k] for k in f] for f in faces], anchors)

def material_colors(ob):
    result = []
    cache = {}
    uv = ob.data.uv_layers.active
    for mat in ob.data.materials:
        im = next((n.image for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image and n.image.size[0]), None) if mat and mat.use_nodes else None
        if im:
            a = np.empty(len(im.pixels), np.float32)
            im.pixels.foreach_get(a)
            cache[mat.name] = (a.reshape(im.size[1], im.size[0], 4), im.size[:])
    for face in ob.data.polygons:
        mat = ob.data.materials[face.material_index] if face.material_index < len(ob.data.materials) else None
        if mat and mat.name in cache and uv:
            a, (w, h) = cache[mat.name]
            p = np.mean([uv.data[k].uv[:] for k in face.loop_indices], 0) % 1
            result.append(a[min(h - 1, int(p[1] * h)), min(w - 1, int(p[0] * w)), :3])
        else:
            result.append(mat.diffuse_color[:3] if mat else [0.5, 0.5, 0.5])
    return np.asarray(result)

def is_inside(tree, point, eps):
    votes = 0
    for direction in [(1, 0.271, 0.119), (-0.173, 1, 0.233), (0.131, -0.213, 1)]:
        d = Vector(direction).normalized()
        o = Vector(point)
        hits = 0
        for _ in range(64):
            loc, _, _, dist = tree.ray_cast(o, d)
            if loc is None:
                break
            hits += 1
            o = loc + d * eps
        votes += hits % 2
    return votes >= 2

def voxel_clip(vertices, tris, target_tree, h):
    pitch = h / 128
    halfdiag = np.sqrt(3) * pitch
    guard = h * 0.001
    proxy_tree = BVHTree.FromPolygons(vertices.tolist(), tris.tolist(), all_triangles=True)
    lo = np.floor(vertices.min(0) / pitch).astype(int)
    hi = np.ceil(vertices.max(0) / pitch).astype(int)
    cells = set()
    clearance = []
    for z in range(lo[2], hi[2] + 1):
        for y in range(lo[1], hi[1] + 1):
            for x in range(lo[0], hi[0] + 1):
                q = (np.array([x, y, z], float) + 0.5) * pitch
                _, _, _, dist = target_tree.find_nearest(Vector(q))
                if dist is None or dist < halfdiag + guard:
                    continue
                if is_inside(proxy_tree, q, h * 1e-06) and is_inside(target_tree, q, h * 1e-06):
                    cells.add((x, y, z))
                    clearance.append(dist - halfdiag)
    if not cells:
        raise ValueError('No contained proxy cells; more anatomical observations needed')
    original_cells = len(cells)
    neighbours = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    closed = set(cells)
    for _ in range(2):
        closed |= {tuple((c[k] + d[k] for k in range(3))) for c in closed for d in neighbours}
    for _ in range(2):
        closed = {c for c in closed if all((tuple((c[k] + d[k] for k in range(3))) in closed for d in neighbours))}
    for cell in sorted(closed - cells):
        q = (np.array(cell, float) + 0.5) * pitch
        _, _, _, dist = target_tree.find_nearest(Vector(q))
        if dist is not None and dist > halfdiag + guard and is_inside(target_tree, q, h * 1e-06):
            cells.add(cell)
            clearance.append(dist - halfdiag)
    vv, ff = voxel_surface(cells, pitch)
    return (vv, ff, dict(method='conservative CC0-volume / source-volume intersection; manifold marching tetrahedra', pitch_m=pitch, cells=len(cells), closed_crack_cells=len(cells) - original_cells, minimum_surface_support_clearance_m=min(clearance), surface_detail_limited=True))

def segments(points):
    p = points
    hips = (p['leftHip'] + p['rightHip']) / 2
    neck = p['neck']
    rows = [('torso', hips, neck, 'spine'), ('chest', hips * 0.45 + neck * 0.55, hips * 0.2 + neck * 0.8, 'chest'), ('neck', neck, p['head'], 'neck'), ('head', p['head'], p['head_top'] * 0.65 + p['head'] * 0.35, 'head')]
    for s in ['left', 'right']:
        rows.append((s + '_shoulder', neck * 0.6 + p[s + 'Shoulder'] * 0.4, p[s + 'Shoulder'], 'upperChest'))
        for label, a, b, bone in [('upper_arm', 'Shoulder', 'Elbow', 'UpperArm'), ('forearm', 'Elbow', 'Wrist', 'LowerArm'), ('hand', 'Wrist', 'Hand', 'Hand'), ('thigh', 'Hip', 'Knee', 'UpperLeg'), ('shin', 'Knee', 'Ankle', 'LowerLeg'), ('foot', 'Ankle', 'Toe', 'Foot')]:
            rows.append((s + '_' + label, p[s + a], p[s + b], s + bone))
    return rows

def fit_proxy(v, faces, anchors, p, target_v, target_t, h):
    names = {'neck': 'joint-neck', 'head': 'joint-head', 'head_top': 'joint-head-2'}
    for s, short in [('left', 'l'), ('right', 'r')]:
        for dst, src in [('Hip', 'upper-leg'), ('Knee', 'knee'), ('Ankle', 'ankle'), ('Shoulder', 'shoulder'), ('Elbow', 'elbow'), ('Wrist', 'hand'), ('Hand', 'finger-3-3'), ('Toe', 'foot-2')]:
            names[s + dst] = 'joint-' + short + '-' + src
    source_points = {name: anchors[key] for name, key in names.items()}
    source_points['head_top'] = np.array([0, source_points['head'][1], v[:, 2].max()])
    source_segments = segments(source_points)
    target_segments = segments(p)
    triangles = np.array([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)])
    controls = [source_points[n] for n in names]
    destinations = [p[n] for n in names]
    for (name, a, b, _), (_, ta, tb, _) in zip(source_segments, target_segments):
        if 'shoulder' in name or name in ['neck', 'chest']:
            continue
        rotation = np.array(Vector(b - a).rotation_difference(Vector(tb - ta)).to_matrix())
        midpoint = (ta + tb) / 2
        sr = section(v, triangles, (a + b) / 2, b - a, np.ptp(v[:, 2]) * 0.25)
        tr = section(target_v, target_t, midpoint, tb - ta, h * 0.35)
        if not sr or not tr:
            continue
        axis = (b - a) / np.linalg.norm(b - a)
        u = np.array([1.0, 0, 0])
        u -= axis * (u @ axis)
        if np.linalg.norm(u) < 0.1:
            u = np.cross(axis, [0, 1, 0])
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        for d in (u, -u, w, -w):
            sradius = float(np.max((sr['outline'] - (a + b) / 2) @ d))
            td = d @ rotation.T
            tradius = float(np.max((tr['outline'] - midpoint) @ td)) * 0.78
            if name == 'torso':
                tradius = min(tradius, np.linalg.norm(p['leftShoulder'] - p['rightShoulder']) * 0.55)
            if name.endswith('thigh'):
                side = name.split('_')[0]
                knee = section(target_v, target_t, p[side + 'Knee'], p[side + 'Knee'] - p[side + 'Ankle'], h * 0.1)
                if knee:
                    tradius = min(tradius, knee['radius'] * 1.18)
            if sradius <= 0 or tradius <= 0:
                continue
            controls.append((a + b) / 2 + d * sradius)
            destinations.append(midpoint + td * tradius)
    fitted = tps(controls, destinations, v)
    tree = BVHTree.FromPolygons(target_v.tolist(), target_t.tolist(), all_triangles=True)
    seg = segments(p)
    projections = []
    distance = []
    for (_, a, b, _), (_, ta, tb, _) in zip(source_segments, seg):
        q, d, u = segment_projection(v, a, b)
        projections.append(ta + u[:, None] * (tb - ta))
        distance.append(d)
    which = np.argmin(distance, axis=0)
    origins = np.array(projections)[which, np.arange(len(v))]
    inset = h * 0.003
    misses = 0
    outside_origins = 0
    for i, (q, origin) in enumerate(zip(fitted, origins)):
        if not is_inside(tree, origin, h * 1e-06):
            loc, n, _, _ = tree.find_nearest(Vector(origin))
            if loc is not None:
                candidates = [loc - n * inset, loc + n * inset]
                origin = next((np.array(c) for c in candidates if is_inside(tree, c, h * 1e-06)), origin)
            if not is_inside(tree, origin, h * 1e-06):
                outside_origins += 1
        origins[i] = origin
        d = q - origin
        length = np.linalg.norm(d)
        if length < 1e-10:
            continue
        direction = Vector(d / length)
        hit, _, _, dist = tree.ray_cast(Vector(origin), direction)
        if hit is None:
            misses += 1
            fitted[i] = origin
            continue
        fitted[i] = origin + d / length * max(0, min(length * 0.94, dist - inset))
    tris = np.array([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)])
    history = []
    best = fitted.copy()
    best_count = float('inf')
    edges = np.unique(np.sort(np.vstack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])), axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=len(fitted))
    for iteration in range(60):
        proxy_tree = BVHTree.FromPolygons(fitted.tolist(), tris.tolist(), all_triangles=True)
        overlaps = proxy_tree.overlap(tree)
        history.append(len(overlaps))
        if len(overlaps) < best_count:
            best_count = len(overlaps)
            best = fitted.copy()
        if not overlaps:
            break
        bad = np.unique(tris[[x[0] for x in overlaps]])
        if iteration < 24:
            fitted[bad] = origins[bad] + 0.72 * (fitted[bad] - origins[bad])
        else:
            sums = np.zeros_like(fitted)
            np.add.at(sums, edges[:, 0], fitted[edges[:, 1]])
            np.add.at(sums, edges[:, 1], fitted[edges[:, 0]])
            candidates = fitted[bad] * 0.3 + sums[bad] / np.maximum(degree[bad, None], 1) * 0.7
            for k, vi in enumerate(bad):
                q = candidates[k]
                if is_inside(tree, q, h * 1e-06):
                    fitted[vi] = q
                else:
                    fitted[vi] = origins[vi] + 0.9 * (fitted[vi] - origins[vi])
    fitted = best
    fallback = None
    if best_count:
        fitted, faces, fallback = voxel_clip(fitted, tris, tree, h)
        tris = np.array([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)])
        which = np.argmin([segment_projection(fitted, a, b)[1] for _, a, b, _ in seg], axis=0)
    probes = np.vstack((fitted, fitted[tris].mean(1)))
    outside = [i for i, q in enumerate(probes) if not is_inside(tree, q, h * 1e-06)]
    proxy_tree = BVHTree.FromPolygons(fitted.tolist(), tris.tolist(), all_triangles=True)
    crossings = len(proxy_tree.overlap(tree))
    return (fitted, tris, dict(method='three-ray majority; all vertices and triangle centroids; triangle BVH overlap', probes=len(probes), outside_probes=len(outside), triangle_overlap_pairs=crossings, ray_misses=misses, invalid_origins=outside_origins, inset_m=inset, contraction_history=history, voxel_fallback=fallback, passed=len(outside) == 0 and crossings == 0, limitation='Parity is diagnostic on non-watertight or intersecting source meshes'), which, seg, faces)

def classify(ob, p, h, proxy, proxy_tri, palette=None):
    v, tri, _ = arrays(ob)
    centres = np.array([v[list(f.vertices)].mean(0) for f in ob.data.polygons])
    colors = material_colors(ob)
    regions, region_report = arm_region_masks(v, tri, h, p)
    mass = np.maximum(*[regions.get('ownership', {}).get(s, {}).get('mass', np.zeros(len(v))) for s in ['left', 'right']])
    arm_faces = np.array([np.mean(mass[list(f.vertices)]) > 0.55 for f in ob.data.polygons])
    head_span = float(v[:, 2].max() - p['neck'][2])
    neckz = p['neck'][2]
    head = (centres[:, 2] > neckz + 0.2 * head_span) & (centres[:, 2] < neckz + 0.62 * head_span) & (abs(centres[:, 0]) < 0.05 * h) & (centres[:, 1] < p['head'][1])
    choices = colors[head]
    warm = choices[(choices[:, 0] > choices[:, 2] * 1.15) & (choices[:, 0] > 0.25) & (choices[:, 1] > choices[:, 2] * 0.9)] if len(choices) else choices
    if len(warm) > 3:
        warm = warm[warm.mean(1) >= np.quantile(warm.mean(1), 0.4)]
    skin = np.median(warm if len(warm) > 3 else choices, axis=0) if len(choices) else np.array([0.7, 0.45, 0.3])
    if palette:
        swatches = np.asarray(palette, float) / 255
        warm2 = swatches[(swatches[:, 0] > swatches[:, 2] * 1.15) & (swatches[:, 0] > 0.25) & (swatches[:, 1] > swatches[:, 2] * 0.9)]
        if len(warm2):
            swatch = warm2[np.argmin(np.linalg.norm(warm2 - skin, axis=1))]
            if np.linalg.norm(swatch - skin) < 0.15:
                skin = 0.75 * skin + 0.25 * swatch
    skin_distance = np.linalg.norm(colors - skin, axis=1)
    scalp = centres[:, 2] > neckz + 0.82 * head_span
    hair_color = np.median(colors[scalp], axis=0)
    hair_distance = np.linalg.norm(colors - hair_color, axis=1)
    possible_hair = (centres[:, 2] > neckz - 0.055 * h) & (hair_distance < 0.28) & (hair_distance < skin_distance * 0.9)
    edges = {}
    adj = [[] for _ in ob.data.polygons]
    for f in ob.data.polygons:
        ids = list(f.vertices)
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edge = tuple(sorted((a, b)))
            if edge in edges:
                j = edges[edge]
                adj[j].append(f.index)
                adj[f.index].append(j)
            else:
                edges[edge] = f.index
    hair_mask = scalp & possible_hair
    queue = deque(np.flatnonzero(hair_mask).tolist())
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if possible_hair[j] and (not hair_mask[j]):
                hair_mask[j] = True
                queue.append(j)
    tree = BVHTree.FromPolygons(proxy.tolist(), proxy_tri.tolist(), all_triangles=True)
    gaps = np.array([tree.find_nearest(Vector(q))[3] for q in centres])
    labels = []
    seg = segments(p)
    dist = []
    for _, a, b, _ in seg:
        dist.append(segment_projection(centres, a, b)[1])
    nearest = np.argmin(dist, axis=0)
    hipz = (p['leftHip'][2] + p['rightHip'][2]) / 2
    torso = (centres[:, 2] > hipz + 0.08 * h) & (centres[:, 2] < neckz - 0.08 * h) & (abs(centres[:, 0]) < 0.055 * h) & (centres[:, 1] < p['head'][1])
    top_color = np.median(colors[torso], axis=0) if torso.any() else np.array([0.0, 0.0, 0.0])
    top_distance = np.linalg.norm(colors - top_color, axis=1)
    ring = section(v, np.array([[f.vertices[0], f.vertices[k], f.vertices[k + 1]] for f in ob.data.polygons for k in range(1, len(f.vertices) - 1)]), [0, 0, hipz - 0.025 * h], [0, 0, 1], h * 0.4)
    skirt = bool(ring and ring['widths'][0] > abs(p['leftHip'][0] - p['rightHip'][0]) * 2.3)
    for i, q in enumerate(centres):
        branch = seg[nearest[i]][0]
        col = colors[i]
        face_front = q[2] > neckz - 0.015 * h and q[2] < neckz + 0.82 * head_span and (q[1] < p['head'][1])
        if hair_mask[i]:
            label = 'hair'
        elif face_front:
            label = 'skin'
        elif skin_distance[i] < 0.27 and gaps[i] < 0.08 * h and (q[2] > neckz - 0.025 * h or arm_faces[i] or 'shin' in branch):
            label = 'skin'
        elif q[2] > neckz:
            label = 'hair'
        elif arm_faces[i]:
            label = 'sleeve'
        elif q[2] < hipz and ('foot' in branch or q[2] < h * 0.13):
            label = 'footwear'
        elif skirt and hipz - 0.15 * h < q[2] < hipz + 0.12 * h:
            label = 'skirt'
        elif q[2] < hipz and top_distance[i] < 0.2:
            label = 'top'
        elif q[2] < hipz:
            label = 'skirt' if skirt and q[2] > hipz - 0.15 * h else 'trousers'
        else:
            label = 'top'
        labels.append(label)
    return (np.array(labels), dict(skin_rgb=skin.tolist(), hair_rgb=hair_color.tolist(), top_rgb=top_color.tolist(), head_span_m=head_span, s0_palette=palette, skin_seed_faces=int(head.sum()), proxy_distance_p50_m=float(np.median(gaps)), proxy_distance_p95_m=float(np.quantile(gaps, 0.95)), method='head-scaled skin seeds + scalp-connected texture hair + original fused surface arm ownership + torso colour + anatomical branch + proxy distance; S0 palette evidence', semantic_status='unreviewed', skirt_hypothesis=skirt))

def shell_copy(ob, label, labels):
    new = ob.copy()
    new.data = ob.data.copy()
    new.name = 'Shell_' + label
    bpy.context.collection.objects.link(new)
    bm = bmesh.new()
    bm.from_mesh(new.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if labels[f.index] != label], context='FACES')
    bm.to_mesh(new.data)
    bm.free()
    new.data.update()
    new['charforge_part'] = label
    return new

def render_review(objects, proxy, arm, out, h, p):
    sc = bpy.context.scene
    glass = bpy.data.materials.new('Proxy_xray_shell')
    glass.use_nodes = True
    n = glass.node_tree.nodes
    n.clear()
    output = n.new('ShaderNodeOutputMaterial')
    mix = n.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = 0.2
    transparent = n.new('ShaderNodeBsdfTransparent')
    em = n.new('ShaderNodeEmission')
    em.inputs[0].default_value = (0.2, 0.43, 0.65, 1)
    glass.node_tree.links.new(transparent.outputs[0], mix.inputs[1])
    glass.node_tree.links.new(em.outputs[0], mix.inputs[2])
    glass.node_tree.links.new(mix.outputs[0], output.inputs[0])
    pm = bpy.data.materials.new('Proxy_proxy_cyan')
    pm.diffuse_color = (0.03, 0.7, 0.45, 1)
    pm.use_nodes = True
    pm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.03, 0.7, 0.45, 1)
    proxy.data.materials.clear()
    proxy.data.materials.append(pm)
    proxy.hide_render = False
    saved = {o.name: list(o.data.materials) for o in objects}
    for o in objects:
        for k in range(len(o.data.materials)):
            o.data.materials[k] = glass
    for o in sc.objects:
        if o.type in ('CAMERA', 'LIGHT'):
            o.hide_render = True
    if sc.world is None:
        sc.world = bpy.data.worlds.new('Proxy_world')
    sc.world.color = (0.3, 0.3, 0.3)
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 8
    sc.render.resolution_x = 600
    sc.render.resolution_y = 700
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'Standard'
    for loc, energy in [((h * 2, -h * 3, h * 3), 450), ((-h * 2, -h, h * 2), 200)]:
        data = bpy.data.lights.new('Proxy_light', 'AREA')
        ob = bpy.data.objects.new('Proxy_light', data)
        sc.collection.objects.link(ob)
        ob.location = loc
        data.energy = energy
        data.shape = 'DISK'
        data.size = h * 3
        ob.rotation_euler = (Vector((0, 0, h * 0.5)) - ob.location).to_track_quat('-Z', 'Y').to_euler()
    data = bpy.data.cameras.new('Proxy_camera')
    cam = bpy.data.objects.new('Proxy_camera', data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    data.type = 'ORTHO'
    data.ortho_scale = h * 1.18
    for name, pos in [('front', (0, -h * 3, h * 0.5)), ('side', (h * 3, 0, h * 0.5))]:
        cam.location = pos
        cam.rotation_euler = (Vector((0, 0, h * 0.5)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = str(out / ('xray-' + name + '.png'))
        bpy.ops.render.render(write_still=True)
    proxy.hide_render = True
    for o in objects:
        for k, m in enumerate(saved[o.name]):
            o.data.materials[k] = m

def run(cfg):
    validate_input(cfg)
    out = Path(cfg['out']).resolve()
    scratch = output_root()
    if not out.is_relative_to(scratch) or out == scratch:
        raise ValueError('Expected an output child of CHARFORGE_SCRATCH')
    if out.name != cfg['character_id']:
        raise ValueError('Output leaf must match character_id; refuse cross-character overwrite')
    if shutil.disk_usage(scratch).free < 40 * 1024 ** 3:
        raise RuntimeError('BLOCKED disk')
    out.mkdir(parents=True, exist_ok=True)
    source = Path(cfg['source'])
    before = sha(source)
    h = cfg['height_cm'] / 100
    if source.suffix == '.blend':
        bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(source), use_image_search=False)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and (not o.hide_render) and (not o.name.startswith('WGT'))]
    if cfg.get('mesh_objects'):
        meshes = [bpy.data.objects[n] for n in cfg['mesh_objects']]
    for a in [o for o in bpy.data.objects if o.type == 'ARMATURE']:
        a.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    land = json.loads(Path(cfg['landmarks']).read_text())
    p = {k: np.array(v, float) for k, v in land['points'].items()}
    details = land['details']
    if cfg.get('canonical_source'):
        basis = np.eye(3)
        origin = np.zeros(3)
        scale = 1
    else:
        transform = land['source_transform']
        basis = np.array(transform['basis'])
        origin = np.array(transform['origin'])
        scale = transform['scale']
    raw = {o.name: arrays(o) for o in meshes}
    for o in meshes:
        co, tri, edges = raw[o.name]
        co = (co @ basis - origin) * scale
        if o.data.shape_keys:
            o.shape_key_clear()
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        o.modifiers.clear()
        o.vertex_groups.clear()
        o.data.vertices.foreach_set('co', co.ravel())
        o.data.update()
        raw[o.name] = (co, tri, edges)
    for o in list(bpy.data.objects):
        if o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    allv = np.concatenate([r[0] for r in raw.values()])
    allt = []
    offset = 0
    for v, t, e in raw.values():
        allt.extend((t + offset).tolist())
        offset += len(v)
    allt = np.array(allt)
    old = {k: v.copy() for k, v in p.items()}
    for s in ['left', 'right']:
        p[s + 'Hip'][0] = fit_hip_from_leg(p[s + 'Knee'], p[s + 'Ankle'], p[s + 'Hip'][2])
        details[s + 'Hip'] = {**details[s + 'Hip'], 'source': 'leg-centreline constrained CC0 proxy', 'confidence': 0.55}
    if cfg.get('reuse_proxy'):
        prior = json.loads((out / 'layers.json').read_text())
        if prior['source_sha256'] != before or prior['height_cm'] != cfg['height_cm'] or prior['character_id'] != cfg['character_id']:
            raise ValueError('Proxy cache identity mismatch')
        with np.load(out / 'proxy.npz') as cached:
            pv = cached['vertices']
            pt = cached['triangles']
        bf = pt.tolist()
        contain = prior['proxy']['containment']
        seg = segments(p)
        which = np.argmin([segment_projection(pv, a, b)[1] for _, a, b, _ in seg], axis=0)
    else:
        bv, bf, anchors = base_obj(cfg['mpfb_base'])
        pv, pt, contain, which, seg, bf = fit_proxy(bv, bf, anchors, p, allv, allt, h)
    me = bpy.data.meshes.new('CC0_inner_body')
    me.from_pydata(pv.tolist(), [], bf)
    me.update()
    proxy = bpy.data.objects.new('CC0_inner_body', me)
    bpy.context.collection.objects.link(proxy)
    proxy.hide_render = True
    proxy['charforge_collision_only'] = True
    print('PROXY', cfg['character_id'], contain, flush=True)
    parts = []
    shells = []
    classifications = []
    labelsets = {}
    for ob in meshes:
        labels, audit = classify(ob, p, h, pv, pt, cfg.get('s0_palette'))
        classifications.append(audit)
        labelsets[ob.name] = labels
        np.savez_compressed(out / (ob.name.replace('/', '_') + '-face-labels.npz'), labels=labels)
    bones = skeleton(p, details)
    arm = create_armature(bones, p, details)
    supports, twists = helpers(arm, bones, p, details)
    reports = {}
    for ob in meshes:
        co, tr, ed = arrays(ob)
        regions, rr = arm_region_masks(co, tr, h, p)
        reports[ob.name] = apply_weights(ob, co, ed, bones, arm, h, 'body', supports, twists, regions, rr)
        labels = labelsets[ob.name]
        for label in sorted(set(labels)):
            shell = shell_copy(ob, label, labels)
            shells.append(shell)
            counts = {}
            for f in shell.data.polygons:
                for a, b in zip(f.vertices, list(f.vertices)[1:] + [f.vertices[0]]):
                    counts[tuple(sorted((a, b)))] = counts.get(tuple(sorted((a, b))), 0) + 1
            parts.append(dict(object=shell.name, label=label, faces=len(shell.data.polygons), confidence=0.5, closed=all((x == 2 for x in counts.values()))))
        bpy.data.objects.remove(ob, do_unlink=True)
    co, tr, ed = arrays(proxy)
    apply_weights(proxy, co, ed, bones, arm, h, 'body', supports, twists)
    colliders = []
    for i, (name, a, b, bone) in enumerate(seg):
        ix = which == i
        _, d, _ = segment_projection(pv[ix], a, b)
        radius = float(np.quantile(d, 0.15)) if len(d) else 0.0
        colliders.append(dict(name=name, head=a.tolist(), tail=b.tolist(), radius_m=radius, bone=bone, provenance='15th percentile inscribed proxy radial distance', purpose='body collision, not garment clearance', requires_review=True))
    render_review(shells, proxy, arm, out, h, p)
    arm.data.pose_position = 'POSE'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'layers.blend'))
    report = dict(schema='charforge.layers/1', character_id=cfg['character_id'], height_cm=cfg['height_cm'], source_sha256=before, status='review_required' if contain['passed'] else 'failed', coordinate_system='metres; X-left Y-back Z-up', source_manifest=cfg.get('source_manifest'), proxy=dict(asset_sha256=sha(cfg['mpfb_base']), visible=False, vertices=len(pv), triangles=len(pt), containment=contain, licence='CC0-1.0'), joints={k: v.tolist() for k, v in p.items()}, original_joints={k: v.tolist() for k, v in old.items()}, colliders=colliders, parts=parts, classification=classifications, implementation_sha256=sha(__file__), artifact_sha256=sha(out / 'layers.blend'), limitations=['Hidden anatomy is estimated, not observed.', 'Semantic masks require review; open shells are not complete cloth.', 'Collision capsules do not enclose loose garments.', 'No community release acceptance is claimed.'])
    write(out / 'layers.json', report)
    write(out / 'landmarks.json', {**land, 'points': report['joints'], 'details': details})
    write(out / 'rig.json', dict(schema_version=1, height_cm=cfg['height_cm'], character_id=cfg['character_id'], output_sha256=sha(out / 'layers.blend'), humanoid={b['name']: b['name'] for b in bones}, bones=bones, correctives=supports, twist_helpers=twists, weights=reports))
    np.savez_compressed(out / 'proxy.npz', vertices=pv, triangles=pt)
    if sha(source) != before:
        raise RuntimeError('Input changed')
    print('Proxy_FIT', cfg['character_id'], report['status'], flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(json.loads(Path(a.input).read_text()))

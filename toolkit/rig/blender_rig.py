import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, json, hashlib, re, shutil, os
import bpy
import numpy as np
from mathutils import Vector, Matrix
from toolkit.rig.math_core import frame, vec, perpendicular, corrective_profile
from toolkit.rig.geometry import detect, skeleton, arm_region_masks
from toolkit.rig.weights import diffuse
from toolkit.rig.foot_math import bind_shoes

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

from toolkit.runtime import output_root, safe_output

def validate(record):
    if record.get('schema_version') != 1:
        raise ValueError('Unsupported measurement schema')
    if not re.fullmatch('[a-zA-Z0-9_-]+', record.get('character_id', '')):
        raise ValueError('Invalid character_id')
    height = record.get('height_cm')
    if isinstance(height, bool) or not isinstance(height, (int, float)) or (not np.isfinite(height)) or (height <= 0):
        raise ValueError('height_cm is mandatory and must be positive')
    frame(record['orientation']['up'], record['orientation']['forward'])
    path = Path(record['mesh_path']).resolve()
    if 'private' in [p.lower() for p in path.parts]:
        raise ValueError('Private input requires a separate metadata review; not accepted here')
    if not path.is_file():
        raise ValueError('Mesh file not found')
    for key in ['landmarks', 'silhouette_landmarks']:
        for p in record.get(key, {}).values():
            vec(p)
    return path

def arrays(ob):
    m = ob.data
    co = np.empty((len(m.vertices), 3))
    m.vertices.foreach_get('co', co.ravel())
    mw = np.array(ob.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    m.calc_loop_triangles()
    tri = np.empty((len(m.loop_triangles), 3), dtype=np.int32)
    m.loop_triangles.foreach_get('vertices', tri.ravel())
    edges = np.empty((len(m.edges), 2), dtype=np.int32)
    m.edges.foreach_get('vertices', edges.ravel())
    return (co, tri, edges)

def create_armature(bones, points, details):
    data = bpy.data.armatures.new('CharForgeHumanoid')
    arm = bpy.data.objects.new('CharForgeRig', data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for b in bones:
        eb = data.edit_bones.new(b['name'])
        eb.head = b['head']
        eb.tail = b['tail']
        if b['parent']:
            eb.parent = data.edit_bones[b['parent']]
        eb.use_connect = False
        direction = np.array(b['tail']) - b['head']
        if b['role'] == 'leg':
            side = 'left' if b['name'].startswith('left') else 'right'
            hint = np.array(points[side + 'Toe']) - points[side + 'Heel']
        else:
            hint = np.array([0, -1, 0])
        eb.align_roll(Vector(perpendicular(direction, hint)))
    bpy.ops.object.mode_set(mode='OBJECT')
    for p in arm.pose.bones:
        p.rotation_mode = 'QUATERNION'
    arm.show_in_front = True
    return arm

def helpers(arm, bones, points, details):
    configs = []
    twists = []
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for s in ['left', 'right']:
        for distal, joint in [('LowerLeg', 'Knee'), ('LowerArm', 'Elbow'), ('UpperLeg', 'Hip'), ('UpperArm', 'Shoulder'), ('Hand', 'Wrist')]:
            name = s + distal
            eb = arm.data.edit_bones[name]
            parent = eb.parent
            radius = details[s + joint].get('radius_m') or min(eb.length, parent.length) * 0.15
            cfg = corrective_profile(radius, parent.length, eb.length)
            hb = arm.data.edit_bones.new('support_' + name)
            hb.head = eb.head
            hb.tail = eb.head + (eb.tail - eb.head) * 0.2
            hb.roll = eb.roll
            hb.parent = parent
            configs.append(dict(bone=hb.name, target=name, joint=s + joint, radius=radius, **cfg))
        name = s + 'LowerArm'
        eb = arm.data.edit_bones[name]
        hb = arm.data.edit_bones.new('twist_' + name)
        hb.head = eb.head
        hb.tail = eb.tail
        hb.roll = eb.roll
        hb.parent = eb
        twists.append(dict(bone=hb.name, target=s + 'Hand', base=name))
    eb = arm.data.edit_bones['neck']
    hb = arm.data.edit_bones.new('support_neck')
    hb.head = eb.head
    hb.tail = eb.tail
    hb.roll = eb.roll
    hb.parent = eb.parent
    cfg = corrective_profile(details['neck'].get('radius_m') or eb.length * 0.25, eb.parent.length, eb.length)
    configs.append(dict(bone=hb.name, target='neck', joint='neck', radius=details['neck'].get('radius_m') or eb.length * 0.25, **cfg))
    bpy.ops.object.mode_set(mode='OBJECT')
    for cfg in configs:
        pb = arm.pose.bones[cfg['bone']]
        pb.rotation_mode = 'QUATERNION'
        pb.bone.inherit_scale = 'NONE'
        c = pb.constraints.new('COPY_ROTATION')
        c.target = arm
        c.subtarget = cfg['target']
        c.target_space = 'LOCAL'
        c.owner_space = 'LOCAL'
        c.influence = 0.5
        for index in [0, 2]:
            driver = pb.driver_add('scale', index).driver
            driver.type = 'SCRIPTED'
            for name, axis in [('x', 'ROT_X'), ('z', 'ROT_Z')]:
                variable = driver.variables.new()
                variable.name = name
                variable.type = 'TRANSFORMS'
                t = variable.targets[0]
                t.id = arm
                t.bone_target = cfg['target']
                t.transform_type = axis
                t.transform_space = 'LOCAL_SPACE'
            start = cfg['start_angle']
            gain = cfg['max_radial_scale'] - 1
            driver.expression = f'1+{gain:.9g}*min(1,max(0,(sqrt(x*x+z*z)-{start:.9g})/(1.570796327-{start:.9g})))'
    for cfg in twists:
        pb = arm.pose.bones[cfg['bone']]
        c = pb.constraints.new('COPY_ROTATION')
        c.target = arm
        c.subtarget = cfg['target']
        c.target_space = 'LOCAL'
        c.owner_space = 'LOCAL'
        c.use_x = False
        c.use_y = True
        c.use_z = False
        c.influence = 0.5
    return (configs, twists)

def apply_weights(ob, co, edges, bones, arm, h, part, supports, twists, arm_regions=None, region_report=None):
    w, report = diffuse(co, edges, bones, h, part=part, arm_regions=arm_regions)
    if part not in ('hair', 'rigid_head'):
        w, report['shoe_binding'] = bind_shoes(co, w, bones, h)
    if region_report:
        report['arm_region_mask'] = region_report
    names = [b['name'] for b in bones]
    extra = []
    for cfg in supports:
        target = names.index(cfg['target'])
        parent = next((b['parent'] for b in bones if b['name'] == cfg['target']))
        j = names.index(parent)
        centre = np.array(arm.data.bones[cfg['target']].head_local)
        radius = cfg['support']
        blend = 0.6 * np.maximum(0, 1 - np.linalg.norm(co - centre, axis=1) / radius) ** 2
        if cfg['target'].startswith('left'):
            blend[co[:, 0] < -0.008 * h] = 0
        if cfg['target'].startswith('right'):
            blend[co[:, 0] > 0.008 * h] = 0
        influence = (w[:, target] + w[:, j]) * blend
        w[:, target] *= 1 - blend
        w[:, j] *= 1 - blend
        extra.append((cfg['bone'], influence))
    for cfg in twists:
        j = names.index(cfg['base'])
        b = arm.data.bones[cfg['base']]
        a = np.array(b.head_local)
        d = np.array(b.tail_local) - a
        u = np.clip((co - a) @ d / (d @ d), 0, 1)
        fraction = u * u * (3 - 2 * u)
        extra.append((cfg['bone'], w[:, j] * fraction))
        w[:, j] *= 1 - fraction
    full = np.column_stack([w] + [x[1] for x in extra])
    names += [x[0] for x in extra]
    order = np.argsort(-full, axis=1, kind='stable')
    keep = np.zeros_like(full, bool)
    np.put_along_axis(keep, order[:, :8], True, axis=1)
    full *= keep
    full[full < 1e-05] = 0
    full /= full.sum(axis=1, keepdims=True)
    ob.vertex_groups.clear()
    for j, name in enumerate(names):
        group = ob.vertex_groups.new(name=name)
        for idx in np.flatnonzero(full[:, j] > 0):
            group.add([int(idx)], float(full[idx, j]), 'REPLACE')
    mod = ob.modifiers.new('CharForgeDeform', 'ARMATURE')
    mod.object = arm
    mod.use_deform_preserve_volume = True
    ob.parent = arm
    ob.matrix_parent_inverse = Matrix.Identity(4)
    opposite = 0.0
    for j, name in enumerate(names):
        semantic = name.removeprefix('support_').removeprefix('twist_')
        if semantic.startswith('left'):
            opposite += float(full[co[:, 0] < -0.008 * h, j].sum())
        if semantic.startswith('right'):
            opposite += float(full[co[:, 0] > 0.008 * h, j].sum())
    report.update(max_influences=int((full > 0).sum(axis=1).max()), normalization_error=float(abs(full.sum(axis=1) - 1).max()), vertex_count=len(co), opposite_side_weight=opposite)
    return report

def run(record, out):
    source = validate(record)
    out = safe_output(out)
    if source == out / 'rig.blend':
        raise ValueError('Input and output must differ')
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 40 * 1024 ** 3:
        raise RuntimeError('BLOCKED disk')
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise RuntimeError('Run Blender with -Y')
    if source.suffix.lower() == '.blend':
        bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    elif source.suffix.lower() in ['.glb', '.gltf']:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source))
    elif source.suffix.lower() == '.fbx':
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(source), use_image_search=False)
    else:
        raise ValueError('Expected .blend/.glb/.gltf/.fbx data')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and (not o.hide_render) and o.visible_get()]
    if record.get('mesh_objects'):
        meshes = [bpy.data.objects[n] for n in record['mesh_objects']]
    if not meshes:
        raise ValueError('No visible mesh')
    if any((m.type == 'ARMATURE' for o in meshes for m in o.modifiers)):
        raise ValueError('Input still rigged; provide a mesh-only S3 artifact')
    if any((len(o.vertex_groups) for o in meshes)):
        raise ValueError('Input still has weights; strip before S4')
    raw = {o.name: arrays(o) for o in meshes}
    allco = np.concatenate([r[0] for r in raw.values()])
    basis = frame(record['orientation']['up'], record['orientation']['forward'])
    canonical = allco @ basis
    lo = canonical.min(axis=0)
    hi = canonical.max(axis=0)
    scale = record['height_cm'] / 100 / (hi[2] - lo[2])
    origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    h = record['height_cm'] / 100
    for o in meshes:
        co, tri, edges = raw[o.name]
        co = (co @ basis - origin) * scale
        if o.data.shape_keys:
            o.shape_key_clear()
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        o.data.vertices.foreach_set('co', co.ravel())
        o.data.update()
        raw[o.name] = (co, tri, edges)
    names = record.get('body_objects')
    if names:
        if any((n not in raw for n in names)):
            raise ValueError('Missing declared body object')
    else:
        names = [max(raw, key=lambda n: np.ptp(raw[n][0][:, 2]) * np.sqrt(len(raw[n][0])))]
    co = []
    tri = []
    offset = 0
    for name in names:
        c, t, _ = raw[name]
        co.append(c)
        tri.append(t + offset)
        offset += len(c)
    co = np.concatenate(co)
    tri = np.concatenate(tri)
    if record.get('landmark_space', 'source') not in ('source', 'canonical'):
        raise ValueError('Unknown landmark_space')
    observations = {k: vec(v) if record.get('landmark_space') == 'canonical' else (vec(v) @ basis - origin) * scale for k, v in record.get('landmarks', {}).items()}
    for k, v in record.get('silhouette_landmarks', {}).items():
        observations.setdefault(k, vec(v) * h)
    points, details, warnings = detect(co, tri, h, observations, record.get('silhouette_evidence'))
    for k, confidence in record.get('landmark_confidence', {}).items():
        if k not in observations or not isinstance(confidence, (int, float)) or (not 0 <= confidence <= 1):
            raise ValueError('Invalid landmark confidence')
        details[k]['confidence'] = confidence
        if confidence < 0.8:
            warnings.append(k + ': supplied observation confidence below .8')
    bones = skeleton(points, details)
    write(out / 'landmarks.json', dict(schema_version=1, stage='S3', height_cm=record['height_cm'], character_id=record['character_id'], coordinate_system='metres; X-left Y-back Z-up', points={k: v.tolist() for k, v in points.items()}, details=details, body_objects=names, source_transform=dict(basis=basis.tolist(), origin=origin.tolist(), scale=scale), warnings=warnings))
    for o in bpy.context.scene.objects:
        o.select_set(False)
    arm = create_armature(bones, points, details)
    supports, twists = helpers(arm, bones, points, details)
    weight_reports = {}
    for o in meshes:
        c, t, e = raw[o.name]
        part = record.get('parts', {}).get(o.name, 'body')
        regions, region_report = arm_region_masks(c, t, h, points) if part not in ('hair', 'rigid_head') else (None, None)
        weight_reports[o.name] = apply_weights(o, c, e, bones, arm, h, part, supports, twists, regions, region_report)
    mapping = {b['name']: b['name'] for b in bones}
    arm['charforge_humanoid'] = json.dumps(mapping)
    arm['charforge_status'] = 'candidate'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'rig.blend'))
    manifest = dict(schema_version=1, stage='S4', character_id=record['character_id'], status='review_required', height_cm=record['height_cm'], input_sha256=sha(source), output_sha256=sha(out / 'rig.blend'), artifact='rig.blend', landmarks='landmarks.json', humanoid=mapping, bones=bones, correctives=supports, twist_helpers=twists, weights=weight_reports, preserve_volume=True, implementation_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}, runtime=dict(blender=bpy.app.version_string, numpy=np.__version__), gates=dict(landmarks='BLOCKED' if warnings else 'REVIEW_REQUIRED', motion_suite='BLOCKED', visual_review='PENDING'), unvalidated=['S8 motion suite', 'finger articulation', 'face', 'garment collisions', 'export helper baking'], warnings=warnings)
    write(out / 'rig.json', manifest)
    print('CHARFORGE_S4_CANDIDATE', record['character_id'], len(bones), len(supports), flush=True)
    return manifest
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(json.loads(Path(args.measurements).read_text(encoding='utf-8-sig')), args.out)

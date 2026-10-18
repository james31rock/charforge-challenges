import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, json, sys, hashlib
from pathlib import Path
import bpy, numpy as np
from mathutils import Matrix, Vector

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--height', type=float, default=1.6)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = a.height
    bpy.ops.wm.read_factory_settings(use_empty=True)
    data = bpy.data.armatures.new('C7Original')
    arm = bpy.data.objects.new('C7Original', data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    mp = {}
    fingers = {}
    rest = {}

    def bone(name, head, tail, parent=None):
        b = data.edit_bones.new(name)
        b.head = Vector(head) * h
        b.tail = Vector(tail) * h
        if parent:
            b.parent = data.edit_bones[parent]
        return name
    mp['hips'] = bone('hips', (0, 0, 0.5), (0, 0, 0.6))
    mp['chest'] = bone('chest', (0, 0, 0.7), (0, 0, 0.8))
    mp['head'] = bone('head', (0, 0, 0.9), (0, 0, 1.0))
    for s, sign in [('L', 1), ('R', -1)]:
        x = 0.22 * sign
        for k, head, tail in [('hip', (0.07 * sign, 0, 0.5), (0.07 * sign, 0, 0.3)), ('knee', (0.07 * sign, 0, 0.3), (0.07 * sign, 0, 0.05)), ('ankle', (0.07 * sign, 0, 0.05), (0.07 * sign, 0.03, 0.02)), ('toe', (0.07 * sign, 0.03, 0.02), (0.07 * sign, 0.08, 0.02)), ('shoulder', (x, -0.3, 0.5), (x, -0.15, 0.5)), ('elbow', (x, -0.15, 0.5), (x, 0, 0.5))]:
            mp[s + k] = bone(s + k, head, tail)
        mp[s + 'wrist'] = bone(s + 'wrist', (x, 0, 0.5), (x, 0.06, 0.5), s + 'elbow')
        fingers[s] = []
        for digit, offset in enumerate((0.015, 0.005, -0.005, -0.015)):
            chain = []
            for j in range(3):
                name = f'{s}digit{digit}_{j}'
                parent = chain[-1] if chain else s + 'wrist'
                bone(name, (x + offset * sign, 0.065 + j * 0.02, 0.5), (x + offset * sign, 0.085 + j * 0.02, 0.5), parent)
                chain.append(name)
            fingers[s].append(chain)
    bpy.ops.object.mode_set(mode='OBJECT')

    def skin_box(name, centre, size, group, sub=6):
        c = np.array(centre) * h
        size = np.array(size) * h
        vertices = []
        faces = []
        for z in (-0.5, 0.5):
            for y in np.linspace(-0.5, 0.5, sub + 1):
                for x in np.linspace(-0.5, 0.5, sub + 1):
                    vertices.append((c + np.array([x, y, z]) * size).tolist())
        n = sub + 1
        layer = n * n
        for k in (0, 1):
            for j in range(sub):
                for i in range(sub):
                    q = k * layer + j * n + i
                    faces.append([q, q + 1, q + n + 1, q + n] if k else [q, q + n, q + n + 1, q + 1])
        perimeter = list(range(n)) + [j * n + sub for j in range(1, n)] + [sub * n + i for i in range(sub - 1, -1, -1)] + [j * n for j in range(sub - 1, 0, -1)]
        for i, j in zip(perimeter, perimeter[1:] + perimeter[:1]):
            faces.append([i, j, j + layer, i + layer])
        me = bpy.data.meshes.new(name)
        me.from_pydata(vertices, [], faces)
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(ob)
        vg = ob.vertex_groups.new(name=group)
        vg.add(list(range(len(vertices))), 1.0, 'REPLACE')
        mod = ob.modifiers.new('Skin', 'ARMATURE')
        mod.object = arm
    skin_box('height marker', (0, 0.5, 0.5), (0.01, 0.01, 1.0), 'hips', 1)
    for s, sign in [('L', 1), ('R', -1)]:
        x = 0.22 * sign
        skin_box('palm' + s, (x, 0.0325, 0.5), (0.04, 0.065, 0.012), s + 'wrist')
        skin_box('forearm' + s, (x, -0.075, 0.5), (0.027, 0.15, 0.027), s + 'elbow', 1)
        for d, offset in enumerate((0.015, 0.005, -0.005, -0.015)):
            for j in range(3):
                skin_box(f'finger{s}{d}{j}', (x + offset * sign, 0.075 + j * 0.02, 0.5), (0.008, 0.02, 0.012), fingers[s][d][j], 1)
    sc = bpy.context.scene
    sc.render.fps = 24
    sc.frame_start = 1
    sc.frame_end = 24
    for s, sign in [('L', 1), ('R', -1)]:
        wrist = Vector((0.22 * sign * h, 0, 0.006 * h))
        rot = Matrix.Rotation(np.radians(-80), 4, 'X')
        direction = rot.to_3x3() @ Vector((0, 1, 0))
        elbow = wrist - direction * 0.15 * h
        arm.pose.bones[s + 'elbow'].matrix = Matrix.Translation(elbow) @ rot
        bpy.context.view_layer.update()
        arm.pose.bones[s + 'wrist'].matrix = Matrix.Translation(wrist)
        bpy.context.view_layer.update()
        for key in (s + 'elbow', s + 'wrist'):
            pb = arm.pose.bones[key]
            pb.rotation_mode = 'QUATERNION'
            for frame in (1, 24):
                for prop in ('location', 'rotation_quaternion', 'scale'):
                    pb.keyframe_insert(prop, frame=frame)
    arm.animation_data.action.name = 'C7_planted'
    (out / 'mapping.json').write_text(json.dumps(dict(kind='custom', points=mp, fingers=fingers), indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'planted.blend'))
    names = [b.name for b in arm.pose.bones]
    basis = []
    world = []
    for frame in range(1, 25):
        sc.frame_set(frame)
        bpy.context.view_layer.update()
        basis.append([np.array(arm.pose.bones[n].matrix_basis) for n in names])
        world.append(np.array(arm.matrix_world))
    np.savez_compressed(out / 'poses.npz', names=names, basis=basis, world=world, times=np.arange(24) / 24)
    plan = dict(fps=24, hands={s: dict(joint=s + 'wrist', intervals=[[0, 24]], method='Original synthetic intended stationary palm support') for s in 'LR'}, feet={s: dict(intervals=[]) for s in 'LR'})
    (out / 'source-intent.json').write_text(json.dumps(plan, indent=2))
    (out / 'contact.json').write_text(json.dumps(dict(source_plan=plan, cache_sha256=hashlib.sha256((out / 'poses.npz').read_bytes()).hexdigest(), source_sha256=hashlib.sha256((out / 'source-intent.json').read_bytes()).hexdigest()), indent=2))
    print('C7_FIXTURE_CREATED', flush=True)
if __name__ == '__main__':
    main()

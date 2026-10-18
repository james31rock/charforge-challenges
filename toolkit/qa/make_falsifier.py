import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sys, argparse, math, json
from pathlib import Path
import bpy
from mathutils import Vector

def build(out, height):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = 24
    sc.frame_start = 1
    sc.frame_end = 24
    data = bpy.data.armatures.new('QAFixture')
    arm = bpy.data.objects.new('QAFixture', data)
    sc.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    coords = {'hips': ((0, 0, 0.5), (0, 0, 0.58), None), 'spine': ((0, 0, 0.58), (0, 0, 0.66), 'hips'), 'chest': ((0, 0, 0.66), (0, 0, 0.74), 'spine'), 'upperChest': ((0, 0, 0.74), (0, 0, 0.81), 'chest'), 'neck': ((0, 0, 0.81), (0, 0, 0.85), 'upperChest'), 'head': ((0, 0, 0.85), (0, 0, 0.98), 'neck')}
    for side, sign in [('left', 1), ('right', -1)]:
        for name, head, tail, parent in [('UpperArm', (0.16, 0, 0.76), (0.25, 0, 0.63), 'upperChest'), ('LowerArm', (0.25, 0, 0.63), (0.31, 0, 0.5), side + 'UpperArm'), ('Hand', (0.31, 0, 0.5), (0.35, 0, 0.43), side + 'LowerArm'), ('UpperLeg', (0.08, 0, 0.5), (0.08, 0, 0.28), 'hips'), ('LowerLeg', (0.08, 0, 0.28), (0.08, 0, 0.06), side + 'UpperLeg'), ('Foot', (0.08, 0, 0.06), (0.08, -0.08, 0.03), side + 'LowerLeg'), ('Toes', (0.08, -0.08, 0.03), (0.08, -0.13, 0.03), side + 'Foot')]:
            coords[side + name] = (tuple([head[0] * sign, *head[1:]]), tuple([tail[0] * sign, *tail[1:]]), parent)
    for name, (a, b, parent) in coords.items():
        bone = data.edit_bones.new(name)
        bone.head = Vector(a) * height
        bone.tail = Vector(b) * height
        if parent:
            bone.parent = data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')

    def box(name, centre, size, bone):
        bpy.ops.mesh.primitive_cube_add(size=1, location=Vector(centre) * height)
        o = bpy.context.object
        o.name = name
        o.scale = Vector(size) * height
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        group = o.vertex_groups.new(name=bone)
        group.add(list(range(len(o.data.vertices))), 1, 'REPLACE')
        mod = o.modifiers.new('Skin', 'ARMATURE')
        mod.object = arm
    box('Body', (0, 0, 0.65), (0.25, 0.17, 0.3), 'hips')
    box('Garment', (0, 0, 0.62), (0.29, 0.21, 0.31), 'hips')
    box('Head', (0, 0, 0.9), (0.22, 0.18, 0.18), 'head')
    for side, s in [('left', 1), ('right', -1)]:
        box(side + 'Hand', (s * 0.335, 0, 0.46), (0.05, 0.045, 0.09), side + 'Hand')
        box(side + 'Shoe', (s * 0.08, -0.035, 0.025), (0.075, 0.2, 0.05), side + 'Foot')
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
    for frame in range(1, 25):
        sc.frame_set(frame)
        for side, s in [('left', 1), ('right', -1)]:
            hand = arm.pose.bones[side + 'Hand']
            hand.location = (0, 0, 0)
            if 9 <= frame <= 16:
                world_delta = Vector((-s * 0.28 * height, 0, 0.1 * height))
                hand.location = hand.bone.matrix_local.to_3x3().inverted() @ world_delta
            hand.keyframe_insert('location', frame=frame)
            foot = arm.pose.bones[side + 'Foot']
            foot.rotation_euler = (math.radians(35) if frame >= 17 else 0, 0, 0)
            foot.keyframe_insert('rotation_euler', frame=frame)
    arm.animation_data.action.name = 'QA_seeded'
    for layer in arm.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = 'CONSTANT'
    sc.frame_set(1)
    bpy.context.preferences.filepaths.save_version = 0
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'seeded.blend'))
    (out / 'truth.json').write_text(json.dumps(dict(seed=61017, height_m=height, clean_frames=[1, 8], hand_penetration_frames=[9, 16], tilted_foot_frames=[17, 24], purpose='End-to-end extraction falsifier; original boxes, not a professional positive control'), indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--height', type=float, default=1.6)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    build(Path(a.out), a.height)

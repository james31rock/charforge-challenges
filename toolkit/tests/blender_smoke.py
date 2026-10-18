"""Native rig adapter smoke test on the public mannequin; output is disposable."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import bpy
import numpy as np
from mathutils import Vector
from toolkit.rig.blender_rig import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    with np.load(root/'garment-separation/assets/oracle.npz', allow_pickle=False) as data:
        vertices, faces = data['body_vertices'], data['body_faces']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new('PublicMannequin')
    mesh.from_pydata(vertices.tolist(), [], faces.tolist()); mesh.update()
    ob = bpy.data.objects.new('PublicMannequin', mesh); bpy.context.collection.objects.link(ob)
    source = out/'unrigged.blend'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(source))
    measurements = dict(schema_version=1, character_id='public-mannequin', height_cm=float(np.ptp(vertices[:,2]))*100,
                        mesh_path=str(source), orientation=dict(up=[0,0,1], forward=[0,-1,0]))
    report = run(measurements, out/'candidate')
    assert report['status'] == 'review_required'
    assert len(report['bones']) >= 20 and len(report['correctives']) >= 8
    assert all(row['normalization_error'] < 1e-10 and row['max_influences'] <= 8 for row in report['weights'].values())
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    for name, angle in [('leftLowerLeg', -.55), ('leftLowerArm', .65)]:
        bone = arm.pose.bones[name]; bone.rotation_mode = 'XYZ'; bone.rotation_euler[0] = angle
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
        ev = ob.evaluated_get(dg); result = ev.to_mesh()
        assert all(np.isfinite(v.co).all() for v in result.vertices)
        ev.to_mesh_clear()
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 8
    sc.render.threads_mode = 'FIXED'; sc.render.threads = 2
    material = bpy.data.materials.new('ReviewSurface'); material.use_nodes = True
    material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.22,.55,.7,1)
    for obj in [o for o in sc.objects if o.type == 'MESH']:
        obj.data.materials.clear(); obj.data.materials.append(material)
    sc.world = bpy.data.worlds.new('ReviewWorld'); sc.world.use_nodes = True
    sc.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (.5,.5,.5,1)
    light_data = bpy.data.lights.new('ReviewLight', 'AREA'); light_data.energy = 450; light_data.size = 4
    light = bpy.data.objects.new('ReviewLight', light_data); sc.collection.objects.link(light)
    light.location = (2,-3,4); light.rotation_euler = (Vector((0,0,.8))-light.location).to_track_quat('-Z','Y').to_euler()
    sc.render.resolution_x = 512; sc.render.resolution_y = 640; sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    camera_data = bpy.data.cameras.new('ReviewCamera'); camera = bpy.data.objects.new('ReviewCamera',camera_data)
    sc.collection.objects.link(camera); sc.camera = camera; camera_data.type = 'ORTHO'; camera_data.ortho_scale = 2.2
    for label, location in [('front',(0,-4,.9)), ('oblique',(3,-4,1.3)), ('side',(4,0,.9))]:
        camera.location = location; camera.rotation_euler = (Vector((0,0,.85))-camera.location).to_track_quat('-Z','Y').to_euler()
        sc.render.filepath = str(out/(label+'.png')); bpy.ops.render.render(write_still=True)
    (out/'smoke.json').write_text(json.dumps(dict(status='PASS', bones=len(report['bones']), correctives=len(report['correctives']),
        scope='Native construction and finite deformation only; visual quality requires review.'), indent=2))


if __name__ == '__main__':
    main()

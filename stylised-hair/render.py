"""Trusted CPU witness renderer for the exact scored data. No blend input."""
import sys,argparse
from pathlib import Path
sys.dont_write_bytecode=True
import bpy,numpy as np
from mathutils import Vector
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from common import load_npz
p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('--out',required=True);p.add_argument('--control',action='store_true');p.add_argument('--rear',action='store_true');p.add_argument('--all',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
assets=HERE/('tests/control/assets' if a.control else 'assets');fixture=load_npz(assets/'fixture.npz');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
mesh=bpy.data.meshes.new('benchmark');mesh.from_pydata(fixture['rest'].tolist(),[],fixture['triangles'].tolist());mesh.update()
obj=bpy.data.objects.new('benchmark',mesh);bpy.context.collection.objects.link(obj)
attr=mesh.color_attributes.new(name='colour',type='FLOAT_COLOR',domain='POINT');rgba=np.c_[fixture['colours'],np.ones(len(fixture['rest']))];attr.data.foreach_set('color',rgba.ravel())
mat=bpy.data.materials.new('surface');mat.use_nodes=True;nodes=mat.node_tree.nodes;bs=nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.7
vc=nodes.new('ShaderNodeVertexColor');vc.layer_name='colour';mat.node_tree.links.new(vc.outputs['Color'],bs.inputs['Base Color']);mesh.materials.append(mat)
control_body=None
if a.control:
    initial=load_npz(assets/'walk.npz')['target_0'][0]
    control_body=bpy.data.meshes.new('control body');control_body.from_pydata(initial.tolist(),[],fixture['target_triangles_0'].tolist());control_body.update()
    body_object=bpy.data.objects.new('control body',control_body);bpy.context.collection.objects.link(body_object)
sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=8;sc.cycles.use_denoising=False
sc.render.resolution_x=600;sc.render.resolution_y=600;sc.render.resolution_percentage=100;sc.world.color=(.7,.7,.7);sc.view_settings.view_transform='Standard'
bpy.ops.object.camera_add();sc.camera=bpy.context.object;sc.camera.data.type='ORTHO';h=float(fixture['height_m']);sc.camera.data.ortho_scale=h*.65
bpy.ops.object.light_add(type='AREA',location=(1,-2,3));light=bpy.context.object;light.data.energy=250;light.data.shape='DISK';light.data.size=3
for clip,frames in [('walk',[7,37,74]),('jump',[24,37,65])]:
    drive=load_npz(assets/(clip+'.npz'));hair=load_npz(Path(a.submission)/(clip+'.npz'))['hair']
    if a.all:frames=range(1,len(hair)+1)
    for frame in frames:
        f=frame-1;v=drive['vertices'][f].copy();v[fixture['hair_ids']]=hair[f];mesh.vertices.foreach_set('co',v.ravel());mesh.update()
        if control_body:
            control_body.vertices.foreach_set('co',drive['target_0'][f].ravel());control_body.update()
        target=Vector(drive['head'][f,:3,3])+Vector((0,0,h*.04));sc.camera.location=target+Vector((h*1.4,h*(2.0 if a.rear else -2.0),h*.4));sc.camera.rotation_euler=(target-sc.camera.location).to_track_quat('-Z','Y').to_euler()
        sc.render.filepath=str(out/(f'{clip}-{frame:03d}'+('-rear' if a.rear else '')+'.png'))
        sc.render.use_stamp=False;bpy.ops.render.render(write_still=True)

"""Trusted CPU witness renderer for fixed native rig/pose data."""
import sys,math
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector
root=Path(__file__).resolve().parent
folder,pose,out,frame,side=sys.argv[sys.argv.index('--')+1:];folder=Path(folder);out=Path(out);frame=int(frame)
bpy.ops.wm.open_mainfile(filepath=str(folder/'rig.blend'),use_scripts=False)
arm=next(o for o in bpy.data.objects if o.type=='ARMATURE')
if arm.animation_data:
    arm.animation_data.action=None
    for t in arm.animation_data.nla_tracks:t.mute=True
for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
with np.load(pose,allow_pickle=False) as z:
    if frame:
        for i,name in enumerate(z['names']):arm.pose.bones[str(name)].matrix_basis=Matrix(z['basis'][frame-1,i].tolist())
    arm.matrix_world=Matrix(z['world'][max(0,frame-1)].tolist())
bpy.context.view_layer.update()
name=side+'wrist' if side+'wrist' in arm.pose.bones else f'J_Bip_{side}_Hand'
target=arm.matrix_world@arm.pose.bones[name].head
def mat(name,color):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=.8
    return m
skin=mat('Witness neutral',(.60,.72,.82))
for ob in list(bpy.context.scene.objects):
    if ob.type=='MESH':
        ob.data.materials.clear();ob.data.materials.append(skin)
    elif ob.type in ('CAMERA','LIGHT'):bpy.data.objects.remove(ob,do_unlink=True)
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.0001));bpy.context.object.data.materials.append(mat('Floor',(.30,.32,.36)))
bpy.ops.object.camera_add(location=target+Vector((.34,-.17,.095)))
cam=bpy.context.object;cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=.34
s=bpy.context.scene;s.camera=cam
bpy.ops.object.light_add(type='AREA',location=target+Vector((.7,-.5,1)));bpy.context.object.data.energy=90;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=.7
if not s.world:s.world=bpy.data.worlds.new('Witness world')
s.world.color=(.4,.4,.4);s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=12
s.render.resolution_x=512;s.render.resolution_y=384;s.render.resolution_percentage=100;s.render.filepath=str(out)
bpy.ops.render.render(write_still=True)


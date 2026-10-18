"""Original procedural test mannequin; run in Blender 5.x, CPU only.

Known hidden geometry is an evaluation oracle, not inferred anatomy.
This test fixture is not a production character creation recipe.
"""
import sys,json
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)

def ellipsoid(name,center,scale):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40,ring_count=24,location=center)
    ob=bpy.context.object;ob.name=name;ob.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return ob
def join(objects,name,voxel=None):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:o.select_set(True)
    bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join();ob=objects[0];ob.name=name
    if voxel:
        m=ob.modifiers.new('Union','REMESH');m.mode='VOXEL';m.voxel_size=voxel;m.use_smooth_shade=True
        bpy.ops.object.modifier_apply(modifier=m.name)
    return ob
body=join([ellipsoid('torso',(0,0,.96),(.22,.13,.36)),ellipsoid('neck',(0,0,1.31),(.08,.075,.12)),
           ellipsoid('head',(0,0,1.49),(.16,.14,.21))]+
          [ellipsoid('leg',(s*.105,0,.40),(.083,.09,.4)) for s in (-1,1)]+
          [ellipsoid('arm',(s*.31,0,1.05),(.16,.065,.075)) for s in (-1,1)],'body',.006)
garment=ellipsoid('garment',(0,0,.97),(.255,.165,.33))
# A shell with real interior: inner, inverted surface plus outer surface.
inner=ellipsoid('inside',(0,0,.97),(.238,.148,.313))
for poly in inner.data.polygons:poly.flip()
garment=join([garment,inner],'garment')
hair=ellipsoid('hair',(0,.025,1.585),(.177,.155,.14))

def arrays(ob):
    me=ob.data;me.calc_loop_triangles()
    return np.array([ob.matrix_world@v.co for v in me.vertices]),np.array([t.vertices[:] for t in me.loop_triangles],dtype=np.int32)
parts={}
for label,ob in [('body',body),('garment',garment),('hair',hair)]:
    v,f=arrays(ob);parts[label+'_vertices']=v;parts[label+'_faces']=f
np.savez_compressed(ROOT/'tests/good.npz',**parts)
# Voxel union destroys semantic boundaries and covered body, like a fused output.
outer=ellipsoid('fused garment',(0,0,.97),(.255,.165,.33))
fused=join([body,outer,hair],'fused',.006)
v,f=arrays(fused);np.savez_compressed(ROOT/'assets/fused.npz',vertices=v,faces=f)
np.savez_compressed(ROOT/'baseline/parts.npz',body_vertices=v,body_faces=f)
bpy.ops.wm.obj_export(filepath=str(ROOT/'assets/fused.obj'),export_selected_objects=True,forward_axis='Y',up_axis='Z',export_materials=False)
np.savez_compressed(ROOT/'assets/oracle.npz',**parts)
(ROOT/'assets/fixture.json').write_text(json.dumps({'version':1,'height_m':float(np.ptp(v[:,2])),'units':'metres-z-up','kind':'self-made clothed test mannequin','source':'voxel union of original geometric parts','voxel_pitch_m':.006,'oracle':'known construction geometry; not reconstructed biological anatomy'},indent=2))
print('GARMENT_FIXTURE_BUILT')


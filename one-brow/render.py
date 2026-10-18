"""Render fixed C8 views from the supplied head with scripts disabled."""
import sys,json,math
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parent
source,out,assets=map(Path,sys.argv[sys.argv.index('--')+1:]);out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(source),use_scripts=False)
if bpy.app.version[0]!=5:raise RuntimeError('Blender 5.x required')
b=json.loads((assets/'reference/binding.json').read_text());s=bpy.context.scene
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=1
s.render.resolution_x=s.render.resolution_y=512;s.render.resolution_percentage=100;s.render.use_persistent_data=True
ctl=bpy.data.objects['CharForgeFaceControls'];ctl.animation_data_clear();ctl['mouth/animated']=0;ctl['mouth/viseme']=0
cam=s.camera;center=Vector((b['camera']['center_x'],0,b['camera']['center_z']))
meshes=[o for o in s.objects if o.type=='MESH'];ref=dict(np.load(assets/'geometry.npz',allow_pickle=False))
if len(meshes)!=1:raise ValueError('Keep the supplied head mesh and topology')
ob=meshes[0];v=np.array([ob.matrix_world@v.co for v in ob.data.vertices]);ob.data.calc_loop_triangles();tri=np.array([t.vertices[:] for t in ob.data.loop_triangles])
if v.shape!=ref['vertices'].shape or not np.allclose(v,ref['vertices'],atol=1e-6) or not np.array_equal(tri,ref['triangles']):raise ValueError('Head geometry changed')
if ob.hide_render:raise ValueError('Head hidden')
records=[]
def render(mode,index,yaw):
    rad=math.radians(yaw);cam.location=center+Vector((math.sin(rad)*3,-math.cos(rad)*3,0));cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler()
    ctl.update_tag();s.frame_set(1);bpy.context.view_layer.update()
    name=f'{mode}-{index}-{yaw}.png';s.render.filepath=str(out/name);bpy.ops.render.render(write_still=True)
    records.append(dict(mode=mode,index=index,yaw=yaw,path=name))
for mode,count in [('expressions',6),('blink',3)]:
    for index in range(count):
        ctl['features/enabled']=1;ctl['expression/name']=index if mode=='expressions' else 0;ctl['brows/state']=ctl['expression/name'];ctl['eyes/state']=index if mode=='blink' else 0
        for yaw in (0,35):render(mode,index,yaw)
ctl['features/enabled']=0
for yaw in (0,35):render('composite-static',0,yaw)
for mat in {m for o in meshes for m in o.data.materials if m}:
    nt=mat.node_tree;original=[n for n in nt.nodes if n.type=='TEX_IMAGE' and not n.label.startswith('S5 ')]
    if len(original)!=1:raise ValueError('Ambiguous original static image inventory')
    emission=nt.nodes.new('ShaderNodeEmission');nt.links.new(original[0].outputs['Color'],emission.inputs['Color'])
    output=next(n for n in nt.nodes if n.type=='OUTPUT_MATERIAL');nt.links.new(emission.outputs[0],output.inputs['Surface'])
for yaw in (0,35):render('raw-static',0,yaw)
(out/'renders.json').write_text(json.dumps(records))


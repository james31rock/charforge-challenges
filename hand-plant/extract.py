"""Trusted Blender-side measurement of a fixed rig and data-only pose matrices."""
import json, sys, hashlib
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from skin import measured_anatomy,select_patches,measure_surfaces
from geometry import geometry
folder,source,out=map(Path,sys.argv[sys.argv.index('--')+1:])
if bpy.app.version[0]!=5:raise RuntimeError('Blender 5.x required')
bpy.ops.wm.open_mainfile(filepath=str(folder/'rig.blend'),use_scripts=False)
mp=json.loads((folder/'mapping.json').read_text());plan=json.loads((folder/'contact-plan.json').read_text())
arm=next(o for o in bpy.data.objects if o.type=='ARMATURE')
objects=[o for o in bpy.data.objects if o.type=='MESH' and not o.hide_render and any(m.type=='ARMATURE' and m.object==arm for m in o.modifiers)]
with np.load(source,allow_pickle=False) as z:cache={k:z[k] for k in ('names','basis','world','times')}
if arm.animation_data:
    arm.animation_data.action=None
    for t in arm.animation_data.nla_tracks:t.mute=True
for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
arm.matrix_world=Matrix(cache['world'][0].tolist())
arm.data.pose_position='POSE';bpy.context.view_layer.update()
rest=geometry(objects,True);height=float(np.ptp(np.concatenate([g['v'] for g in rest.values()])[:,2]))
neutral=measured_anatomy(arm,mp['points'],mp['fingers']);selection=select_patches(rest,arm,mp['points'],mp['fingers'],neutral)
def head(name):return np.array(arm.matrix_world@arm.pose.bones[name].head)
pairs=[(f'J_Bip_{s}_{a}',f'J_Bip_{s}_{b}') for s in 'LR' for a,b in [('UpperArm','LowerArm'),('LowerArm','Hand'),('UpperLeg','LowerLeg'),('LowerLeg','Foot')]]
pairs=[(a,b) for a,b in pairs if a in arm.pose.bones and b in arm.pose.bones]
lengths={(a,b):float(np.linalg.norm(head(a)-head(b))) for a,b in pairs};attachment=0.
digest=hashlib.sha256(source.read_bytes()).hexdigest()
support=dict(schema='qa-hand-support/1',bound=True,fps=24,frame_count=len(cache['basis']),hands=plan['hands'],source_sha256=plan['source_sha256'],cache_sha256=digest)
trace=dict(height_m=height,fps=24,source=dict(clip='cartwheel',cache_sha256=digest),hand_support_required=True,hand_support=support,rest_hands=neutral,frames=[])
selected_names={name for s in selection.values() for name,ids in s['all_hand'].items() if ids}
objects=[o for o in objects if o.name in selected_names]
for i in range(len(cache['basis'])):
    bpy.context.scene.frame_set(i+1)
    for j,name in enumerate(cache['names']):arm.pose.bones[str(name)].matrix_basis=Matrix(cache['basis'][i,j].tolist())
    arm.matrix_world=Matrix(cache['world'][i].tolist());bpy.context.view_layer.update()
    for a,b in pairs:attachment=max(attachment,abs(float(np.linalg.norm(head(a)-head(b)))-lengths[a,b])/height)
    anatomical=measured_anatomy(arm,mp['points'],mp['fingers'])
    hands=measure_surfaces(geometry(objects),selection,anatomical,1.,0.)
    trace['frames'].append(dict(frame=i+1,hands=hands))
trace['limb_attachment_error_height']=attachment
out.write_text(json.dumps(trace,allow_nan=False))


import bpy
import numpy as np

def geometry(objects, weights=False):
    result={};dg=bpy.context.evaluated_depsgraph_get()
    for ob in objects:
        ev=ob.evaluated_get(dg);me=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
        v=np.empty(len(me.vertices)*3);me.vertices.foreach_get('co',v);v=v.reshape(-1,3)
        mw=np.array(ev.matrix_world);v=v@mw[:3,:3].T+mw[:3,3]
        me.calc_loop_triangles();tri=np.array([x.vertices[:] for x in me.loop_triangles],dtype=int)
        labels=np.array([ob.data.materials[me.polygons[x.polygon_index].material_index].name
                         if me.polygons[x.polygon_index].material_index<len(ob.data.materials) and ob.data.materials[me.polygons[x.polygon_index].material_index] else '' for x in me.loop_triangles])
        row=dict(v=v,t=tri,labels=labels)
        if weights:
            arm=next(m.object for m in ob.modifiers if m.type=='ARMATURE' and m.object)
            groups={g.index:g.name for g in ob.vertex_groups if g.name in arm.data.bones and arm.data.bones[g.name].use_deform}
            row['dominant']=[];row['influences']=[]
            for vertex in me.vertices:
                influences=[g for g in vertex.groups if g.group in groups]
                row['dominant'].append(groups[max(influences,key=lambda g:g.weight).group] if influences else '')
                row['influences'].append([groups[g.group] for g in influences if g.weight>.01])
        result[ob.name]=row;ev.to_mesh_clear()
    return result

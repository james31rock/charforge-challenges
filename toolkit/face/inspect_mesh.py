import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, json, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--out', required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(Path(a.source).resolve()), use_scripts=False)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
verts = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
lo = [min((v[i] for v in verts)) for i in range(3)]
hi = [max((v[i] for v in verts)) for i in range(3)]
record = {'source': str(Path(a.source).resolve()), 'bounds': [lo, hi], 'meshes': []}
for o in meshes:
    mats = []
    for m in o.data.materials:
        imgs = []
        if m and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image:
                    name = f'{o.name}-{m.name}-{n.name}'.replace('/', '_').replace(':', '_') + '.png'
                    n.image.filepath_raw = str(out / name)
                    n.image.file_format = 'PNG'
                    n.image.save()
                    imgs.append({'node': n.name, 'image': name, 'size': list(n.image.size)})
        mats.append({'name': m.name if m else None, 'images': imgs})
    record['meshes'].append({'name': o.name, 'vertices': len(o.data.vertices), 'polygons': len(o.data.polygons), 'matrix': [list(row) for row in o.matrix_world], 'materials': mats, 'uvs': [u.name for u in o.data.uv_layers]})
    o.data.calc_loop_triangles()
    tris = o.data.loop_triangles
    vertices = np.array([list(o.matrix_world @ v.co) for v in o.data.vertices], dtype=np.float32)
    normals = np.array([list(o.matrix_world.to_3x3() @ v.normal) for v in o.data.vertices], dtype=np.float32)
    triangles = np.array([list(t.vertices) for t in tris], dtype=np.int32)
    uv = np.array([[list(o.data.uv_layers.active.data[i].uv) for i in t.loops] for t in tris], dtype=np.float32)
    np.savez_compressed(out / (o.name + '.npz'), vertices=vertices, normals=normals, triangles=triangles, uv=uv)
(out / 'mesh.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))

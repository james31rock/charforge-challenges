import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector
import numpy as np
MOUTHS = ['closed', 'mbp', 'small-a', 'wide-a', 'ee', 'o', 'fv', 'lth']
EXPS = ['neutral', 'happy', 'worried', 'surprised', 'sad', 'determined']
EYES = ['open', 'half', 'closed']

def driver(socket, ctl, expr, variables):
    d = socket.driver_add('default_value').driver
    d.type = 'SCRIPTED'
    d.expression = expr
    for name, prop in variables.items():
        v = d.variables.new()
        v.name = name
        v.type = 'SINGLE_PROP'
        v.targets[0].id = ctl
        v.targets[0].data_path = '["' + prop + '"]'

def install(source, folder, track, out, view=0):
    folder = Path(folder)
    b = json.loads((folder / 'binding.json').read_text())
    t = json.loads(Path(track).read_text())
    bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()), use_scripts=False)
    sc = bpy.context.scene
    if not b.get('static_uv'):
        raise ValueError('C8: original static UV erasure is mandatory; run erase_static.py first')
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 4
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = 4
    sc.render.resolution_x = 512
    sc.render.resolution_y = 512
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.fps = t['fps']
    sc.frame_start = 1
    sc.frame_end = t['frame_count']
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = 0
    sc.view_settings.gamma = 1
    ctl = bpy.data.objects.new('CharForgeFaceControls', None)
    sc.collection.objects.link(ctl)
    ctl.empty_display_type = 'PLAIN_AXES'
    ctl.empty_display_size = 0.03
    controls = {'mouth/viseme': MOUTHS, 'mouth/sample': b['atlases']['mouth']['states'], 'mouth/animated': ['manual viseme', 'eased track', 'expression mouth'], 'features/enabled': ['all animated features off; inspect raw static separately', 'enabled'], 'eyes/state': EYES, 'expression/name': EXPS, 'brows/state': EXPS}
    for prop, values in controls.items():
        ctl[prop] = 0
        ctl.id_properties_ui(prop).update(min=0, max=len(values) - 1, description=' | '.join((f'{i}: {x}' for i, x in enumerate(values))))
    ctl['mouth/animated'] = int(t.get('schema_version') == 'charforge.face.track/2')
    ctl['features/enabled'] = 1
    meshes = [o for o in sc.objects if o.type == 'MESH']
    m = np.array(b['front_to_source'])
    w, h = b['canvas']
    cam = b['camera']
    summary = []
    override = None
    if b.get('visibility_override'):
        im = bpy.data.images.load(str((folder / b['visibility_override']).resolve()))
        pixels = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(pixels)
        override = pixels.reshape(h, w, 4)[::-1, :, 0]
    for obj in meshes:
        data = obj.data
        old_uvs = [x.name for x in data.uv_layers]
        uv = data.uv_layers.get('CharForgeFaceProjection') or data.uv_layers.new(name='CharForgeFaceProjection')
        attr = data.attributes.get('CharForgeFaceVisibility') or data.attributes.new('CharForgeFaceVisibility', 'FLOAT', 'POINT')
        positions = [obj.matrix_world @ v.co for v in data.vertices]
        normals = [obj.matrix_world.to_3x3() @ v.normal for v in data.vertices]
        coords = []
        vis = []
        for p, n in zip(positions, normals):
            fx = (p.x - cam['center_x']) / cam['span'] * w + w / 2
            fy = (cam['center_z'] - p.z) / cam['span'] * h + h / 2
            sx = m[0] @ [fx, fy, 1]
            sy = m[1] @ [fx, fy, 1]
            coords.append((sx / w, 1 - sy / h))
            weight = max(0, min(1, (-n.y - 0.08) / 0.18)) if p.y < 0 and 1 < sx < w - 1 and (1 < sy < h - 1) else 0
            if override is not None and p.y < 0 and (1 < sx < w - 1) and (1 < sy < h - 1):
                weight = max(weight, float(override[int(sy), int(sx)]))
            vis.append(weight)
        uv.data.foreach_set('uv', np.array([coords[loop.vertex_index] for loop in data.loops], dtype=np.float32).ravel())
        attr.data.foreach_set('value', vis)
        print('S5 projection fitted', obj.name, flush=True)
        data.uv_layers.active_index = next((i for i, x in enumerate(data.uv_layers) if x.name == old_uvs[0]))
        for oldmat in list(data.materials):
            if oldmat is None:
                continue
            mat = oldmat.copy()
            mat.name = oldmat.name + '_CharForgeFace'
            slot = next((i for i, x in enumerate(data.materials) if x == oldmat))
            data.materials[slot] = mat
            nt = mat.node_tree
            nodes = nt.nodes
            links = nt.links
            if b.get('static_uv'):
                original_images = [n for n in nodes if n.type == 'TEX_IMAGE']
                if len(original_images) != b['static_uv']['source_image_count']:
                    raise ValueError('C8: unhandled static texture, AO or line map')
                original_images[0].image = bpy.data.images.load(str((folder / b['static_uv']['path']).resolve()), check_existing=True)
                original_images[0].label = 'C8 cleaned native UV'
                if b['static_uv'].get('alpha_is_feature_visibility'):
                    original_images[0].image.alpha_mode = 'CHANNEL_PACKED'
            principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if not principled or not principled.inputs['Base Color'].is_linked:
                raise ValueError('Expected textured Principled material')
            color = principled.inputs['Base Color'].links[0].from_socket
            projected = nodes.new('ShaderNodeUVMap')
            projected.uv_map = uv.name
            projected.label = 'Rest-space face projection'
            visibility = nodes.new('ShaderNodeAttribute')
            visibility.attribute_name = attr.name
            for key in ('base', 'blush-left', 'blush-right', 'brow-left', 'brow-right', 'eye-left', 'eye-right', 'mouth'):
                if key == 'base' and b.get('static_uv') and (not b['static_uv']['base_overlay_enabled']):
                    continue
                spec = b['atlases'][key]
                names = spec['states']
                cols = spec['cols']
                rows = spec['rows']
                if key == 'mouth':
                    expr = '(' + ','.join((str(names.index('expression-' + x)) for x in EXPS)) + ',)[int(e)] if a==2 else s if a==1 else (' + ','.join((str(names.index('mouth-' + x)) for x in MOUTHS)) + ',)[int(m)]'
                    vars = {'m': 'mouth/viseme', 's': 'mouth/sample', 'a': 'mouth/animated', 'e': 'expression/name'}
                elif key.startswith('eye-'):
                    ex = '(' + ','.join((str(names.index('expression-' + x)) for x in EXPS)) + ',)[int(e)]'
                    closed = b.get('native_closed_expressions', {}).get(key, [])
                    condition = 'k==0' + ''.join((' or e==' + str(EXPS.index(x)) for x in closed))
                    expr = f"{ex} if ({condition}) else ({names.index('eyes-half')} if k==1 else {names.index('eyes-closed')})"
                    vars = {'e': 'expression/name', 'k': 'eyes/state'}
                else:
                    expr = '(' + ','.join((str(names.index('expression-' + x)) for x in EXPS)) + ',)[int(e)]'
                    vars = {'e': 'brows/state' if key.startswith('brow-') else 'expression/name'}
                index = nodes.new('ShaderNodeValue')
                index.label = key + ' state'
                driver(index.outputs[0], ctl, expr, vars)
                x0, y0, x1, y1 = spec.get('source_rect', [0, 0, w, h])
                cw, ch = (x1 - x0, y1 - y0)
                local_scale = nodes.new('ShaderNodeVectorMath')
                local_scale.operation = 'MULTIPLY'
                local_scale.inputs[1].default_value = (w / cw, h / ch, 1)
                links.new(projected.outputs['UV'], local_scale.inputs[0])
                local = nodes.new('ShaderNodeVectorMath')
                local.operation = 'ADD'
                local.inputs[1].default_value = (-x0 / cw, -(h - y1) / ch, 0)
                links.new(local_scale.outputs[0], local.inputs[0])
                separate = nodes.new('ShaderNodeSeparateXYZ')
                links.new(local.outputs[0], separate.inputs[0])
                crop = None
                for coordinate in ('X', 'Y'):
                    for op, threshold in [('GREATER_THAN', 0), ('LESS_THAN', 1)]:
                        check = nodes.new('ShaderNodeMath')
                        check.operation = op
                        check.inputs[1].default_value = threshold
                        links.new(separate.outputs[coordinate], check.inputs[0])
                        if crop is None:
                            crop = check.outputs[0]
                        else:
                            both = nodes.new('ShaderNodeMath')
                            both.operation = 'MULTIPLY'
                            links.new(crop, both.inputs[0])
                            links.new(check.outputs[0], both.inputs[1])
                            crop = both.outputs[0]
                scale = nodes.new('ShaderNodeVectorMath')
                scale.operation = 'MULTIPLY'
                scale.inputs[1].default_value = (1 / cols, 1 / rows, 1)
                links.new(local.outputs[0], scale.inputs[0])
                offset = nodes.new('ShaderNodeCombineXYZ')
                driver(offset.inputs['X'], ctl, f'(({expr})%{cols})/{cols}', vars)
                driver(offset.inputs['Y'], ctl, f'({rows}-1-floor(({expr})/{cols}))/{rows}', vars)
                add = nodes.new('ShaderNodeVectorMath')
                add.operation = 'ADD'
                links.new(scale.outputs['Vector'], add.inputs[0])
                links.new(offset.outputs[0], add.inputs[1])
                tex = nodes.new('ShaderNodeTexImage')
                tex.label = 'S5 ' + key
                tex.image = bpy.data.images.load(str((folder / spec['path']).resolve()), check_existing=True)
                tex.image.alpha_mode = 'STRAIGHT'
                tex.extension = 'CLIP'
                tex.interpolation = 'Linear'
                links.new(add.outputs['Vector'], tex.inputs['Vector'])
                alpha = nodes.new('ShaderNodeMath')
                alpha.operation = 'MULTIPLY'
                links.new(tex.outputs['Alpha'], alpha.inputs[0])
                links.new(visibility.outputs['Fac'], alpha.inputs[1])
                if b['static_uv'].get('alpha_is_feature_visibility'):
                    skin_only = nodes.new('ShaderNodeMath')
                    skin_only.operation = 'MULTIPLY'
                    skin_only.label = 'Preserve native hair occlusion'
                    links.new(alpha.outputs[0], skin_only.inputs[0])
                    links.new(original_images[0].outputs['Alpha'], skin_only.inputs[1])
                    alpha = skin_only
                clipped = nodes.new('ShaderNodeMath')
                clipped.operation = 'MULTIPLY'
                links.new(alpha.outputs[0], clipped.inputs[0])
                links.new(crop, clipped.inputs[1])
                if key != 'base':
                    enabled = nodes.new('ShaderNodeMath')
                    enabled.operation = 'MULTIPLY'
                    links.new(clipped.outputs[0], enabled.inputs[0])
                    driver(enabled.inputs[1], ctl, 'f', {'f': 'features/enabled'})
                    clipped = enabled
                mix = nodes.new('ShaderNodeMixRGB')
                mix.blend_type = 'MIX'
                links.new(clipped.outputs[0], mix.inputs[0])
                links.new(color, mix.inputs[1])
                links.new(tex.outputs['Color'], mix.inputs[2])
                color = mix.outputs[0]
            links.new(color, principled.inputs['Base Color'])
            emission = nodes.new('ShaderNodeEmission')
            links.new(color, emission.inputs['Color'])
            emission.inputs['Strength'].default_value = 1
            output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'))
            links.new(emission.outputs[0], output.inputs['Surface'])
        summary.append({'mesh': obj.name, 'vertices': len(data.vertices), 'face_support_vertices': sum((x > 0 for x in vis)), 'existing_uvs': old_uvs, 'projection_uv': uv.name})
    print('S5 materials installed', flush=True)
    for row in t['frames']:
        for prop, value in [('mouth/viseme', MOUTHS.index(row['mouth'])), ('eyes/state', EYES.index(row['eyes'])), ('expression/name', EXPS.index(row['expression'])), ('brows/state', EXPS.index(row['brows']))]:
            ctl[prop] = value
            ctl.keyframe_insert(data_path='["' + prop + '"]', frame=row['frame'])
        ctl['mouth/sample'] = b['atlases']['mouth']['states'].index(row.get('mouth_sample', 'mouth-' + row['mouth']))
        ctl.keyframe_insert(data_path='["mouth/sample"]', frame=row['frame'])
    action = ctl.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for k in fc.keyframe_points:
                        k.interpolation = 'CONSTANT'
    for o in list(sc.objects):
        if o.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(o, do_unlink=True)
    camera_data = bpy.data.cameras.new('FaceReviewCamera')
    camera = bpy.data.objects.new('FaceReviewCamera', camera_data)
    sc.collection.objects.link(camera)
    angle = math.radians(view)
    center = Vector((cam['center_x'], 0, cam['center_z']))
    camera.location = center + Vector((math.sin(angle) * 3, -math.cos(angle) * 3, 0))
    camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = cam['span']
    sc.camera = camera
    if sc.world is None:
        sc.world = bpy.data.worlds.new('FaceReviewWorld')
    sc.world.color = (0.8, 0.8, 0.8)
    sc.world.use_nodes = True
    sc.world.node_tree.nodes.get('Background').inputs[0].default_value = (0.85, 0.86, 0.89, 1)
    sc.frame_set(1)
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(out).resolve()))
    Path(out).with_suffix('.binding.json').write_text(json.dumps({'meshes': summary, 'controls': controls, 'track': str(Path(track).resolve()), 'notes': ['Texture projection is attached to existing rest geometry and survives skinning.', 'Blender shader animation is not a portable GLB/FBX animation; engines need the supplied atlas/control track.', 'Source S3 mesh is static; S4 integration requires the same rest-space frame.']}, indent=2))
    return sc
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--folder', required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--view', type=float, default=0)
    p.add_argument('--render-frame', type=int)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    sc = install(a.source, a.folder, a.track, a.out, a.view)
    if a.render_frame:
        sc.frame_set(a.render_frame)
        sc.render.filepath = str(Path(a.out).with_suffix('.png'))
        bpy.ops.render.render(write_still=True)

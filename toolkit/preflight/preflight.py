from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import bpy
import bmesh
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
CANONICAL_CANDIDATES: Dict[str, List[str]] = {'hips': ['hips', 'DEF-spine', 'J_Bip_C_Hips', 'Hips', 'mixamorig:Hips', 'pelvis', 'root_hips'], 'spine': ['spine', 'DEF-spine.001', 'J_Bip_C_Spine', 'Spine', 'mixamorig:Spine', 'Spine01'], 'chest': ['chest', 'DEF-spine.002', 'J_Bip_C_Chest', 'Spine1', 'Spine02', 'mixamorig:Spine1'], 'upperChest': ['upperChest', 'DEF-spine.003', 'J_Bip_C_UpperChest', 'Spine2', 'mixamorig:Spine2'], 'neck': ['neck', 'DEF-spine.004', 'J_Bip_C_Neck', 'Neck', 'mixamorig:Neck'], 'head': ['head', 'DEF-spine.006', 'DEF-spine.005', 'J_Bip_C_Head', 'Head', 'mixamorig:Head'], 'leftShoulder': ['leftShoulder', 'DEF-shoulder.L', 'J_Bip_L_Shoulder', 'LeftShoulder', 'mixamorig:LeftShoulder', 'Shoulder.L'], 'rightShoulder': ['rightShoulder', 'DEF-shoulder.R', 'J_Bip_R_Shoulder', 'RightShoulder', 'mixamorig:RightShoulder', 'Shoulder.R'], 'leftUpperArm': ['leftUpperArm', 'DEF-upper_arm.L', 'J_Bip_L_UpperArm', 'LeftArm', 'LeftUpperArm', 'mixamorig:LeftArm', 'UpperArm.L'], 'rightUpperArm': ['rightUpperArm', 'DEF-upper_arm.R', 'J_Bip_R_UpperArm', 'RightArm', 'RightUpperArm', 'mixamorig:RightArm', 'UpperArm.R'], 'leftLowerArm': ['leftLowerArm', 'DEF-forearm.L', 'J_Bip_L_LowerArm', 'LeftForeArm', 'LeftLowerArm', 'mixamorig:LeftForeArm', 'ForeArm.L'], 'rightLowerArm': ['rightLowerArm', 'DEF-forearm.R', 'J_Bip_R_LowerArm', 'RightForeArm', 'RightLowerArm', 'mixamorig:RightForeArm', 'ForeArm.R'], 'leftHand': ['leftHand', 'DEF-hand.L', 'J_Bip_L_Hand', 'LeftHand', 'mixamorig:LeftHand', 'Hand.L'], 'rightHand': ['rightHand', 'DEF-hand.R', 'J_Bip_R_Hand', 'RightHand', 'mixamorig:RightHand', 'Hand.R'], 'leftUpperLeg': ['leftUpperLeg', 'DEF-thigh.L', 'J_Bip_L_UpperLeg', 'LeftUpLeg', 'LeftUpperLeg', 'mixamorig:LeftUpLeg', 'Thigh.L'], 'rightUpperLeg': ['rightUpperLeg', 'DEF-thigh.R', 'J_Bip_R_UpperLeg', 'RightUpLeg', 'RightUpperLeg', 'mixamorig:RightUpLeg', 'Thigh.R'], 'leftLowerLeg': ['leftLowerLeg', 'DEF-shin.L', 'J_Bip_L_LowerLeg', 'LeftLeg', 'LeftLowerLeg', 'mixamorig:LeftLeg', 'Shin.L'], 'rightLowerLeg': ['rightLowerLeg', 'DEF-shin.R', 'J_Bip_R_LowerLeg', 'RightLeg', 'RightLowerLeg', 'mixamorig:RightLeg', 'Shin.R'], 'leftFoot': ['leftFoot', 'DEF-foot.L', 'J_Bip_L_Foot', 'LeftFoot', 'mixamorig:LeftFoot', 'Foot.L'], 'rightFoot': ['rightFoot', 'DEF-foot.R', 'J_Bip_R_Foot', 'RightFoot', 'mixamorig:RightFoot', 'Foot.R'], 'leftToes': ['leftToes', 'DEF-toe.L', 'J_Bip_L_ToeBase', 'LeftToeBase', 'LeftToes', 'mixamorig:LeftToeBase', 'Toe.L'], 'rightToes': ['rightToes', 'DEF-toe.R', 'J_Bip_R_ToeBase', 'RightToeBase', 'RightToes', 'mixamorig:RightToeBase', 'Toe.R']}

def load_character(filepath: Path) -> Tuple[Optional[bpy.types.Object], List[bpy.types.Object]]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = filepath.suffix.lower()
    if ext == '.blend':
        bpy.ops.wm.open_mainfile(filepath=str(filepath), load_ui=False, use_scripts=False)
    elif ext in ('.glb', '.gltf', '.vrm'):
        bpy.ops.import_scene.gltf(filepath=str(filepath))
    else:
        raise ValueError(f'Unsupported file extension: {ext}')
    for scene in bpy.data.scenes:
        if hasattr(scene, 'cycles'):
            scene.cycles.device = 'CPU'
    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    if not armatures:
        return (None, [])

    def arm_score(a: bpy.types.Object) -> Tuple[int, int]:
        meshes_bound = sum((1 for m in bpy.data.objects if m.type == 'MESH' and (m.find_armature() == a or m.parent == a)))
        return (meshes_bound, len(a.data.bones))
    arm = max(armatures, key=arm_score)
    meshes: List[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.name.startswith('WGT-') or 'widget' in obj.name.lower():
            continue
        if len(obj.data.vertices) == 0:
            continue
        bound_arm = obj.find_armature()
        if bound_arm == arm or obj.parent == arm or len(obj.vertex_groups) > 0:
            meshes.append(obj)
    return (arm, meshes)

def reset_to_rest_pose(arm: bpy.types.Object) -> Dict[str, bool]:
    if arm.animation_data:
        arm.animation_data.action = None
        for track in arm.animation_data.nla_tracks:
            track.mute = True
    arm.data.pose_position = 'POSE'
    muted_constraints: Dict[str, bool] = {}
    for pbone in arm.pose.bones:
        pbone.location = (0.0, 0.0, 0.0)
        pbone.rotation_mode = 'QUATERNION'
        pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        pbone.scale = (1.0, 1.0, 1.0)
        for c in pbone.constraints:
            key = f'{pbone.name}::{c.name}'
            muted_constraints[key] = c.mute
            c.mute = True
    bpy.context.view_layer.update()
    return muted_constraints

def resolve_bone_map(arm: bpy.types.Object) -> Tuple[str, Dict[str, str]]:
    bone_names = {b.name for b in arm.data.bones}
    lower_lookup = {b.name.lower(): b.name for b in arm.data.bones}
    resolved: Dict[str, str] = {}
    for slot, candidates in CANONICAL_CANDIDATES.items():
        for cand in candidates:
            if cand in bone_names:
                resolved[slot] = cand
                break
            if cand.lower() in lower_lookup:
                resolved[slot] = lower_lookup[cand.lower()]
                break
    if 'DEF-thigh.L' in bone_names:
        convention = 'rigify'
    elif 'J_Bip_L_UpperLeg' in bone_names:
        convention = 'vrm'
    elif 'leftUpperLeg' in bone_names:
        convention = 'charforge'
    else:
        convention = 'mixamo_or_generic'
    return (convention, resolved)

def classify_vertex_groups(arm: bpy.types.Object, bone_map: Dict[str, str]) -> Dict[str, set]:
    bone_names = [b.name for b in arm.data.bones]

    def descendants_of(roots: List[str], stop_at: set) -> set:
        result = set()
        queue = [r for r in roots if r and r in arm.data.bones]
        while queue:
            curr = queue.pop(0)
            if curr in result:
                continue
            result.add(curr)
            b = arm.data.bones[curr]
            for ch in b.children:
                if ch.name not in stop_at:
                    queue.append(ch.name)
        return result
    l_uarm = bone_map.get('leftUpperArm', '')
    r_uarm = bone_map.get('rightUpperArm', '')
    l_hand = bone_map.get('leftHand', '')
    r_hand = bone_map.get('rightHand', '')
    l_uleg = bone_map.get('leftUpperLeg', '')
    r_uleg = bone_map.get('rightUpperLeg', '')
    l_foot = bone_map.get('leftFoot', '')
    r_foot = bone_map.get('rightFoot', '')
    neck = bone_map.get('neck', '')
    head = bone_map.get('head', '')
    left_arm_bones = descendants_of([l_uarm], set())
    right_arm_bones = descendants_of([r_uarm], set())
    left_hand_bones = descendants_of([l_hand], set())
    right_hand_bones = descendants_of([r_hand], set())
    left_leg_bones = descendants_of([l_uleg], set())
    right_leg_bones = descendants_of([r_uleg], set())
    left_foot_bones = descendants_of([l_foot], set())
    right_foot_bones = descendants_of([r_foot], set())
    for bname in bone_names:
        low = bname.lower()
        if any((k in low for k in ('upper_arm', 'forearm', 'upperarm', 'lowerarm', 'arm_twist', 'forearm_twist'))):
            if '.l' in low or 'left' in low or '_l_' in low:
                left_arm_bones.add(bname)
            elif '.r' in low or 'right' in low or '_r_' in low:
                right_arm_bones.add(bname)
        if any((k in low for k in ('thigh', 'shin', 'upperleg', 'lowerleg', 'upleg'))):
            if '.l' in low or 'left' in low or '_l_' in low:
                left_leg_bones.add(bname)
            elif '.r' in low or 'right' in low or '_r_' in low:
                right_leg_bones.add(bname)
        if any((k in low for k in ('foot', 'toe', 'ball'))):
            if '.l' in low or 'left' in low or '_l_' in low:
                left_foot_bones.add(bname)
                left_leg_bones.add(bname)
            elif '.r' in low or 'right' in low or '_r_' in low:
                right_foot_bones.add(bname)
                right_leg_bones.add(bname)
    hair_spring_bones = {bname for bname in bone_names if any((k in bname.lower() for k in ('hair', 'j_sec_hair', 'ponytail', 'bang', 'braid', 'strand')))}
    torso_pelvis_bones = set()
    for slot in ('hips', 'spine', 'chest', 'upperChest'):
        if slot in bone_map:
            torso_pelvis_bones.add(bone_map[slot])
    for bname in bone_names:
        low = bname.lower()
        if any((k in low for k in ('hips', 'pelvis', 'def-spine', 'j_bip_c_hips', 'j_bip_c_spine', 'j_bip_c_chest', 'skirt', 'coat', 'hem'))):
            if bname not in (neck, head) and '004' not in low and ('005' not in low) and ('006' not in low):
                torso_pelvis_bones.add(bname)
    pelvis_and_legs = set(left_leg_bones | right_leg_bones)
    if 'hips' in bone_map:
        pelvis_and_legs.add(bone_map['hips'])
    for bname in bone_names:
        low = bname.lower()
        if any((k in low for k in ('skirt', 'hem', 'pelvis', 'thigh', 'shin', 'upperleg', 'lowerleg', 'upleg'))):
            pelvis_and_legs.add(bname)
    return {'all_armature_bones': set(bone_names), 'left_arm': left_arm_bones, 'right_arm': right_arm_bones, 'all_arms': left_arm_bones | right_arm_bones, 'left_hand': left_hand_bones, 'right_hand': right_hand_bones, 'all_hands': left_hand_bones | right_hand_bones, 'left_leg': left_leg_bones, 'right_leg': right_leg_bones, 'all_legs': left_leg_bones | right_leg_bones, 'left_foot': left_foot_bones, 'right_foot': right_foot_bones, 'all_feet': left_foot_bones | right_foot_bones, 'torso_pelvis': torso_pelvis_bones, 'lower_body_and_hips': pelvis_and_legs, 'hair_spring': hair_spring_bones, 'head': {head} if head else set()}

def get_evaluated_world_verts_and_tris(meshes: List[bpy.types.Object], depsgraph: bpy.types.Depsgraph, valid_bone_names: Optional[set]=None) -> Tuple[List[Vector], List[Tuple[int, int, int]], List[Tuple[int, int]], List[Dict[str, float]]]:
    all_verts: List[Vector] = []
    all_tris: List[Tuple[int, int, int]] = []
    all_edges: List[Tuple[int, int]] = []
    all_weights: List[Dict[str, float]] = []
    v_offset = 0
    for obj in meshes:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.calc_loop_triangles()
        mw = eval_obj.matrix_world
        vg_names = {vg.index: vg.name for vg in obj.vertex_groups}
        for v in mesh.vertices:
            all_verts.append(mw @ v.co)
            w_dict: Dict[str, float] = {}
            for g in v.groups:
                gname = vg_names.get(g.group)
                if gname and g.weight > 0.0001:
                    if valid_bone_names is None or gname in valid_bone_names:
                        w_dict[gname] = w_dict.get(gname, 0.0) + float(g.weight)
            all_weights.append(w_dict)
        for e in mesh.edges:
            all_edges.append((v_offset + e.vertices[0], v_offset + e.vertices[1]))
        for tri in mesh.loop_triangles:
            all_tris.append((v_offset + tri.vertices[0], v_offset + tri.vertices[1], v_offset + tri.vertices[2]))
        v_offset += len(mesh.vertices)
        eval_obj.to_mesh_clear()
    return (all_verts, all_tris, all_edges, all_weights)

def build_filtered_bvh(verts: List[Vector], tris: List[Tuple[int, int, int]], weights: List[Dict[str, float]], allowed_bones: Optional[set]=None, min_weight: float=0.15) -> Optional[BVHTree]:
    if allowed_bones is None:
        if not tris:
            return None
        return BVHTree.FromPolygons(verts, tris)
    sub_tris = []
    for i0, i1, i2 in tris:
        w0 = sum((weights[i0].get(b, 0.0) for b in allowed_bones))
        w1 = sum((weights[i1].get(b, 0.0) for b in allowed_bones))
        w2 = sum((weights[i2].get(b, 0.0) for b in allowed_bones))
        if max(w0, w1, w2) >= min_weight:
            sub_tris.append((i0, i1, i2))
    if not sub_tris:
        return None
    return BVHTree.FromPolygons(verts, sub_tris)

def cast_bisecting_fan_24(bvh: Optional[BVHTree], origin: Vector, u_parent: Vector, u_child: Vector, max_dist: float, n_rays: int=24) -> Dict[str, Any]:
    if bvh is None:
        return {'valid': False, 'hit_count': 0}
    n_axis = u_parent.normalized() + u_child.normalized()
    if n_axis.length < 1e-05:
        n_axis = u_child.normalized()
    else:
        n_axis.normalize()
    ref_y = Vector((0.0, -1.0, 0.0))
    if abs(n_axis.dot(ref_y)) > 0.95:
        ref_y = Vector((1.0, 0.0, 0.0))
    e1 = (ref_y - ref_y.dot(n_axis) * n_axis).normalized()
    e2 = n_axis.cross(e1).normalized()
    radii: List[float] = []
    hit_pts_2d: List[Tuple[float, float]] = []
    hit_pts_3d: List[Vector] = []
    for k in range(n_rays):
        theta = 2.0 * math.pi * k / n_rays
        d = math.cos(theta) * e1 + math.sin(theta) * e2
        loc, _norm, _idx, dist = bvh.ray_cast(origin + d * 0.0001, d, max_dist)
        if loc is not None and dist is not None and (dist <= max_dist):
            radii.append(float(dist))
            hit_pts_2d.append((float(dist) * math.cos(theta), float(dist) * math.sin(theta)))
            hit_pts_3d.append(loc)
    hit_fraction = len(radii) / float(n_rays)
    if len(radii) < 16:
        return {'valid': False, 'hit_count': len(radii), 'hit_fraction': round(hit_fraction, 4)}
    r_mean = sum(radii) / len(radii)
    r_min = min(radii)
    r_max = max(radii)
    area = 0.0
    cx = 0.0
    cy = 0.0
    m = len(hit_pts_2d)
    for i in range(m):
        x0, y0 = hit_pts_2d[i]
        x1, y1 = hit_pts_2d[(i + 1) % m]
        cross = x0 * y1 - x1 * y0
        area += 0.5 * cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) > 1e-09:
        cx /= 6.0 * area
        cy /= 6.0 * area
        area = abs(area)
    else:
        cx = sum((p[0] for p in hit_pts_2d)) / m
        cy = sum((p[1] for p in hit_pts_2d)) / m
        area = 0.0
    centroid_3d = origin + cx * e1 + cy * e2
    offset_vec = centroid_3d - origin
    offset_dist = math.hypot(cx, cy)
    return {'valid': True, 'hit_count': len(radii), 'hit_fraction': round(hit_fraction, 4), 'r_mean_m': round(r_mean, 5), 'r_min_m': round(r_min, 5), 'r_max_m': round(r_max, 5), 'area_m2': round(area, 6), 'centroid_world': [round(c, 5) for c in centroid_3d], 'pivot_offset_m': round(offset_dist, 5), 'pivot_offset_mm': round(offset_dist * 1000.0, 2), 'pivot_offset_share': round(offset_dist / max(r_mean, 1e-06), 4), 'offset_world_xyz_mm': [round(c * 1000.0, 2) for c in offset_vec]}

def rotate_bone_world(arm: bpy.types.Object, bone_name: str, axis_world: Vector, angle_deg: float) -> None:
    if not bone_name or bone_name not in arm.pose.bones:
        return
    pbone = arm.pose.bones[bone_name]
    mw = arm.matrix_world
    head_world = mw @ pbone.head
    rot_world = Matrix.Translation(head_world) @ Quaternion(axis_world.normalized(), math.radians(angle_deg)).to_matrix().to_4x4() @ Matrix.Translation(-head_world)
    pbone.matrix = mw.inverted() @ (rot_world @ (mw @ pbone.matrix))
    twist_sub = f'{bone_name}.001'
    if twist_sub in arm.pose.bones:
        sub_pb = arm.pose.bones[twist_sub]
        if sub_pb.parent != pbone:
            sub_pb.matrix = mw.inverted() @ (rot_world @ (mw @ sub_pb.matrix))
    bpy.context.view_layer.update()

def count_self_intersections_near_region(verts: List[Vector], tris: List[Tuple[int, int, int]], z_min: float, z_max: float) -> int:
    band_tris: List[Tuple[int, int, int]] = []
    for t in tris:
        zs = (verts[t[0]].z, verts[t[1]].z, verts[t[2]].z)
        if max(zs) >= z_min and min(zs) <= z_max:
            band_tris.append(t)
    if len(band_tris) < 2:
        return 0
    bvh = BVHTree.FromPolygons(verts, band_tris)
    pairs = bvh.overlap(bvh)
    non_adjacent = 0
    for i, j in pairs:
        if i >= j:
            continue
        t0 = band_tris[i]
        t1 = band_tris[j]
        if t0[0] in t1 or t0[1] in t1 or t0[2] in t1:
            continue
        non_adjacent += 1
    return non_adjacent

def run_preflight_on_rig(filepath: Path, name_override: Optional[str]=None) -> Dict[str, Any]:
    t0 = time.time()
    rig_id = name_override or filepath.stem
    arm, meshes = load_character(filepath)
    if arm is None or not meshes:
        return {'rig_id': rig_id, 'path': str(filepath), 'error': 'No armature or deforming mesh found'}
    reset_to_rest_pose(arm)
    convention, bone_map = resolve_bone_map(arm)
    branch_groups = classify_vertex_groups(arm, bone_map)
    valid_bones = branch_groups['all_armature_bones']
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rest_verts, rest_tris, rest_edges, rest_weights = get_evaluated_world_verts_and_tris(meshes, depsgraph, valid_bone_names=valid_bones)
    zs = [v.z for v in rest_verts]
    xs = [v.x for v in rest_verts]
    ys = [v.y for v in rest_verts]
    stature_m = max(zs) - min(zs) if zs else 1.0
    floor_z = min(zs) if zs else 0.0
    mw = arm.matrix_world

    def b_head(slot: str) -> Optional[Vector]:
        bname = bone_map.get(slot)
        if bname and bname in arm.pose.bones:
            return mw @ arm.pose.bones[bname].head
        return None

    def b_tail(slot: str) -> Optional[Vector]:
        bname = bone_map.get(slot)
        if bname and bname in arm.pose.bones:
            return mw @ arm.pose.bones[bname].tail
        return None
    l_foot_h = b_head('leftFoot')
    l_foot_t = b_tail('leftFoot')
    l_toe_h = b_head('leftToes')
    forward_sign_y = -1.0
    if l_foot_h is not None:
        target_pt = l_toe_h if l_toe_h is not None else l_foot_t
        if target_pt is not None and abs(target_pt.y - l_foot_h.y) > 0.0001:
            forward_sign_y = 1.0 if target_pt.y - l_foot_h.y > 0 else -1.0
    rest_edge_lengths = [(rest_verts[i] - rest_verts[j]).length for i, j in rest_edges]
    l_hip = b_head('leftUpperLeg')
    r_hip = b_head('rightUpperLeg')
    l_knee = b_head('leftLowerLeg')
    r_knee = b_head('rightLowerLeg')
    l_ankle = b_head('leftFoot')
    r_ankle = b_head('rightFoot')
    leg_bvh_l = build_filtered_bvh(rest_verts, rest_tris, rest_weights, branch_groups['left_leg'], min_weight=0.15)
    leg_bvh_r = build_filtered_bvh(rest_verts, rest_tris, rest_weights, branch_groups['right_leg'], min_weight=0.15)
    full_bvh = build_filtered_bvh(rest_verts, rest_tris, rest_weights, None)
    if leg_bvh_l is None:
        leg_bvh_l = full_bvh
    if leg_bvh_r is None:
        leg_bvh_r = full_bvh
    fp01_metrics: Dict[str, Any] = {}
    fp02_metrics: Dict[str, Any] = {}
    if l_hip and l_knee and l_ankle and r_hip and r_knee and r_ankle:
        hip_x = 0.5 * (abs(l_hip.x) + abs(r_hip.x))
        knee_x = 0.5 * (abs(l_knee.x) + abs(r_knee.x))
        ankle_x = 0.5 * (abs(l_ankle.x) + abs(r_ankle.x))
        hip_to_knee_x_ratio = hip_x / max(knee_x, 1e-05)
        alpha_l = (l_hip.z - l_knee.z) / max(l_hip.z - l_ankle.z, 1e-05)
        alpha_r = (r_hip.z - r_knee.z) / max(r_hip.z - r_ankle.z, 1e-05)
        line_x_l = (1.0 - alpha_l) * abs(l_hip.x) + alpha_l * abs(l_ankle.x)
        line_x_r = (1.0 - alpha_r) * abs(r_hip.x) + alpha_r * abs(r_ankle.x)
        coronal_offset_l_mm = (abs(l_knee.x) - line_x_l) * 1000.0
        coronal_offset_r_mm = (abs(r_knee.x) - line_x_r) * 1000.0
        coronal_offset_mean_mm = 0.5 * (coronal_offset_l_mm + coronal_offset_r_mm)
        thigh_vec_l = l_knee - l_hip
        femur_coronal_angle_deg = math.degrees(math.atan2(abs(l_knee.x) - abs(l_hip.x), max(-thigh_vec_l.z, 1e-05)))
        ik_sagittal_plane_knee_shift_mm = abs(coronal_offset_mean_mm)
        ik_hip_scaled_gait_knee_shift_mm = max(0.0, (hip_x - knee_x) * 1000.0)
        l_thigh_len = (l_knee - l_hip).length
        ik_thigh_abduction_delta_deg = math.degrees(math.asin(min(1.0, max(-1.0, (line_x_l - abs(l_knee.x)) / max(l_thigh_len, 1e-05)))))
        u_thigh_l = (l_knee - l_hip).normalized()
        u_shin_l = (l_ankle - l_knee).normalized()
        fan_rest_l = cast_bisecting_fan_24(leg_bvh_l, l_knee, u_thigh_l, u_shin_l, max_dist=0.22 * stature_m)
        u_thigh_r = (r_knee - r_hip).normalized()
        u_shin_r = (r_ankle - r_knee).normalized()
        fan_rest_r = cast_bisecting_fan_24(leg_bvh_r, r_knee, u_thigh_r, u_shin_r, max_dist=0.22 * stature_m)
        r_knee_mean_m = fan_rest_l.get('r_mean_m', 0.04 * stature_m) if fan_rest_l.get('valid') else 0.04 * stature_m
        coronal_offset_radius_share = abs(coronal_offset_mean_mm / 1000.0) / max(r_knee_mean_m, 1e-05)
        fp01_metrics = {'hip_x_mm': round(hip_x * 1000.0, 2), 'knee_x_mm': round(knee_x * 1000.0, 2), 'ankle_x_mm': round(ankle_x * 1000.0, 2), 'hip_width_stature_ratio': round(2.0 * hip_x / max(stature_m, 1e-05), 4), 'hip_to_knee_x_ratio': round(hip_to_knee_x_ratio, 4), 'knee_coronal_offset_mm': round(coronal_offset_mean_mm, 2), 'knee_coronal_offset_radius_share': round(coronal_offset_radius_share, 4), 'femur_coronal_angle_deg': round(femur_coronal_angle_deg, 2), 'ik_sagittal_plane_knee_shift_mm': round(ik_sagittal_plane_knee_shift_mm, 2), 'ik_hip_scaled_gait_knee_shift_mm': round(ik_hip_scaled_gait_knee_shift_mm, 2), 'ik_thigh_abduction_delta_deg': round(ik_thigh_abduction_delta_deg, 2)}
        fp02_metrics = {'rest_knee_fan_left': fan_rest_l, 'rest_knee_fan_right': fan_rest_r, 'knee_pivot_offset_mm': fan_rest_l.get('pivot_offset_mm', None), 'knee_pivot_offset_share': fan_rest_l.get('pivot_offset_share', None), 'knee_height_stature_share': round((l_knee.z - floor_z) / max(stature_m, 1e-05), 4)}
    fp03_metrics: Dict[str, Any] = {}
    if l_ankle and (l_toe_h or l_foot_t):
        foot_end = l_toe_h if l_toe_h is not None else l_foot_t
        foot_vec = foot_end - l_ankle
        horiz_len = math.hypot(foot_vec.x, foot_vec.y)
        foot_bone_pitch_deg = math.degrees(math.atan2(-foot_vec.z, max(horiz_len, 1e-05)))
        foot_vert_indices = [idx for idx, w in enumerate(rest_weights) if sum((w.get(b, 0.0) for b in branch_groups['left_foot'])) >= 0.25]
        if not foot_vert_indices:
            foot_vert_indices = [idx for idx, v in enumerate(rest_verts) if v.x > 0.0 and v.z <= l_ankle.z + 0.02 * stature_m]
        if foot_vert_indices:
            f_verts = [rest_verts[i] for i in foot_vert_indices]
            f_ys = [v.y * forward_sign_y for v in f_verts]
            f_zs = [v.z for v in f_verts]
            heel_y = min(f_ys)
            tip_y = max(f_ys)
            sole_z = min(f_zs)
            shoe_length_m = max(tip_y - heel_y, 0.0001)
            if l_toe_h is not None:
                toe_fwd = l_toe_h.y * forward_sign_y
                ball_station_ratio = (toe_fwd - heel_y) / shoe_length_m
                toe_height_above_sole_mm = (l_toe_h.z - sole_z) * 1000.0
            else:
                ball_station_ratio = None
                toe_height_above_sole_mm = None
            ankle_to_ball_lever_m = max(0.01, heel_y + 0.76 * shoe_length_m - l_ankle.y * forward_sign_y)
            naive_align_pitch_lift_mm = ankle_to_ball_lever_m * math.sin(math.radians(max(0.0, foot_bone_pitch_deg))) * 1000.0
            fp03_metrics = {'foot_bone_pitch_deg': round(foot_bone_pitch_deg, 2), 'shoe_length_mm': round(shoe_length_m * 1000.0, 2), 'ankle_height_above_sole_mm': round((l_ankle.z - sole_z) * 1000.0, 2), 'has_toe_bone': l_toe_h is not None, 'toe_ball_station_ratio': round(ball_station_ratio, 4) if ball_station_ratio is not None else None, 'toe_height_above_sole_mm': round(toe_height_above_sole_mm, 2) if toe_height_above_sole_mm is not None else None, 'naive_align_tiptoe_lift_mm': round(naive_align_pitch_lift_mm, 2)}
    arm_bleed_hips_leg_verts = 0
    lower_arm_bleed_hips_leg_verts = 0
    max_arm_hips_leg_bleed = 0.0
    worst_bleed_examples: List[Dict[str, Any]] = []
    lower_arm_bones = branch_groups['left_hand'] | branch_groups['right_hand'] | {b for b in branch_groups['all_arms'] if any((k in b.lower() for k in ('lowerarm', 'forearm', 'hand')))}
    for idx, w in enumerate(rest_weights):
        w_arm = sum((w.get(b, 0.0) for b in branch_groups['all_arms']))
        if w_arm < 0.15:
            continue
        w_lower_arm = sum((w.get(b, 0.0) for b in lower_arm_bones))
        w_hips_leg = sum((w.get(b, 0.0) for b in branch_groups['lower_body_and_hips']))
        v_co = rest_verts[idx]
        if w_hips_leg >= 0.05:
            arm_bleed_hips_leg_verts += 1
            if w_lower_arm >= 0.1:
                lower_arm_bleed_hips_leg_verts += 1
            if w_hips_leg > max_arm_hips_leg_bleed:
                max_arm_hips_leg_bleed = w_hips_leg
            if len(worst_bleed_examples) < 5 and w_hips_leg >= 0.12:
                worst_bleed_examples.append({'vertex': idx, 'xyz_m': [round(c, 4) for c in v_co], 'w_arm': round(w_arm, 4), 'w_lower_arm': round(w_lower_arm, 4), 'w_hips_leg': round(w_hips_leg, 4), 'top_groups': sorted([(k, round(val, 4)) for k, val in w.items()], key=lambda x: -x[1])[:4]})
    l_uarm_h = b_head('leftUpperArm')
    l_larm_h = b_head('leftLowerArm')
    l_hand_h = b_head('leftHand')
    arm_bvh_l = build_filtered_bvh(rest_verts, rest_tris, rest_weights, branch_groups['left_arm'], min_weight=0.15)
    if arm_bvh_l is None:
        arm_bvh_l = full_bvh
    elbow_fan_rest = {'valid': False, 'hit_count': 0}
    if l_uarm_h and l_larm_h and l_hand_h:
        u_ua = (l_larm_h - l_uarm_h).normalized()
        u_la = (l_hand_h - l_larm_h).normalized()
        elbow_fan_rest = cast_bisecting_fan_24(arm_bvh_l, l_larm_h, u_ua, u_la, max_dist=0.18 * stature_m)
    fp04_metrics: Dict[str, Any] = {'arm_hips_leg_bleed_vertex_count': arm_bleed_hips_leg_verts, 'lower_arm_hips_leg_bleed_vertex_count': lower_arm_bleed_hips_leg_verts, 'max_arm_hips_leg_bleed_share': round(max_arm_hips_leg_bleed, 4), 'worst_bleed_examples': worst_bleed_examples, 'rest_elbow_fan': elbow_fan_rest}
    open_boundary_loops = 0
    for obj in meshes:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        boundary_edges = [e for e in bm.edges if e.is_boundary]
        visited_e = set()
        for e in boundary_edges:
            if e.index in visited_e:
                continue
            open_boundary_loops += 1
            stack = [e]
            while stack:
                curr_e = stack.pop()
                if curr_e.index in visited_e:
                    continue
                visited_e.add(curr_e.index)
                for v in curr_e.verts:
                    for link_e in v.link_edges:
                        if link_e.is_boundary and link_e.index not in visited_e:
                            stack.append(link_e)
        bm.free()
    non_arm_tris: List[Tuple[int, int, int]] = []
    for i0, i1, i2 in rest_tris:
        wa0 = sum((rest_weights[i0].get(b, 0.0) for b in branch_groups['all_arms']))
        wa1 = sum((rest_weights[i1].get(b, 0.0) for b in branch_groups['all_arms']))
        wa2 = sum((rest_weights[i2].get(b, 0.0) for b in branch_groups['all_arms']))
        if max(wa0, wa1, wa2) < 0.25:
            non_arm_tris.append((i0, i1, i2))
    body_garment_bvh = BVHTree.FromPolygons(rest_verts, non_arm_tris) if non_arm_tris else None
    hand_forearm_indices = [idx for idx, w in enumerate(rest_weights) if sum((w.get(b, 0.0) for b in lower_arm_bones)) >= 0.35]
    wrist_z = l_hand_h.z if l_hand_h else floor_z + 0.45 * stature_m
    elbow_z = l_larm_h.z if l_larm_h else floor_z + 0.58 * stature_m
    band_z_min = min(wrist_z, elbow_z) - 0.05 * stature_m
    band_z_max = max(wrist_z, elbow_z)
    garment_band_xs = [abs(rest_verts[idx].x) for idx in range(len(rest_verts)) if band_z_min <= rest_verts[idx].z <= band_z_max and sum((rest_weights[idx].get(b, 0.0) for b in branch_groups['all_arms'])) < 0.15]
    garment_max_half_width_m = max(garment_band_xs) if garment_band_xs else 0.0
    hand_inner_x_m = min((abs(rest_verts[idx].x) for idx in hand_forearm_indices), default=abs(l_hand_h.x) if l_hand_h else 0.15)
    rest_coronal_swing_margin_mm = (hand_inner_x_m - garment_max_half_width_m) * 1000.0
    rest_min_surface_dist_mm = 999.0
    if body_garment_bvh and hand_forearm_indices:
        for idx in hand_forearm_indices:
            loc, _n, _i, dist = body_garment_bvh.find_nearest(rest_verts[idx])
            if dist is not None and dist * 1000.0 < rest_min_surface_dist_mm:
                rest_min_surface_dist_mm = dist * 1000.0
    fp05_metrics: Dict[str, Any] = {'open_boundary_loops': open_boundary_loops, 'garment_max_half_width_at_wrist_mm': round(garment_max_half_width_m * 1000.0, 2), 'hand_inner_x_mm': round(hand_inner_x_m * 1000.0, 2), 'rest_coronal_swing_margin_mm': round(rest_coronal_swing_margin_mm, 2), 'rest_min_hand_to_body_dist_mm': round(rest_min_surface_dist_mm, 2)}
    spring_bone_count = len(branch_groups['hair_spring'])
    head_hair_verts = [idx for idx, w in enumerate(rest_weights) if sum((w.get(b, 0.0) for b in branch_groups['head'] | branch_groups['hair_spring'])) >= 0.25]
    dynamic_hair_verts = [idx for idx in head_hair_verts if sum((rest_weights[idx].get(b, 0.0) for b in branch_groups['hair_spring'])) >= 0.15]
    neck_h = b_head('neck')
    neck_z = neck_h.z if neck_h else floor_z + 0.72 * stature_m
    low_hanging_head_verts = [idx for idx in head_hair_verts if rest_verts[idx].z < neck_z]
    fp06_metrics: Dict[str, Any] = {'spring_hair_bone_count': spring_bone_count, 'head_and_hair_vertex_count': len(head_hair_verts), 'dynamic_hair_vertex_count': len(dynamic_hair_verts), 'dynamic_hair_share': round(len(dynamic_hair_verts) / max(len(head_hair_verts), 1), 4), 'low_hanging_hair_verts_below_neck': len(low_hanging_head_verts)}
    fp07_metrics: Dict[str, Any] = {'deforming_mesh_count': len(meshes), 'is_single_fused_mesh': len(meshes) == 1}
    fp08_metrics: Dict[str, Any] = {}
    if l_hip and l_knee and l_ankle:
        l_upper = (l_knee - l_hip).length
        l_lower = (l_ankle - l_knee).length
        chain_len = l_upper + l_lower
        chord_len = (l_ankle - l_hip).length
        vert_span = max(l_hip.z - l_ankle.z, 1e-05)
        leg_reach_slack_ratio = chain_len / max(chord_len, 1e-05) - 1.0
        vertical_leg_slack_ratio = chain_len / vert_span - 1.0
        u_th = (l_knee - l_hip).normalized()
        u_sh = (l_ankle - l_knee).normalized()
        rest_knee_flexion_deg = math.degrees(math.acos(min(1.0, max(-1.0, u_th.dot(u_sh)))))
        step_horiz = 0.3 * chain_len
        max_vert_at_step = math.sqrt(max(0.0, (0.995 * chain_len) ** 2 - step_horiz ** 2))
        stride_reach_drop_m = max(0.0, vert_span - max_vert_at_step)
        shoe_len_m = (fp03_metrics.get('shoe_length_mm', 0.0) or 0.0) / 1000.0
        sole_lever_to_leg_ratio = shoe_len_m / max(chain_len, 1e-05)
        sole_pitch25_z_swing_mm = shoe_len_m * 0.5 * math.sin(math.radians(25.0)) * 1000.0
        fp08_metrics = {'thigh_len_mm': round(l_upper * 1000.0, 2), 'shin_len_mm': round(l_lower * 1000.0, 2), 'leg_chain_len_mm': round(chain_len * 1000.0, 2), 'hip_to_ankle_chord_mm': round(chord_len * 1000.0, 2), 'rest_knee_flexion_deg': round(rest_knee_flexion_deg, 2), 'leg_reach_slack_ratio': round(leg_reach_slack_ratio, 5), 'vertical_leg_slack_ratio': round(vertical_leg_slack_ratio, 5), 'stride30_required_pelvis_drop_mm': round(stride_reach_drop_m * 1000.0, 2), 'sole_lever_to_leg_ratio': round(sole_lever_to_leg_ratio, 4), 'sole_pitch25_z_swing_mm': round(sole_pitch25_z_swing_mm, 2)}
    reset_to_rest_pose(arm)
    if 'leftLowerLeg' in bone_map:
        knee_sign = 1.0 if forward_sign_y < 0 else -1.0
        rotate_bone_world(arm, bone_map['leftLowerLeg'], Vector((1.0, 0.0, 0.0)), knee_sign * -90.0)
        deps = bpy.context.evaluated_depsgraph_get()
        k90_verts, k90_tris, _, _ = get_evaluated_world_verts_and_tris(meshes, deps, valid_bone_names=valid_bones)
        k90_bvh = build_filtered_bvh(k90_verts, k90_tris, rest_weights, branch_groups['left_leg'], min_weight=0.15)
        if k90_bvh is None:
            k90_bvh = BVHTree.FromPolygons(k90_verts, k90_tris)
        l_hip_k = b_head('leftUpperLeg')
        l_knee_k = b_head('leftLowerLeg')
        l_ankle_k = b_head('leftFoot')
        if l_hip_k and l_knee_k and l_ankle_k:
            fan_k90 = cast_bisecting_fan_24(k90_bvh, l_knee_k, (l_knee_k - l_hip_k).normalized(), (l_ankle_k - l_knee_k).normalized(), max_dist=0.22 * stature_m)
            fp02_metrics['knee90_fan_left'] = fan_k90
            r_rest = fp02_metrics.get('rest_knee_fan_left', {}).get('r_mean_m')
            r_k90 = fan_k90.get('r_mean_m')
            if r_rest and r_k90:
                fp02_metrics['knee90_section_ratio'] = round(r_k90 / max(r_rest, 1e-06), 4)
    reset_to_rest_pose(arm)
    if 'leftUpperArm' in bone_map and 'leftLowerArm' in bone_map:
        fwd_rot_sign = 1.0 if forward_sign_y < 0 else -1.0
        rotate_bone_world(arm, bone_map['leftUpperArm'], Vector((0.0, 1.0, 0.0)), -20.0)
        rotate_bone_world(arm, bone_map['leftUpperArm'], Vector((1.0, 0.0, 0.0)), fwd_rot_sign * 35.0)
        rotate_bone_world(arm, bone_map['leftLowerArm'], Vector((1.0, 0.0, 0.0)), fwd_rot_sign * 60.0)
        if 'rightUpperArm' in bone_map and 'rightLowerArm' in bone_map:
            rotate_bone_world(arm, bone_map['rightUpperArm'], Vector((0.0, 1.0, 0.0)), 20.0)
            rotate_bone_world(arm, bone_map['rightUpperArm'], Vector((1.0, 0.0, 0.0)), -fwd_rot_sign * 35.0)
            rotate_bone_world(arm, bone_map['rightLowerArm'], Vector((1.0, 0.0, 0.0)), fwd_rot_sign * 60.0)
        deps = bpy.context.evaluated_depsgraph_get()
        arm_verts, arm_tris, _, _ = get_evaluated_world_verts_and_tris(meshes, deps, valid_bone_names=valid_bones)
        arm_stretches: List[float] = []
        for e_idx, (i0, i1) in enumerate(rest_edges):
            l0 = rest_edge_lengths[e_idx]
            if l0 < 1e-05:
                continue
            w0 = sum((rest_weights[i0].get(b, 0.0) for b in branch_groups['all_arms']))
            w1 = sum((rest_weights[i1].get(b, 0.0) for b in branch_groups['all_arms']))
            if max(w0, w1) >= 0.15:
                posed_length = (arm_verts[i0] - arm_verts[i1]).length
                arm_stretches.append(posed_length / l0)
        if arm_stretches:
            arm_stretches.sort()
            p99_idx = min(len(arm_stretches) - 1, int(0.99 * len(arm_stretches)))
            fp04_metrics['arm_swing_edge_stretch_p99'] = round(arm_stretches[p99_idx], 4)
            fp04_metrics['arm_swing_edge_stretch_max'] = round(arm_stretches[-1], 4)
    min_swing_clearance_mm = rest_min_surface_dist_mm
    swing_penetrating_verts = 0
    if 'leftUpperArm' in bone_map and body_garment_bvh and hand_forearm_indices:
        fwd_rot_sign = 1.0 if forward_sign_y < 0 else -1.0
        for swing_deg in (-25.0, 0.0, 25.0):
            reset_to_rest_pose(arm)
            rotate_bone_world(arm, bone_map['leftUpperArm'], Vector((1.0, 0.0, 0.0)), fwd_rot_sign * swing_deg)
            if 'leftLowerArm' in bone_map:
                rotate_bone_world(arm, bone_map['leftLowerArm'], Vector((1.0, 0.0, 0.0)), fwd_rot_sign * 15.0)
            deps = bpy.context.evaluated_depsgraph_get()
            sw_verts, _, _, _ = get_evaluated_world_verts_and_tris(meshes, deps, valid_bone_names=valid_bones)
            for idx in hand_forearm_indices:
                pt = sw_verts[idx]
                loc, norm, _i, dist = body_garment_bvh.find_nearest(pt)
                if loc is not None and dist is not None:
                    signed_d_mm = dist * 1000.0
                    if abs(pt.x) < abs(loc.x) - 0.002 and abs(pt.y - loc.y) < 0.04 * stature_m:
                        signed_d_mm = -signed_d_mm
                        swing_penetrating_verts += 1
                    if signed_d_mm < min_swing_clearance_mm:
                        min_swing_clearance_mm = signed_d_mm
    fp05_metrics['sagittal_swing_min_clearance_mm'] = round(min_swing_clearance_mm, 2)
    fp05_metrics['sagittal_swing_penetrating_vertex_hits'] = swing_penetrating_verts
    reset_to_rest_pose(arm)
    if 'leftUpperLeg' in bone_map and l_hip:
        fwd_rot_sign = 1.0 if forward_sign_y < 0 else -1.0
        rest_si = count_self_intersections_near_region(rest_verts, rest_tris, l_hip.z - 0.25 * stature_m, l_hip.z + 0.15 * stature_m)
        rotate_bone_world(arm, bone_map['leftUpperLeg'], Vector((1.0, 0.0, 0.0)), fwd_rot_sign * 90.0)
        deps = bpy.context.evaluated_depsgraph_get()
        h90_verts, h90_tris, _, _ = get_evaluated_world_verts_and_tris(meshes, deps, valid_bone_names=valid_bones)
        h90_si = count_self_intersections_near_region(h90_verts, h90_tris, l_hip.z - 0.25 * stature_m, l_hip.z + 0.15 * stature_m)
        fp07_metrics['rest_hip_self_intersections'] = rest_si
        fp07_metrics['hip90_self_intersections'] = h90_si
        fp07_metrics['hip90_self_intersection_delta'] = h90_si - rest_si
    reset_to_rest_pose(arm)
    predictions: Dict[str, Dict[str, Any]] = {}
    hip_knee_ratio = fp01_metrics.get('hip_to_knee_x_ratio', 1.0)
    coronal_rad_share = fp01_metrics.get('knee_coronal_offset_radius_share', 0.0)
    coronal_mm = fp01_metrics.get('knee_coronal_offset_mm', 0.0)
    fp01_likely = hip_knee_ratio > 1.18 or (coronal_mm < -8.0 and coronal_rad_share > 0.22)
    predictions['FP-01_outward_knees_coronal_splay'] = {'status': 'LIKELY' if fp01_likely else 'CLEAR', 'severity': 'HIGH' if hip_knee_ratio > 1.3 else 'MEDIUM' if fp01_likely else 'NONE', 'reason': f"hip_to_knee_x_ratio={hip_knee_ratio:.3f} (threshold <=1.18), knee_coronal_offset={coronal_mm:.2f} mm ({coronal_rad_share * 100:.1f}% of knee radius), sagittal IK shifts knee +{fp01_metrics.get('ik_sagittal_plane_knee_shift_mm', 0.0):.2f} mm outward", 'metrics': fp01_metrics}
    kp_share = fp02_metrics.get('knee_pivot_offset_share')
    k90_sec = fp02_metrics.get('knee90_section_ratio')
    fp02_likely = kp_share is not None and kp_share > 0.3 or (k90_sec is not None and k90_sec < 0.55)
    predictions['FP-02_sagittal_knee_pivot_offset'] = {'status': 'LIKELY' if fp02_likely else 'CLEAR', 'severity': 'HIGH' if (kp_share or 0.0) > 0.4 else 'MEDIUM' if fp02_likely else 'NONE', 'reason': f'knee_pivot_offset_share={kp_share} (provisional threshold <=0.30), knee90_section_ratio={k90_sec}', 'metrics': fp02_metrics}
    foot_pitch = fp03_metrics.get('foot_bone_pitch_deg', 0.0)
    has_toe = fp03_metrics.get('has_toe_bone', False)
    ball_ratio = fp03_metrics.get('toe_ball_station_ratio')
    fp03_likely = not has_toe or (ball_ratio is not None and (ball_ratio < 0.45 or ball_ratio > 0.85))
    predictions['FP-03_tiptoe_foot_bind_pitch'] = {'status': 'LIKELY' if fp03_likely else 'CLEAR', 'severity': 'HIGH' if not has_toe or (ball_ratio or 0.75) > 0.9 else 'MEDIUM' if fp03_likely else 'NONE', 'reason': f"has_toe_bone={has_toe}, ball_station_ratio={ball_ratio} (anatomical ball band 0.45..0.85; >0.90 means toe bone is at extreme shoe tip), foot_bone_pitch_deg={foot_pitch:.2f} deg (naive direction align lifts heel/drops toe by {fp03_metrics.get('naive_align_tiptoe_lift_mm', 0.0):.1f} mm)", 'metrics': fp03_metrics}
    bleed_cnt = fp04_metrics.get('arm_hips_leg_bleed_vertex_count', 0)
    larm_bleed_cnt = fp04_metrics.get('lower_arm_hips_leg_bleed_vertex_count', 0)
    max_bleed = fp04_metrics.get('max_arm_hips_leg_bleed_share', 0.0)
    arm_stretch_p99 = fp04_metrics.get('arm_swing_edge_stretch_p99', 1.0)
    arm_stretch_max = fp04_metrics.get('arm_swing_edge_stretch_max', 1.0)
    elbow_valid = fp04_metrics.get('rest_elbow_fan', {}).get('valid', True)
    fp04_likely = larm_bleed_cnt > 0 or max_bleed >= 0.12 or arm_stretch_p99 > 2.2 or (not elbow_valid)
    predictions['FP-04_webbed_sleeve_weight_bleed'] = {'status': 'LIKELY' if fp04_likely else 'CLEAR', 'severity': 'HIGH' if max_bleed >= 0.25 or arm_stretch_p99 > 3.0 else 'MEDIUM' if fp04_likely else 'NONE', 'reason': f'lower_arm_hips_leg_bleed_verts={larm_bleed_cnt} (total_arm_bleed={bleed_cnt}), max_bleed={max_bleed:.3f}, arm_swing_edge_stretch_p99={arm_stretch_p99:.2f}x (max={arm_stretch_max:.2f}x), elbow_fan_valid={elbow_valid}', 'metrics': fp04_metrics}
    coronal_margin = fp05_metrics.get('rest_coronal_swing_margin_mm', 99.0)
    swing_clear = fp05_metrics.get('sagittal_swing_min_clearance_mm', 99.0)
    swing_pen = fp05_metrics.get('sagittal_swing_penetrating_vertex_hits', 0)
    fp05_likely = coronal_margin < 5.0 or swing_clear < 6.0 or swing_pen > 0
    predictions['FP-05_hand_garment_clearance_starvation'] = {'status': 'LIKELY' if fp05_likely else 'CLEAR', 'severity': 'HIGH' if swing_pen > 0 or coronal_margin < 0.0 else 'MEDIUM' if fp05_likely else 'NONE', 'reason': f'rest_coronal_swing_margin={coronal_margin:.2f} mm, sagittal_swing_min_clearance={swing_clear:.2f} mm, penetrating_hits={swing_pen}, open_boundary_loops={open_boundary_loops}', 'metrics': fp05_metrics}
    fp06_likely = spring_bone_count == 0 or fp06_metrics.get('dynamic_hair_share', 0.0) < 0.05
    predictions['FP-06_hair_secondary_and_collision'] = {'status': 'LIKELY' if fp06_likely else 'CLEAR', 'severity': 'MEDIUM' if fp06_likely else 'NONE', 'reason': f"spring_hair_bone_count={spring_bone_count}, dynamic_hair_share={fp06_metrics.get('dynamic_hair_share', 0.0):.3f}", 'metrics': fp06_metrics}
    si_delta = fp07_metrics.get('hip90_self_intersection_delta', 0)
    is_fused = fp07_metrics.get('is_single_fused_mesh', False)
    fp07_likely = is_fused and si_delta > 350
    predictions['FP-07_fused_garment_hip_self_intersection'] = {'status': 'LIKELY' if fp07_likely else 'CLEAR', 'severity': 'HIGH' if si_delta > 800 else 'MEDIUM' if fp07_likely else 'NONE', 'reason': f'deforming_mesh_count={len(meshes)}, hip90_self_intersection_delta=+{si_delta}', 'metrics': fp07_metrics}
    vert_slack = fp08_metrics.get('vertical_leg_slack_ratio', 0.0)
    sole_ratio = fp08_metrics.get('sole_lever_to_leg_ratio', 0.0)
    sole_swing = fp08_metrics.get('sole_pitch25_z_swing_mm', 0.0)
    fp08_likely = vert_slack < 0.0085 and sole_ratio > 0.48 or (vert_slack < 0.005 and sole_ratio > 0.35)
    predictions['FP-08_pelvis_reach_and_sole_bounce'] = {'status': 'LIKELY' if fp08_likely else 'CLEAR', 'severity': 'HIGH' if vert_slack < 0.0085 and sole_ratio > 0.48 else 'MEDIUM' if fp08_likely else 'NONE', 'reason': f"vertical_leg_slack_ratio={vert_slack:.5f} (rest_knee_flexion={fp08_metrics.get('rest_knee_flexion_deg', 0.0):.2f} deg), sole_lever_to_leg_ratio={sole_ratio:.3f} (>0.48 means oversized cartoon shoe lever causes per-frame min(sole_z) pelvis bounce), sole_pitch25_z_swing={sole_swing:.2f} mm", 'metrics': fp08_metrics}
    likely_list = [k for k, v in predictions.items() if v['status'] == 'LIKELY']
    clear_list = [k for k, v in predictions.items() if v['status'] == 'CLEAR']
    knee_clear = predictions['FP-01_outward_knees_coronal_splay']['status'] == 'CLEAR' and predictions['FP-02_sagittal_knee_pivot_offset']['status'] == 'CLEAR'
    return {'rig_id': rig_id, 'path': str(filepath).replace('\\', '/'), 'convention': convention, 'stature_m': round(stature_m, 4), 'vertex_count': len(rest_verts), 'triangle_count': len(rest_tris), 'deforming_mesh_count': len(meshes), 'elapsed_sec': round(time.time() - t0, 3), 'knee_preflight_pass': knee_clear, 'likely_failure_patterns': likely_list, 'clear_failure_patterns': clear_list, 'predictions': predictions}

def main() -> int:
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(description='CharForge Rig Pre-Flight Diagnostic Tool')
    parser.add_argument('--input', type=str, help='Path to rigged character (.blend, .glb, .vrm)')
    parser.add_argument('--name', type=str, default=None, help='Optional identifier for the rig')
    parser.add_argument('--batch', type=str, default=None, help="JSON file with list of {'name', 'path'} entries")
    parser.add_argument('--output', type=str, required=True, help='Output JSON report path')
    args = parser.parse_args(argv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.batch:
        batch_items = json.loads(Path(args.batch).read_text(encoding='utf-8'))
        results = []
        for item in batch_items:
            p = Path(item['path'])
            nm = item.get('name', p.stem)
            print(f'[preflight] Evaluating {nm} ({p}) ...', flush=True)
            rep = run_preflight_on_rig(p, name_override=nm)
            results.append(rep)
            print(f"[preflight] -> {nm}: knee_pass={rep.get('knee_preflight_pass')}, likely={rep.get('likely_failure_patterns')}", flush=True)
        payload = {'generated_by': 'toolkit/preflight/preflight.py',  'rig_count': len(results), 'results': results}
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'[preflight] Wrote batch report to {out_path}', flush=True)
        return 0
    if not args.input:
        raise SystemExit('Must provide either --input or --batch')
    rep = run_preflight_on_rig(Path(args.input), name_override=args.name)
    out_path.write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(json.dumps({'rig_id': rep.get('rig_id'), 'knee_preflight_pass': rep.get('knee_preflight_pass'), 'likely_failure_patterns': rep.get('likely_failure_patterns')}, indent=2), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

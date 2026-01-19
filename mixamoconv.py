# -*- coding: utf-8 -*-

'''
    Copyright (C) 2017-2018  Antonio 'GNUton' Aloisio
    Copyright (C) 2017-2018  Enzio Probst
    Copyright (C) 2025 MaksKraft

    Created by Enzio Probst, Edited and fixed by MaksKraft
    Patched for Blender 5.0+ (slotted actions/channelbags) by ChatGPT.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
'''

from pathlib import Path
import re
import logging

import bpy
from math import pi
from mathutils import Vector, Quaternion

# Blender 5.0+ helper (channelbags for slotted actions)
try:
    from bpy_extras import anim_utils  # Blender 5.0+ docs & community examples
except Exception:
    anim_utils = None

log = logging.getLogger(__name__)
# log.setLevel('DEBUG')


# -----------------------------------------------------------------------------
# F-CURVE / ACTION COMPAT LAYER (Blender 2.8+ legacy + Blender 5.0+ slotted)
# -----------------------------------------------------------------------------

def _get_anim_data(datablock):
    return getattr(datablock, "animation_data", None)

def _get_action(datablock):
    ad = _get_anim_data(datablock)
    if ad is None:
        return None
    return getattr(ad, "action", None)

def _get_action_slot(datablock):
    """
    In Blender 5.0+ animation_data has action_slot. In older versions it doesn't exist.
    """
    ad = _get_anim_data(datablock)
    if ad is None:
        return None
    return getattr(ad, "action_slot", None)

def _is_legacy_action(action):
    """
    Blender <=4.5 (and some 4.x) exposed action.fcurves directly.
    Blender 5.0 removed Action.fcurves; fcurves live in channelbags/slots.
    """
    return hasattr(action, "fcurves")

def _get_channelbag(action, datablock):
    """
    Returns the channelbag for (action, slot) if available (Blender 5.0+),
    otherwise returns None.
    """
    if action is None:
        return None
    # If legacy, we won't use channelbags.
    if _is_legacy_action(action):
        return None

    if anim_utils is None:
        # Blender 5.0+ should have it, but just in case:
        raise RuntimeError("Blender 5.0+ detected (no action.fcurves), but bpy_extras.anim_utils is unavailable")

    slot = _get_action_slot(datablock)
    if slot is None:
        # Some data might not have an explicit slot assigned; try best effort:
        # action_get_channelbag_for_slot requires a slot; without it it's ambiguous.
        # We'll fallback to scanning layers/strips if present.
        # This is defensive; typical object animation should have action_slot.
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for cb in getattr(strip, "channelbags", []):
                    # no reliable filter without slot, return first
                    return cb
        return None

    return anim_utils.action_get_channelbag_for_slot(action, slot)

def _iter_fcurves_for_datablock(datablock):
    """
    Iterate all fcurves for datablock's current action/slot.
    """
    action = _get_action(datablock)
    if action is None:
        return []
    if _is_legacy_action(action):
        return list(action.fcurves)

    cb = _get_channelbag(action, datablock)
    if cb is None:
        return []
    return list(cb.fcurves)

def _fcurve_find(datablock, data_path, index=0):
    """
    Find an fcurve for datablock's action/slot.
    """
    action = _get_action(datablock)
    if action is None:
        return None

    if _is_legacy_action(action):
        return action.fcurves.find(data_path, index=index)

    cb = _get_channelbag(action, datablock)
    if cb is None:
        return None
    # ActionChannelbagFCurves has .find(data_path, index=...)
    return cb.fcurves.find(data_path, index=index)

def _fcurve_remove(datablock, fcurve):
    """
    Remove an fcurve from datablock's action/slot.
    """
    if fcurve is None:
        return

    action = _get_action(datablock)
    if action is None:
        return

    if _is_legacy_action(action):
        action.fcurves.remove(fcurve)
        return

    cb = _get_channelbag(action, datablock)
    if cb is None:
        return
    cb.fcurves.remove(fcurve)

def _fcurve_ensure(action, datablock, data_path, index=0, group_name=None):
    """
    Blender 5.0+: Use Action.fcurve_ensure_for_datablock().
    Blender legacy: Find; if missing, create via drivers not desired here.
    We'll return existing curve on legacy.
    """
    if action is None:
        return None

    if hasattr(action, "fcurve_ensure_for_datablock"):
        # Blender 5.0 signature supports index= and group_name=
        kwargs = {"datablock": datablock, "data_path": data_path, "index": index}
        if group_name is not None:
            kwargs["group_name"] = group_name
        return action.fcurve_ensure_for_datablock(**kwargs)

    # Legacy fallback
    return _fcurve_find(datablock, data_path, index=index)


# -----------------------------------------------------------------------------
# ORIGINAL LOGIC
# -----------------------------------------------------------------------------

def remove_namespace(s=''):
    """Removing namespaces from strings, objects or armature bones."""
    if isinstance(s, str):
        i = re.search(r"[:_]", s[::-1])
        if i:
            return s[-(i.start())::]
        return s

    if isinstance(s, bpy.types.Object):
        if s.type == 'ARMATURE':
            for bone in s.data.bones:
                bone.name = remove_namespace(bone.name)
        s.name = remove_namespace(s.name)
        return 1

    return -1


def rename_bones(s='', t='unreal'):
    """Renaming the armature bones to a target skeleton."""
    unreal = {
        'root': 'Root',
        'Hips': 'Pelvis',
        'Spine': 'spine_01',
        'Spine1': 'spine_02',
        'Spine2': 'spine_03',
        'LeftShoulder': 'clavicle_l',
        'LeftArm': 'upperarm_l',
        'LeftForeArm': 'lowerarm_l',
        'LeftHand': 'hand_l',
        'RightShoulder': 'clavicle_r',
        'RightArm': 'upperarm_r',
        'RightForeArm': 'lowerarm_r',
        'RightHand': 'hand_r',
        'Neck1': 'neck_01',
        'Neck': 'neck_01',
        'Head': 'head',
        'LeftUpLeg': 'thigh_l',
        'LeftLeg': 'calf_l',
        'LeftFoot': 'foot_l',
        'RightUpLeg': 'thigh_r',
        'RightLeg': 'calf_r',
        'RightFoot': 'foot_r',
        'LeftHandIndex1': 'index_01_l',
        'LeftHandIndex2': 'index_02_l',
        'LeftHandIndex3': 'index_03_l',
        'LeftHandMiddle1': 'middle_01_l',
        'LeftHandMiddle2': 'middle_02_l',
        'LeftHandMiddle3': 'middle_03_l',
        'LeftHandPinky1': 'pinky_01_l',
        'LeftHandPinky2': 'pinky_02_l',
        'LeftHandPinky3': 'pinky_03_l',
        'LeftHandRing1': 'ring_01_l',
        'LeftHandRing2': 'ring_02_l',
        'LeftHandRing3': 'ring_03_l',
        'LeftHandThumb1': 'thumb_01_l',
        'LeftHandThumb2': 'thumb_02_l',
        'LeftHandThumb3': 'thumb_03_l',
        'RightHandIndex1': 'index_01_r',
        'RightHandIndex2': 'index_02_r',
        'RightHandIndex3': 'index_03_r',
        'RightHandMiddle1': 'middle_01_r',
        'RightHandMiddle2': 'middle_02_r',
        'RightHandMiddle3': 'middle_03_r',
        'RightHandPinky1': 'pinky_01_r',
        'RightHandPinky2': 'pinky_02_r',
        'RightHandPinky3': 'pinky_03_r',
        'RightHandRing1': 'ring_01_r',
        'RightHandRing2': 'ring_02_r',
        'RightHandRing3': 'ring_03_r',
        'RightHandThumb1': 'thumb_01_r',
        'RightHandThumb2': 'thumb_02_r',
        'RightHandThumb3': 'thumb_03_r',
        'LeftToeBase': 'ball_l',
        'RightToeBase': 'ball_r'
    }
    schema = {'unreal': unreal}

    if isinstance(s, str):
        i = schema[t].get(s)
        if i:
            return i
        log.warning('WARNING %s bone is missing', s)
        return s

    if isinstance(s, bpy.types.Object):
        if s.type == 'ARMATURE':
            for bone in s.data.bones:
                bone.name = rename_bones(remove_namespace(bone.name))
        s.name = rename_bones(s.name)
        return 1

    return -1


def key_all_bones(armature, frame_range=(1, 2)):
    """Sets keys for all Bones in frame_range."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    for i in range(*frame_range):
        bpy.context.scene.frame_current = i
        bpy.ops.anim.keyframe_insert_menu(type='BUILTIN_KSI_LocRot')
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_restoffset(armature, hipbone, restoffset):
    """Apply restoffset to rig and correct hip animation."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.transform.translate(
        value=restoffset,
        constraint_axis=(False, False, False),
        orient_type='GLOBAL',
        mirror=False,
        use_proportional_edit=False,
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    # apply restoffset to animation of hip
    action = _get_action(armature)
    if action is None:
        return 1

    restoffset_local = (restoffset[0], restoffset[2], -restoffset[1])
    path = f'pose.bones["{hipbone.name}"].location'
    for axis in range(3):
        fcurve = _fcurve_find(armature, path, index=axis)
        if fcurve is None:
            # ensure curve exists for safety (Blender 5.0+)
            fcurve = _fcurve_ensure(action, armature, path, index=axis)
        if fcurve is None:
            continue

        for kpi in range(len(fcurve.keyframe_points)):
            fcurve.keyframe_points[kpi].co.y -= restoffset_local[axis] / armature.scale.x
    return 1


def apply_kneefix(armature, offset, bonenames=('RightUpLeg', 'LeftUpLeg')):
    """Workaround for flickering knees after export."""
    if bpy.context.scene.mixamo.b_unreal_bones:
        bonenames = ("calf_r", "calf_l")

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    for name in bonenames:
        armature.data.edit_bones[name].select_tail = True
    bpy.ops.transform.translate(value=offset, use_proportional_edit=False, release_confirm=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    return 1


def get_all_quaternion_curves(obj):
    """Returns all quaternion fcurves of object/bones packed in a tuple per object/bone."""
    action = _get_action(obj)
    if action is None:
        return

    # object rotation_quaternion
    dp = 'rotation_quaternion'
    c0 = _fcurve_find(obj, dp, 0)
    if c0:
        yield (_fcurve_find(obj, dp, 0),
               _fcurve_find(obj, dp, 1),
               _fcurve_find(obj, dp, 2),
               _fcurve_find(obj, dp, 3))

    if obj.type == 'ARMATURE':
        for bone in obj.pose.bones:
            data_path = f'pose.bones["{bone.name}"].rotation_quaternion'
            c = _fcurve_find(obj, data_path, 0)
            if c:
                yield (_fcurve_find(obj, data_path, 0),
                       _fcurve_find(obj, data_path, 1),
                       _fcurve_find(obj, data_path, 2),
                       _fcurve_find(obj, data_path, 3))


def quaternion_cleanup(obj, prevent_flips=True, prevent_inverts=True):
    """Fixes signs in quaternion fcurves swapping from one frame to another."""
    for curves in get_all_quaternion_curves(obj):
        if not curves or any(c is None for c in curves):
            continue

        start = int(min((curves[i].keyframe_points[0].co.x for i in range(4) if curves[i].keyframe_points)))
        end = int(max((curves[i].keyframe_points[-1].co.x for i in range(4) if curves[i].keyframe_points)))

        for curve in curves:
            for i in range(start, end):
                curve.keyframe_points.insert(i, curve.evaluate(i)).interpolation = 'LINEAR'

        zipped = list(zip(
            curves[0].keyframe_points,
            curves[1].keyframe_points,
            curves[2].keyframe_points,
            curves[3].keyframe_points
        ))

        for i in range(1, len(zipped)):
            if prevent_flips:
                rot_prev = Quaternion((zipped[i - 1][j].co.y for j in range(4)))
                rot_cur = Quaternion((zipped[i][j].co.y for j in range(4)))
                diff = rot_prev.rotation_difference(rot_cur)
                if abs(diff.angle - pi) < 0.5:
                    rot_cur.rotate(Quaternion(diff.axis, pi))
                    for j in range(4):
                        zipped[i][j].co.y = rot_cur[j]

            if prevent_inverts:
                change_amount = 0.0
                for j in range(4):
                    change_amount += abs(zipped[i - 1][j].co.y - zipped[i][j].co.y)
                if change_amount > 1.0:
                    for j in range(4):
                        zipped[i][j].co.y *= -1.0


def apply_foot_bone_workaround(armature, bonenames=('RightToeBase', 'LeftToeBase')):
    """Workaround for the twisting of the foot bones in some skeletons."""
    if bpy.context.scene.mixamo.b_unreal_bones:
        bonenames = ("ball_r", "ball_l")

    bpy.ops.object.mode_set(mode='EDIT')
    for name in bonenames:
        armature.data.edit_bones[name].roll = pi


class Status:
    def __init__(self, msg, status_type='default'):
        self.msg = msg
        self.status_type = status_type

    def __str__(self):
        return str(self.msg)


def hip_to_root(armature, use_x=True, use_y=True, use_z=True, on_ground=True, use_rotation=True, scale=1.0,
               restoffset=(0, 0, 0), hipname='', fixbind=True, apply_rotation=True, apply_scale=False,
               quaternion_clean_pre=True, quaternion_clean_post=True, foot_bone_workaround=False):
    """Bake hip motion to RootMotion in Mixamo rigs."""
    yield Status("starting hip_to_root")

    root = armature
    root.name = "root"
    root.rotation_mode = 'QUATERNION'

    if root.animation_data is None or root.animation_data.action is None:
        raise ValueError("Armature has no animation action to process")

    action = root.animation_data.action
    framerange = action.frame_range

    hips = None
    for hn in ('Hips', 'mixamorig:Hips', 'mixamorig_Hips', 'Pelvis', hipname):
        hips = root.pose.bones.get(hn)
        if hips is not None:
            break

    if hips is None:
        log.warning('WARNING I have not found any hip bone for %s and the conversion is stopping here', root.pose.bones)
        raise ValueError("no hips found")
    yield Status("hips found")

    key_all_bones(root, (1, 2))

    # Scale by ScaleFactor (remove existing scale curves first)
    if scale != 1.0:
        for i in range(3):
            fcurve = _fcurve_find(root, 'scale', index=i)
            if fcurve is not None:
                _fcurve_remove(root, fcurve)
        root.scale *= scale
        yield Status("scaling")

    # fix quaternion sign swapping
    if quaternion_clean_pre:
        quaternion_cleanup(root)
        yield Status("quaternion clean pre")

    if foot_bone_workaround:
        apply_foot_bone_workaround(armature)

    # apply restoffset to restpose and correct animation
    apply_restoffset(root, hips, restoffset)
    yield Status("restoffset")

    hiplocation_world = root.matrix_local @ hips.bone.head
    z_offset = hiplocation_world[2]

    # Create helper to bake the root motion
    rootbaker = bpy.data.objects.new(name="rootbaker", object_data=None)
    rootbaker.rotation_mode = 'QUATERNION'

    if use_z:
        c_rootbaker_copy_z_loc = rootbaker.constraints.new(type='COPY_LOCATION')
        c_rootbaker_copy_z_loc.name = "Copy Z_Loc"
        c_rootbaker_copy_z_loc.target = root
        c_rootbaker_copy_z_loc.subtarget = hips.name
        c_rootbaker_copy_z_loc.use_x = False
        c_rootbaker_copy_z_loc.use_y = False
        c_rootbaker_copy_z_loc.use_z = True
        c_rootbaker_copy_z_loc.use_offset = True
        if on_ground:
            rootbaker.location[2] = -z_offset
            c_on_ground = rootbaker.constraints.new(type='LIMIT_LOCATION')
            c_on_ground.name = "On Ground"
            c_on_ground.use_min_z = True

    c_rootbaker_copy_loc = rootbaker.constraints.new(type='COPY_LOCATION')
    c_rootbaker_copy_loc.use_x = use_x
    c_rootbaker_copy_loc.use_y = use_y
    c_rootbaker_copy_loc.use_z = False
    c_rootbaker_copy_loc.target = root
    c_rootbaker_copy_loc.subtarget = hips.name

    c_rootbaker_copy_rot = rootbaker.constraints.new(type='COPY_ROTATION')
    c_rootbaker_copy_rot.target = root
    c_rootbaker_copy_rot.subtarget = hips.name
    c_rootbaker_copy_rot.use_y = False
    c_rootbaker_copy_rot.use_x = False
    c_rootbaker_copy_rot.use_z = use_rotation

    bpy.context.scene.collection.objects.link(rootbaker)
    yield Status("rootbaker created")

    bpy.ops.object.select_all(action='DESELECT')
    rootbaker.select_set(True)
    bpy.context.view_layer.objects.active = rootbaker

    bpy.ops.nla.bake(
        frame_start=int(framerange[0]), frame_end=int(framerange[1]),
        step=1, only_selected=True, visual_keying=True,
        clear_constraints=True, clear_parents=False,
        use_current_action=False, bake_types={'OBJECT'}
    )
    yield Status("rootbaker baked")
    quaternion_cleanup(rootbaker)
    yield Status("rootbaker quat_cleanup")

    # Create helper to bake hip motion in Worldspace
    hipsbaker = bpy.data.objects.new(name="hipsbaker", object_data=None)
    hipsbaker.rotation_mode = 'QUATERNION'

    c_hipsbaker_copy_loc = hipsbaker.constraints.new(type='COPY_LOCATION')
    c_hipsbaker_copy_loc.target = root
    c_hipsbaker_copy_loc.subtarget = hips.name

    c_hipsbaker_copy_rot = hipsbaker.constraints.new(type='COPY_ROTATION')
    c_hipsbaker_copy_rot.target = root
    c_hipsbaker_copy_rot.subtarget = hips.name

    bpy.context.scene.collection.objects.link(hipsbaker)
    yield Status("hipsbaker created")

    bpy.ops.object.select_all(action='DESELECT')
    hipsbaker.select_set(True)
    bpy.context.view_layer.objects.active = hipsbaker

    bpy.ops.nla.bake(
        frame_start=int(framerange[0]), frame_end=int(framerange[1]),
        step=1, only_selected=True, visual_keying=True,
        clear_constraints=True, clear_parents=False,
        use_current_action=False, bake_types={'OBJECT'}
    )
    yield Status("hipsbaker baked")
    quaternion_cleanup(hipsbaker)
    yield Status("hipsbaker quatCleanup")

    # select armature
    bpy.ops.object.select_all(action='DESELECT')
    root.select_set(True)
    bpy.context.view_layer.objects.active = root

    if apply_rotation or apply_scale:
        bpy.ops.object.transform_apply(location=False, rotation=apply_rotation, scale=apply_scale)
        yield Status("apply transform")

    # Bake Root motion to Armature (root)
    c_root_copy_loc = root.constraints.new(type='COPY_LOCATION')
    c_root_copy_loc.target = rootbaker

    c_root_copy_rot = root.constraints.new(type='COPY_ROTATION')
    c_root_copy_rot.target = rootbaker
    c_root_copy_rot.use_offset = True
    yield Status("root constrained to rootbaker")

    bpy.ops.nla.bake(
        frame_start=int(framerange[0]), frame_end=int(framerange[1]),
        step=1, only_selected=True, visual_keying=True,
        clear_constraints=True, clear_parents=False,
        use_current_action=True, bake_types={'OBJECT'}
    )

    yield Status("rootbaker baked back")
    quaternion_cleanup(root)
    yield Status("root quaternion cleanup")
    hipsbaker.select_set(False)

    bpy.ops.object.mode_set(mode='POSE')
    hips.bone.select = True
    root.data.bones.active = hips.bone

    c_hips_copy_loc = hips.constraints.new(type='COPY_LOCATION')
    c_hips_copy_loc.target = hipsbaker
    c_hips_copy_rot = hips.constraints.new(type='COPY_ROTATION')
    c_hips_copy_rot.target = hipsbaker
    yield Status("hips constrained to hipsbaker")

    bpy.ops.nla.bake(
        frame_start=int(framerange[0]), frame_end=int(framerange[1]),
        step=1, only_selected=True, visual_keying=True,
        clear_constraints=True, clear_parents=False,
        use_current_action=True, bake_types={'POSE'}
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    yield Status("hipsbaker baked back")

    if quaternion_clean_post:
        quaternion_cleanup(root)
        yield Status("root quaternion cleanup")

    # Delete helpers
    if hipsbaker.animation_data and hipsbaker.animation_data.action:
        bpy.data.actions.remove(hipsbaker.animation_data.action)
    if rootbaker.animation_data and rootbaker.animation_data.action:
        bpy.data.actions.remove(rootbaker.animation_data.action)

    bpy.data.objects.remove(hipsbaker)
    bpy.data.objects.remove(rootbaker)
    yield Status("bakers deleted")

    # bind armature to dummy mesh if it doesn't have any
    if fixbind:
        bindmesh = None
        for child in root.children:
            for mod in child.modifiers:
                if mod.type == 'ARMATURE' and mod.object == root:
                    bindmesh = child
                    break

        if bindmesh is None:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.mesh.primitive_plane_add(size=1, align='WORLD', enter_editmode=False, location=(0, 0, 0))
            binddummy = bpy.context.object
            binddummy.name = 'binddummy'
            root.select_set(True)
            bpy.context.view_layer.objects.active = root
            bpy.ops.object.parent_set(type='ARMATURE')
            yield Status("binddummy created")
        elif apply_rotation or apply_scale:
            bindmesh.select_set(True)
            bpy.context.view_layer.objects.active = bindmesh
            bpy.ops.object.transform_apply(location=False, rotation=apply_rotation, scale=apply_scale)
            yield Status("apply transform to bindmesh")

    return 1


def batch_hip_to_root(source_dir, dest_dir, use_x=True, use_y=True, use_z=True, on_ground=True, use_rotation=True, scale=1.0,
                      restoffset=(0, 0, 0), hipname='', fixbind=True, apply_rotation=True, apply_scale=False,
                      b_remove_namespace=True, b_unreal_bones=False, add_leaf_bones=False, knee_offset=(0, 0, 0),
                      ignore_leaf_bones=True, automatic_bone_orientation=True, quaternion_clean_pre=True,
                      quaternion_clean_post=True, foot_bone_workaround=False):
    """Batch Convert Mixamo Rigs."""

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1

    numfiles = 0
    for file in source_dir.iterdir():
        if not file.is_file():
            continue

        file_ext = file.suffix.lower()
        file_loader = {
            ".fbx": lambda filename: bpy.ops.import_scene.fbx(
                filepath=str(filename),
                axis_forward='-Z',
                axis_up='Y',
                directory="",
                filter_glob="*.fbx",
                ui_tab='MAIN',
                use_manual_orientation=False,
                global_scale=1.0,
                bake_space_transform=False,
                use_custom_normals=True,
                use_image_search=True,
                use_alpha_decals=False,
                decal_offset=0.0,
                use_anim=True,
                anim_offset=1.0,
                use_custom_props=True,
                use_custom_props_enum_as_string=True,
                ignore_leaf_bones=ignore_leaf_bones,
                force_connect_children=False,
                automatic_bone_orientation=automatic_bone_orientation,
                primary_bone_axis='Y',
                secondary_bone_axis='X',
                use_prepost_rot=True
            ),
            ".dae": lambda filename: bpy.ops.wm.collada_import(
                filepath=str(filename),
                filter_blender=False,
                filter_backup=False,
                filter_image=False,
                filter_movie=False,
                filter_python=False,
                filter_font=False,
                filter_sound=False,
                filter_text=False,
                filter_btx=False,
                filter_collada=True,
                filter_alembic=False,
                filter_folder=True,
                filter_blenlib=False,
                filemode=8,
                display_type='DEFAULT',
                sort_method='FILE_SORT_ALPHA',
                import_units=False,
                fix_orientation=True,
                find_chains=True,
                auto_connect=True,
                min_chain_length=0
            )
        }

        if file_ext not in file_loader:
            continue

        numfiles += 1

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=True)

        # remove all datablocks
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material, do_unlink=True)
        for action in list(bpy.data.actions):
            bpy.data.actions.remove(action, do_unlink=True)

        bpy.ops.outliner.orphans_purge(do_recursive=True)

        for data_list in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.armatures,
            bpy.data.actions,
            bpy.data.materials,
            bpy.data.textures,
            bpy.data.images,
            bpy.data.node_groups,
            bpy.data.curves,
            bpy.data.cameras,
            bpy.data.lights,
            bpy.data.collections,
            bpy.data.shape_keys,
        ):
            for datablock in list(data_list):
                if datablock.users == 0:
                    data_list.remove(datablock)

        # import file
        file_loader[file_ext](file)

        # namespace removal / rename
        if b_remove_namespace:
            for obj in bpy.context.selected_objects:
                remove_namespace(obj)
        elif b_unreal_bones:
            for obj in bpy.context.selected_objects:
                rename_bones(obj, 'unreal')

        def getArmature(objects):
            for a in objects:
                if a.type == 'ARMATURE':
                    return a
            raise TypeError("No Armature found")

        armature = getArmature(bpy.context.selected_objects)

        # do hip to Root conversion
        try:
            for _step in hip_to_root(
                armature,
                use_x=use_x, use_y=use_y, use_z=use_z,
                on_ground=on_ground, use_rotation=use_rotation,
                scale=scale, restoffset=restoffset, hipname=hipname,
                fixbind=fixbind, apply_rotation=apply_rotation,
                apply_scale=apply_scale,
                quaternion_clean_pre=quaternion_clean_pre,
                quaternion_clean_post=quaternion_clean_post,
                foot_bone_workaround=foot_bone_workaround
            ):
                pass
        except Exception as e:
            log.error("ERROR hip_to_root raised %s when processing %s" % (str(e), file.name))
            return -1

        if Vector(knee_offset).length > 0.0:
            apply_kneefix(
                armature, knee_offset,
                bonenames=bpy.context.scene.mixamo.knee_bones.split(',')
            )

        # remove newly created orphan actions
        if armature.animation_data and armature.animation_data.action:
            keep_action = armature.animation_data.action
            for act in list(bpy.data.actions):
                if act != keep_action:
                    bpy.data.actions.remove(act, do_unlink=True)

        # export file
        output_file = dest_dir.joinpath(file.stem + ".fbx")
        bpy.ops.export_scene.fbx(
            filepath=str(output_file),
            use_selection=False,
            apply_unit_scale=False,
            add_leaf_bones=add_leaf_bones,
            axis_forward='-Z',
            axis_up='Y',
            mesh_smooth_type='FACE'
        )

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    return numfiles


if __name__ == "__main__":
    print("mixamoconv Hello.")

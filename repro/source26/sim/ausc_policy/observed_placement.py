"""Pure rigid coordinate transform for already verified observed SMPL arrays.

Point-SMPL source coordinates may be camera coordinates. No source gravity or
floor is inferred from the pelvis, neck or anatomical frame. A caller supplies
an explicit unit plane normal and a ``floor_normal_assuming_level`` binding.
The hashes are syntactically checked, not read or authenticated by this kernel;
the level-plane assumption remains unverified and is never called gravity.

A minimal proper rotation maps the supplied normal to world +Z. The caller's
explicit yaw is then applied around world +Z, and one translation maps the
source anchor to the world anchor. All points receive the same transform;
normals receive its rotation only. No joints are authored; no pose/torso
leveling, scaling or scene/support/placement logic is performed.

There is no model/dataset/mesh/file I/O, simulator, GPU or CLI here. Reported
invariance is purely numerical. Support, placement, IK, physics and training
qualification always remain false.
"""
from collections.abc import Mapping
from copy import deepcopy
import re

import numpy as np

SCHEMA = 'observed_source_plane_rigid_transform_v1'
# Standard first-24 SMPL hierarchy, used only to report rigid bone invariance.
SMPL_PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])


def _numeric(value,name):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(name+' must contain finite numeric values')
    with np.errstate(over='ignore',invalid='ignore'):
        result=raw.astype(np.float64,copy=True)
    if not np.isfinite(result).all():raise ValueError(name+' must be representable in float64')
    return result


def _vector(value,name):
    result=_numeric(value,name)
    if result.shape!=(3,):raise ValueError(name+' must have shape (3,)')
    return result


def _points(value,name,minimum=1):
    result=_numeric(value,name)
    if result.ndim!=2 or result.shape[1]!=3 or len(result)<minimum:
        raise ValueError(name+' must be finite Nx3 points')
    return result


def _binding(value):
    required={'method','source_plane_sha256','source_report_sha256'}
    if not isinstance(value,Mapping) or not required<=set(value):
        raise ValueError('Explicit source floor binding with method and plane/report SHA256 required')
    if value['method']!='floor_normal_assuming_level':
        raise ValueError('Source floor binding method must be floor_normal_assuming_level')
    for key in ('source_plane_sha256','source_report_sha256'):
        if not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{64}',value[key]):
            raise ValueError('Malformed source floor binding SHA256: '+key)
    if set(value)-required-{'source_frame'}:
        raise ValueError('Unsupported source floor binding fields')
    if 'source_frame' in value and (not isinstance(value['source_frame'],str) or not value['source_frame'].strip()):
        raise ValueError('Source floor binding frame must be explicit nonempty text')
    return deepcopy(dict(value))


def _cross_matrix(axis):
    x,y,z=axis
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def _minimal_to_world_z(normal):
    target=np.array([0.,0.,1.]);cross=np.cross(normal,target)
    sine=float(np.linalg.norm(cross));cosine=float(np.clip(normal@target,-1.,1.))
    if sine>1e-12:
        axis=cross/sine
        # Axis-angle form avoids division by 1+cos(theta) near antiparallel.
        return cosine*np.eye(3)+(1-cosine)*np.outer(axis,axis)+sine*_cross_matrix(axis)
    if cosine>0:return np.eye(3)
    # At exactly pi every orthogonal axis is minimal. Pick a deterministic
    # least-aligned basis; this is a proper rotation, never a reflection.
    basis=np.eye(3)[int(np.argmin(np.abs(normal)))]
    axis=np.cross(normal,basis);axis/=np.linalg.norm(axis)
    return 2*np.outer(axis,axis)-np.eye(3)


def _angle_degrees(vector,normal):
    length=float(np.linalg.norm(vector))
    if not np.isfinite(length) or length<1e-12:raise ValueError('Degenerate/nonfinite torso vector')
    return float(np.degrees(np.arccos(np.clip(vector@normal/length,-1.,1.))))


def transform_observed_arrays(vertices,joints,anchor_positions,anchor_normals,*,source_floor_normal,
                              world_yaw_rad,source_anchor_m,world_anchor_m,source_floor_binding):
    """Return fresh transformed arrays, world-from-source matrix and invariance.

    Joints follow the SMPL convention (at least 24, pelvis=0, neck=12). Unit
    source-plane normals are checked to 1e-8; admitted serialization roundoff
    is removed for constructing the rotation. Anchor normals must already be
    unit to 1e-7 and are rotated without normalization or translation. Inputs
    may be read-only; no returned mutable array aliases the supplied arrays.
    """
    provenance=_binding(source_floor_binding)
    source=_vector(source_floor_normal,'source_floor_normal')
    length=float(np.linalg.norm(source))
    if not np.isfinite(length) or abs(length-1.)>1e-8:
        raise ValueError('source_floor_normal must be an explicit unit normal')
    normal=source/length
    source_anchor=_vector(source_anchor_m,'source_anchor_m')
    world_anchor=_vector(world_anchor_m,'world_anchor_m')
    if (isinstance(world_yaw_rad,(bool,np.bool_)) or
            not isinstance(world_yaw_rad,(int,float,np.integer,np.floating)) or not np.isfinite(world_yaw_rad)):
        raise ValueError('world_yaw_rad must be finite numeric')
    yaw=float(world_yaw_rad)
    arrays=dict(vertices=_points(vertices,'vertices'),joints=_points(joints,'joints',24),
                anchor_positions=_points(anchor_positions,'anchor_positions'))
    normals=_points(anchor_normals,'anchor_normals')
    if normals.shape!=arrays['anchor_positions'].shape or not np.allclose(
            np.linalg.norm(normals,axis=1),1.,atol=1e-7,rtol=0):
        raise ValueError('anchor_normals must match anchor_positions and already be unit')
    c,s=np.cos(yaw),np.sin(yaw)
    yaw_rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    rotation=yaw_rotation@_minimal_to_world_z(normal)
    translation=world_anchor-rotation@source_anchor
    world_from_source=np.eye(4);world_from_source[:3,:3]=rotation;world_from_source[:3,3]=translation
    transformed={key:value@rotation.T+translation for key,value in arrays.items()}
    transformed['anchor_normals']=normals@rotation.T
    if not all(np.isfinite(value).all() for value in (*transformed.values(),world_from_source)):
        raise ValueError('Rigid transform produced nonfinite coordinates')
    children=np.arange(1,24);parents=SMPL_PARENTS[1:]
    before_bones=np.linalg.norm(arrays['joints'][children]-arrays['joints'][parents],axis=1)
    after_bones=np.linalg.norm(transformed['joints'][children]-transformed['joints'][parents],axis=1)
    before_angle=_angle_degrees(arrays['joints'][12]-arrays['joints'][0],normal)
    after_angle=_angle_degrees(transformed['joints'][12]-transformed['joints'][0],np.array([0.,0.,1.]))
    invariance=dict(rotation_determinant=float(np.linalg.det(rotation)),
        rotation_orthogonality_max_abs=float(np.max(np.abs(rotation.T@rotation-np.eye(3)))),
        source_normal_alignment_error=float(np.linalg.norm(rotation@normal-[0.,0.,1.])),
        anchor_mapping_error_m=float(np.linalg.norm(rotation@source_anchor+translation-world_anchor)),
        source_bone_lengths_m=before_bones.tolist(),world_bone_lengths_m=after_bones.tolist(),
        max_bone_length_error_m=float(np.max(np.abs(before_bones-after_bones))),
        source_torso_to_reference_normal_deg=before_angle,world_torso_to_reference_normal_deg=after_angle,
        torso_angle_error_deg=abs(before_angle-after_angle),
        max_normal_norm_error=float(np.max(np.abs(np.linalg.norm(normals,axis=1)-
                                                      np.linalg.norm(transformed['anchor_normals'],axis=1)))),
        max_normal_reference_cosine_error=float(np.max(np.abs(normals@normal-transformed['anchor_normals'][:,2]))))
    if not np.isfinite([value for value in invariance.values() if isinstance(value,float)]).all():
        raise ValueError('Nonfinite rigid invariance measurements')
    return dict(schema=SCHEMA,**transformed,world_from_source=world_from_source,
        source_floor_normal=source,aligned_reference_normal_world=np.array([0.,0.,1.]),world_yaw_rad=yaw,
        source_anchor_m=source_anchor,world_anchor_m=world_anchor,source_floor_binding=provenance,
        reference_scope='Explicit floor_normal_assuming_level provenance; no gravity or semantic floor verification',
        binding_validation='Method and hash syntax only; source artifact bytes are not read or authenticated',
        invariance=invariance,torso_leveling_applied=False,scale_applied=1.,
        placement_support_qualified=False,placement_qualified=False,ik_qualified=False,
        physics_qualified=False,training_eligible=False,source_floor_semantically_verified=False,
        source_floor_artifact_hashes_verified=False)

"""Pure synthetic rigid-transform tests; no datasets, meshes or native work."""
import importlib
from types import MappingProxyType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def module():
    assert importlib.util.find_spec('sim.ausc_policy.observed_placement') is not None, 'observed rigid transform kernel is missing'
    return importlib.import_module('sim.ausc_policy.observed_placement')


def binding():
    return dict(method='floor_normal_assuming_level',source_plane_sha256='a'*64,
                source_report_sha256='b'*64,source_frame='kinect_color_000')


def fixture():
    rotation=Rotation.from_euler('y',31,degrees=True).as_matrix()
    rng=np.random.default_rng(92)
    joints=rng.normal(0,.1,(24,3));joints[0]=[0,0,0]
    joints[12]=np.array([np.sin(np.radians(23)),0,np.cos(np.radians(23))])*.65
    vertices=rng.normal(0,.3,(15,3))
    normals=rng.normal(size=(4,3));normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    source_translation=np.array([1.2,-.8,2.4])
    return dict(vertices=vertices@rotation.T+source_translation,
        joints=joints@rotation.T+source_translation,
        anchor_positions=vertices[:4]@rotation.T+source_translation,
        anchor_normals=normals@rotation.T,source_floor_normal=rotation@[0.,0.,1.],
        world_yaw_rad=.4,source_anchor_m=source_translation,
        world_anchor_m=np.array([.3,.2,.45]),source_floor_binding=binding())


def transform(**overrides):
    args=fixture();args.update(overrides)
    return module().transform_observed_arrays(**args)


def test_nonvertical_source_plane_alignment_retains_observed_torso_tilt_and_lengths():
    args=fixture();result=module().transform_observed_arrays(**args)
    matrix=result['world_from_source'];rotation=matrix[:3,:3]
    np.testing.assert_allclose(rotation@args['source_floor_normal'],[0,0,1],atol=1e-12)
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
    assert np.linalg.det(rotation)==pytest.approx(1.,abs=1e-12)
    np.testing.assert_allclose(rotation@args['source_anchor_m']+matrix[:3,3],args['world_anchor_m'],atol=1e-12)
    for key in ('vertices','joints','anchor_positions'):
        np.testing.assert_allclose(result[key],args[key]@rotation.T+matrix[:3,3],atol=1e-12)
    torso=result['joints'][12]-result['joints'][0]
    assert np.degrees(np.arccos(torso[2]/np.linalg.norm(torso)))==pytest.approx(23.,abs=1e-10)
    metrics=result['invariance']
    assert metrics['source_torso_to_reference_normal_deg']==pytest.approx(23.,abs=1e-10)
    assert metrics['world_torso_to_reference_normal_deg']==pytest.approx(23.,abs=1e-10)
    assert metrics['max_bone_length_error_m']<1e-12
    assert metrics['torso_angle_error_deg']<1e-10
    assert metrics['max_normal_norm_error']<1e-12
    assert metrics['max_normal_reference_cosine_error']<1e-12
    assert not result['torso_leveling_applied']
    assert not any(result[key] for key in ('placement_support_qualified','placement_qualified',
        'ik_qualified','physics_qualified','training_eligible','source_floor_semantically_verified',
        'source_floor_artifact_hashes_verified'))
    assert result['source_floor_binding']==binding()
    assert 'assuming' in result['reference_scope'] and 'gravity' in result['reference_scope']


def test_source_already_world_up_applies_exactly_explicit_world_yaw():
    yaw=.73;result=transform(source_floor_normal=np.array([0.,0.,1.]),world_yaw_rad=yaw)
    expected=Rotation.from_rotvec([0,0,yaw]).as_matrix()
    np.testing.assert_allclose(result['world_from_source'][:3,:3],expected,atol=1e-12)


def test_minimal_plane_rotation_has_no_hidden_yaw_when_caller_yaw_is_zero():
    args=fixture();result=module().transform_observed_arrays(**dict(args,world_yaw_rad=0.))
    np.testing.assert_allclose(result['world_from_source'][:3,:3],Rotation.from_euler('y',-31,degrees=True).as_matrix(),atol=1e-12)


@pytest.mark.parametrize('normal',[np.array([0.,0.,-1.]),np.array([1e-9,0.,-1.])])
def test_antiparallel_and_near_antiparallel_normals_get_proper_deterministic_rotation(normal):
    normal=normal/np.linalg.norm(normal)
    first=transform(source_floor_normal=normal,world_yaw_rad=0.)
    second=transform(source_floor_normal=normal,world_yaw_rad=0.)
    rotation=first['world_from_source'][:3,:3]
    np.testing.assert_allclose(rotation@normal,[0,0,1],atol=1e-12)
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
    assert np.linalg.det(rotation)==pytest.approx(1.,abs=1e-12)
    np.testing.assert_array_equal(first['world_from_source'],second['world_from_source'])


def test_anchor_normals_rotate_without_translation_or_renormalization():
    args=fixture();first=module().transform_observed_arrays(**args)
    second=module().transform_observed_arrays(**dict(args,world_anchor_m=np.array([100.,-40.,2.])))
    rotation=first['world_from_source'][:3,:3]
    np.testing.assert_allclose(first['anchor_normals'],args['anchor_normals']@rotation.T,atol=1e-12)
    np.testing.assert_array_equal(first['anchor_normals'],second['anchor_normals'])
    np.testing.assert_allclose(second['anchor_positions']-first['anchor_positions'],np.tile(np.array([100.,-40.,2.])-args['world_anchor_m'],(4,1)),atol=1e-12)


def test_readonly_inputs_and_binding_are_preserved_and_outputs_do_not_alias():
    args=fixture();saved={key:np.array(value,copy=True) for key,value in args.items() if isinstance(value,np.ndarray)}
    for value in args.values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    args['source_floor_binding']=MappingProxyType(binding())
    result=module().transform_observed_arrays(**args)
    for key,value in saved.items():np.testing.assert_array_equal(args[key],value)
    for key in ('vertices','joints','anchor_positions','anchor_normals'):
        assert not np.shares_memory(result[key],args[key])
        result[key][:]=0
        np.testing.assert_array_equal(args[key],saved[key])
    result['source_floor_binding']['method']='changed-output'
    assert args['source_floor_binding']['method']=='floor_normal_assuming_level'


@pytest.mark.parametrize('value',[None,{},dict(method='torso_inferred',source_plane_sha256='a'*64,source_report_sha256='b'*64),
    dict(method='floor_normal_assuming_level',source_plane_sha256='a'*63,source_report_sha256='b'*64),
    dict(method='floor_normal_assuming_level',source_plane_sha256='z'*64,source_report_sha256='b'*64),
    dict(method='floor_normal_assuming_level',source_plane_sha256='a'*64),
    dict(method='floor_normal_assuming_level',source_plane_sha256='a'*64,source_report_sha256=np.nan)])
def test_missing_or_malformed_floor_provenance_is_rejected(value):
    with pytest.raises(ValueError,match='binding|SHA|method'):transform(source_floor_binding=value)


@pytest.mark.parametrize('key,value',[
    ('source_floor_normal',[0,0,0]),('source_floor_normal',[0,0,2]),('source_floor_normal',[0,np.nan,1]),
    ('source_floor_normal',[0,1]),('world_yaw_rad',np.inf),('world_yaw_rad',True),
    ('source_anchor_m',[0,np.nan,1]),('world_anchor_m',[0,0]),
    ('vertices',np.full((3,3),np.inf)),('joints',np.zeros((23,3))),
    ('anchor_positions',np.zeros((5,3))),('anchor_normals',np.zeros((4,3))),
])
def test_invalid_numeric_inputs_cannot_generate_rigid_transform(key,value):
    with pytest.raises(ValueError):transform(**{key:value})


def test_degenerate_torso_cannot_claim_angle_invariance():
    args=fixture();args['joints'][12]=args['joints'][0]
    with pytest.raises(ValueError,match='torso'):module().transform_observed_arrays(**args)

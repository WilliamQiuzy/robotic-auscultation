"""Tiny in-memory meshes; no FCL, IK, USD, disk assets or native work."""
from copy import deepcopy
import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.ausc_policy.geometry_epoch import GEOMETRY_ID,GEOMETRY_SOURCE_SHA256
from sim.ausc_policy.surface_anchors import evaluate_anchors

PARTS=('seat','leg0','leg1','leg2','leg3','rung_x0','rung_x1','rung_y0','rung_y1')


def module():
    assert importlib.util.find_spec('sim.ausc_policy.cpu_scene_variant') is not None, 'CPU scene variant builder is missing'
    return importlib.import_module('sim.ausc_policy.cpu_scene_variant')


def fixture():
    entries=[];arrays={}
    paths=['/World/Stool/'+name for name in PARTS]+['/World/Patient/body','/World/Patient/shirt','/World/Table/top','/World/Robot/link/collider']
    for index,path in enumerate(paths):
        key=f'c{index:03d}';v=np.array([[0.,0.,.4],[.1,0.,.4],[0.,.1,.4]])+[index*.01,0,0]
        arrays[key+'_v']=v;arrays[key+'_f']=np.array([[0,1,2]],np.int32)
        entries.append(dict(key=key,path=path,body='world',world_from_body=np.eye(4).tolist(),
                            approximation='none',bounds_local=[v.min(0).tolist(),v.max(0).tolist()]))
    metadata=dict(colliders=entries,allowed_body_pairs=[['/World/Robot/link1','/World/Robot/link2']],
        analytic_planes=[dict(path='/World/Ground/plane',normal=[0,0,1],point=[0,0,0])],
        unsupported=[dict(path='/World/Unsupported/thing',type='Torus')],custom_source='preserve this metadata')
    vertices=np.array([[1.123456789,2.,.501234567],[1.223456789,2.,.501234567],[1.123456789,2.1,.551234567]])
    faces=np.array([[0,1,2]],np.int64)
    anchors={side+str(i):dict(smpl_vertex_indices=[0,1,2],barycentric_weights=[.2,.3,.5]) for side in 'LR' for i in range(1,6)}
    bindings=dict(geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,freshly_generated=True,
        collision_geometry_sha256='a'*64,collision_arrays_sha256='b'*64,
        observed_placement_sha256='c'*64,anchor_inventory_sha256='d'*64)
    return dict(metadata=metadata,arrays=arrays,patient_vertices_world=vertices,patient_faces=faces,
                anchors=anchors,stool_xy_translation_m=np.array([.12,-.07]),source_bindings=bindings)


def build(**changes):
    args=fixture();args.update(changes)
    return module().build_cpu_scene_variant(**args)


def test_replaces_all_old_patient_parts_and_derives_targets_from_float32_mesh():
    args=fixture();result=module().build_cpu_scene_variant(**args)
    assert result['kind']=='cpu_collision_bundle_proposal'
    entries=result['metadata']['colliders'];patients=[row for row in entries if row['path'].startswith('/World/Patient/')]
    assert len(patients)==1 and patients[0]['path']=='/World/Patient/SMPLSurface'
    patient=patients[0];key=patient['key']
    assert patient['body']=='world' and patient['approximation']=='none'
    np.testing.assert_array_equal(patient['world_from_body'],np.eye(4))
    quantized=args['patient_vertices_world'].astype(np.float32)
    assert result['arrays'][key+'_v'].dtype==np.dtype('<f4')
    np.testing.assert_array_equal(result['arrays'][key+'_v'],quantized)
    np.testing.assert_array_equal(result['arrays'][key+'_f'],args['patient_faces'])
    np.testing.assert_array_equal(patient['bounds_local'],[quantized.min(0),quantized.max(0)])
    actual=evaluate_anchors(quantized,args['patient_faces'],args['anchors'])
    assert len(result['targets'])==10
    for target in result['targets']:
        np.testing.assert_array_equal(target['position_world_m'],actual[target['name']]['position_m'])
        np.testing.assert_array_equal(target['normal_world'],actual[target['name']]['normal'])
    old_keys=[row['key'] for row in args['metadata']['colliders'] if row['path'].startswith('/World/Patient/')]
    assert all(key+'_v' not in result['arrays'] and key+'_f' not in result['arrays'] for key in old_keys)
    for claim in ('patient_receipt','non_patient_scene_unchanged'):
        assert claim not in result and claim not in result['metadata']
    assert not any(result[key] for key in ('static_support_qualified','ik_qualified','physics_qualified',
                                          'training_eligible','source_artifact_hashes_verified','stool_height_tested'))


def test_all_nine_stool_parts_shift_world_xy_and_keep_height_other_objects_and_policy():
    args=fixture();result=module().build_cpu_scene_variant(**args)
    new={row['path']:row for row in result['metadata']['colliders']}
    for original in args['metadata']['colliders']:
        path,key=original['path'],original['key']
        if path.startswith('/World/Patient/'):continue
        if path.startswith('/World/Stool/'):
            translated=args['arrays'][key+'_v']+np.array([.12,-.07,0])
            np.testing.assert_array_equal(result['arrays'][key+'_v'],translated)
            np.testing.assert_array_equal(result['arrays'][key+'_v'][:,2],args['arrays'][key+'_v'][:,2])
            np.testing.assert_array_equal(new[path]['bounds_local'],[translated.min(0),translated.max(0)])
            assert new[path]['world_from_body']==original['world_from_body']
        else:
            assert new[path]==original
            np.testing.assert_array_equal(result['arrays'][key+'_v'],args['arrays'][key+'_v'])
        np.testing.assert_array_equal(result['arrays'][key+'_f'],args['arrays'][key+'_f'])
    for name in ('allowed_body_pairs','analytic_planes','unsupported','custom_source'):
        assert result['metadata'][name]==args['metadata'][name]
    assert result['stool_translation_world_m']==[.12,-.07,0.]
    assert len(result['changed_stool_paths'])==9 and result['stool_height_tested'] is False


def test_stool_translation_is_world_xy_even_if_stored_local_frame_is_rotated():
    args=fixture();entry=args['metadata']['colliders'][0]
    rotation=Rotation.from_euler('x',37,degrees=True).as_matrix()
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=[1,2,3]
    entry['body']='/World/Stool';entry['world_from_body']=transform.tolist()
    result=module().build_cpu_scene_variant(**args)
    before=args['arrays'][entry['key']+'_v']@rotation.T+transform[:3,3]
    after=result['arrays'][entry['key']+'_v']@rotation.T+transform[:3,3]
    np.testing.assert_allclose(after-before,np.tile([.12,-.07,0],(3,1)),atol=1e-12)


def test_every_entry_and_array_is_canonically_bound_and_order_layout_independent():
    args=fixture();mod=module();result=mod.build_cpu_scene_variant(**args)
    bound=result['content_bindings']['proposal']
    assert set(bound['entry_sha256'])=={entry['key'] for entry in result['metadata']['colliders']}
    assert set(bound['array_sha256'])==set(result['arrays'])
    for entry in result['metadata']['colliders']:
        assert bound['entry_sha256'][entry['key']]==mod.canonical_entry_sha256(entry)
    for key,value in result['arrays'].items():
        digest=mod.canonical_array_sha256(value)
        assert bound['array_sha256'][key]==digest
        assert mod.canonical_array_sha256(np.asfortranarray(value))==digest
        swapped=value.byteswap().view(value.dtype.newbyteorder())
        assert mod.canonical_array_sha256(swapped)==digest
    entry=result['metadata']['colliders'][0]
    assert mod.canonical_entry_sha256(dict(reversed(list(entry.items()))))==mod.canonical_entry_sha256(entry)
    altered=deepcopy(entry);altered['bounds_local'][0][0]-=.01
    assert mod.canonical_entry_sha256(altered)!=mod.canonical_entry_sha256(entry)
    altered_array=result['arrays'][entry['key']+'_v'].copy();altered_array[0,0]+=.01
    assert mod.canonical_array_sha256(altered_array)!=bound['array_sha256'][entry['key']+'_v']


def test_readonly_input_arrays_and_nested_metadata_do_not_alias_output():
    args=fixture();original=deepcopy(args)
    for value in args['arrays'].values():value.setflags(write=False)
    args['patient_vertices_world'].setflags(write=False);args['patient_faces'].setflags(write=False)
    result=module().build_cpu_scene_variant(**args)
    assert args['metadata']==original['metadata'] and args['source_bindings']==original['source_bindings']
    for key,value in args['arrays'].items():
        np.testing.assert_array_equal(value,original['arrays'][key])
        if key in result['arrays']:
            assert not np.shares_memory(value,result['arrays'][key])
            result['arrays'][key][:]=0
            np.testing.assert_array_equal(value,original['arrays'][key])
    result['metadata']['allowed_body_pairs'][0][0]='changed'
    result['source_bindings']['collision_geometry_sha256']='0'*64
    assert args['metadata']==original['metadata'] and args['source_bindings']==original['source_bindings']


@pytest.mark.parametrize('kind',['duplicate_path','duplicate_key','missing_stool','missing_array','orphan_array'])
def test_invalid_collision_inventory_is_rejected(kind):
    args=fixture()
    if kind=='duplicate_path':args['metadata']['colliders'][1]['path']=args['metadata']['colliders'][0]['path']
    elif kind=='duplicate_key':args['metadata']['colliders'][1]['key']=args['metadata']['colliders'][0]['key']
    elif kind=='missing_stool':
        removed=args['metadata']['colliders'].pop(0)
        del args['arrays'][removed['key']+'_v'];del args['arrays'][removed['key']+'_f']
    elif kind=='missing_array':del args['arrays']['c000_f']
    else:args['arrays']['orphan_v']=np.zeros((3,3))
    with pytest.raises(ValueError):module().build_cpu_scene_variant(**args)


@pytest.mark.parametrize('key,value',[
    ('stool_xy_translation_m',[.1,.2,.3]),('stool_xy_translation_m',[np.nan,0]),
    ('patient_vertices_world',np.full((3,3),np.nan)),('patient_vertices_world',np.full((3,3),1e40)),
    ('patient_faces',np.array([[0,1,3]])),('patient_faces',np.array([[0,1,-1]])),
    ('patient_faces',np.array([[0.,1.,2.]])),('patient_faces',np.array([[0,0,1]])),
])
def test_nonfinite_wrong_faces_or_patient_float32_overflow_are_rejected(key,value):
    with pytest.raises(ValueError):build(**{key:value})


@pytest.mark.parametrize('field',['collision_geometry_sha256','collision_arrays_sha256','observed_placement_sha256','anchor_inventory_sha256'])
def test_missing_or_malformed_external_source_hashes_fail(field):
    args=fixture();args['source_bindings'][field]='not-a-hash'
    with pytest.raises(ValueError,match='SHA|hash'):module().build_cpu_scene_variant(**args)


def test_retired_geometry_and_forged_native_receipts_are_not_promoted():
    args=fixture();args['source_bindings']['geometry_id']='retired'
    with pytest.raises(ValueError,match='geometry'):module().build_cpu_scene_variant(**args)
    args=fixture();args['metadata']['patient_receipt']={'non_patient_scene_unchanged':True}
    with pytest.raises(ValueError,match='receipt|native'):module().build_cpu_scene_variant(**args)


def test_rounded_rigid_source_frame_still_gets_only_world_xy_translation():
    args=fixture();entry=args['metadata']['colliders'][0]
    rotation=np.round(Rotation.from_euler('xyz',[37,19,13],degrees=True).as_matrix(),8)
    transform=np.eye(4);transform[:3,:3]=rotation
    entry['world_from_body']=transform.tolist();entry['body']='/World/Stool'
    result=module().build_cpu_scene_variant(**args)
    before=args['arrays'][entry['key']+'_v']@rotation.T
    after=result['arrays'][entry['key']+'_v']@rotation.T
    np.testing.assert_allclose(after-before,np.tile([.12,-.07,0],(3,1)),atol=1e-14,rtol=0)


@pytest.mark.parametrize('kind',['array_nan','source_face','matrix_nan','matrix_scale',
                                 'bounds_nan','plane_nan','bad_anchor','missing_anchor',
                                 'extra_stool','native_claim','reserved_key'])
def test_invalid_source_geometry_and_anchor_contracts_are_rejected(kind):
    args=fixture();entry=args['metadata']['colliders'][0]
    if kind=='array_nan':args['arrays']['c000_v'][0,0]=np.nan
    elif kind=='source_face':args['arrays']['c000_f'][0,2]=99
    elif kind=='matrix_nan':entry['world_from_body'][0][0]=np.nan
    elif kind=='matrix_scale':entry['world_from_body'][0][0]=2
    elif kind=='bounds_nan':entry['bounds_local'][0][0]=np.nan
    elif kind=='plane_nan':args['metadata']['analytic_planes'][0]['normal'][0]=np.nan
    elif kind=='bad_anchor':args['anchors']['L1']['barycentric_weights']=[np.nan,0,1]
    elif kind=='missing_anchor':del args['anchors']['L1']
    elif kind=='extra_stool':args['metadata']['colliders'][-1]['path']='/World/Stool/extra'
    elif kind=='native_claim':args['metadata']['non_patient_scene_unchanged']=True
    else:
        entry['key']='cpu_patient_smpl'
        for suffix in ('_v','_f'):args['arrays']['cpu_patient_smpl'+suffix]=args['arrays'].pop('c000'+suffix)
    with pytest.raises(ValueError):module().build_cpu_scene_variant(**args)


def test_quantization_degenerate_anchor_triangle_is_not_accepted_from_prequantized_mesh():
    # Distinct float64 X coordinates collapse to one float32 value at this scale.
    args=fixture();args['patient_vertices_world']=np.array([[1e8,0,0],[1e8+1,0,0],[1e8,1,0]])
    assert evaluate_anchors(args['patient_vertices_world'],args['patient_faces'],args['anchors'])
    with pytest.raises(ValueError,match='Degenerate'):module().build_cpu_scene_variant(**args)


def test_determinism_and_source_bindings_cover_original_entries_arrays_and_targets():
    args=fixture();mod=module();one=mod.build_cpu_scene_variant(**args);two=mod.build_cpu_scene_variant(**args)
    assert one['content_bindings']==two['content_bindings'] and one['targets']==two['targets']
    source=one['content_bindings']['source']
    assert source['metadata_sha256']==mod.canonical_entry_sha256(args['metadata'])
    for entry in args['metadata']['colliders']:
        assert source['entry_sha256'][entry['key']]==mod.canonical_entry_sha256(entry)
    for key,value in args['arrays'].items():
        assert source['array_sha256'][key]==mod.canonical_array_sha256(value)
    changed=deepcopy(args);changed['anchors']['L1']['barycentric_weights']=[.3,.3,.4]
    other=mod.build_cpu_scene_variant(**changed)
    assert other['content_bindings']['anchors_sha256']!=one['content_bindings']['anchors_sha256']
    assert other['content_bindings']['targets_sha256']!=one['content_bindings']['targets_sha256']
    key=one['metadata']['colliders'][-1]['key']
    assert not np.shares_memory(one['arrays'][key+'_v'],args['patient_vertices_world'])
    assert not np.shares_memory(one['arrays'][key+'_f'],args['patient_faces'])

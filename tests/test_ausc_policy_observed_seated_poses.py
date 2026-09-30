"""Tiny synthetic fixtures only; no actual SMPL dataset/model or mesh output."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def module():
    assert importlib.util.find_spec('sim.ausc_policy.observed_seated_poses') is not None, 'observed numeric pose candidate generator is missing'
    return importlib.import_module('sim.ausc_policy.observed_seated_poses')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')


def joints(seated=True):
    value=np.zeros((24,3));value[1]=[.1,0,0];value[2]=[-.1,0,0];value[12]=[0,.7,0]
    if seated:
        value[4]=[.1,0,.4];value[5]=[-.1,0,.4]
        value[7]=[.1,-.4,.4];value[8]=[-.1,-.4,.4]
    else:
        value[4]=[.1,-.4,0];value[5]=[-.1,-.4,0]
        value[7]=[.1,-.8,0];value[8]=[-.1,-.8,0]
    return value


class FakeModel:
    def __init__(self,mode='seated'):
        self.mode=mode;self.calls=[]
    def __call__(self,**inputs):
        self.calls.append({k:np.array(v,copy=True) for k,v in inputs.items()})
        if self.mode=='raise':raise RuntimeError('injected model failure')
        orientation=Rotation.from_rotvec(inputs['global_orient']).as_matrix()
        skeleton=joints(self.mode!='standing')@orientation.T+inputs['transl']
        vertices=np.array([[0,.4,-.1],[.2,.4,-.1],[0,.6,-.1]])@orientation.T+inputs['transl']
        if self.mode=='invalid_geometry':skeleton[4]=skeleton[1]
        return dict(vertices=vertices,faces=np.array([[0,1,2]]),joints=skeleton)


@pytest.fixture
def fixture(tmp_path):
    source=tmp_path/'params';source.mkdir()
    inventory=[]
    for identity,beta in [('p0001',.025),('p0002',-.02),('p0003',.03)]:
        directory=source/(identity+'_a000001');directory.mkdir()
        path=directory/'smpl_params.npz';count=5
        pose=np.tile(np.arange(69)*.001,(count,1));pose[:,0]=np.arange(count)*.03
        orient=np.array([[.1+i*.01,.2,-.3] for i in range(count)])
        translation=np.array([[1+i*.001,2.,3.] for i in range(count)])
        betas=np.full((count,10),beta);betas[:,0]+=np.array([-4e-5,0.,0.,0.,4e-5])
        np.savez(path,body_pose=pose,global_orient=orient,transl=translation,betas=betas)
        inventory.append(dict(path=path.relative_to(source).as_posix(),identity=identity,frames=count,
                              sha256=digest(path),beta_within_sequence_max_abs=4e-5))
    model=tmp_path/'SMPL_NEUTRAL.pkl';model.write_bytes(b'fake model bytes only')
    inventory_path=tmp_path/'source_inventory.json';write_json(inventory_path,inventory)
    split=tmp_path/'identity_split.json';write_json(split,dict(seed=20260930,splits=dict(
        train=['p0001','p0002'],validation=['p0003'],test=['p0004'])))
    quality=tmp_path/'identity_quality.json';write_json(quality,dict(identity_split_sha256=digest(split),
        eligible_shape_betas={'p0001':[.025]*10,'p0002':[-.02]*10,'p0003':[.03]*10},quarantined_identities=[]))
    anchors=tmp_path/'anchors.json';write_json(anchors,dict(anchors={side+str(i):dict(
        smpl_vertex_indices=[0,1,2],barycentric_weights=[.2,.3,.5]) for side in 'LR' for i in range(1,6)}))
    return dict(source_root=source,model_path=model,inventory_path=inventory_path,
                split_path=split,quality_path=quality,anchor_path=anchors,output=tmp_path/'out')


def generate(fixture,**kwargs):
    options=dict(fixture,expected_split='train',candidates_per_identity=2,max_evaluations_per_identity=4,
                 seed=20260930,model_adapter=FakeModel())
    for key in ('model','inventory','split','quality','anchor'):options[key+'_sha256']=digest(options[key+'_path'])
    options.update(kwargs)
    return module().generate_candidates(**options)


def test_generator_preserves_observed_parameters_provenance_and_never_writes_mesh(fixture):
    adapter=FakeModel();summary=generate(fixture,model_adapter=adapter)
    manifest=json.loads((fixture['output']/'candidate_manifest.json').read_text())
    assert summary['requested_identities']==['p0001','p0002']
    assert summary['denominators']==dict(requested=2,shape_eligible=2,selected=2,skipped=0)
    assert summary['candidates']==4 and len(adapter.calls)==4
    assert all('numeric_candidate' in row['name'] for row in manifest['candidates'])
    assert {row['identity'] for row in manifest['candidates']}=={'p0001','p0002'}
    for row,call in zip(manifest['candidates'],adapter.calls):
        source=fixture['source_root']/row['source']['sequence_path']
        assert row['source']['sequence_sha256']==digest(source)
        with np.load(source,allow_pickle=False) as arrays:
            frame=row['source']['frame_index']
            for key in ('body_pose','global_orient','transl'):
                np.testing.assert_array_equal(row['observed_parameters'][key],arrays[key][frame])
                np.testing.assert_array_equal(call[key],arrays[key][frame])
            np.testing.assert_array_equal(row['observed_parameters']['betas'],arrays['betas'][frame])
        assert row['shape_source']['kind']=='frozen_identity_quality_canonical_beta'
        assert row['source']['identity']==row['identity'] and row['split']=='train'
        assert row['metrics']['numeric_criterion_passed'] and len(row['anchors'])==10
        assert not any(row[key] for key in ('training_eligible','physics_qualified','placement_qualified'))
        assert row['global_placement_applied'] is False and row['torso_leveling_applied'] is False
    assert {p.name for p in fixture['output'].iterdir()}=={'summary.json','candidate_manifest.json'}
    assert not any(summary[key] for key in ('training_eligible','physics_qualified','placement_qualified'))
    assert 'squat' in summary['limitations'] and 'support' in summary['limitations']


def test_selected_identity_pilot_retains_requested_denominator(fixture):
    summary=generate(fixture,selected_identities=['p0002'],candidates_per_identity=1)
    assert summary['requested_identities']==['p0002']
    assert summary['denominators']==dict(requested=1,shape_eligible=1,selected=1,skipped=0)
    assert [row['identity'] for row in summary['by_identity']]==['p0002']


@pytest.mark.parametrize('selected',[['p0003'],['p0001','p0001'],['p9999']])
def test_selected_identities_cannot_leak_other_split_duplicate_or_unknown(fixture,selected):
    with pytest.raises(ValueError,match='identity|identities|split'):generate(fixture,selected_identities=selected)
    assert not fixture['output'].exists()


def test_default_reports_quarantined_identity_but_explicit_request_rejected(fixture):
    path=fixture['quality_path'];quality=json.loads(path.read_text())
    del quality['eligible_shape_betas']['p0002'];quality['quarantined_identities']=['p0002'];write_json(path,quality)
    summary=generate(fixture)
    assert summary['denominators']==dict(requested=2,shape_eligible=1,selected=1,skipped=1)
    assert next(row for row in summary['by_identity'] if row['identity']=='p0002')['status']=='skipped_shape_ineligible'
    with pytest.raises(ValueError,match='eligible'):
        generate(fixture,output=fixture['output'].with_name('explicit'),selected_identities=['p0002'])


def test_inventory_identity_must_match_its_own_sequence_path(fixture):
    path=fixture['inventory_path'];inventory=json.loads(path.read_text());inventory[0]['identity']='p0002';write_json(path,inventory)
    with pytest.raises(ValueError,match='identity'):generate(fixture)


@pytest.mark.parametrize('kind',['model','inventory','split','quality','anchor'])
def test_changed_explicit_source_hash_is_rejected_before_output(fixture,kind):
    expected=digest(fixture[kind+'_path']);fixture[kind+'_path'].write_bytes(fixture[kind+'_path'].read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash|SHA'):generate(fixture,**{kind+'_sha256':expected})
    assert not fixture['output'].exists()


def test_changed_sequence_hash_is_not_silently_used(fixture):
    path=fixture['source_root']/'p0001_a000001/smpl_params.npz';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash|SHA'):generate(fixture,selected_identities=['p0001'])


@pytest.mark.parametrize('key',['body_pose','global_orient','transl','betas'])
def test_nonfinite_observed_source_is_rejected_even_with_updated_inventory_hash(fixture,key):
    path=fixture['source_root']/'p0001_a000001/smpl_params.npz'
    with np.load(path,allow_pickle=False) as arrays:values={name:arrays[name] for name in arrays.files}
    values[key][2,0]=np.nan;np.savez(path,**values)
    inventory=json.loads(fixture['inventory_path'].read_text());inventory[0]['sha256']=digest(path)
    write_json(fixture['inventory_path'],inventory)
    with pytest.raises(ValueError,match='nonfinite|finite'):generate(fixture,selected_identities=['p0001'])


def test_anatomical_criterion_uses_actual_joint_geometry_and_is_rigid_invariant():
    mod=module();seated=joints();standing=joints(False)
    accepted=mod.classify_numeric_seated(seated)
    assert accepted['numeric_criterion_passed']
    np.testing.assert_allclose(accepted['knee_flexion_deg'],[90.,90.])
    assert not mod.classify_numeric_seated(standing)['numeric_criterion_passed']
    rotation=Rotation.from_rotvec([.6,.3,-.8]).as_matrix()
    transformed=mod.classify_numeric_seated(seated@rotation.T+[1.,2.,3.])
    np.testing.assert_allclose(transformed['thigh_pitch_deg'],accepted['thigh_pitch_deg'],atol=1e-12)
    np.testing.assert_allclose(transformed['knee_flexion_deg'],accepted['knee_flexion_deg'],atol=1e-12)
    assert transformed['numeric_criterion_passed']


def test_axis_angles_alone_cannot_override_standing_joint_geometry(fixture):
    summary=generate(fixture,model_adapter=FakeModel('standing'),selected_identities=['p0001'])
    row=summary['by_identity'][0]
    assert summary['candidates']==0 and row['status']=='none_found_in_bounded_scan'
    assert row['evaluated_frames']==4 and row['rejection_counts']['numeric_criterion_failed']==4


@pytest.mark.parametrize('mode',['raise','invalid_geometry'])
def test_failed_model_and_invalid_geometry_consume_whole_identity_budget(fixture,mode):
    adapter=FakeModel(mode);summary=generate(fixture,model_adapter=adapter,selected_identities=['p0001'],
                                            max_evaluations_per_identity=3,candidates_per_identity=1)
    row=summary['by_identity'][0]
    assert row['status']=='none_found_in_bounded_scan'
    assert row['evaluated_frames']==3 and len(adapter.calls)==3
    assert sum(row['rejection_counts'].values())==3


def test_deterministic_selection_ignores_inventory_order_and_preserves_original_shape_option(fixture):
    generate(fixture,use_canonical_betas=False)
    first=json.loads((fixture['output']/'candidate_manifest.json').read_text())['candidates']
    path=fixture['inventory_path'];inventory=json.loads(path.read_text());write_json(path,list(reversed(inventory)))
    generate(fixture,use_canonical_betas=False,output=fixture['output'].with_name('second'))
    second=json.loads((fixture['output'].with_name('second')/'candidate_manifest.json').read_text())['candidates']
    assert [(r['identity'],r['source']['sequence_path'],r['source']['frame_index']) for r in first]==[(r['identity'],r['source']['sequence_path'],r['source']['frame_index']) for r in second]
    for record in second:
        assert record['shape_source']['kind']=='observed_frame_beta'
        np.testing.assert_array_equal(record['model_parameters']['betas'],record['observed_parameters']['betas'])


def test_max_evaluations_is_across_sequences_not_reset_per_sequence(fixture):
    source=fixture['source_root']/'p0001_a000001/smpl_params.npz'
    extra=fixture['source_root']/'p0001_a000002';extra.mkdir();path=extra/'smpl_params.npz';path.write_bytes(source.read_bytes())
    inventory=json.loads(fixture['inventory_path'].read_text());row=dict(inventory[0],path='p0001_a000002/smpl_params.npz')
    inventory.append(row);write_json(fixture['inventory_path'],inventory)
    model=FakeModel('standing');summary=generate(fixture,selected_identities=['p0001'],model_adapter=model,max_evaluations_per_identity=3)
    assert len(model.calls)==3 and summary['by_identity'][0]['available_frames']==10


def test_cli_blocks_nonlinux_before_reading_model_or_dataset(monkeypatch):
    mod=module();monkeypatch.setattr(mod.sys,'platform','darwin')
    with pytest.raises(RuntimeError,match='Linux'):mod.require_linux_cli()


def test_candidate_manifest_explicitly_labels_injected_cpu_fixture(fixture):
    generate(fixture)
    manifest=json.loads((fixture['output']/'candidate_manifest.json').read_text())
    assert manifest['injected_model_adapter'] is True
    assert all(row['injected_model_adapter'] is True for row in manifest['candidates'])


def test_heldout_sequence_bytes_are_never_loaded_for_train_candidates(fixture):
    (fixture['source_root']/'p0003_a000001/smpl_params.npz').write_bytes(b'heldout bytes must not be opened')
    summary=generate(fixture)
    assert summary['candidates']==4
    assert all(row['identity'] in ('p0001','p0002') for row in summary['by_identity'])


def test_identity_without_source_rows_has_explicit_zero_scan_result(fixture):
    inventory=json.loads(fixture['inventory_path'].read_text())
    write_json(fixture['inventory_path'],[row for row in inventory if row['identity']!='p0002'])
    summary=generate(fixture,selected_identities=['p0002'])
    assert summary['denominators']['requested']==1 and summary['candidates']==0
    assert summary['by_identity'][0]['status']=='none_found_in_bounded_scan'
    assert summary['by_identity'][0]['available_frames']==0
    assert summary['by_identity'][0]['evaluated_frames']==0


@pytest.mark.parametrize('kind',['knee_too_straight','knee_overflexed','thigh_backward','nonfinite'])
def test_actual_joint_geometric_failures_cannot_be_numeric_candidates(kind):
    value=joints()
    if kind=='knee_too_straight':value[7]=value[4]+[0,0,.4]
    elif kind=='knee_overflexed':value[7]=value[4]+[0,0,-.4]
    elif kind=='thigh_backward':value[[4,5,7,8],2]*=-1
    else:value[7,0]=np.nan
    if kind=='nonfinite':
        with pytest.raises(ValueError,match='Finite'):module().classify_numeric_seated(value)
    else:assert not module().classify_numeric_seated(value)['numeric_criterion_passed']


def test_observed_torso_lean_remains_in_model_output_anchor_coordinates(fixture):
    generate(fixture,selected_identities=['p0001'],candidates_per_identity=1)
    row=json.loads((fixture['output']/'candidate_manifest.json').read_text())['candidates'][0]
    parameters=row['observed_parameters'];rotation=Rotation.from_rotvec(parameters['global_orient']).as_matrix()
    vertices=np.array([[0,.4,-.1],[.2,.4,-.1],[0,.6,-.1]])@rotation.T+parameters['transl']
    np.testing.assert_allclose(row['anchors']['R3']['position_m'],np.array([.2,.3,.5])@vertices,atol=1e-12)
    np.testing.assert_allclose(row['metrics']['torso_axis_in_source_coordinates'],rotation@[0,1,0],atol=1e-12)

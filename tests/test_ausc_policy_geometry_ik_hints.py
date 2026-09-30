"""Bound seed forwarding only; CPU fixtures establish no geometric qualification."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.ausc_policy import geometry_epoch, geometry_plan
from test_ausc_policy_geometry_plan import candidate_report, CONTACT_SURFACE


def module():
    assert importlib.util.find_spec('sim.ausc_policy.geometry_ik_hints'), 'Bound IK hint helper is missing'
    return importlib.import_module('sim.ausc_policy.geometry_ik_hints')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def fixture(tmp_path):
    artifacts=dict(collision_geometry_sha256='a'*64, collision_arrays_sha256='b'*64)
    target=dict(name='L3', position_world_m=[0.,0.,0.], normal_world=[0.,0.,1.])
    config=dict(geometry_id=geometry_epoch.GEOMETRY_ID, freshly_generated=True,
        geometry_source_sha256=geometry_epoch.GEOMETRY_SOURCE_SHA256, **artifacts,
        world_from_left_base=np.eye(4).tolist(), loaded_compression_m=.0025, targets=[target])
    report=candidate_report()
    report.update(geometry_artifacts=artifacts, loaded_compression_m=.0025)
    report['candidates'][0]['raw_anchor_world_m']=target['position_world_m']
    source_fit=tmp_path/'source/L3_fits.json';fresh_fit=tmp_path/'output/L3_fits.json'
    write(source_fit,report);write(fresh_fit,report)
    document=dict(schema='current_geometry_ik_hints_v1', geometry_id=config['geometry_id'],
        geometry_source_sha256=config['geometry_source_sha256'], geometry_artifacts=artifacts,
        contact_surface=CONTACT_SURFACE, target=target, world_from_left_base=config['world_from_left_base'],
        loaded_compression_m=.0025, patient_provenance=None,
        candidate_file=source_fit.name, candidate_file_sha256=sha(source_fit),
        hints=[dict(candidate_index=0, loaded_q=[.1]*7, pre_q=[.2]*7)],
        collision_checked=False, path_checked=False, physics_executed=False, training_eligible=False)
    source_hint=tmp_path/'source/L3_hints.json';write(source_hint,document)
    config['ik_hints']={'L3':dict(file=str(source_hint), sha256=sha(source_hint))}
    return SimpleNamespace(config=config, artifacts=artifacts, target=target, report=report,
        source_fit=source_fit, fresh_fit=fresh_fit, source_hint=source_hint, document=document)


def repin(data):
    write(data.source_hint,data.document)
    data.config['ik_hints']['L3']['sha256']=sha(data.source_hint)


def binder(data):
    return module().BoundIKHints(data.config,data.artifacts,None)


def test_exact_fit_bytes_bind_seed_copy_receipt_and_final_verification(tmp_path):
    data=fixture(tmp_path);bound=binder(data)
    original=data.source_hint.read_bytes();reference=bound.bind(data.target,data.fresh_fit)
    hints=bound.planner_hints('L3')
    assert hints==data.document['hints']
    hints[0]['loaded_q'][0]=999.
    assert bound.planner_hints('L3')==data.document['hints']
    assert (data.fresh_fit.parent/'L3_ik_hints.json').read_bytes()==original
    receipt_path=data.fresh_fit.parent/reference['file']
    assert sha(receipt_path)==reference['sha256']
    receipt=json.loads(receipt_path.read_text())
    assert receipt['source_hint']['sha256']==sha(data.source_hint)
    assert receipt['source_fit']['sha256']==receipt['fresh_fit']['sha256']==sha(data.source_fit)
    assert not receipt['training_eligible'] and not receipt['physics_executed']
    final=bound.finalize()
    assert final['verified'] is True and final['bindings']=={'L3':reference}


@pytest.mark.parametrize('field,value', [
    ('geometry_id','old_geometry'), ('geometry_source_sha256','c'*64),
    ('geometry_artifacts',dict(collision_geometry_sha256='c'*64,collision_arrays_sha256='b'*64)),
    ('contact_surface','membrane_only'),
    ('target',dict(name='R3',position_world_m=[0.,0.,0.],normal_world=[0.,0.,1.])),
    ('target',dict(name='L3',position_world_m=[.001,0.,0.],normal_world=[0.,0.,1.])),
    ('target',dict(name='L3',position_world_m=[0.,0.,0.],normal_world=[0.,1.,0.])),
    ('world_from_left_base',np.diag([1.,-1.,-1.,1.]).tolist()),
    ('loaded_compression_m',.003), ('patient_provenance',{'identity':'other-body'}),
    ('schema','unqualified_current_geometry_ik_seed_hints_v1'), ('training_eligible',True),
    ('collision_checked',True), ('path_checked',True), ('physics_executed',True)])
def test_wrong_epoch_body_point_frame_or_qualification_is_rejected(tmp_path,field,value):
    data=fixture(tmp_path);data.document[field]=value;repin(data)
    with pytest.raises(ValueError):binder(data).bind(data.target,data.fresh_fit)
    assert not (data.fresh_fit.parent/'L3_ik_hint_receipt.json').exists()


@pytest.mark.parametrize('fault',['source_hint','source_fit','fresh_fit_value','fresh_fit_format'])
def test_changed_input_or_fresh_fit_bytes_cannot_reinterpret_indices(tmp_path,fault):
    data=fixture(tmp_path)
    path={'source_hint':data.source_hint,'source_fit':data.source_fit,
          'fresh_fit_value':data.fresh_fit,'fresh_fit_format':data.fresh_fit}[fault]
    if fault=='fresh_fit_value':
        changed=deepcopy(data.report);changed['candidates'][0]['surface_gap_m']+=1e-6;write(path,changed)
    else:path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='SHA|bytes|changed'):binder(data).bind(data.target,data.fresh_fit)


@pytest.mark.parametrize('hints', [[],[{'candidate_index':True,'loaded_q':[0.]*7}],
    [{'candidate_index':-1,'loaded_q':[0.]*7}],[{'candidate_index':1,'loaded_q':[0.]*7}],
    [{'candidate_index':0,'loaded_q':[0.]*6}],[{'candidate_index':0,'loaded_q':['0']*7}],
    [{'candidate_index':0,'loaded_q':[True]+[0.]*6}],
    [{'candidate_index':0,'pre_q':[0.]*7}],
    [{'candidate_index':0,'loaded_q':[0.]*7,'bypass_collision':True}],
    [{'candidate_index':0,'loaded_q':[0.]*7}]*2])
def test_indices_and_seed_schema_are_strict(tmp_path,hints):
    data=fixture(tmp_path);data.document['hints']=hints;repin(data)
    with pytest.raises(ValueError):binder(data).bind(data.target,data.fresh_fit)


def test_seed_values_are_not_clipped_or_treated_as_qualified_poses(tmp_path):
    data=fixture(tmp_path);data.document['hints'][0]['loaded_q']=[100.]*7;repin(data)
    bound=binder(data);bound.bind(data.target,data.fresh_fit)
    assert bound.planner_hints('L3')[0]['loaded_q']==[100.]*7


@pytest.mark.parametrize('fault',['duplicate_key','nan'])
def test_hint_json_cannot_hide_duplicate_keys_or_nonfinite_seeds(tmp_path,fault):
    data=fixture(tmp_path)
    text=data.source_hint.read_text()
    if fault=='duplicate_key':text=text.replace('"candidate_index": 0','"candidate_index": 0, "candidate_index": 0')
    else:text=text.replace('0.1','NaN',1)
    data.source_hint.write_text(text);data.config['ik_hints']['L3']['sha256']=sha(data.source_hint)
    with pytest.raises(ValueError):binder(data).bind(data.target,data.fresh_fit)


@pytest.mark.parametrize('changed',['source_hint','source_fit','fresh_fit','hint_copy','receipt'])
@pytest.mark.parametrize('check',['planner_hints','finalize'])
def test_every_bound_file_is_rechecked_before_use_and_at_completion(tmp_path,changed,check):
    data=fixture(tmp_path);bound=binder(data);reference=bound.bind(data.target,data.fresh_fit)
    path={'source_hint':data.source_hint,'source_fit':data.source_fit,'fresh_fit':data.fresh_fit,
          'hint_copy':data.fresh_fit.parent/'L3_ik_hints.json','receipt':data.fresh_fit.parent/reference['file']}[changed]
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='changed|SHA'):
        getattr(bound,check)(*(['L3'] if check=='planner_hints' else []))


@pytest.mark.parametrize('configuration',[None,[],{'R4':{'file':'unused','sha256':'a'*64}},
                                           {'L3':{'file':'relative.json','sha256':'a'*64}}])
def test_invalid_or_unused_config_bindings_are_not_silently_ignored(tmp_path,configuration):
    data=fixture(tmp_path);data.config['ik_hints']=configuration
    with pytest.raises(ValueError):binder(data)


def test_requested_hints_must_be_bound_before_completion(tmp_path):
    with pytest.raises(ValueError,match='unused|bound'):
        binder(fixture(tmp_path)).finalize()


def test_refinement_only_mode_cannot_silently_ignore_requested_ik_hints(tmp_path):
    data=fixture(tmp_path);data.config['fit_refinement_diagnostic']={}
    with pytest.raises(ValueError,match='refinement'):binder(data)


def test_absent_hints_are_a_noop_without_new_outputs_or_metadata(tmp_path):
    data=fixture(tmp_path);del data.config['ik_hints'];bound=binder(data)
    assert bound.bind(data.target,data.fresh_fit) is None
    assert bound.finalize() is None
    assert sorted(p.name for p in data.fresh_fit.parent.iterdir())==['L3_fits.json']


def run_geometry_plan(data,monkeypatch,*,with_hints=True,mutate_during_plan=False,mutate_geometry_source=False):
    """Exercise real orchestration/provenance, injecting only numerical engines."""
    import sim.ausc_policy.finite_head as finite_head
    import sim.ausc_policy.contact_collision as collision
    import sim.auscultation_expert_layout_v1.kinematics as kinematics
    import sim.auscultation_expert_layout_v1.posture as posture
    source=data.source_hint.parent/'candidate.usdc';source.write_bytes(b'isolated CPU source fixture')
    monkeypatch.setattr(geometry_epoch,'GEOMETRY_SOURCE_SHA256',sha(source))
    data.config['geometry_source_sha256']=data.document['geometry_source_sha256']=sha(source)
    directory=data.source_hint.parent/'geometry';directory.mkdir()
    for name,key in [('collision_geometry.json','collision_geometry_sha256'),
                     ('collision_geometry.npz','collision_arrays_sha256')]:
        path=directory/name;path.write_bytes(('CPU '+name).encode())
        data.artifacts[key]=data.config[key]=sha(path)
    write(data.source_fit,data.report);data.document['candidate_file_sha256']=sha(data.source_fit);repin(data)
    data.config.update(geometry_source=str(source),collision_directory=str(directory),posture_limits={'sentinel':True},
        seeds_per_candidate=4,per_gap_seconds=180,fresh_pre_gap_search_m=[.04,.03])
    if not with_hints:del data.config['ik_hints']
    output=data.fresh_fit.parent.parent/'planned';config_path=data.source_hint.parent/'config.json';write(config_path,data.config)
    monkeypatch.setattr(sys,'argv',['geometry_plan','--config',str(config_path),'--output',str(output)])
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setattr(finite_head,'FiniteHeadFit',lambda *a:SimpleNamespace(candidates=lambda *a,**k:deepcopy(data.report)))
    checker=SimpleNamespace(check=lambda q:{'clear':True});limits=object();kin=object()
    monkeypatch.setattr(collision,'ContactCollisionChecker',lambda *a:checker)
    monkeypatch.setattr(kinematics,'OpenArmKinematics',lambda **k:kin)
    monkeypatch.setattr(posture,'PostureLimits',SimpleNamespace(from_dict=lambda *a:limits))
    monkeypatch.setattr(posture,'PostureChecker',lambda *a:checker)
    calls=[]
    def plan(target,audit,base,actual_kin,actual_checker,actual_limits,**kwargs):
        assert actual_kin is kin and actual_checker is checker and actual_limits is limits
        assert kwargs['seeds_per_candidate']==4 and kwargs['max_seconds']==180 and kwargs['max_routes']==2
        assert audit['candidates'][0]['finite_head_fit']['passed']
        calls.append(deepcopy(kwargs))
        if mutate_during_plan:data.source_hint.write_bytes(data.source_hint.read_bytes()+b'\n')
        if mutate_geometry_source:source.write_bytes(source.read_bytes()+b'\n')
        return dict(success=False,attempts=[dict(failure='loaded_ik_or_posture')],
                    failure_counts={'loaded_ik_or_posture':1},search_hints=kwargs.get('hints',[]))
    monkeypatch.setitem(sys.modules,'sim.auscultation_expert_layout_v1.plan_low_elbow',SimpleNamespace(plan_target=plan))
    return output,calls


@pytest.mark.parametrize('with_hints',[False,True])
def test_geometry_plan_forwards_only_verified_seeds_and_keeps_existing_checks(tmp_path,monkeypatch,with_hints):
    data=fixture(tmp_path);output,calls=run_geometry_plan(data,monkeypatch,with_hints=with_hints)
    geometry_plan.main()
    assert len(calls)==1
    result=json.loads((output/'COMPLETE.json').read_text())
    assert result['targets'][0]['geometry_path_found'] is False
    assert result['physics_executed'] is False and result['training_eligible'] is False
    if with_hints:
        assert calls[0]['hints']==data.document['hints']
        assert result['ik_hint_verification']['verified']
        assert (output/'L3_ik_hints.json').read_bytes()==data.source_hint.read_bytes()
        attempts=json.loads((output/'L3_planning_attempts.json').read_text())
        assert attempts[0]['ik_hint_receipt']['sha256']==sha(output/'L3_ik_hint_receipt.json')
    else:
        assert 'hints' not in calls[0] and 'ik_hint_verification' not in result
        assert not list(output.glob('*ik_hint*'))


def test_changed_hint_during_search_prevents_complete_receipt(tmp_path,monkeypatch):
    data=fixture(tmp_path);output,calls=run_geometry_plan(data,monkeypatch,mutate_during_plan=True)
    with pytest.raises(ValueError,match='changed|SHA'):geometry_plan.main()
    assert len(calls)==1 and not (output/'COMPLETE.json').exists()


def test_changed_geometry_source_during_hinted_search_prevents_final_receipts(tmp_path,monkeypatch):
    data=fixture(tmp_path);output,calls=run_geometry_plan(data,monkeypatch,mutate_geometry_source=True)
    with pytest.raises(ValueError,match='Geometry source changed'):geometry_plan.main()
    assert len(calls)==1 and not (output/'COMPLETE.json').exists()
    assert not (output/'IK_HINTS_FINAL.json').exists()

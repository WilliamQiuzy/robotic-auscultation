"""CPU-only binding tests. No real experiment, credentials, SDK or model call."""
import ast
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def mod(name):
    full='sim.auscultation_pi05_v7.'+name
    try:result=importlib.import_module(full)
    except ModuleNotFoundError:result=None
    assert result is not None, 'Independent expanded runtime module required: '+name
    return result


def fixture_binding():
    from sim.auscultation_pi05_v7.tests.test_expanded_runtime import complete_binding
    from sim.auscultation_pi05_v7.expanded_contract import experiment_source_hashes
    b=complete_binding();b['experiment_source_sha256']=experiment_source_hashes()
    a=dict(schema='wam_v7_expanded_selected_runtime_admission_v1',
        checkpoint_metadata_sha256=b['checkpoint_metadata_sha256'],data_hash=b['data_hash'],
        experiment_binding=b['experiment_binding'],experiment_source_sha256=b['experiment_source_sha256'],
        dynamics_recipe_sha256=b['dynamics_recipe_sha256'],dynamics_contract_sha256=b['dynamics_contract_sha256'],
        execution_contract_sha256=b['execution_contract_sha256'],
        selection_lock_sha256='9'*64)
    return b,a


def test_runtime_contract_keeps_actual_headroom_and_trained_execution_distinct():
    m=mod('expanded_runtime_contract');d=m.runtime_connection_contract()
    assert d['scope']=='zero_home_full_single_or_continuous_v1'
    assert d['controller_class']=='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
    assert d['executed_runner_module']=='sim.auscultation_pi05_v7.run_expanded_headroom_isaac'
    assert len(d['connection_source_sha256'])==4 and len(d['experiment_source_sha256'])==9
    assert d['headroom_implementation']['runtime_execution_contract']['jerk_rad_s3']==[7.99999]*7
    assert d['headroom_implementation']['original_whole_stream_gate']['joint_jerk_rad_s3']==8.
    assert d['start_family_preparation_supported'] is False


def test_selected_binding_rejects_missing_or_other_experiment():
    m=mod('expanded_runtime_contract');b,a=fixture_binding();m.verify_policy_against_admission(b,a)
    for field in ('checkpoint_metadata_sha256','data_hash','experiment_binding','experiment_source_sha256'):
        changed=copy.deepcopy(b);changed.pop(field)
        with pytest.raises(ValueError):m.verify_policy_against_admission(changed,a)
    changed=copy.deepcopy(a);changed['checkpoint_metadata_sha256']='b'*64
    with pytest.raises(ValueError):m.verify_policy_against_admission(b,changed)


def test_every_query_and_server_receipt_requires_extension_not_only_outcome():
    m=mod('audit_expanded_headroom_actions');b,a=fixture_binding()
    inference=dict(b,request_id='one');learned=[dict(index=0,inference=inference)]
    requests=[dict(request_id='one',experiment_binding=b['experiment_binding'],experiment_source_sha256=b['experiment_source_sha256'],checkpoint_metadata_sha256=b['checkpoint_metadata_sha256'])]
    assert m.verify_experiment_requests(learned,requests,b,a)==1
    for source in ('query','server'):
        q=copy.deepcopy(learned);r=copy.deepcopy(requests)
        (q[0]['inference'] if source=='query' else r[0]).pop('experiment_source_sha256')
        with pytest.raises(ValueError,match='experiment|source'):m.verify_experiment_requests(q,r,b,a)


def test_unmatched_or_duplicate_server_request_rejected():
    m=mod('audit_expanded_headroom_actions');b,a=fixture_binding();q=[dict(index=0,inference=dict(b,request_id='one'))]
    row=dict(request_id='two',checkpoint_metadata_sha256=b['checkpoint_metadata_sha256'],experiment_binding=b['experiment_binding'],experiment_source_sha256=b['experiment_source_sha256'])
    with pytest.raises(ValueError):m.verify_experiment_requests(q,[row],b,a)
    with pytest.raises(ValueError):m.verify_experiment_requests(q,[row,row],b,a)


def test_new_assessment_cannot_accept_old_audit_or_failed_inherited_replay():
    m=mod('assess_expanded_headroom_trials')
    with pytest.raises(ValueError,match='expanded'):m.verify_outer_audit({'schema':'wam_v7_headroom_learned_action_audit_v1','passed':True})
    value=dict(schema='wam_v7_expanded_headroom_action_audit_v1',passed=True,
        experiment_binding_verified=True,selected_checkpoint_lock_verified=True,connection_source_snapshot_verified=True,
        experiment_request_bindings_verified=True,inherited_action_audit={'schema':'wam_v7_headroom_learned_action_audit_v1','passed':False})
    with pytest.raises(ValueError,match='inherited'):m.verify_outer_audit(value)


def test_runner_reuses_true_controller_and_has_no_tick_override():
    r=mod('run_expanded_headroom_isaac')
    controller,trajectory=r.physical_classes()
    assert controller.__module__=='sim.auscultation_pi05_v7.headroom_learned_controller'
    assert trajectory.__module__=='sim.auscultation_pi05_v7.headroom_execution'
    source=ast.parse(Path(r.__file__).read_text())
    assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(source))


def test_runner_requires_full_zero_home_and_explicit_selected_experiment(tmp_path):
    r=mod('run_expanded_headroom_isaac')
    with pytest.raises(SystemExit):r.parse_args(['--point','R4'])
    args=SimpleNamespace(task_scope='contact')
    with pytest.raises(ValueError,match='full'):r.verify_expanded_scope(args)
    args=SimpleNamespace(task_scope='full',start_family='test_fold_v1')
    with pytest.raises(ValueError,match='[Ss]tart'):r.verify_expanded_scope(args)


def outcome_fixture():
    c=mod('expanded_runtime_contract');b,a=fixture_binding();contract=c.runtime_connection_contract()
    value=dict(executed_runner_module=c.RUNNER,executed_controller_class=c.CONTROLLER,
        executed_runner_source_sha256=contract['connection_source_sha256']['v7/run_expanded_headroom_isaac.py'],
        executed_policy_client_class=c.CLIENT,runtime_connection_contract=contract,
        runtime_connection_contract_sha256=c.E.json_sha(contract),experiment_runtime_binding=a,
        task_scope='full',initial_state_scope='zero_home',learned_policy_binding=b)
    return value,a


def test_actual_runner_bytes_and_identity_not_just_class_label():
    c=mod('expanded_runtime_contract');value,a=outcome_fixture();c.verify_connection_outcome(value,a)
    for key,bad in [('executed_runner_source_sha256','f'*64),
            ('executed_controller_class','sim.auscultation_pi05_v7.learned_controller.LearnedController'),
            ('executed_policy_client_class','sim.auscultation_pi05_v7.policy_client.PolicyClient'),
            ('initial_state_scope','start_family')]:
        changed=copy.deepcopy(value);changed[key]=bad
        with pytest.raises(ValueError):c.verify_connection_outcome(changed,a)


def snapshot_fixture(tmp_path):
    m=mod('audit_expanded_headroom_actions');c=mod('expanded_runtime_contract');_,a=fixture_binding()
    root=tmp_path/'episode';directory=root/'input_snapshot';objects=directory/'objects';objects.mkdir(parents=True)
    files={}
    def add(key,data):
        import hashlib
        digest=hashlib.sha256(data).hexdigest();(objects/digest).write_bytes(data);files[key]={'sha256':digest}
        return digest
    for label,path in {**m.experiment_source_paths(),**c.connection_source_paths()}.items():
        generation,name=label.split('/');add('source/pi05_'+generation+'/'+name,Path(path).read_bytes())
    for key,field in [('expanded_selection_lock','selection_lock_sha256'),
            ('expanded_experiment','experiment_sha256'),('expanded_manifest','manifest_sha256')]:
        a[field]=add('source/pi05_v7/'+key,b'actual bound '+key.encode())
    import json
    add('source/pi05_v7/expanded_selected_admission',json.dumps(a).encode())
    contract_sha=add('source/pi05_v7/expanded_connection_contract',json.dumps(c.runtime_connection_contract()).encode())
    manifest=directory/'manifest.json';manifest.write_text(json.dumps({'files':files}))
    outcome=dict(input_snapshot='input_snapshot/manifest.json',input_snapshot_manifest_sha256=c.E.sha(manifest),
        runtime_connection_contract_file_sha256=contract_sha)
    return m,c,root,outcome,a,manifest,files,objects


def test_snapshot_binds_all_nine_plus_four_actual_sources_and_selected_inputs(tmp_path):
    m,c,root,outcome,a,manifest,files,objects=snapshot_fixture(tmp_path)
    artifacts={};m.verify_connection_snapshot(root,outcome,a,artifacts)
    assert len(artifacts)==19  # manifest + thirteen source objects + five selection/contract objects
    import hashlib,json
    changed=b'not the selected actual runner';digest=hashlib.sha256(changed).hexdigest()
    (objects/digest).write_bytes(changed)
    files['source/pi05_v7/run_expanded_headroom_isaac.py']['sha256']=digest
    manifest.write_text(json.dumps({'files':files}));outcome['input_snapshot_manifest_sha256']=c.E.sha(manifest)
    with pytest.raises(ValueError,match='source differs'):m.verify_connection_snapshot(root,outcome,a,{})


def test_snapshot_selection_substitution_and_escape_rejected(tmp_path):
    m,c,root,outcome,a,manifest,files,objects=snapshot_fixture(tmp_path)
    altered=copy.deepcopy(a);altered['selection_lock_sha256']='0'*64
    with pytest.raises(ValueError,match='snapshot differs'):m.verify_connection_snapshot(root,outcome,altered,{})
    outside=tmp_path/'outside.json';outside.write_text('{}');outcome['input_snapshot']='../outside.json'
    with pytest.raises(ValueError,match='escapes'):m.verify_connection_snapshot(root,outcome,a,{})


def test_fixed_denominator_rejects_partial_and_start_family_plan(monkeypatch):
    m=mod('assess_expanded_headroom_trials')
    records=[dict(case_id=p,directory='/unused/'+p,points=[p],scope='full',role='nominal_single_point') for p in m.POINTS]
    plan=dict(schema='wam_v7_expanded_headroom_acceptance_plan_v1',checkpoint_metadata_sha256='a'*64,
        selection_lock_sha256='b'*64,cases=records)
    with pytest.raises(ValueError,match='ten full singles'):m.assess(plan)
    plan['cases']=records+[dict(case_id='sequence',directory='/unused/sequence',points=list(m.POINTS),
        scope='full',role='nominal_continuous_sequence')]
    monkeypatch.setattr(m,'assess_case',lambda record,checkpoint:dict(passed=True,errors=[],selection_lock_sha256='b'*64))
    result=m.assess(plan);assert result['requested_cases']==11 and result['all_declared_tests_passed'] is True
    plan['cases'][0]['role']='start_family'
    with pytest.raises(ValueError):m.assess(plan)


def test_runner_imports_and_injects_expanded_client_without_algorithm_copy():
    r=mod('run_expanded_headroom_isaac');source=ast.parse(Path(r.__file__).read_text())
    capture=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    imports=[n for n in ast.walk(capture) if isinstance(n,ast.ImportFrom)]
    assert any(n.module=='expanded_policy_client' and any(a.name=='PolicyClient' for a in n.names) for n in imports)
    constructor=next(n for n in ast.walk(capture) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='LearnedController')
    assert any(k.arg=='client' and isinstance(k.value,ast.Name) and k.value.id=='client' for k in constructor.keywords)
    assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(source))
    main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    checked=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='verify_expanded_inputs')
    kit=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom) and n.module=='isaacsim')
    assert checked<kit

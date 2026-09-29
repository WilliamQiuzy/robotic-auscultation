"""CPU-only binding tests. No real experiment, credentials, SDK or model call."""
import ast
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def mod(name):
    if sys.platform=='darwin':pytest.skip('Real parent/runtime imports run in isolated cloud CPU overlay')
    full='sim.auscultation_pi05_v7.'+name
    try:result=importlib.import_module(full)
    except ModuleNotFoundError:result=None
    assert result is not None, 'Independent coverage_pilot runtime module required: '+name
    return result


def fixture_binding():
    from sim.auscultation_pi05_v7.tests.test_coverage_pilot_binding import valid
    from sim.auscultation_pi05_v7 import coverage_pilot_policy_client as client
    from sim.auscultation_pi05_v7.coverage_pilot_schedule import source_hashes
    b=valid(client);b['pilot_source_sha256']=source_hashes()
    a=dict(schema='wam_v7_coverage_pilot_selected_runtime_admission_v1',
        **{k:b[k] for k in ('checkpoint_metadata_sha256','data_hash','pilot_binding','pilot_source_sha256',
                          'pilot_selection_sha256','dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256')},
        selection_lock_sha256=b['pilot_selection_sha256'])
    return b,a


def test_runtime_contract_keeps_actual_headroom_and_trained_execution_distinct():
    m=mod('coverage_pilot_runtime_contract');d=m.runtime_connection_contract()
    assert d['scope']=='zero_home_full_single_or_continuous_v1'
    assert d['controller_class']=='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
    assert d['executed_runner_module']=='sim.auscultation_pi05_v7.run_coverage_pilot_headroom_isaac'
    assert len(d['connection_source_sha256'])==4 and len(d['pilot_source_sha256'])==7
    assert d['headroom_implementation']['runtime_execution_contract']['jerk_rad_s3']==[7.99999]*7
    assert d['headroom_implementation']['original_whole_stream_gate']['joint_jerk_rad_s3']==8.
    assert d['start_family_preparation_supported'] is False


def test_selected_binding_rejects_missing_or_other_experiment():
    m=mod('coverage_pilot_runtime_contract');b,a=fixture_binding();m.verify_policy_against_admission(b,a)
    for field in ('checkpoint_metadata_sha256','data_hash','pilot_binding','pilot_source_sha256'):
        changed=copy.deepcopy(b);changed.pop(field)
        with pytest.raises(ValueError):m.verify_policy_against_admission(changed,a)
    changed=copy.deepcopy(a);changed['checkpoint_metadata_sha256']='b'*64
    with pytest.raises(ValueError):m.verify_policy_against_admission(b,changed)


def test_every_query_and_server_receipt_requires_extension_not_only_outcome():
    m=mod('audit_coverage_pilot_headroom_actions');b,a=fixture_binding()
    inference=dict(b,request_id='one');learned=[dict(index=0,inference=inference)]
    requests=[dict(request_id='one',pilot_binding=b['pilot_binding'],pilot_source_sha256=b['pilot_source_sha256'],checkpoint_metadata_sha256=b['checkpoint_metadata_sha256'],pilot_selection_sha256=b['pilot_selection_sha256'])]
    assert m.verify_experiment_requests(learned,requests,b,a)==1
    for source in ('query','server'):
        q=copy.deepcopy(learned);r=copy.deepcopy(requests)
        (q[0]['inference'] if source=='query' else r[0]).pop('pilot_source_sha256')
        with pytest.raises(ValueError,match='experiment|source'):m.verify_experiment_requests(q,r,b,a)


def test_unmatched_or_duplicate_server_request_rejected():
    m=mod('audit_coverage_pilot_headroom_actions');b,a=fixture_binding();q=[dict(index=0,inference=dict(b,request_id='one'))]
    row=dict(request_id='two',checkpoint_metadata_sha256=b['checkpoint_metadata_sha256'],pilot_binding=b['pilot_binding'],pilot_source_sha256=b['pilot_source_sha256'],pilot_selection_sha256=b['pilot_selection_sha256'])
    with pytest.raises(ValueError):m.verify_experiment_requests(q,[row],b,a)
    with pytest.raises(ValueError):m.verify_experiment_requests(q,[row,row],b,a)


def test_new_assessment_cannot_accept_old_audit_or_failed_inherited_replay():
    m=mod('assess_coverage_pilot_headroom_trials')
    with pytest.raises(ValueError,match='pilot'):m.verify_outer_audit({'schema':'wam_v7_headroom_learned_action_audit_v1','passed':True})
    value=dict(schema='wam_v7_coverage_pilot_headroom_action_audit_v1',passed=True,
        pilot_binding_verified=True,selected_checkpoint_lock_verified=True,connection_source_snapshot_verified=True,
        pilot_request_bindings_verified=True,inherited_action_audit={'schema':'wam_v7_coverage_pilot_headroom_replay_v1','passed':False})
    with pytest.raises(ValueError,match='inherited'):m.verify_outer_audit(value)


def test_runner_reuses_true_controller_and_has_no_tick_override():
    r=mod('run_coverage_pilot_headroom_isaac')
    controller,trajectory=r.physical_classes()
    assert controller.__module__=='sim.auscultation_pi05_v7.headroom_learned_controller'
    assert trajectory.__module__=='sim.auscultation_pi05_v7.headroom_execution'
    source=ast.parse(Path(r.__file__).read_text())
    assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(source))


def test_runner_requires_full_zero_home_and_explicit_selected_experiment(tmp_path):
    r=mod('run_coverage_pilot_headroom_isaac')
    with pytest.raises(SystemExit):r.parse_args(['--point','R4'])
    args=SimpleNamespace(task_scope='contact')
    with pytest.raises(ValueError,match='full'):r.verify_coverage_pilot_scope(args)
    args=SimpleNamespace(task_scope='full',start_family='test_fold_v1')
    with pytest.raises(ValueError,match='[Ss]tart'):r.verify_coverage_pilot_scope(args)


def outcome_fixture():
    c=mod('coverage_pilot_runtime_contract');b,a=fixture_binding();contract=c.runtime_connection_contract()
    value=dict(executed_runner_module=c.RUNNER,executed_controller_class=c.CONTROLLER,
        executed_runner_source_sha256=contract['connection_source_sha256']['v7/run_coverage_pilot_headroom_isaac.py'],
        executed_policy_client_class=c.CLIENT,runtime_connection_contract=contract,
        runtime_connection_contract_sha256=c.E.digest(contract),pilot_runtime_binding=a,
        task_scope='full',initial_state_scope='zero_home',learned_policy_binding=b)
    return value,a


def test_actual_runner_bytes_and_identity_not_just_class_label():
    c=mod('coverage_pilot_runtime_contract');value,a=outcome_fixture();c.verify_connection_outcome(value,a)
    for key,bad in [('executed_runner_source_sha256','f'*64),
            ('executed_controller_class','sim.auscultation_pi05_v7.learned_controller.LearnedController'),
            ('executed_policy_client_class','sim.auscultation_pi05_v7.policy_client.PolicyClient'),
            ('initial_state_scope','start_family')]:
        changed=copy.deepcopy(value);changed[key]=bad
        with pytest.raises(ValueError):c.verify_connection_outcome(changed,a)


def snapshot_fixture(tmp_path):
    m=mod('audit_coverage_pilot_headroom_actions');c=mod('coverage_pilot_runtime_contract');_,a=fixture_binding()
    root=tmp_path/'episode';directory=root/'input_snapshot';objects=directory/'objects';objects.mkdir(parents=True)
    files={}
    def add(key,data):
        import hashlib
        digest=hashlib.sha256(data).hexdigest();(objects/digest).write_bytes(data);files[key]={'sha256':digest}
        return digest
    for label,path in {**m.pilot_source_paths(),**c.connection_source_paths()}.items():
        generation,name=label.split('/');add('source/pi05_'+generation+'/'+name,Path(path).read_bytes())
    for key,field in [('coverage_pilot_selection_lock','selection_lock_sha256'),
            ('coverage_pilot_manifest','manifest_sha256')]:
        a[field]=add('source/pi05_v7/'+key,b'actual bound '+key.encode())
    import json
    add('source/pi05_v7/coverage_pilot_selected_admission',json.dumps(a).encode())
    contract_sha=add('source/pi05_v7/coverage_pilot_connection_contract',json.dumps(c.runtime_connection_contract()).encode())
    manifest=directory/'manifest.json';manifest.write_text(json.dumps({'files':files}))
    outcome=dict(input_snapshot='input_snapshot/manifest.json',input_snapshot_manifest_sha256=c.E.file_sha(manifest),
        runtime_connection_contract_file_sha256=contract_sha)
    return m,c,root,outcome,a,manifest,files,objects


def test_snapshot_binds_all_nine_plus_four_actual_sources_and_selected_inputs(tmp_path):
    m,c,root,outcome,a,manifest,files,objects=snapshot_fixture(tmp_path)
    artifacts={};m.verify_connection_snapshot(root,outcome,a,artifacts)
    assert len(artifacts)==16  # manifest + thirteen source objects + five selection/contract objects
    import hashlib,json
    changed=b'not the selected actual runner';digest=hashlib.sha256(changed).hexdigest()
    (objects/digest).write_bytes(changed)
    files['source/pi05_v7/run_coverage_pilot_headroom_isaac.py']['sha256']=digest
    manifest.write_text(json.dumps({'files':files}));outcome['input_snapshot_manifest_sha256']=c.E.file_sha(manifest)
    with pytest.raises(ValueError,match='source differs'):m.verify_connection_snapshot(root,outcome,a,{})


def test_snapshot_selection_substitution_and_escape_rejected(tmp_path):
    m,c,root,outcome,a,manifest,files,objects=snapshot_fixture(tmp_path)
    altered=copy.deepcopy(a);altered['selection_lock_sha256']='0'*64
    with pytest.raises(ValueError,match='snapshot differs'):m.verify_connection_snapshot(root,outcome,altered,{})
    outside=tmp_path/'outside.json';outside.write_text('{}');outcome['input_snapshot']='../outside.json'
    with pytest.raises(ValueError,match='escapes'):m.verify_connection_snapshot(root,outcome,a,{})


def test_fixed_denominator_rejects_partial_and_start_family_plan(monkeypatch):
    m=mod('assess_coverage_pilot_headroom_trials')
    records=[dict(case_id=p,directory='/unused/'+p,points=[p],scope='full',role='nominal_single_point') for p in m.POINTS]
    plan=dict(schema='wam_v7_coverage_pilot_headroom_acceptance_plan_v1',checkpoint_metadata_sha256='a'*64,
        selection_lock_sha256='b'*64,cases=records)
    with pytest.raises(ValueError,match='ten full singles'):m.assess(plan)
    plan['cases']=records+[dict(case_id='sequence',directory='/unused/sequence',points=list(m.POINTS),
        scope='full',role='nominal_continuous_sequence')]
    monkeypatch.setattr(m,'assess_case',lambda record,checkpoint:dict(passed=True,errors=[],selection_lock_sha256='b'*64))
    result=m.assess(plan);assert result['requested_cases']==11 and result['all_declared_tests_passed'] is True
    plan['cases'][0]['role']='start_family'
    with pytest.raises(ValueError):m.assess(plan)


def test_runner_imports_and_injects_coverage_pilot_client_without_algorithm_copy():
    r=mod('run_coverage_pilot_headroom_isaac');source=ast.parse(Path(r.__file__).read_text())
    capture=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    imports=[n for n in ast.walk(capture) if isinstance(n,ast.ImportFrom)]
    assert any(n.module=='coverage_pilot_policy_client' and any(a.name=='PolicyClient' for a in n.names) for n in imports)
    constructor=next(n for n in ast.walk(capture) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='LearnedController')
    assert any(k.arg=='client' and isinstance(k.value,ast.Name) and k.value.id=='client' for k in constructor.keywords)
    assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(source))
    main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    checked=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='verify_coverage_pilot_inputs')
    kit=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom) and n.module=='isaacsim')
    assert checked<kit


def parsed(name):
    return ast.parse((Path(__file__).parents[1]/name).read_text())


def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}


def parent_source(name):
    # On Mac use exact frozen326 CPU-reviewed source copies, avoiding iCloud
    # hydration of old parent files. Cloud overlay supplies the frozen originals.
    copied=Path(__file__).parents[3]/'results/pi05_v7_repair/coverage_pilot_runtime_preparation/ancestry'/name
    if copied.is_file():return copied
    import sim.auscultation_pi05_v7 as package
    for folder in package.__path__:
        p=Path(folder)/name
        if p.is_file():return p
    raise AssertionError('Exact frozen parent source unavailable')


def test_every_copied_action_replay_function_unchanged_except_declared_identity():
    import hashlib
    p=parent_source('audit_headroom_learned_actions.py')
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='771a788d9041f12268631196504ddfdf7d5ca606542a80d259aa870a2ee552e2'
    old=functions(ast.parse(p.read_text()));new=functions(parsed('audit_coverage_pilot_headroom_actions.py'))
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='pi05_v7_original8_coverage_continuation_v1':n.value='pi05_v7_fixed_observation_joint_residual_v1'
            if n.value=='wam_v7_coverage_pilot_headroom_replay_v1':n.value='wam_v7_headroom_learned_action_audit_v1'
            return n
        def visit_Name(self,n):
            if n.id=='inherited_audit_source_paths':n.id='audit_source_paths'
            return n
    for name,node in old.items():
        if name=='main':continue
        renamed={'audit_source_paths':'inherited_audit_source_paths','audit_episode':'_audit_episode_replay'}.get(name,name)
        actual=copy.deepcopy(new[renamed]);actual.name=name
        assert ast.dump(Normalize().visit(actual))==ast.dump(node),name
    assert all(key in Path(new_source:=Path(__file__).parents[1]/'audit_coverage_pilot_headroom_actions.py').read_text()
               for key in ("'pilot_binding'","'pilot_source_sha256'","'pilot_selection_sha256'"))


def test_copied_acceptance_physics_adc_and_completion_checks_unchanged():
    import hashlib
    p=parent_source('assess_headroom_trials.py')
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='d731ba715f713925086744588504d87589003ea29b7a6e588f91da1bdfad640e'
    old=functions(ast.parse(p.read_text()))['check_documents']
    new=copy.deepcopy(functions(parsed('assess_coverage_pilot_headroom_trials.py'))['inherited_check_documents']);new.name=old.name
    for n in ast.walk(new):
        if isinstance(n,ast.Constant) and n.value=='wam_v7_coverage_pilot_headroom_replay_v1':n.value='wam_v7_headroom_learned_action_audit_v1'
    assert ast.dump(old)==ast.dump(new)


def test_legacy_binding_is_not_rewritten_or_monkeypatched_for_pilot():
    audit=parsed('audit_coverage_pilot_headroom_actions.py');assessor=parsed('assess_coverage_pilot_headroom_trials.py')
    for tree in (audit,assessor):
        assert any(isinstance(n,ast.ImportFrom) and n.module=='coverage_pilot_policy_client' for n in tree.body)
        assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='verify_policy_binding'
            for t in n.targets) for n in ast.walk(tree))
    runner=parsed('run_coverage_pilot_headroom_isaac.py')
    assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(runner))
    capture=functions(runner)['capture']
    assert any(isinstance(n,ast.ImportFrom) and n.module=='coverage_pilot_policy_client' for n in ast.walk(capture))

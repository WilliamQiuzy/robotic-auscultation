"""Finite orchestration and explicit admission tests, with no native/model calls."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt


def module():
    name='scripts.rebuild_eval_chain_20261105'
    assert importlib.util.find_spec(name) is not None,'Finite post-training evaluation continuation missing'
    return importlib.import_module(name)


def owner():
    return dict(run_id='rebuilt-act-20261105-v2',completed=True,gpu_released=True,
        stage=dict(passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[]))


def test_explicit_smoke_episode_ids_resolve_two_pose_duplicates_without_fallback():
    m=module();rows=[]
    for i,(split,subject,target,eid) in enumerate(m.SMOKES):
        rows.extend([dict(episode_id=eid,subject_id=subject,split=split,target_id=target,pose_id=m.SMOKE_POSE,windows=12,config={}),
            dict(episode_id='other-pose-'+str(i),subject_id=subject,split=split,target_id=target,pose_id='other-pose',windows=12,config={})])
    chosen=m.select_smokes(dict(episodes=rows))
    assert [r['episode_id'] for r in chosen]==[r[3] for r in m.SMOKES]
    for fault in ('missing','duplicate','zero_windows','wrong_subject'):
        changed=deepcopy(rows)
        if fault=='missing':changed.pop(0)
        if fault=='duplicate':changed.append(deepcopy(changed[0]))
        if fault=='zero_windows':changed[0]['windows']=0
        if fault=='wrong_subject':changed[0]['subject_id']='p001030'
        with pytest.raises(ValueError):m.select_smokes(dict(episodes=changed))


@pytest.mark.parametrize('fault',['not_terminal','not_released','not_drained','nonzero','signals','wrong_steps'])
def test_training_admission_refuses_incomplete_owner_or_schedule_before_frozen_access(tmp_path,monkeypatch,fault):
    m=module();folder=tmp_path/'training';folder.mkdir();(folder/'output').mkdir();value=owner()
    complete=dict(completed_steps=20000,schedule={'steps':20000})
    if fault=='not_terminal':value['completed']=False
    elif fault=='not_released':value['gpu_released']=False
    elif fault=='not_drained':value['stage']['process_group_drained']=False
    elif fault=='nonzero':value['stage']['exit_code']=1
    elif fault=='signals':value['stage']['signals']=['SIGTERM']
    else:complete['completed_steps']=19999
    rt.write_new(folder/'result.json',value);rt.write_new(folder/'output/COMPLETE.json',complete)
    monkeypatch.setattr(m.expanded,'model_binding',lambda ref:({},complete))
    with pytest.raises(ValueError):m.admit_training(folder/'result.json')


def test_wait_is_finite_and_never_launches_training_or_native_work(tmp_path,monkeypatch):
    m=module();ticks=iter([0.,2.]);monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(m.time,'sleep',lambda *args:pytest.fail('Wait exceeded deadline'))
    with pytest.raises(TimeoutError):m.wait_terminal(tmp_path/'absent/result.json',seconds=1)


def chain_fixture(tmp_path,monkeypatch,*,smoke_pass=True,unsafe_arm=None):
    m=module();monkeypatch.setattr(m,'CONTROL',tmp_path/'control');monkeypatch.setattr(m,'BENCHMARK',tmp_path/'benchmark')
    monkeypatch.setattr(m,'BASE',tmp_path/'base');monkeypatch.setenv('AUSC_ASTRA_RUN_TOKEN','secret-token-that-must-never-be-written')
    calls=[];ref=rt.write_new(tmp_path/'artifact.json',{'actual':'bound fixture'})
    monkeypatch.setattr(m,'source_snapshot',lambda:{'fixture':ref});monkeypatch.setattr(m,'check_source',lambda snapshot:None)
    monkeypatch.setattr(m,'wait_terminal',lambda *a,**k:None)
    monkeypatch.setattr(m,'admit_training',lambda path:dict(complete=ref,model={'checkpoint':ref},owner=ref))
    monkeypatch.setattr(m,'prepare_smokes',lambda training,output:dict(batch=ref,cases=[{'id':'train-r3'},{'id':'val-r3'}]))
    def run_owner(config,run_id,phase,source):
        calls.append(phase)
        return dict(normal=True,safe_to_continue=phase!=unsafe_arm,run_id=run_id,terminal=ref)
    monkeypatch.setattr(m,'run_owner',run_owner)
    monkeypatch.setattr(m,'qualify_smokes',lambda prepared,result:dict(passed=smoke_pass,cases=[dict(passed=smoke_pass)]*2))
    def prepare_frozen(training):
        calls.append('prepare_frozen')
        return dict(protocol=ref,receipt=ref,batches={arm:ref for arm in m.paired.ARMS},fixed_denominator=8)
    monkeypatch.setattr(m,'prepare_frozen',prepare_frozen)
    monkeypatch.setattr(m,'qualify_arm',lambda prepared,arm:dict(arm=arm))
    def summarize(prepared):calls.append('summarize');return ref
    monkeypatch.setattr(m,'summarize_benchmark',summarize)
    return m,calls


def test_failed_act_native_admission_stops_before_any_frozen_source_or_outcome(tmp_path,monkeypatch):
    m,calls=chain_fixture(tmp_path,monkeypatch,smoke_pass=False)
    with pytest.raises(ValueError,match='admission'):m.run_chain()
    assert calls==['smoke']
    failed=json.loads((m.CONTROL/'FAILED.json').read_text())
    assert failed['frozen_sources_opened'] is False
    assert not (m.CONTROL/'COMPLETE.json').exists()


def test_both_actual_smokes_gate_preparation_then_sequential_all_arms_and_summary(tmp_path,monkeypatch):
    m,calls=chain_fixture(tmp_path,monkeypatch)
    result=m.run_chain()
    assert calls==['smoke','prepare_frozen','expert','act','astra','summarize']
    assert result['fixed_denominator']==8 and result['planned_arm_denominator']==24
    assert result['physical_success_claimed'] is False
    assert 'secret-token' not in ''.join(p.read_text() for p in m.CONTROL.glob('*.json'))


def test_undrained_owner_prevents_next_gpu_launch_but_preserves_fixed_summary(tmp_path,monkeypatch):
    m,calls=chain_fixture(tmp_path,monkeypatch,unsafe_arm='act')
    qualified=[];monkeypatch.setattr(m,'qualify_arm',lambda prepared,arm:qualified.append(arm) or {})
    with pytest.raises(ValueError,match='drain'):m.run_chain()
    assert calls==['smoke','prepare_frozen','expert','act','summarize']
    assert qualified==['expert']
    failed=json.loads((m.CONTROL/'FAILED.json').read_text())
    assert failed['summary'] and failed['frozen_sources_opened'] is True


def test_missing_astra_token_is_not_logged_or_replaced_by_another_model(tmp_path,monkeypatch):
    m,calls=chain_fixture(tmp_path,monkeypatch);monkeypatch.delenv('AUSC_ASTRA_RUN_TOKEN')
    with pytest.raises(ValueError,match='token'):m.run_chain()
    assert calls==['smoke']
    assert json.loads((m.CONTROL/'FAILED.json').read_text())['frozen_sources_opened'] is False


def test_normal_training_owner_rebinds_actual_config_process_guard_and_model_without_rewriting(tmp_path,monkeypatch):
    m=module();base=tmp_path/'base';monkeypatch.setattr(m,'BASE',base)
    folder=base/'runs/ausc_policy'/m.TRAIN_RUN;folder.mkdir(parents=True);(folder/'output').mkdir()
    dataset=rt.write_new(tmp_path/'dataset.json',dict(rebuild_epoch='observed-reconstruction-20261105',original_cases=0))
    config=dict(base=str(base),experiment='m2_given_contact_expanded_pilot_train',steps=20000)
    config_ref=rt.write_new(tmp_path/'config.json',config)
    identity=dict(pid=123,pgid=123,sid=123,start_ticks='456')
    cpu=rt.write_new(tmp_path/'cpu.json',{'passed':True})
    complete=dict(completed_steps=20000,schedule=config,inputs=dict(config=config_ref,dataset=dataset),
        scope=dict(train={'subject_ids':list(m.paired.COHORTS['train'])},validation={'subject_ids':['p000923']}),
        identity=identity,cpu_equivalence=cpu,selected_step=19500)
    complete_ref=rt.write_new(folder/'output/COMPLETE.json',complete)
    command=['/actual/python','-u','-B','-m',m.expanded.pilot.MODULE,'--config',config_ref['path']]
    value=owner();value.update(config=config,config_sha256=config_ref['sha256'],command=command,
        resource_guard={'sha256':rt.reference(m.process_guard.__file__)['sha256']})
    value['stage'].update(command=command,process_identity=dict(identity,members={'123':'456'}))
    terminal=rt.write_new(folder/'result.json',value);calls=[]
    def binding(ref):calls.append(ref);return {'checkpoint':cpu},deepcopy(complete)
    monkeypatch.setattr(m.expanded,'model_binding',binding)
    admitted=m.admit_training(terminal['path'])
    assert calls==[complete_ref] and admitted['complete']==complete_ref and admitted['dataset']==dataset
    assert rt.read_bound(complete_ref)==complete


def test_owner_launcher_uses_original_guard_command_without_manufacturing_marker_or_logging_token(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m,'BASE',tmp_path/'base');monkeypatch.setattr(m,'CODE',tmp_path/'source8')
    monkeypatch.setattr(m,'CONTROL',tmp_path/'control');m.CONTROL.mkdir()
    token='secret-value-for-inherited-astra-token';monkeypatch.setenv('AUSC_ASTRA_RUN_TOKEN',token);monkeypatch.setenv('AUSC_OWNED_RUN','1')
    config=dict(experiment='isaac_batch',base=str(m.BASE),time_limit_seconds=9000,jobs=[])
    config_ref=rt.write_new(tmp_path/'batch.json',config);guard=rt.reference(m.process_guard.__file__);calls=[]
    monkeypatch.setattr(m,'check_source',lambda source:None);monkeypatch.setattr(m,'empty_gpu',lambda:{'compute_pids':[]})
    monkeypatch.setattr(m.isaac_batch,'validate_jobs',lambda value:[])
    class Child:
        pid=987
        def wait(self,timeout):
            assert timeout==9900
            value=owner();value.update(run_id=m.SMOKE_RUN,config=config,config_sha256=config_ref['sha256'],resource_guard=guard)
            rt.write_new(m.BASE/'runs/ausc_policy'/m.SMOKE_RUN/'result.json',value)
            return 0
    def launch(command,**kwargs):calls.append((command,kwargs));return Child()
    monkeypatch.setattr(m.subprocess,'Popen',launch)
    result=m.run_owner(config_ref,m.SMOKE_RUN,'smoke',{'native_guard':guard})
    command,kwargs=calls[0]
    assert command[:4]==[m.PY_NATIVE,'-B','-m','sim.ausc_policy.run']
    assert kwargs['start_new_session'] is True and kwargs['cwd']==m.CODE
    assert kwargs['env']['AUSC_ASTRA_RUN_TOKEN']==token and 'AUSC_OWNED_RUN' not in kwargs['env']
    assert result['normal'] is True and result['safe_to_continue'] is True
    assert token not in ''.join(p.read_text() for p in m.CONTROL.glob('*.json'))

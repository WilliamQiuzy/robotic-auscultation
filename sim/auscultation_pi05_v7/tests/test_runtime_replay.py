"""Independent real-Ruckig command evidence, with nonzero measured origins."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
pytest.importorskip('ruckig',reason='Real command replay requires pinned cloud CPU Ruckig 0.12.2')

from sim.auscultation_pi05_v7.tests.test_runtime_binding import binding,module

spec=importlib.util.spec_from_file_location('independent_absolute_fixture',Path(__file__).with_name('_absolute_fixture.py'))
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)


def evidence(tmp_path,**kwargs):
    data=fixture_module.fixture(tmp_path,**kwargs)
    data['binding']=binding()
    for row in data['learned']:
        row['inference'].update(data['binding'])
    return data


def test_real_continuous_replay_uses_decoded_absolute_targets(tmp_path):
    auditor=module('audit_learned_actions');data=evidence(tmp_path)
    expected=fixture_module.SEED+np.array([1.]*6+[-1.])*8*fixture_module.DT**3/6
    np.testing.assert_allclose(data['physics'][3]['q_command'],expected,atol=1e-14,rtol=0)
    report=auditor.audit_records(**data)
    assert report['passed'],report
    assert report['verified_executed_steps']==24
    assert report['schema']=='wam_learned_absolute_joint_action_evidence_audit_v7'
    assert report['replay']['goal_composition']=='raw_actions[0]'


@pytest.mark.parametrize('case',['double_add','reset','legacy_recipe','missing_source','tamper_seventh'])
def test_independent_replay_rejects_semantically_wrong_consistent_records(tmp_path,case):
    auditor=module('audit_learned_actions')
    data=evidence(tmp_path,add_measured_origin=case=='double_add',reset_between=case=='reset')
    if case=='legacy_recipe':
        data['binding']['training_recipe']='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1'
        for row in data['learned']:row['inference']['training_recipe']=data['binding']['training_recipe']
    if case=='missing_source':
        data['binding']['inference_source_sha256'].pop('v6/execution.py')
        for row in data['learned']:row['inference']['inference_source_sha256']=data['binding']['inference_source_sha256']
    if case=='tamper_seventh':data['physics'][3]['q_command'][6]+=.00001
    report=auditor.audit_records(**data)
    assert not report['passed'],report


def test_failed_query_keeps_partial_prefix_without_success_claim(tmp_path):
    data=evidence(tmp_path);last=data['learned'][-1];end=last['physics_row_start']+5
    data['physics']=data['physics'][:end]
    last.update(physics_row_end_exclusive=end,failure_reason='independent safety rejection')
    report=module('audit_learned_actions').audit_records(**data)
    assert report['passed'] and report['failed_queries']==1
    assert report['verified_executed_steps']==17
    assert report['policy_acceptance']=='not_evaluated'


def test_bound_source_snapshot_rejects_changed_reused_execution(tmp_path):
    auditor=module('audit_learned_actions');client=module('policy_client')
    source=tmp_path/'input_snapshot';(source/'objects').mkdir(parents=True)
    info=binding();snapshot={'files':{}}
    for label,path in client.inference_source_paths().items():
        raw=path.read_bytes();digest=auditor.digest(raw)
        (source/'objects'/digest).write_bytes(raw)
        generation,name=label.split('/')
        snapshot['files']['source/pi05_'+generation+'/'+name]={'sha256':digest}
        info['inference_source_sha256'][label]=digest
    auditor.verify_inference_snapshot(snapshot,source,info,{})
    label='source/pi05_v6/execution.py';digest=snapshot['files'][label]['sha256']
    (source/'objects'/digest).write_text('changed execution')
    with pytest.raises(ValueError):auditor.verify_inference_snapshot(snapshot,source,info,{})

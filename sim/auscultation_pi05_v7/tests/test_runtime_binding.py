"""CPU wire-contract tests. No trained model or physics process is started."""
import copy
import hashlib
import importlib
import importlib.util
import io
import json

import numpy as np
import pytest


def module(name):
    full = 'sim.auscultation_pi05_v7.' + name
    assert importlib.util.find_spec(full) is not None, 'V7 runtime component required: ' + name
    return importlib.import_module(full)


def binding():
    client = module('policy_client')
    from sim.auscultation_pi05_v6.training_data import MODALITIES
    from sim.auscultation_pi05_v6.task_observation import observation_binding
    from sim.auscultation_pi05_v7.action_encoding import action_encoding_binding
    return dict(checkpoint_metadata_sha256='a'*64, data_hash='b'*64,
        inference_source_sha256={name:'c'*64 for name in client.inference_source_paths()},
        sampling_steps=10, sampling_seed=917, sampling_policy='fixed common Gaussian seed',
        action_semantics='absolute_joint_command_v1', model_action_semantics='future_command_minus_fixed_observation_measured_q', training_recipe='pi05_v7_fixed_observation_joint_residual_v1',
        modalities=MODALITIES, audio='absent', dynamics_recipe_sha256='d'*64,
        dynamics_contract_sha256='e'*64, execution_contract_sha256='f'*64,
        observation_contract=observation_binding(), action_encoding_binding=action_encoding_binding())


def config(info):
    return dict(token='synthetic-test-fixture-only', client_port=1,
                **{'expected_'+key:info[key] for key in module('policy_client').CONFIG_FIELDS})


def test_missing_component_is_not_silently_v6():
    assert module('policy_client').RECIPE == 'pi05_v7_fixed_observation_joint_residual_v1'


@pytest.mark.parametrize('change',['recipe','semantics','internal_semantics','source_missing','source_extra','source_invalid','encoding','steps','dynamics'])
def test_rejects_wrong_recipe_and_incomplete_exact_wire_binding(change):
    client=module('policy_client'); info=binding()
    if change=='recipe':info['training_recipe']='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1'
    elif change=='semantics':info['action_semantics']='fixed_observation_joint_residual_v1'
    elif change=='internal_semantics':info.pop('model_action_semantics')
    elif change=='source_missing':info['inference_source_sha256'].pop(next(iter(info['inference_source_sha256'])))
    elif change=='source_extra':info['inference_source_sha256']['actor.py']='c'*64
    elif change=='source_invalid':info['inference_source_sha256'][next(iter(info['inference_source_sha256']))]='not-a-sha'
    elif change=='encoding':info['action_encoding_binding']={'version':'v6'}
    elif change=='steps':info['sampling_steps']=9
    elif change=='dynamics':info.pop('execution_contract_sha256')
    with pytest.raises(ValueError):client.verify_policy_binding(info,'a'*64)


def test_decoded_absolute_reply_not_added_to_nonzero_measured_origin(tmp_path,monkeypatch):
    client=module('policy_client'); info=binding(); path=tmp_path/'config.json';path.write_text(json.dumps(config(info)))
    state=np.linspace(-.6,.8,41,dtype=np.float32);rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    # Already decoded, fixed-query-root absolute output; no client composition.
    absolute=np.tile(np.array([.15,-.2,.3,1.2,-.4,.1,.2],np.float32),(10,1))
    def send(request,timeout):
        if request.data is None:return io.BytesIO(json.dumps(info).encode())
        req=json.loads(request.data)
        reply=dict(info,request_id=req['request_id'],raw_actions=absolute.tolist(),
            request_payload_sha256=hashlib.sha256(request.data).hexdigest(),
            rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),rgb_shape=list(rgb.shape),
            state_f32_sha256=hashlib.sha256(np.asarray(req['state'],dtype='<f4').tobytes()).hexdigest(),
            raw_action_f32_sha256=hashlib.sha256(absolute.astype('<f4').tobytes()).hexdigest())
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr(client.urllib.request,'urlopen',send)
    actor=client.PolicyClient(path);actual,_=actor.predict(rgb,state,'FREE','R4')
    np.testing.assert_array_equal(actual,absolute)
    assert not np.allclose(actual[0],absolute[0]+state[:7])


def test_client_rejects_reply_source_change(tmp_path,monkeypatch):
    client=module('policy_client');info=binding();path=tmp_path/'config.json';path.write_text(json.dumps(config(info)))
    def send(request,timeout):
        if request.data is None:return io.BytesIO(json.dumps(info).encode())
        reply=copy.deepcopy(info);reply['inference_source_sha256']['v7/actor.py']='0'*64
        reply['request_id']=json.loads(request.data)['request_id']
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr(client.urllib.request,'urlopen',send)
    actor=client.PolicyClient(path)
    with pytest.raises(ValueError,match='binding changed'):
        actor.predict(np.zeros((3,4,3),np.uint8),np.zeros(41),'FREE','R4')


def test_service_snapshot_namespaces_are_preserved(tmp_path):
    service=module('policy_service');paths={}
    for name,raw in [('v7/actor.py',b'new actor'),('v6/action_contract.py',b'frozen absolute contract')]:
        p=tmp_path/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw);paths[name]=p
    output=tmp_path/'snapshot';result=service.write_source_snapshot(output,paths)
    assert result=={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    assert (output/'v7/actor.py').read_bytes()==b'new actor'
    assert (output/'v6/action_contract.py').read_bytes()==b'frozen absolute contract'
    with pytest.raises(ValueError):service.write_source_snapshot(tmp_path/'bad',{'../escape.py':paths['v7/actor.py']})


def test_runner_reuses_actual_frozen_physical_classes():
    pytest.importorskip('ruckig',reason='Real Ruckig parent-runtime check runs in pinned cloud CPU environment')
    runner=module('run_isaac')
    from sim.auscultation_pi05_v6.learned_controller import LearnedController
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    assert runner.physical_classes()==(LearnedController,JointTrajectory)
    sources=runner.runtime_source_paths()
    for key in ('pi05_v7/run_isaac.py','pi05_v7/policy_client.py','pi05_v6/learned_controller.py',
                'pi05_v6/execution.py','pi05_v6/observed_supervisor.py','pi05_v6/task_observation.py'):
        assert key in sources and sources[key].is_file()


def test_runner_import_does_not_initialize_physics_before_kit():
    import subprocess
    import sys
    code="import sys; import sim.auscultation_pi05_v7.run_isaac; assert not any(k.startswith(('pxr','omni.','isaacsim','sim.auscultation_pi05_v6.learned_controller')) for k in sys.modules)"
    subprocess.run([sys.executable,'-B','-c',code],check=True,timeout=15)


def test_runtime_rejects_service_bound_to_other_local_source_bytes():
    client=module('policy_client');info=binding()
    info['inference_source_sha256']=client.inference_source_hashes()
    client.verify_runtime_sources(info)
    info['inference_source_sha256']['v6/execution.py']='0'*64
    with pytest.raises(ValueError,match='source bytes differ'):client.verify_runtime_sources(info)

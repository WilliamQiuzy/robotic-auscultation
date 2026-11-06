"""CPU checks that the model-visible payload explains actual sensor quantities."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_ausc_policy_astra_sensors import make


def approved_preload(tmp_path):
    from sim.ausc_policy import passive_preload
    (tmp_path/'fresh_expert_inputs').mkdir()
    config=dict(fresh_expert=dict(passive_preload_N=1.5,passive_preload_model=dict(mode=passive_preload.MODE)))
    receipt=dict(schema=passive_preload.MODE,mode=passive_preload.MODE,nominal_preload_N=1.5,
        explicit_effort_N=-1.5,source_sha256=hashlib.sha256(Path(passive_preload.__file__).read_bytes()).hexdigest(),
        other_stage_attributes_unchanged=True,protected_before_sha256='a'*64,protected_after_sha256='a'*64)
    startup=dict(mode=passive_preload.MODE,nominal_preload_N=1.5,explicit_effort_N=-1.5,
        scope='live_usd_after_native_initialization',read_only_usd=True,applied_effort_measured=False)
    for rel,value in [('fresh_expert_inputs/runtime_config.json',config),
                      ('passive_preload_authoring.json',receipt),('passive_preload_startup_usd_readback.json',startup)]:
        (tmp_path/rel).write_text(json.dumps(value))
    return config,receipt,startup


def test_native_context_uses_existing_approved_preload_config_and_receipt_hashes(tmp_path):
    from sim.ausc_policy.astra_native import preload_observation_context
    approved_preload(tmp_path)
    result=preload_observation_context(tmp_path)
    assert result['nominal_preload_N']==1.5 and result['native_authoring_readback_verified'] is True
    assert result['applied_effort_measured'] is False
    for item in result['sources'].values():
        assert item['sha256']==hashlib.sha256((tmp_path/item['file']).read_bytes()).hexdigest()


@pytest.mark.parametrize('file,field,value',[
    ('passive_preload_authoring.json','nominal_preload_N',0.),
    ('passive_preload_authoring.json','source_sha256','0'*64),
    ('passive_preload_startup_usd_readback.json','read_only_usd',False)])
def test_native_preload_semantics_cannot_use_unbound_or_changed_authoring(tmp_path,file,field,value):
    from sim.ausc_policy.astra_native import preload_observation_context
    approved_preload(tmp_path);path=tmp_path/file;data=json.loads(path.read_text());data[field]=value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):preload_observation_context(tmp_path)


def test_sensor_boundary_explains_preload_warn_and_occluded_probe_without_new_force_labels(tmp_path):
    sensor,env,_=make(tmp_path)
    bundle,_=sensor.capture(env.state(),0,np.eye(4))
    meaning=bundle['hall']['interpretation']
    assert meaning['nominal_preload_N']==1.5
    assert meaning['rest_value_indicates_patient_normal_contact'] is False
    assert meaning['decoded_spring_load_is_patient_normal_force'] is False
    assert meaning['preload_subtracted'] is False
    assert meaning['WARN_means']=='outside_target_force_window; can include unloaded rest'
    assert meaning['WARN_alone_is_safety_trip'] is False
    assert meaning['status_thresholds_N']['target_force_N']==4.
    assert meaning['status_thresholds_N']['target_tolerance_N']==.5
    probe=bundle['depth']['prescribed_projection_probe']
    assert probe['may_be_occluded'] is True and probe['patient_surface_verified'] is False
    assert probe['pixel_validity_verifies_anatomical_surface'] is False
    sensor.close()


def test_native_http_payload_carries_interpretation_and_preserves_decoded_value(tmp_path):
    from sim.ausc_policy.astra_native import native_observation
    from sim.ausc_policy.astra_supervisor import request_payload
    sensor,env,_=make(tmp_path)
    controller=SimpleNamespace(env=env,astra_sensors=sensor,episode_id='episode',
        manager=SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='APPROACH'))),
        target=SimpleNamespace(name='L1'),reference=SimpleNamespace(sha256='a'*64,pose_base=np.eye(4)))
    obs,rgb=native_observation(controller,'HOME');payload=request_payload(obs,rgb)
    assert payload['observation']['sensor_bundle']['hall']['interpretation']['preload_subtracted'] is False
    assert obs['hall_force_N']==obs['sensor_bundle']['hall']['decoded']['spring_force_estimate_n']
    assert obs['sensor_bundle']['hall']['force_label_used'] is False
    assert 'diaphragm_N' not in json.dumps(payload) and 'normal_force_N' not in json.dumps(payload)
    sensor.close()


def test_unloaded_warn_and_preload_context_reach_boundary_without_subtraction(tmp_path):
    from sim.ausc_policy.astra_native import preload_observation_context
    from sim.ausc_policy.astra_sensors import AstraSensors
    from test_ausc_policy_m2_capture import FakeEnv,FakeBackend,CAMERA,CALIBRATION
    approved_preload(tmp_path);context=preload_observation_context(tmp_path)
    env=FakeEnv();env.rows=[];env.base=np.eye(4);env.value['compression_m']=0.
    sensor=AstraSensors(env,tmp_path/'sensors',camera_bytes=CAMERA,calibration_bytes=CALIBRATION,
                        backend_factory=FakeBackend,preload_context=context)
    bundle,_=sensor.capture(env.state(),0,np.eye(4));hall=bundle['hall']
    assert hall['decoded']['spring_force_estimate_n']==pytest.approx(1.4975641978331033)
    assert hall['decoded']['status']=='WARN'
    assert hall['interpretation']['preload_context']==context
    assert hall['interpretation']['WARN_alone_is_safety_trip'] is False
    manifest=json.loads((sensor.output/'manifest.json').read_text())
    assert manifest['hall_interpretation']==hall['interpretation']
    sensor.close()

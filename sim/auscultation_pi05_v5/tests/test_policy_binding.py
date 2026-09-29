import io,json
import pytest
from sim.auscultation_pi05_v5.policy_client import PolicyClient
from sim.auscultation_pi05_v5.training_data import MODALITIES


def config_and_reply(tmp_path,monkeypatch,**overrides):
    config=dict(token='test-token',client_port=18915,expected_checkpoint_metadata_sha256='checkpoint',
        expected_inference_source_sha256={'actor.py':'source'},expected_data_hash='data')
    reply=dict(checkpoint_metadata_sha256='checkpoint',inference_source_sha256={'actor.py':'source'},
        data_hash='data',action_semantics='measured_joint_command_offset_v1',sampling_steps=10,
        sampling_seed=917,training_recipe='phase_balanced_measured_joint_actions_v5',modalities=MODALITIES,audio='absent')
    reply.update(overrides)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:io.BytesIO(json.dumps(reply).encode()))
    p=tmp_path/'private.json';p.write_text(json.dumps(config));return p


@pytest.mark.parametrize('bad',[
    {'action_semantics':'measured_tcp_command_offset_v4'}, {'sampling_steps':1},
    {'training_recipe':'phase_balanced_phase_action_scaling_v4'},
    {'inference_source_sha256':{'actor.py':'changed'}}, {'data_hash':'other'},
    {'modalities':dict(MODALITIES,hall='ideal_force')},
])
def test_rejects_wrong_representation_or_runtime_despite_matching_checkpoint(tmp_path,monkeypatch,bad):
    p=config_and_reply(tmp_path,monkeypatch,**bad)
    with pytest.raises(ValueError):PolicyClient(p)


def test_accepts_only_explicit_joint_checkpoint_and_runtime(tmp_path,monkeypatch):
    p=config_and_reply(tmp_path,monkeypatch)
    assert PolicyClient(p).binding['action_semantics']=='measured_joint_command_offset_v1'


def test_saved_instruction_change_is_rejected_before_model_loading():
    from sim.auscultation_pi05_v5.openpi_adapter import INFERENCE_RECIPE,verify_inference_recipe
    saved=dict(INFERENCE_RECIPE,canonical_phase_instructions={'HOLD':'Maintain contact.'})
    verify_inference_recipe(saved,{'HOLD':'Maintain contact.'})
    with pytest.raises(ValueError,match='instruction'):
        verify_inference_recipe(saved,{'HOLD':'Retract immediately.'})
    with pytest.raises(ValueError,match='action_horizon'):
        verify_inference_recipe(dict(saved,action_horizon=1),saved['canonical_phase_instructions'])


@pytest.mark.parametrize('tamper',[None,'request_payload_sha256','rgb_pixel_sha256','state_f32_sha256','raw_action_f32_sha256'])
def test_reply_must_attest_the_actual_request_pixels_state_and_actions(monkeypatch,tamper):
    import hashlib,base64
    import numpy as np
    from PIL import Image
    client=PolicyClient.__new__(PolicyClient);client.url='http://127.0.0.1:0';client.headers={};client.count=0
    client.binding=dict(checkpoint_metadata_sha256='checkpoint')
    def response(request,**kwargs):
        req=json.loads(request.data);rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(req['rgb_png']))))
        raw=np.zeros((10,7),dtype='<f4')
        reply=dict(client.binding,request_id=req['request_id'],raw_actions=raw.tolist(),
            request_payload_sha256=hashlib.sha256(request.data).hexdigest(),rgb_shape=list(rgb.shape),
            rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            state_f32_sha256=hashlib.sha256(np.asarray(req['state'],dtype='<f4').tobytes()).hexdigest(),
            raw_action_f32_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        if tamper:reply[tamper]='changed'
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr('urllib.request.urlopen',response)
    args=(np.zeros((2,3,3),np.uint8),np.arange(32,dtype=np.float32),'FREE','R4')
    if tamper:
        with pytest.raises(ValueError,match='attestation'):client.predict(*args)
    else:
        result,_=client.predict(*args);assert result.shape==(10,7)

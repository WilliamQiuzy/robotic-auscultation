"""Synthetic V11 transport/identity tests; no scientific admission or model."""
import copy
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import pytest


def module(name):
    name='sim.auscultation_pi05_v11.'+name
    assert importlib.util.find_spec(name) is not None, 'Missing accepted V11 runtime module'
    return importlib.import_module(name)


def fixture(monkeypatch):
    C=module('policy_contract')
    checkpoint=dict(schema=C.CHECKPOINT_SCHEMA,checkpoint_path='/synthetic/run/model/step-000000000500',local_step=500,optimizer_step=500,schedule_cursor=2000,cumulative_trainable_update_age=7800,metadata_sha256='a'*64,arrays_sha256='b'*64,complete_sha256='c'*64,data_hash='d'*64,normalization_sha256='e'*64,config_sha256='f'*64,schedule_sha256='1'*64,source_sha256={'sim/synthetic_training.py':'2'*64})
    dataset=dict(mode=C.codec.MODE,schema='wam_pi05_v11_bounded_dataset_binding_v1',data_hash='d'*64,normalization_file_binding=dict(path='/synthetic/norm.json',sha256='e'*64),normalization_object_sha256='3'*64)
    info=dict(schema=C.RUNTIME_SCHEMA,candidate_admitted=True,service_eligible=True,checkpoint_binding=checkpoint,dataset_binding=dataset,admission_binding=dict(path='/synthetic/run/postlock/admission.json',sha256='4'*64),selection_lock_binding=dict(path='/synthetic/run/selection/selection.lock.json',sha256='5'*64),normalization_object_sha256='3'*64,postlock_producer_source_binding={'/synthetic/producer.py':'6'*64},runtime_inventory_binding=dict(path='/synthetic/inventory.json',sha256='7'*64),training_source_sha256=checkpoint['source_sha256'])
    metadata=dict(config=dict(training_schema=C.TRAINING_SCHEMA,source_sha256=checkpoint['source_sha256'],**{k:'8'*64 for k in C.DYNAMICS_FIELDS}))
    monkeypatch.setattr(C,'inference_source_hashes',lambda:{'sim/auscultation_pi05_v11/'+n:'9'*64 for n in C.INFERENCE_MODULES})
    monkeypatch.setattr(C,'connection_source_hashes',lambda:{'sim/auscultation_pi05_v11/'+n:'9'*64 for n in C.CONNECTION_MODULES})
    return C,info,C.make_binding(info,'4'*64,metadata)


def test_truthful_v10_binding_has_independent_identities(monkeypatch):
    C,a,b=fixture(monkeypatch);C.verify_against_admission(b,a,'4'*64)
    assert b['schema']==C.POLICY_SCHEMA and b['checkpoint_step']==500 and b['cumulative_trainable_update_age']==7800
    assert b['action_encoding_binding']['mode']==C.codec.MODE and b['bounded_by_codec'] and not b['learned_safety_validated']
    assert not any(k.startswith('pair_') for k in b)


@pytest.mark.parametrize('field,value',[('checkpoint_step',500.),('checkpoint_optimizer_step',7800),('training_recipe','old-pair'),('admission_binding',{'path':'/foreign','sha256':'4'*64}),('normalization_object_sha256','0'*64)])
def test_wire_alias_and_cross_generation_reject(monkeypatch,field,value):
    C,a,b=fixture(monkeypatch);b=copy.deepcopy(b);b[field]=value
    with pytest.raises(ValueError):C.verify_against_admission(b,a,'4'*64)


def test_complete_expected_configuration_required(monkeypatch):
    C,a,b=fixture(monkeypatch);c={'expected_'+k:b[k] for k in C.CONFIG_FIELDS}
    C.verify_config_binding(c,b);c.pop('expected_admission_binding')
    with pytest.raises(ValueError):C.verify_config_binding(c,b)


def test_transport_preserves_unsafe_raw_and_rejects_subfloat32_and_binding(monkeypatch):
    C,a,b=fixture(monkeypatch);client=module('policy_client');v=client.PolicyClient.__new__(client.PolicyClient)
    v.binding=b;v.url='http://127.0.0.1:1';v.headers={};v.count=0
    rgb=np.zeros((4,5,3),np.uint8);state=np.arange(41,dtype=np.float32);raw=np.full((10,7),20,np.float32);mutate=['none']
    def response(request,timeout):
        q=json.loads(request.data);r=copy.deepcopy(b);r.update(request_id=q['request_id'],codec_target=q['target'],codec_phase=q['phase'],raw_actions=raw.tolist(),request_payload_sha256=hashlib.sha256(request.data).hexdigest(),rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),rgb_shape=list(rgb.shape),state_f32_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest(),raw_action_f32_sha256=hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest())
        if mutate[0]=='float':r['raw_actions'][0][0]=20.00000001
        if mutate[0]=='alias':r['checkpoint_step']=500.
        if mutate[0]=='nested':r['checkpoint_binding']['optimizer_step']=500.
        return io.BytesIO(json.dumps(r).encode())
    monkeypatch.setattr(client.urllib.request,'urlopen',response)
    actual,_=v.predict(rgb,state,'FREE','R4');np.testing.assert_array_equal(actual,raw)
    for change in ('float','alias','nested'):
        mutate[0]=change
        with pytest.raises(ValueError):v.predict(rgb,state,'FREE','R4')
    assert v.count==1


@pytest.mark.parametrize('port',[True,'8830',80,65536,'1@foreign'])
def test_port_rejected_without_request(tmp_path,monkeypatch,port):
    client=module('policy_client');p=tmp_path/'private.json';p.write_text(json.dumps(dict(client_port=port,token='synthetic')))
    monkeypatch.setattr(client.urllib.request,'urlopen',lambda *a,**k:pytest.fail('network forbidden'))
    with pytest.raises(ValueError):client.PolicyClient(p)


def test_negative_admission_precedes_model(tmp_path,monkeypatch):
    actor=module('actor');B=module('admission_bridge')
    monkeypatch.setattr(B,'selected_admission_with_receipt',lambda *a,**k:({'candidate_admitted':False,'service_eligible':False},{}))
    with pytest.raises(ValueError,match='Positive'):actor.load_actor('/synthetic/admission','a'*64,verifier_binding={})


def test_bound_json_rejects_changed_bytes(tmp_path):
    actor=module('actor');p=tmp_path/'x.json';p.write_text('{"x":1}');h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actor.read_bound_json(p,h)=={'x':1};p.write_text('{"x":2}')
    with pytest.raises(ValueError):actor.read_bound_json(p,h)


def test_nested_json_source_snapshot_and_escape(tmp_path,monkeypatch):
    service=module('policy_service');root=tmp_path/'source';f=root/'sim/nested/posture.json';f.parent.mkdir(parents=True);f.write_text('{}')
    output=tmp_path/'snap';h=hashlib.sha256(f.read_bytes()).hexdigest()
    service.write_source_snapshot(output,{'sim/nested/posture.json':h},source_project=root)
    assert (output/'sim/nested/posture.json').read_bytes()==b'{}'
    with pytest.raises(ValueError):service.write_source_snapshot(tmp_path/'bad',{'../escape':'a'*64},source_project=root)
    assert not (tmp_path/'bad').exists()


def test_original_absolute_producer_snapshot_keeps_paths_and_bytes(tmp_path):
    service=module('policy_service');C=module('policy_contract')
    source=tmp_path/'original376/sim/config/plan.json';source.parent.mkdir(parents=True);source.write_text('{"synthetic":true}')
    external=tmp_path/'external/postlock.py';external.parent.mkdir();external.write_text('# synthetic producer')
    expected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,external)}
    records=service.write_producer_snapshot(tmp_path/'snap',expected)
    assert set(records)==set(expected)
    for p,record in records.items():
        assert record==dict(relative_path=C.producer_snapshot_key(p),sha256=expected[p])
        assert (tmp_path/'snap'/record['relative_path']).read_bytes()==Path(p).read_bytes()
    source.write_text('changed')
    with pytest.raises(ValueError):service.write_producer_snapshot(tmp_path/'bad',expected)
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('field,value',[('bounded_by_codec',False),('action_clipping',True),('posthoc_projection',True),('learned_safety_validated',True),('model_action_semantics','future_command_minus_fixed_observation_measured_q')])
def test_architectural_identity_is_strict(monkeypatch,field,value):
    C,a,b=fixture(monkeypatch);b[field]=value
    with pytest.raises(ValueError):C.verify_against_admission(b,a,'4'*64)


def test_decoder_target_reply_must_match_actual_request(monkeypatch):
    C,a,b=fixture(monkeypatch);client=module('policy_client');v=client.PolicyClient.__new__(client.PolicyClient)
    v.binding=b;v.url='http://127.0.0.1:1';v.headers={};v.count=0
    def response(request,timeout):
        q=json.loads(request.data);r=copy.deepcopy(b);r.update(request_id=q['request_id'],codec_phase=q['phase'],codec_target='L4')
        return io.BytesIO(json.dumps(r).encode())
    monkeypatch.setattr(client.urllib.request,'urlopen',response)
    with pytest.raises(ValueError,match='phase/target'):v.predict(np.zeros((2,2,3),np.uint8),np.zeros(41,np.float32),'FREE','R4')
    assert v.count==0


@pytest.mark.parametrize('target,phase',[('R4','FREE'),('L4','CONTACT'),('unknown','FREE'),('R4','invalid')])
def test_actual_actor_predict_body_uses_explicit_interval_and_retains_latent(target,phase):
    """Exact nested production body with fake sampler only; no model initialized."""
    import ast,time
    from types import SimpleNamespace
    from sim.auscultation_pi05_v11.tests.test_admission_bridge import synthetic_normalization
    actor=module('actor');norm=synthetic_normalization();tree=ast.parse(Path(actor.__file__).read_text())
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_actor')
    body=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    z=np.linspace(-3,3,70,dtype=np.float32).reshape(1,10,7);calls=[]
    def sample(*args):calls.append(args);return z.copy()
    scope=dict(np=np,contract=actor.contract,codec=actor.codec,norm=norm,time=time,sample=sample,runtime={'tokenizer':None,'trainable':None,'frozen':None},input_contract=actor.input_contract,
      observation=lambda rgb,state,p,t,token,n:{'state':state[None].copy(),'rgb':rgb[None].copy()},stack_observations=lambda rows:rows[0],
      jax=SimpleNamespace(device_put=lambda x:x,random=SimpleNamespace(key=lambda seed:seed)),model_api=SimpleNamespace(Observation=lambda **obs:obs))
    exec(compile(ast.Module(body=[body],type_ignores=[]),str(actor.__file__),'exec'),scope)
    state=np.arange(41,dtype=np.float32)/100;rgb=np.zeros((2,2,3),np.uint8)
    if target=='unknown' or phase=='invalid':
        with pytest.raises(ValueError):scope['predict'](rgb,state,phase,target)
        assert not calls;return
    raw,record=scope['predict'](rgb,state,phase,target)
    expected=actor.codec.decode_chunk(z[0],state[:7],phase,target,norm['residual_actions'])
    assert raw.tobytes()==expected.tobytes() and raw.dtype==np.float32 and calls[0][-1]==917
    assert record['codec_target']==target and record['codec_phase']==phase and record['observation_anchor_q']==state[:7].tolist()
    latent=np.asarray(record['model_latent_residual'],np.float64)
    assert latent.tobytes()==actor.codec.denormalize_chunk(z[0],phase,target,norm['residual_actions']).tobytes()

import importlib
from pathlib import Path
import copy
import pytest

def module():
 assert (Path(__file__).parents[1]/'coverage_pilot_policy_client.py').is_file(),'Pilot wire binding is not implemented'
 return importlib.import_module('sim.auscultation_pi05_v7.coverage_pilot_policy_client')

def valid(m):
 from sim.auscultation_pi05_v7 import coverage_pilot_train as t
 from sim.auscultation_pi05_v7 import coverage_pilot_schedule as s
 sha='a'*64
 pb=dict(schema=t.VERSION,training_recipe=t.RECIPE,manifest_sha256=t.MANIFEST_SHA256,data_hash=t.DATA_HASH,
  normalization_sha256=sha,bootstrap_metadata_sha256=t.BOOTSTRAP_METADATA_SHA256,bootstrap_arrays_sha256=t.BOOTSTRAP_ARRAYS_SHA256,
  bootstrap_complete_sha256=sha,bootstrap_optimizer_step=4000,schedule_sha256=sha,records_sha256=sha,additional_steps=14600,
  train_episode_ids=['ep'+str(i) for i in range(8)],train_episodes=8,diagnostic_only=True,
  model_representation='unchanged fixed observed q residual decoded to absolute command',normalization_refit=False,
  initialization='exact old4000 parameters, Adam state/count, and JAX flow RNG; new frozen schedule cursor zero',
  sampling_steps=10,sampling_seed=917,heldout_prediction=False,closed_loop_acceptance=False)
 return dict(checkpoint_metadata_sha256=sha,action_semantics=m.VERSION,model_action_semantics='future_command_minus_fixed_observation_measured_q',
  sampling_steps=10,sampling_seed=917,training_recipe=t.RECIPE,modalities=m.MODALITIES,audio='absent',observation_contract=m.observation_binding(),action_encoding_binding=m.action_encoding_binding(),
  inference_source_sha256={k:sha for k in m.inference_source_paths()},data_hash=t.DATA_HASH,dynamics_recipe_sha256=sha,dynamics_contract_sha256=sha,execution_contract_sha256=sha,
  sampling_policy='same fixed common Gaussian seed at every receding-horizon query',pilot_binding=pb,pilot_source_sha256={k:sha for k in s.source_paths()},pilot_selection_sha256=sha)

def test_pilot_binding_accepts_complete_new_layer():
 m=module();x=valid(m);m.verify_policy_binding(x,'a'*64)

@pytest.mark.parametrize('change',['old_recipe','missing_pilot','missing_source','missing_selection','wrong_bootstrap','wrong_model','old_checkpoint','test','refit'])
def test_old_or_incomplete_policy_is_rejected(change):
 m=module();x=valid(m)
 if change=='old_recipe':x['training_recipe']='pi05_v7_fixed_observation_joint_residual_v1'
 elif change=='missing_pilot':del x['pilot_binding']
 elif change=='missing_source':x['pilot_source_sha256'].pop(next(iter(x['pilot_source_sha256'])))
 elif change=='missing_selection':del x['pilot_selection_sha256']
 elif change=='wrong_bootstrap':x['pilot_binding']['bootstrap_optimizer_step']=0
 elif change=='wrong_model':x['model_action_semantics']='delta_from_previous_command'
 elif change=='old_checkpoint':x['checkpoint_metadata_sha256']='b'*64
 elif change=='test':x['pilot_binding']['heldout_prediction']=True
 elif change=='refit':x['pilot_binding']['normalization_refit']=True
 with pytest.raises(ValueError):m.verify_policy_binding(x,'a'*64)

def test_private_config_must_pin_all_three_pilot_fields():
 m=module();x=valid(m);config={'expected_'+f:x[f] for f in m.CONFIG_FIELDS}
 m.verify_config_binding(config,x)
 for field in ['pilot_binding','pilot_source_sha256','pilot_selection_sha256']:
  bad=copy.deepcopy(config);bad.pop('expected_'+field)
  with pytest.raises(ValueError):m.verify_config_binding(bad,x)

def test_outer_data_hash_cannot_disagree_with_pilot_ancestry():
 m=module();x=valid(m);x['data_hash']='b'*64
 with pytest.raises(ValueError):m.verify_policy_binding(x,'a'*64)

@pytest.mark.parametrize('tamper',[None,'pilot_binding','pilot_source_sha256','pilot_selection_sha256','training_recipe'])
def test_actual_client_reply_requires_pilot_attestation(monkeypatch,tamper):
 import numpy as np,io,json,hashlib
 m=module();binding=valid(m);client=m.PolicyClient.__new__(m.PolicyClient)
 client.url='http://127.0.0.1:1';client.headers={};client.binding=binding;client.count=0
 rgb=np.zeros((2,2,3),np.uint8);state=np.zeros(41,np.float32);action=np.zeros((10,7),np.float32)
 def fake_open(request,timeout):
  body=request.data;req=json.loads(body)
  reply=dict(binding,request_id=req['request_id'],raw_actions=action.tolist(),request_payload_sha256=hashlib.sha256(body).hexdigest(),
    rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),rgb_shape=list(rgb.shape),state_f32_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest(),raw_action_f32_sha256=hashlib.sha256(action.astype('<f4').tobytes()).hexdigest())
  if tamper:reply.pop(tamper)
  return io.BytesIO(json.dumps(reply).encode())
 monkeypatch.setattr(m.urllib.request,'urlopen',fake_open)
 if tamper:
  with pytest.raises(ValueError):client.predict(rgb,state,'FREE','R4')
  assert client.count==0
 else:
  returned,_=client.predict(rgb,state,'FREE','R4');assert np.array_equal(returned,action);assert client.count==1

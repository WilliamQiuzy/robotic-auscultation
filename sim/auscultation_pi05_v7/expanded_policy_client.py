import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
import urllib.request
import uuid
import re
import numpy as np
from PIL import Image
from sim.auscultation_pi05_v6.action_contract import VERSION
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding
from .action_encoding import action_encoding_binding
from .expanded_contract import verify_experiment_binding,verify_source_map,experiment_source_paths,experiment_source_hashes

RECIPE = 'pi05_v7_fixed_observation_joint_residual_v1'
CONFIG_FIELDS = ('checkpoint_metadata_sha256','inference_source_sha256','data_hash',
                 'dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256','experiment_binding','experiment_source_sha256')
INFERENCE_SOURCES = {
    'v7': ('actor.py','policy_service.py','policy_client.py','action_encoding.py',
           'train_openpi.py','training_data.py','model.py'),
    'v6': ('openpi_adapter.py','train_openpi.py','training_data.py','action_contract.py',
           'action_encoding.py','state_encoding.py','dynamics_binding.py','dynamics_contract.py',
           'phase_contract.py','observed_supervisor.py','supervision.py','task_observation.py',
           'derive_task_dataset.py','export_joint_dataset.py','evaluation.py','learned_controller.py','execution.py','live_capture.py'),
}


def package_path(generation):
    if generation not in {'v6','v7'}:raise ValueError('Unknown runtime generation')
    spec=importlib.util.find_spec('sim.auscultation_pi05_'+generation)
    if spec is None or not spec.origin:raise ValueError('Missing explicit runtime package: '+generation)
    return Path(spec.origin).resolve().parent


def inference_source_paths():
    return {generation+'/'+name:package_path(generation)/name
            for generation,names in INFERENCE_SOURCES.items() for name in names}


def inference_source_hashes():
    return {name:hashlib.sha256(path.read_bytes()).hexdigest()
            for name,path in inference_source_paths().items()}


def verify_runtime_sources(binding):
    verify_source_map(binding.get('experiment_source_sha256'),local=True)
    if binding.get('inference_source_sha256')!=inference_source_hashes():
        raise ValueError('Runtime source bytes differ from the explicitly bound service')


def verify_policy_binding(binding,expected_checkpoint_metadata_sha256):
    if not isinstance(binding,dict):raise ValueError('Missing expanded experiment binding')
    verify_experiment_binding(binding.get('experiment_binding'))
    verify_source_map(binding.get('experiment_source_sha256'))
    def sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
    if not isinstance(binding,dict) or not sha(expected_checkpoint_metadata_sha256):
        raise ValueError('Explicit complete policy binding required')
    expected=dict(checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
        action_semantics=VERSION,model_action_semantics='future_command_minus_fixed_observation_measured_q',sampling_steps=10,sampling_seed=917,training_recipe=RECIPE,
        modalities=MODALITIES,audio='absent',observation_contract=observation_binding(),
        action_encoding_binding=action_encoding_binding())
    for key,value in expected.items():
        if binding.get(key)!=value:raise ValueError('Incompatible policy binding: '+key)
    sources=binding.get('inference_source_sha256')
    if (not isinstance(sources,dict) or set(sources)!=set(inference_source_paths())
            or not all(sha(value) for value in sources.values())):
        raise ValueError('Incomplete namespaced inference source binding')
    for key in ('data_hash','dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256'):
        if not sha(binding.get(key)):raise ValueError('Invalid policy binding: '+key)
    if not isinstance(binding.get('sampling_policy'),str) or not binding['sampling_policy']:
        raise ValueError('Missing sampling policy')


def verify_config_binding(config,binding):
    verify_policy_binding(binding,config.get('expected_checkpoint_metadata_sha256'))
    for field in CONFIG_FIELDS:
        expected=config.get('expected_'+field)
        if not expected or binding.get(field)!=expected:
            raise ValueError('Explicit runtime configuration mismatch: '+field)


class PolicyClient:
    def __init__(self,config_path):
        self.config=json.loads(Path(config_path).read_text())
        self.url='http://127.0.0.1:'+str(self.config['client_port'])
        self.headers={'Authorization':'Bearer '+self.config['token'],'Content-Type':'application/json'}
        request=urllib.request.Request(self.url,headers=self.headers)
        with urllib.request.urlopen(request,timeout=30) as r:self.binding=json.load(r)
        verify_config_binding(self.config,self.binding)
        self.count=0

    def predict(self,rgb,state,phase,target):
        buf=io.BytesIO();Image.fromarray(rgb).save(buf,format='PNG')
        identity=str(uuid.uuid4());data=dict(request_id=identity,phase=phase,target=target,
            state=np.asarray(state).tolist(),rgb_png=base64.b64encode(buf.getvalue()).decode())
        body=json.dumps(data).encode()
        expected_inputs=dict(request_payload_sha256=hashlib.sha256(body).hexdigest(),
            rgb_pixel_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),rgb_shape=list(rgb.shape),
            state_f32_sha256=hashlib.sha256(np.asarray(state,dtype='<f4').tobytes()).hexdigest())
        request=urllib.request.Request(self.url,data=body,headers=self.headers)
        started=time.monotonic()
        with urllib.request.urlopen(request,timeout=60) as r:reply=json.load(r)
        if reply['request_id']!=identity or reply['checkpoint_metadata_sha256']!=self.binding['checkpoint_metadata_sha256']:
            raise ValueError('Policy reply binding mismatch')
        for key in ('experiment_binding','experiment_source_sha256','action_encoding_binding','observation_contract','inference_source_sha256','sampling_steps','sampling_seed','sampling_policy','data_hash','action_semantics','model_action_semantics','training_recipe','modalities','audio','dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256'):
            if reply.get(key)!=self.binding.get(key):raise ValueError('Policy runtime binding changed: '+key)
        for key,value in expected_inputs.items():
            if reply.get(key)!=value:raise ValueError('Inference input attestation mismatch: '+key)
        action=np.asarray(reply['raw_actions'],float)
        if action.shape!=(10,7) or not np.isfinite(action).all():raise ValueError('Invalid policy response')
        if reply.get('raw_action_f32_sha256')!=hashlib.sha256(np.asarray(action,dtype='<f4').tobytes()).hexdigest():
            raise ValueError('Inference output attestation mismatch')
        self.count+=1;reply['round_trip_seconds']=time.monotonic()-started
        return action,reply

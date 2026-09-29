import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import uuid
import numpy as np
from PIL import Image
from .action_contract import VERSION
from .training_data import MODALITIES
from .task_observation import observation_binding
from .action_encoding import action_encoding_binding


class PolicyClient:
    def __init__(self,config_path):
        self.config=json.loads(Path(config_path).read_text())
        self.url='http://127.0.0.1:'+str(self.config['client_port'])
        self.headers={'Authorization':'Bearer '+self.config['token'],'Content-Type':'application/json'}
        request=urllib.request.Request(self.url,headers=self.headers)
        with urllib.request.urlopen(request,timeout=30) as r:self.binding=json.load(r)
        if self.binding.get('checkpoint_metadata_sha256')!=self.config.get('expected_checkpoint_metadata_sha256'):
            raise ValueError('Server does not serve the explicitly selected checkpoint')
        expected=dict(action_semantics=VERSION,sampling_steps=10,sampling_seed=917,
            training_recipe='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1',modalities=MODALITIES,audio='absent',
            observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding())
        for key,value in expected.items():
            if self.binding.get(key)!=value:raise ValueError('Incompatible policy binding: '+key)
        for field in ('inference_source_sha256','data_hash','dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256'):
            expected=self.config.get('expected_'+field)
            if not expected or self.binding.get(field)!=expected:
                raise ValueError('Explicit runtime configuration mismatch: '+field)
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
        for key in ('action_encoding_binding','observation_contract','inference_source_sha256','sampling_steps','sampling_seed','sampling_policy','data_hash','action_semantics','training_recipe','modalities','audio','dynamics_recipe_sha256','dynamics_contract_sha256','execution_contract_sha256'):
            if reply.get(key)!=self.binding.get(key):raise ValueError('Policy runtime binding changed: '+key)
        for key,value in expected_inputs.items():
            if reply.get(key)!=value:raise ValueError('Inference input attestation mismatch: '+key)
        action=np.asarray(reply['raw_actions'],float)
        if action.shape!=(10,7) or not np.isfinite(action).all():raise ValueError('Invalid policy response')
        if reply.get('raw_action_f32_sha256')!=hashlib.sha256(np.asarray(action,dtype='<f4').tobytes()).hexdigest():
            raise ValueError('Inference output attestation mismatch')
        self.count+=1;reply['round_trip_seconds']=time.monotonic()-started
        return action,reply

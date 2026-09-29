import base64
import io
import json
from pathlib import Path
import time
import urllib.request
import uuid
import numpy as np
from PIL import Image


class PolicyClient:
    def __init__(self,config_path):
        self.config=json.loads(Path(config_path).read_text())
        self.url='http://127.0.0.1:'+str(self.config['client_port'])
        self.headers={'Authorization':'Bearer '+self.config['token'],'Content-Type':'application/json'}
        request=urllib.request.Request(self.url,headers=self.headers)
        with urllib.request.urlopen(request,timeout=30) as r:self.binding=json.load(r)
        if self.binding.get('checkpoint_metadata_sha256')!=self.config.get('expected_checkpoint_metadata_sha256'):
            raise ValueError('Server does not serve the explicitly selected checkpoint')
        self.count=0

    def predict(self,rgb,state,phase,target):
        buf=io.BytesIO();Image.fromarray(rgb).save(buf,format='PNG')
        identity=str(uuid.uuid4());data=dict(request_id=identity,phase=phase,target=target,
            state=np.asarray(state).tolist(),rgb_png=base64.b64encode(buf.getvalue()).decode())
        request=urllib.request.Request(self.url,data=json.dumps(data).encode(),headers=self.headers)
        started=time.monotonic()
        with urllib.request.urlopen(request,timeout=60) as r:reply=json.load(r)
        if reply['request_id']!=identity or reply['checkpoint_metadata_sha256']!=self.binding['checkpoint_metadata_sha256']:
            raise ValueError('Policy reply binding mismatch')
        for key in ('inference_source_sha256','sampling_steps','sampling_seed','sampling_policy','data_hash','action_semantics'):
            if reply.get(key)!=self.binding.get(key):raise ValueError('Policy runtime binding changed: '+key)
        action=np.asarray(reply['raw_actions'],float)
        if action.shape!=(10,7) or not np.isfinite(action).all():raise ValueError('Invalid policy response')
        self.count+=1;reply['round_trip_seconds']=time.monotonic()-started
        return action,reply

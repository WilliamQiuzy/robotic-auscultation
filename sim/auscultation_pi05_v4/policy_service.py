"""Authenticated loopback-only inference. Accessed through owned SSH tunnels."""
import argparse
import base64
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    from .actor import load_actor
    predict,info=load_actor(a.manifest,a.checkpoint)
    if info['checkpoint_metadata_sha256']!=config['expected_checkpoint_metadata_sha256']:
        raise ValueError('Service checkpoint differs from explicit configuration')
    # Compile against one training observation before exposing the service.
    # This is startup work, never recorded as a live policy rollout.
    manifest=json.loads(a.manifest.read_text())
    record=next(r for r in manifest['episodes'] if r['split']=='train')
    with np.load(a.manifest.parent/record['path'],allow_pickle=False) as episode:
        _,warmup=predict(episode['rgb'][0],episode['state'][0],str(episode['phases'][0]),str(episode['targets'][0]))
    (a.output/'startup_warmup.json').write_text(json.dumps(dict(scope='compile only, training observation',**warmup))+'\n')
    info['inference_source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('actor.py','policy_service.py','openpi_adapter.py','training_data.py','action_contract.py','state_encoding.py')}
    snapshot=a.output/'source_snapshot';snapshot.mkdir()
    for name,expected in info['inference_source_sha256'].items():
        data=Path(__file__).with_name(name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Inference source changed during binding')
        (snapshot/name).write_bytes(data)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,status,result):
            data=json.dumps(result,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+config['token']):
                return self.send_json(401,{'error':'unauthorized'})
            self.send_json(200,info)
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+config['token']):
                return self.send_json(401,{'error':'unauthorized'})
            try:
                size=int(self.headers['Content-Length'])
                if not 1<=size<=4_000_000:raise ValueError('request size')
                req=json.loads(self.rfile.read(size));request_id=req['request_id']
                rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(req['rgb_png'],validate=True))))
                raw,timing=predict(rgb,req['state'],req['phase'],req['target'])
                response=dict(request_id=request_id,raw_actions=raw.tolist(),**timing,**info)
                with (a.output/'requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(request_id=request_id,phase=req['phase'],target=req['target'],**timing))+'\n')
                self.send_json(200,response)
            except Exception as e:self.send_json(422,{'error':type(e).__name__+': '+str(e)})
    (a.output/'binding.json').write_text(json.dumps(info,indent=2)+'\n')
    print('POLICY_READY',json.dumps(info),flush=True)
    HTTPServer(('127.0.0.1',int(config['server_port'])),Handler).serve_forever()


if __name__=='__main__':main()

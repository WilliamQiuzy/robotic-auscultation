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
from .policy_client import inference_source_paths,inference_source_hashes,verify_config_binding


def write_source_snapshot(destination,paths):
    """Preserve both generations without flattened-name collisions."""
    for label,path in paths.items():
        pieces=label.split('/')
        if (len(pieces)!=2 or pieces[0] not in {'v6','v7'} or not pieces[1].endswith('.py')
                or Path(pieces[1]).name!=pieces[1] or pieces[1] in {'.','..'}):
            raise ValueError('Invalid inference source label')
        if not Path(path).is_file():raise ValueError('Missing inference source: '+label)
    destination=Path(destination);destination.mkdir()
    result={}
    for label,path in paths.items():
        raw=Path(path).read_bytes();expected=hashlib.sha256(raw).hexdigest()
        target=destination/label;target.parent.mkdir(exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Inference source changed during snapshot')
        result[label]=expected
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    from .actor import load_actor
    predict,info=load_actor(a.manifest,a.checkpoint)
    info['inference_source_sha256']=inference_source_hashes()
    verify_config_binding(config,info)
    # Compile against one training observation before exposing the service.
    # This is startup work, never recorded as a live policy rollout.
    manifest=json.loads(a.manifest.read_text())
    record=next(r for r in manifest['episodes'] if r['split']=='train')
    with np.load(a.manifest.parent/record['path'],allow_pickle=False) as episode:
        _,warmup=predict(episode['rgb'][0],episode['state'][0],str(episode['phases'][0]),str(episode['targets'][0]))
    (a.output/'startup_warmup.json').write_text(json.dumps(dict(scope='compile only, training observation',**warmup))+'\n')
    if write_source_snapshot(a.output/'source_snapshot',inference_source_paths())!=info['inference_source_sha256']:
        raise ValueError('Inference source changed after actor load')
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
                body=self.rfile.read(size)
                if len(body)!=size:raise ValueError('Truncated request body')
                req=json.loads(body);request_id=req['request_id']
                rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(req['rgb_png'],validate=True))))
                fingerprints=dict(request_payload_sha256=hashlib.sha256(body).hexdigest(),
                    rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),rgb_shape=list(rgb.shape),
                    state_f32_sha256=hashlib.sha256(np.asarray(req['state'],dtype='<f4').tobytes()).hexdigest())
                raw,timing=predict(rgb,req['state'],req['phase'],req['target'])
                fingerprints['raw_action_f32_sha256']=hashlib.sha256(np.asarray(raw,dtype='<f4').tobytes()).hexdigest()
                response=dict(request_id=request_id,raw_actions=raw.tolist(),**fingerprints,**timing,**info)
                with (a.output/'requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(request_id=request_id,phase=req['phase'],target=req['target'],
                        checkpoint_metadata_sha256=info['checkpoint_metadata_sha256'],**fingerprints,**timing))+'\n')
                self.send_json(200,response)
            except Exception as e:self.send_json(422,{'error':type(e).__name__+': '+str(e)})
    (a.output/'binding.json').write_text(json.dumps(info,indent=2)+'\n')
    print('POLICY_READY',json.dumps(info),flush=True)
    HTTPServer(('127.0.0.1',int(config['server_port'])),Handler).serve_forever()


if __name__=='__main__':main()

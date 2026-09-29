"""Authenticated loopback V10 service; scientific verification remains external."""
import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from . import policy_contract as contract,admission_bridge as bridge


def freeze(path,value):
    with Path(path).open('xb') as f:f.write(bridge.json_bytes(value));f.flush();os.fsync(f.fileno())


def write_source_snapshot(destination,expected,*,source_project=None):
    root=contract.project() if source_project is None else Path(source_project);dest=Path(destination);sources={}
    for name,h in expected.items():
        p=root/contract.relative(name);raw=bridge.regular(p).read_bytes();contract.need(hashlib.sha256(raw).hexdigest()==h,'Source changed before snapshot');sources[name]=(p,raw)
    dest.mkdir()
    for name,(p,raw) in sources.items():
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
        contract.need(bridge.file_sha(p)==expected[name] and bridge.file_sha(target)==expected[name],'Source changed during snapshot')
    return dict(expected)


def write_producer_snapshot(destination,expected):
    """Preserve source bytes without pretending original absolute paths moved."""
    dest=Path(destination);sources={};records={}
    for name,h in expected.items():
        p=bridge.regular(name);raw=p.read_bytes();contract.need(hashlib.sha256(raw).hexdigest()==h,'Original producer changed before snapshot')
        key=contract.producer_snapshot_key(name);sources[name]=(p,raw);records[name]=dict(relative_path=key,sha256=h)
    dest.mkdir()
    for name,(p,raw) in sources.items():
        target=dest/records[name]['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw)
        contract.need(bridge.file_sha(p)==expected[name] and bridge.file_sha(target)==expected[name],'Original producer changed during snapshot')
    return records


def startup_warmup(predict,info):
    a=predict.runtime_admission;science=a['scientific_admission'];artifacts=a['artifact_sha256']
    probes=bridge.read_binding(science['probes_binding']);row=probes['rows'][0];d=a['dataset_binding']
    contract.need(row['episode_id'] in d['normalization_fit_episode_ids'],'Warmup must be an original TRAIN probe')
    manifest=bridge.read_binding(d['manifest_binding']);source=next(x for x in manifest['sources'] if x['id']=='original10');sm=bridge.read_binding(source['manifest'])
    rec=next(x for x in sm['episodes'] if x['episode_id']==row['episode_id']);contract.need(rec['split']=='train','Warmup is not TRAIN')
    p=Path(source['manifest']['path']).parent/rec['path'];contract.need(artifacts.get(str(p))==rec['sha256'] and bridge.file_sha(bridge.regular(p))==rec['sha256'],'Warmup source is not admitted')
    with np.load(p,allow_pickle=False) as z:
        start=row['start'];rgb=z['rgb'][start];state=z['state'][start];phase=str(z['phases'][start]);target=str(z['targets'][start])
    contract.need(phase==row['phase'] and target==row['target'],'Warmup descriptor changed')
    raw,receipt=predict(rgb,state,phase,target)
    contract.need(receipt['model_input_leaves']==probes['actual_inputs'][0]['leaves'],'Warmup inputs differ from admitted probe')
    nb=probes['numeric_file_binding'];contract.need(artifacts.get(nb['path'])==nb['sha256'] and bridge.file_sha(bridge.regular(nb['path']))==nb['sha256'],'Admitted probe numeric changed')
    with np.load(nb['path'],allow_pickle=False) as z:
        contract.need(np.asarray(receipt['model_normalized_residual'],np.float32).tobytes()==z['0_000000_normalized'][0].tobytes() and raw.tobytes()==z['0_000000_raw'].tobytes(),'Warmup outputs differ from selected probe')
    bridge.verify_artifacts(a)
    return dict(scope='compile only; original TRAIN probe, not rollout or repeated L5',probe_receipt_binding=science['probes_binding'],row=row,**receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('admission','admission-verifier','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('admission-sha256','admission-verifier-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();output=args.output
    contract.need(output.is_absolute() and output==output.resolve() and not output.exists(),'Fresh canonical service output required')
    for protected in (contract.project(),args.admission.parent,args.admission_verifier.parent,args.config.parent,bridge.BASE/'runs',bridge.BASE/'datasets',bridge.BASE/'public-assets'):
        q=protected.resolve();contract.need(output!=q and q not in output.parents and output not in q.parents,'Service output overlaps bound inputs')
    configpath=bridge.regular(args.config);contract.need(not configpath.stat().st_mode&0o077,'Private0600 configuration required')
    configsha=bridge.file_sha(configpath);config=bridge.read_bound(configpath,configsha)
    contract.need(type(config.get('server_port')) is int and 1024<=config['server_port']<=65535 and type(config.get('client_port')) is int and config['client_port']==config['server_port'] and type(config.get('token')) is str and len(config['token'])>=32,'Explicit private loopback configuration required')
    from .actor import load_actor
    verifier=dict(path=str(args.admission_verifier),sha256=args.admission_verifier_sha256)
    predict,info=load_actor(args.admission,args.admission_sha256,verifier_binding=verifier);contract.verify_config_binding(config,info)
    output.mkdir(parents=True,exist_ok=False);args.output=output
    freeze(output/'startup_warmup.json',startup_warmup(predict,info))
    sources={}
    for mapping in (info['training_source_sha256'],info['inference_source_sha256'],info['connection_source_sha256']):
        for k,h in mapping.items():contract.need(k not in sources or sources[k]==h,'Conflicting source namespace');sources[k]=h
    write_source_snapshot(output/'source_snapshot',sources)
    producer_snapshot=write_producer_snapshot(output/'producer_source_snapshot',info['postlock_producer_source_binding'])
    freeze(output/'producer_sources.json',dict(source_binding=info['postlock_producer_source_binding'],snapshot=producer_snapshot,admission_verifier_binding=verifier))
    freeze(output/'admission_bridge_receipt.json',predict.admission_bridge_receipt)
    contract.verify_runtime_sources(info);bridge.verify_artifacts(predict.runtime_admission);contract.need(bridge.file_sha(configpath)==configsha,'Private configuration changed')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def send_json(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authenticated(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + config['token'])

        def do_GET(self):
            if not self.authenticated():
                return self.send_json(401, {'error': 'unauthorized'})
            self.send_json(200, info)

        def do_POST(self):
            if not self.authenticated():
                return self.send_json(401, {'error': 'unauthorized'})
            try:
                size = int(self.headers['Content-Length'])
                contract.need(1 <= size <= 4_000_000, 'Request size')
                body = self.rfile.read(size)
                contract.need(len(body) == size, 'Truncated request body')
                req = json.loads(body)
                rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(req['rgb_png'], validate=True))))
                contract.need(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'RGB uint8 image required')
                fingerprints = dict(request_payload_sha256=hashlib.sha256(body).hexdigest(),
                    rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), rgb_shape=list(rgb.shape),
                    state_f32_sha256=hashlib.sha256(np.asarray(req['state'], dtype='<f4').tobytes()).hexdigest())
                raw, timing = predict(rgb, req['state'], req['phase'], req['target'])
                fingerprints['raw_action_f32_sha256'] = hashlib.sha256(np.asarray(raw, dtype='<f4').tobytes()).hexdigest()
                response = dict(request_id=req['request_id'], raw_actions=raw.tolist(), **fingerprints, **timing, **info)
                with (args.output / 'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(request_id=req['request_id'], phase=req['phase'], target=req['target'],
                        checkpoint_metadata_sha256=info['checkpoint_metadata_sha256'],
                        admission_binding=info['admission_binding'],
                        training_source_sha256=info['training_source_sha256'], **fingerprints, **timing), allow_nan=False) + '\n')
                self.send_json(200, response)
            except Exception as exc:
                self.send_json(422, {'error': type(exc).__name__ + ': ' + str(exc)})

    freeze(output/'binding.json',info)
    contract.verify_runtime_sources(info);bridge.verify_artifacts(predict.runtime_admission)
    print('POLICY_READY',json.dumps(info,allow_nan=False),flush=True)
    HTTPServer(('127.0.0.1',config['server_port']),Handler).serve_forever()


if __name__=='__main__':main()

"""Only an admitted V8 B checkpoint is exposed by this loopback service."""
import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import policy_contract as contract


def write_source_snapshot(destination, expected):
    destination = Path(destination)
    sources = {}
    for label, digest in expected.items():
        pieces = label.split('/')
        contract.need(len(pieces) == 2 and pieces[0] in {'v6', 'v7', 'v8'} and
            pieces[1].endswith('.py') and Path(pieces[1]).name == pieces[1], 'Invalid namespaced source label')
        path = Path(__file__).resolve().parent.parent / ('auscultation_pi05_' + pieces[0]) / pieces[1]
        data = path.read_bytes()
        contract.need(hashlib.sha256(data).hexdigest() == digest, 'Source changed before snapshot: ' + label)
        sources[label] = (path, data)
    destination.mkdir()
    for label, (path, data) in sources.items():
        target = destination / label
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        contract.need(hashlib.sha256(path.read_bytes()).hexdigest() == expected[label],
                      'Source changed during snapshot: ' + label)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-admission', type=Path, required=True)
    parser.add_argument('--pair-admission-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from . import train as training
    from .actor import load_actor
    training.separate(args.output, [Path(__file__).resolve().parents[2], args.pair_admission.resolve().parent,
                                  args.config.resolve().parent, training.ROOT / 'runs',
                                  training.ROOT / 'datasets', training.ROOT / 'public-assets'])
    config = json.loads(args.config.read_text())
    contract.need(type(config.get('server_port')) is int and 1024 <= config['server_port'] <= 65535 and
                  isinstance(config.get('token'), str) and len(config['token']) >= 32,
                  'Explicit loopback port and nonempty private authentication required')
    args.output.mkdir(parents=True, exist_ok=False)
    predict, info = load_actor(args.pair_admission, args.pair_admission_sha256)
    contract.verify_config_binding(config, info)
    metadata = training.read(Path(info['checkpoint']) / 'metadata.json')
    saved = training.read(Path(metadata['config']['training_run_path']) / 'configuration.json')
    manifest = training.read(saved['manifest'])
    record = next(row for row in manifest['episodes'] if row['split'] == 'train')
    with np.load(Path(saved['manifest']).parent / record['path'], allow_pickle=False) as episode:
        _, warmup = predict(episode['rgb'][0], episode['state'][0], str(episode['phases'][0]), str(episode['targets'][0]))
    training.freeze(args.output / 'startup_warmup.json', dict(scope='compile only, training observation', **warmup))
    write_source_snapshot(args.output / 'source_snapshot', {**info['training_source_sha256'], **info['inference_source_sha256']})
    contract.verify_runtime_sources(info)

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
                        pair_admission_sha256=info['pair_admission_sha256'],
                        training_source_sha256=info['training_source_sha256'], **fingerprints, **timing), allow_nan=False) + '\n')
                self.send_json(200, response)
            except Exception as exc:
                self.send_json(422, {'error': type(exc).__name__ + ': ' + str(exc)})

    training.freeze(args.output / 'binding.json', info)
    print('POLICY_READY', json.dumps(info, allow_nan=False), flush=True)
    HTTPServer(('127.0.0.1', config['server_port']), Handler).serve_forever()


if __name__ == '__main__':
    main()

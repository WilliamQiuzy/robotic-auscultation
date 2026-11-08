"""CPU-only, token-gated loopback Astra inference; never export account auth."""
import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .astra_supervisor import (MODEL, SYSTEM, MAX_IMAGE_BYTES, MAX_REQUEST_BYTES,
    canonical, decision_schema, need, request_payload, validate_decision)

DEFAULT_CODEX = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex'


class CodexClient:
    def __init__(self, executable=DEFAULT_CODEX, timeout_s=120.):
        need(type(timeout_s) in (int, float) and 0 < timeout_s <= 150., 'Bounded inference timeout required')
        self.executable, self.timeout_s = executable, float(timeout_s)

    def __call__(self, obs, rgb):
        schema = decision_schema(obs)
        started = time.monotonic()
        # Linux retains the authoritative observation/image/proposal. Mac image,
        # schema and last-response files exist only during this inference call.
        with tempfile.TemporaryDirectory(prefix='wam-astra-') as tmp:
            folder = Path(tmp); output = folder/'decision.json'; schema_file = folder/'schema.json'
            schema_file.write_bytes(canonical(schema))
            command = [self.executable, 'exec', '--model', MODEL, '--ignore-user-config',
                '--ignore-rules', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only', '--json',
                '--output-schema', str(schema_file), '-o', str(output)]
            if rgb is not None:
                need(isinstance(rgb, bytes) and 0 < len(rgb) <= MAX_IMAGE_BYTES, 'RGB image size invalid')
                path = folder/'rgb.png'; path.write_bytes(rgb); command += ['-i', str(path)]
            command += ['-']
            result = subprocess.run(command, input=SYSTEM+'\n\n'+canonical(obs).decode(),
                capture_output=True, text=True, timeout=self.timeout_s, cwd=folder)
            if result.returncode != 0:
                raise RuntimeError('Astra codex exec failed with exit '+str(result.returncode))
            need(output.is_file() and output.stat().st_size <= 16384, 'Bounded structured last response missing')
            value = validate_decision(json.loads(output.read_text()), obs)
            events = []
            for line in result.stdout.splitlines():
                try: event = json.loads(line)
                except ValueError: continue
                if event.get('type') in ('thread.started', 'turn.started', 'turn.completed', 'turn.failed'):
                    events.append(event)
            need(not any(x.get('type') == 'turn.failed' for x in events), 'Astra inference turn failed')
            return dict(schema='ausc_astra_codex_proposal_v1', model=MODEL, backend='codex_exec',
                decision=value, inference_wall_s=time.monotonic()-started, events=events,
                ephemeral=True, model_fallback=False, executed=False, physical_success=None,
                service_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_source_sha256=hashlib.sha256(Path(__file__).with_name('astra_supervisor.py').read_bytes()).hexdigest(),
                invocation=dict(model=MODEL,sandbox='read-only',ignore_user_config=True,ignore_rules=True,
                    ephemeral=True,output_schema=True,timeout_s=self.timeout_s))


def make_server(port, token, client):
    need(isinstance(token, str) and len(token) >= 24, 'Use a random per-run token of at least 24 characters')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # Never print prompts, images or bearer tokens.

        def respond(self, code, value):
            data = canonical(value); self.send_response(code)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_POST(self):
            if self.path != '/decision':
                self.respond(404, dict(error='unknown endpoint')); return
            supplied = self.headers.get('Authorization', '')
            if not hmac.compare_digest(supplied.encode(), ('Bearer '+token).encode()):
                self.respond(401, dict(error='unauthorized')); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                need(0 < size <= MAX_REQUEST_BYTES, 'Request size invalid')
                self.connection.settimeout(10.)
                data = self.rfile.read(size)
                need(len(data) == size, 'Incomplete request')
                payload = json.loads(data)
                need(isinstance(payload, dict) and set(payload) == {'observation', 'rgb_png_base64'},
                     'Invalid request envelope')
                encoded = payload['rgb_png_base64']
                rgb = None if encoded is None else base64.b64decode(encoded, validate=True)
                request_payload(payload['observation'], rgb)
                result = client(payload['observation'], rgb)
                self.respond(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, dict(error=type(error).__name__, reason=str(error)))
            except Exception as error:
                self.respond(503, dict(error=type(error).__name__, reason='Astra inference failed; no fallback'))
    server = HTTPServer(('127.0.0.1', port), Handler)
    server.timeout = 1.
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8123); p.add_argument('--codex', default=DEFAULT_CODEX)
    p.add_argument('--token-env', default='AUSC_ASTRA_RUN_TOKEN')
    p.add_argument('--timeout-s', type=float, default=120.)
    args = p.parse_args()
    server = make_server(args.port, os.environ.get(args.token_env, ''), CodexClient(args.codex, args.timeout_s))
    print('ASTRA_LOOPBACK_READY '+json.dumps(dict(host='127.0.0.1', port=server.server_port, model=MODEL)), flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == '__main__': main()

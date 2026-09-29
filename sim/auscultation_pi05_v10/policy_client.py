"""Authenticated loopback V10 transport with full admitted-policy identity."""
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
from .policy_contract import verify_config_binding, verify_policy_binding, verify_runtime_sources, need


class PolicyClient:
    def __init__(self, config_path):
        self.config = json.loads(Path(config_path).read_text())
        need(type(self.config.get('client_port')) is int and 1024 <= self.config['client_port'] <= 65535,
             'Explicit loopback client port required')
        self.url = 'http://127.0.0.1:' + str(self.config['client_port'])
        self.headers = {'Authorization': 'Bearer ' + self.config['token'], 'Content-Type': 'application/json'}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=self.headers), timeout=30) as response:
            self.binding = json.load(response)
        verify_config_binding(self.config, self.binding)
        verify_runtime_sources(self.binding)
        self.count = 0

    def predict(self, rgb, state, phase, target):
        pixels = np.asarray(rgb)
        raw_state = np.asarray(state, dtype='<f4')
        need(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3, 'RGB uint8 image required')
        need(raw_state.shape == (41,) and np.isfinite(raw_state).all(), 'Finite original 41D state required')
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format='PNG')
        identity = str(uuid.uuid4())
        payload = dict(request_id=identity, phase=phase, target=target, state=raw_state.tolist(),
                       rgb_png=base64.b64encode(stream.getvalue()).decode())
        body = json.dumps(payload, allow_nan=False).encode()
        inputs = dict(request_payload_sha256=hashlib.sha256(body).hexdigest(),
            rgb_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), rgb_shape=list(pixels.shape),
            state_f32_sha256=hashlib.sha256(raw_state.tobytes()).hexdigest())
        request = urllib.request.Request(self.url, data=body, headers=self.headers)
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.load(response)
        need(reply.get('request_id') == identity, 'Policy response request identity differs')
        verify_policy_binding(reply, self.binding['checkpoint_metadata_sha256'])
        for key, value in self.binding.items():
            need(reply.get(key) == value, 'Policy binding changed: ' + key)
        for key, value in inputs.items():
            need(reply.get(key) == value, 'Inference input attestation differs: ' + key)
        action = np.asarray(reply['raw_actions'], float)
        need(action.shape == (10, 7) and np.isfinite(action).all(), 'Invalid policy response')
        with np.errstate(over='ignore', invalid='ignore'):
            canonical = np.asarray(action, dtype='<f4')
        need(np.isfinite(canonical).all() and np.array_equal(action, canonical.astype(float)),
             'Wire raw actions differ from exact model float32 values')
        need(reply.get('raw_action_f32_sha256') == hashlib.sha256(canonical.tobytes()).hexdigest(),
             'Inference raw output attestation differs')
        self.count += 1
        reply['round_trip_seconds'] = time.monotonic() - started
        return action, reply

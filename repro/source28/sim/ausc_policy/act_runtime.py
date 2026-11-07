"""Local CPU ACT inference only; no robot, simulator, dataset or training API.

Native Python owns `serve`; Isaac Python uses ACTClient. Both speak one bounded
JSON line per connection at 127.0.0.1. A ready receipt pins weights, source config,
the run's action contract and its normalization. verify_reload is an explicit
CPU validation mode; importing this module never imports torch or safetensors.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import sys
import time
import zipfile

import numpy as np

from . import act_baseline as a

HOST = '127.0.0.1'
MAX_MESSAGE_BYTES = 16384
BINDINGS = ('request_id', 'task_id', 'episode_id', 'frame_id', 'target_revision',
            'sim_stamp_s', 'target_id', 'phase')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass
class Artifacts:
    config: dict
    checkpoint: Path
    stats: dict
    receipt: dict
    native_sources: dict


class ReloadMismatch(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def load_artifacts(config_path, checkpoint, checkpoint_sha256, *, environment_width=12):
    """Bind a completed local run. Never consult source datasets or old statistics."""
    a.need(environment_width in (12,40), 'Unknown finite environment width')
    config_path, checkpoint = Path(config_path).resolve(), Path(checkpoint).resolve()
    config = a.read_json(config_path); a.validate_config(config)
    a.need(config['mode'] in ('train32', 'train'), 'Runtime requires a completed finite training configuration')
    output = checkpoint.parent
    a.need(config == a.read_json(output/'configuration.json'), 'Source configuration differs from checkpoint run')
    steps = config['modes'][config['mode']]['steps']
    complete = a.read_json(output/'COMPLETE.json')
    a.need(checkpoint.name == f'checkpoint_{steps:04d}' and complete['completed'] is True
           and complete['mode'] == config['mode'] and complete['steps'] == steps
           and complete['checkpoint_step'] == steps
           and complete['action_representation'] == config['action_representation'],
           'Incomplete or mismatched final checkpoint')
    weights = checkpoint/'model.safetensors'
    a.need(weights.is_file() and not weights.is_symlink() and is_digest(checkpoint_sha256)
           and a.sha256(weights) == checkpoint_sha256, 'Local checkpoint SHA differs')
    policy_config = a.read_json(checkpoint/'config.json')
    expected = dict(type='act', n_obs_steps=1, chunk_size=10, n_action_steps=5,
        input_features={'observation.state': dict(type='STATE', shape=[7]),
                        'observation.environment_state': dict(type='ENV', shape=[environment_width])},
        output_features={'action': dict(type='ACTION', shape=[7])},
        normalization_mapping={key: 'IDENTITY' for key in ('STATE','ENV','ACTION')},
        temporal_ensemble_coeff=None, pretrained_backbone_weights=None, use_amp=False,
        pre_norm=False, feedforward_activation='relu', latent_dim=32, **config['model'])
    a.need(all(policy_config.get(k) == v for k, v in expected.items()), 'Saved ACT policy configuration differs')
    contract_path = output/'action_contract.json'; contract = a.read_json(contract_path)
    relative = config['action_representation'] == a.BLOCK_RELATIVE
    expected_contract = dict(action_representation=config['action_representation'], stored_source_actions=a.ABSOLUTE,
        prediction_npz_raw=a.ABSOLUTE, heldout_arrays_opened=False,
        source_prepared_manifest=config['prepared_manifest'], source_prepared_sha256=config['prepared_manifest_sha256'],
        anchor='acquired measured_q at block origin' if relative else None,
        encoding='a[t+(k+1)*0.05] - measured_q[t]' if relative else 'a[t+(k+1)*0.05]',
        decoding='add the same acquired measured_q[t] once to each horizon' if relative else 'identity',
        action_statistics='recomputed from eligible TRAIN relative blocks' if relative else 'frozen TRAIN absolute blocks')
    a.need(all(contract.get(k) == v for k, v in expected_contract.items()), 'Action contract differs from source configuration')
    normalization = output/'normalization.json'
    a.need(Path(contract['normalization']['path']).resolve() == normalization.resolve(),
           'Runtime normalization must be the action-contract-bound file in this run output')
    stats = a.pinned_json(normalization, contract['normalization']['sha256'])
    for key, width in (('state',19), ('action',7)):
        low, high = (np.asarray(stats[key][k], dtype=float) for k in ('low','high'))
        a.need(low.shape == high.shape == (width,) and np.isfinite(low).all() and np.isfinite(high).all()
               and np.all(high > low) and stats[key]['actual_clamp'] is False, 'Invalid physical normalization range')
    native = a.read_json(output/'native.json')
    a.need(native['upstream_commit'] == a.UPSTREAM_COMMIT, 'Native source commit differs')
    receipt = dict(schema='ausc_act_runtime_receipt_v1', scope='inference_only', device='cpu',
        upstream_commit=a.UPSTREAM_COMMIT, source_config_sha256=a.sha256(config_path),
        checkpoint_sha256=checkpoint_sha256, checkpoint_step=steps,
        policy_config_sha256=a.sha256(checkpoint/'config.json'), action_contract_sha256=a.sha256(contract_path),
        normalization_sha256=contract['normalization']['sha256'],
        action_representation=config['action_representation'], returned_representation=a.ABSOLUTE,
        state_dim=19, horizon=10, execute_steps=5, period_s=.05)
    return Artifacts(config, checkpoint, stats, receipt, native['source_sha256'])


class NativeEngine:
    """Strict local safetensors load into official ACT; CPU and no gradient path."""
    def __init__(self, artifacts, *, environment_width=12):
        a.need(sys.platform == 'linux' and os.environ.get('AUSC_OWNED_RUN') == '1',
               'Native CPU inference requires the Linux experiment owner')
        # In its isolated native process, hide CUDA before importing torch.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        config = copy.deepcopy(artifacts.config)
        config.update(device='cpu', runtime_scope='inference_only')
        self.torch, self.policy, sources = a.make_native(config, environment_width=environment_width)
        a.need(sources == artifacts.native_sources, 'Native runtime sources differ from trained checkpoint')
        from safetensors.torch import load_model
        missing, unexpected = load_model(self.policy, str(artifacts.checkpoint/'model.safetensors'),
                                         strict=True, device='cpu')
        a.need(not missing and not unexpected, 'Checkpoint state dictionary mismatch')
        a.need(a.sha256(artifacts.checkpoint/'model.safetensors') == artifacts.receipt['checkpoint_sha256'],
               'Checkpoint changed during strict reload')
        self.torch.set_num_threads(1)
        self.policy.requires_grad_(False)
        self.policy.eval()
        self._assert_cpu()

    def _assert_cpu(self):
        a.need(self.policy.config.device == 'cpu' and not self.policy.training
               and all(t.device.type == 'cpu' for t in list(self.policy.parameters())+list(self.policy.buffers()))
               and all(not t.requires_grad for t in self.policy.parameters()),
               'Inference policy must stay in CPU eval mode without gradients')

    def predict(self, batch):
        a.need(set(batch) == {'observation.state','observation.environment_state'}, 'Inference is label/image free')
        self._assert_cpu()
        with self.torch.inference_mode():
            prediction = self.policy.predict_action_chunk({k: self.torch.from_numpy(v) for k, v in batch.items()})
        a.need(prediction.device.type == 'cpu', 'Native output migrated off CPU')
        return prediction.detach().numpy().copy()


def validate_state(state):
    raw = np.asarray(state)
    a.need(raw.shape == (19,) and raw.dtype.kind in 'fiu' and np.isfinite(raw).all(), 'Finite physical state19 required')
    result = raw.astype(np.float32)
    a.need(np.isfinite(result).all(), 'State19 overflows float32')
    return result


def input_digest(state):
    return hashlib.sha256(validate_state(state).astype('<f4').tobytes()).hexdigest()


def validate_receipt(receipt):
    a.need(isinstance(receipt, dict) and receipt.get('schema') == 'ausc_act_runtime_receipt_v1'
           and receipt.get('scope') == 'inference_only' and receipt.get('device') == 'cpu'
           and receipt.get('upstream_commit') == a.UPSTREAM_COMMIT
           and receipt.get('action_representation') in (a.ABSOLUTE,a.BLOCK_RELATIVE)
           and receipt.get('returned_representation') == a.ABSOLUTE
           and (receipt.get('state_dim'),receipt.get('horizon'),receipt.get('execute_steps'),receipt.get('period_s')) == (19,10,5,.05)
           and all(is_digest(receipt.get(key)) for key in ('checkpoint_sha256', 'normalization_sha256',
               'source_config_sha256','policy_config_sha256','action_contract_sha256')), 'Invalid pinned runtime receipt')


def validate_request(request, receipt):
    a.need(isinstance(request, dict) and set(request) == {'schema','state','receipt_sha256',*BINDINGS},
           'Request fields differ; labels/images and extra fields are forbidden')
    a.need(request['schema'] == 'ausc_act_request_v1' and request['receipt_sha256'] == digest_json(receipt),
           'Request runtime receipt differs')
    for key in ('request_id','task_id','episode_id'):
        a.need(isinstance(request[key], str) and 0 < len(request[key]) <= 128
               and all(32 <= ord(c) < 127 for c in request[key]), 'Invalid request/task/episode identity')
    for key in ('frame_id','target_revision'):
        a.need(type(request[key]) is int and 0 <= request[key] <= 2**53-1, 'Invalid frame/target revision')
    stamp = request['sim_stamp_s']
    a.need(type(stamp) in (int,float) and np.isfinite(stamp) and stamp >= 0, 'Invalid simulation stamp')
    state = validate_state(request['state'])
    a.need(request['target_id'] in a.TARGETS and request['phase'] in ('FREE','APPROACH'), 'Invalid L2 task')
    a.need(abs(float(state[17])-(2*a.TARGETS.index(request['target_id'])/9-1)) < 1e-6
           and state[18] == (-1 if request['phase'] == 'FREE' else 1), 'State/task binding differs')
    return state


def build_request(state, receipt, **bindings):
    validate_receipt(receipt)
    request = dict(schema='ausc_act_request_v1', state=validate_state(state).tolist(),
                   receipt_sha256=digest_json(receipt), **bindings)
    validate_request(request, receipt)
    return request


class InferenceService:
    def __init__(self, artifacts, engine):
        validate_receipt(artifacts.receipt)
        self.artifacts, self.engine = artifacts, engine

    def predict_state(self, state):
        state = validate_state(state)
        # The original, unnormalized q is independent of all feature transforms.
        anchor = state[:7].copy()[None]
        batch = a.pack_batch([dict(state=state)], self.artifacts.stats, training=False)
        model_output = np.asarray(self.engine.predict(batch))
        a.need(model_output.shape == (1,10,7) and np.isfinite(model_output).all(), 'Invalid ACT chunk output')
        physical = a.unscale(model_output, self.artifacts.stats['action'])
        raw = a.decode_actions(physical, anchor, self.artifacts.config['action_representation'])[0]
        a.need(raw.shape == (10,7) and np.isfinite(raw).all(), 'Invalid absolute chunk output')
        return raw

    def handle(self, request):
        started = time.perf_counter()
        state = validate_request(request, self.artifacts.receipt)
        raw = self.predict_state(state)
        return dict(schema='ausc_act_response_v1', status='ok', raw=raw.tolist(),
            **{k: request[k] for k in BINDINGS}, input_sha256=input_digest(state),
            receipt=copy.deepcopy(self.artifacts.receipt), receipt_sha256=digest_json(self.artifacts.receipt),
            checkpoint_sha256=self.artifacts.receipt['checkpoint_sha256'],
            latency_ms=(time.perf_counter()-started)*1000)


def validate_response(response, request, receipt):
    a.need(isinstance(response, dict) and response.get('schema') == 'ausc_act_response_v1'
           and response.get('status') == 'ok', 'ACT inference failed closed')
    a.need(all(type(response.get(k)) is type(request[k]) and response[k] == request[k] for k in BINDINGS),
           'Response frame/task/request binding differs')
    a.need(response.get('receipt') == receipt and response.get('receipt_sha256') == digest_json(receipt)
           and response.get('checkpoint_sha256') == receipt['checkpoint_sha256'], 'Response checkpoint/receipt differs')
    a.need(response.get('input_sha256') == input_digest(request['state']), 'Response input binding differs')
    raw = np.asarray(response.get('raw'))
    a.need(raw.shape == (10,7) and raw.dtype.kind in 'fiu' and np.isfinite(raw).all(), 'Response raw chunk invalid')
    latency = response.get('latency_ms')
    a.need(type(latency) in (float,int) and np.isfinite(latency) and latency >= 0, 'Response latency invalid')
    return response


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        a.need(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read_message(connection, deadline):
    """One bounded JSON line with an overall deadline, including slow senders."""
    data = bytearray()
    while b'\n' not in data:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('ACT protocol deadline expired')
        connection.settimeout(remaining)
        try:
            part = connection.recv(min(4096, MAX_MESSAGE_BYTES+1-len(data)))
        except socket.timeout as error:
            raise TimeoutError('ACT protocol deadline expired') from error
        a.need(part, 'ACT connection ended before complete response')
        data.extend(part)
        a.need(len(data) <= MAX_MESSAGE_BYTES, 'ACT message too large')
    line, remainder = bytes(data).split(b'\n', 1)
    a.need(not remainder, 'Only one ACT message per connection is permitted')
    return json.loads(line, object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))


def send_message(connection, value, deadline):
    message = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()+b'\n'
    a.need(len(message) <= MAX_MESSAGE_BYTES, 'ACT message too large')
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('ACT protocol deadline expired')
    connection.settimeout(remaining)
    connection.sendall(message)


def make_server(service, *, port=0, request_timeout_s=5.):
    a.need(type(port) is int and 0 <= port <= 65535 and np.isfinite(request_timeout_s)
           and request_timeout_s > 0, 'Invalid localhost server settings')
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            deadline = time.monotonic()+request_timeout_s
            try:
                request = read_message(self.request, deadline)
                response = service.handle(request)
                send_message(self.request, response, deadline)
            except Exception as error:
                # Never return actions on an invalid request, mismatch or inference failure.
                try:
                    send_message(self.request, dict(schema='ausc_act_response_v1', status='error',
                        error=type(error).__name__, reason=str(error)[:256]), time.monotonic()+.1)
                except (OSError, ValueError, TimeoutError):
                    pass
    return socketserver.TCPServer((HOST, port), Handler)


class ACTClient:
    def __init__(self, port, receipt, *, timeout_s=5.):
        validate_receipt(receipt)
        a.need(type(port) is int and 0 < port <= 65535 and np.isfinite(timeout_s) and timeout_s > 0,
               'Invalid localhost client settings')
        self.port, self.receipt, self.timeout_s = port, copy.deepcopy(receipt), float(timeout_s)

    def predict(self, state, **bindings):
        request = build_request(state, self.receipt, **bindings)
        deadline = time.monotonic()+self.timeout_s
        with socket.create_connection((HOST, self.port), timeout=self.timeout_s) as connection:
            send_message(connection, request, deadline)
            response = read_message(connection, deadline)
        return validate_response(response, request, self.receipt)


def verify_reload(artifacts, states, *, engine_factory=NativeEngine, expected_raw=None, atol=1e-6):
    """Construct two independent engines and compare physical outputs, never fit."""
    states = np.asarray(states)
    a.need(states.ndim == 2 and states.shape[1] == 19 and 0 < len(states) <= 32,
           'Reload probe requires 1..32 state19 inputs')
    a.need(np.isfinite(atol) and 0 <= atol <= 1e-4, 'Invalid reload tolerance')
    states = np.stack([validate_state(state) for state in states])
    reference = None if expected_raw is None else np.asarray(expected_raw)
    if reference is not None:
        a.need(reference.shape == (len(states),10,7) and reference.dtype.kind in 'fiu'
               and np.isfinite(reference).all(), 'Invalid reference prediction')
    outputs = []
    for _ in range(2):
        service = InferenceService(artifacts, engine_factory(artifacts))
        outputs.append(np.stack([service.predict_state(state) for state in states]))
        del service
    difference = float(np.max(np.abs(outputs[0]-outputs[1])))
    reference_difference = None
    if reference is not None:
        reference_difference = float(np.max(np.abs(outputs[0]-reference)))
    passed = difference <= atol and (reference_difference is None or reference_difference <= atol)
    evidence = dict(passed=passed, scope='inference_only_reload_check', native_model_loaded=engine_factory is NativeEngine,
        device='cpu', count=len(states), max_reload_difference_rad=difference,
        max_reference_difference_rad=reference_difference, atol_rad=atol,
        receipt=artifacts.receipt, m0_passed=False, closed_loop_evaluated=False)
    if not passed:
        raise ReloadMismatch('CPU reload or saved reference prediction differs beyond tolerance', evidence)
    return evidence


def read_small_npz(path, key, tail_shape):
    """Read only the requested, shape-bounded NPY member; never load other keys."""
    a.need(Path(path).stat().st_size <= 262144, 'Reload input must be a small extracted NPZ')
    with zipfile.ZipFile(path) as archive:
        member = key+'.npy'
        a.need(archive.namelist().count(member) == 1 and archive.getinfo(member).file_size <= 65536,
               'Reload NPZ member is missing, duplicated or too large')
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            a.need(version in ((1,0),(2,0)), 'Unsupported small NPY header')
            reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = reader(stream)
            a.need(len(shape) == len(tail_shape)+1 and shape[1:] == tuple(tail_shape)
                   and 0 < shape[0] <= 32 and dtype.kind in 'fiu' and dtype.itemsize <= 8,
                   'Reload array must contain at most 32 physical observations/chunks')
        with archive.open(member) as stream:
            result = np.lib.format.read_array(stream, allow_pickle=False)
    a.need(np.isfinite(result).all(), 'Nonfinite reload array')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New owner output directory, never overwritten')
    args = parser.parse_args()
    config = a.read_json(args.config)
    a.need(config['experiment'] == 'act_runtime' and config['mode'] in ('verify_reload','serve')
           and config['checkpoint_step'] == 2000, 'Invalid finite runtime configuration')
    if config['mode'] == 'verify_reload':
        a.need(config['atol_rad'] == 1e-6 and config['time_limit_seconds'] == 600,
               'Reload tolerance/budget must remain predeclared')
    args.output.mkdir(parents=True, exist_ok=False)
    a.save_json(args.output/'configuration.json', config)
    try:
        run_root = Path(config['run_root']).resolve()
        source_config = run_root/'output/configuration.json'
        run = a.read_json(run_root/'result.json')
        a.need(run['completed'] is True and run['config'] == a.read_json(source_config),
               'Source training run is incomplete or its configuration differs')
        checkpoint = run_root/f'output/checkpoint_{config["checkpoint_step"]:04d}'
        actual_sha = a.sha256(checkpoint/'model.safetensors')
        artifacts = load_artifacts(source_config, checkpoint, config.get('checkpoint_sha256', actual_sha))
        if config['mode'] == 'verify_reload':
            states = read_small_npz(config['inputs_npz'], 'state19', (19,))
            reference = read_small_npz(config['reference_npz'], 'raw', (10,7))
            result = verify_reload(artifacts, states, expected_raw=reference, atol=config['atol_rad'])
            result.update(completed=True, mode='verify_reload', runtime_config_sha256=a.sha256(args.config),
                inputs_npz_sha256=a.sha256(config['inputs_npz']), reference_npz_sha256=a.sha256(config['reference_npz']))
            a.save_json(args.output/'COMPLETE.json', result)
        else:
            a.need(not {'inputs_npz','reference_npz'} & config.keys(), 'Server accepts no label or dataset input file')
            service = InferenceService(artifacts, NativeEngine(artifacts))
            with make_server(service, port=config.get('port',0),
                             request_timeout_s=config.get('request_timeout_s',5.)) as server:
                ready = dict(host=HOST, port=server.server_address[1], receipt=artifacts.receipt,
                             receipt_sha256=digest_json(artifacts.receipt), scope='inference_only')
                a.save_json(args.output/'ready.json', ready)
                print(json.dumps(ready, allow_nan=False), flush=True)
                try:
                    server.serve_forever(poll_interval=.1)
                except KeyboardInterrupt:
                    pass
    except Exception as error:
        a.save_json(args.output/'FAILED.json', dict(completed=False, mode=config['mode'],
            error=type(error).__name__, reason=str(error), reload_evidence=getattr(error,'evidence',None),
            m0_passed=False))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""CPU contract/transport regressions; no actual trained-policy acceptance."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from sim.auscultation_pi05_v8 import policy_contract as C, policy_client as client, actor


@pytest.fixture
def bound(monkeypatch):
    training_sources = C.source_hashes()
    inference_sources = {key: 'a' * 64 for key in C.inference_source_paths()}
    monkeypatch.setattr(C, 'inference_source_hashes', lambda: inference_sources)
    checkpoint = dict(schema='wam_horizon_pair_checkpoint_binding_v1', mode=C.codec.B,
        checkpoint_path='/synthetic/not-a-real-run/step-000000001217', local_step=1217,
        optimizer_step=1217, metadata_sha256='b' * 64, arrays_sha256='c' * 64,
        complete_sha256='d' * 64, data_hash='e' * 64, normalization_sha256='f' * 64,
        source_sha256=training_sources, config_sha256='1' * 64, schedule_sha256='2' * 64)
    dataset = dict(dataset_binding=dict(mode=C.codec.B, data_hash=checkpoint['data_hash']),
        normalization_sha256=checkpoint['normalization_sha256'], action_encoding_binding=C.codec.action_encoding_binding(C.codec.B))
    admission = dict(candidate_admitted=True, service_eligible=True,
        admitted_checkpoint_binding=checkpoint, admitted_dataset_codec_binding=dataset)
    meta = dict(config={k: '3' * 64 for k in ('dynamics_recipe_sha256', 'dynamics_contract_sha256', 'execution_contract_sha256')})
    binding = C.make_binding(admission, '4' * 64, meta)
    return admission, binding


def test_binding_requires_admitted_B_and_full_sources(bound):
    admission, binding = bound
    C.verify_against_admission(binding, admission, '4' * 64)
    C.verify_runtime_sources(binding)
    bad = copy.deepcopy(binding)
    bad['training_source_sha256'].pop(next(iter(bad['training_source_sha256'])))
    with pytest.raises(ValueError):
        C.verify_policy_binding(bad, 'b' * 64)


@pytest.mark.parametrize('change', ['A', 'age', 'old_recipe', 'norm', 'checkpoint', 'false_admission'])
def test_no_cross_arm_or_legacy_identity(bound, change):
    # Treat the independently frozen admission as immutable while corrupting
    # only the incoming wire copy (the constructor intentionally shares values).
    admission, binding = (copy.deepcopy(value) for value in bound)
    if change == 'A':
        binding['pair_mode'] = C.codec.A
    elif change == 'age':
        binding['checkpoint_binding']['optimizer_step'] += 4000
    elif change == 'old_recipe':
        binding['training_recipe'] = 'pi05_v7_coverage_pilot'
    elif change == 'norm':
        binding['dataset_codec_binding']['normalization_sha256'] = '5' * 64
    elif change == 'checkpoint':
        binding['checkpoint_binding']['arrays_sha256'] = '5' * 64
    else:
        admission['candidate_admitted'] = False
    with pytest.raises(ValueError):
        C.verify_against_admission(binding, admission, '4' * 64)


def test_config_must_pin_new_admission_and_codec(bound):
    _, binding = bound
    config = {'expected_' + key: binding[key] for key in C.CONFIG_FIELDS}
    C.verify_config_binding(config, binding)
    config.pop('expected_pair_admission_sha256')
    with pytest.raises(ValueError):
        C.verify_config_binding(config, binding)


@pytest.mark.parametrize('alias,value', [('checkpoint', '/foreign/old4000/checkpoint'),
    ('checkpoint_step', 4000), ('checkpoint_optimizer_step', 8000), ('checkpoint_step', 1217.)])
def test_checkpoint_aliases_cannot_contradict_admitted_identity(bound, alias, value):
    admission, original = bound
    binding = copy.deepcopy(original)
    binding[alias] = value
    with pytest.raises(ValueError, match='identity alias'):
        C.verify_against_admission(binding, admission, '4' * 64)


def test_transport_returns_unclipped_actions_and_rejects_rebound_reply(monkeypatch, bound):
    _, binding = bound
    instance = client.PolicyClient.__new__(client.PolicyClient)
    instance.binding = binding
    instance.url = 'http://127.0.0.1:1'
    instance.headers = {}
    instance.count = 0
    pixels = np.zeros((8, 9, 3), np.uint8)
    state = np.arange(41, dtype=np.float32) / 100
    raw = np.full((10, 7), 20., np.float32)  # Deliberately unsafe; transport must not hide it.
    mutate = ['none']

    def response(request, timeout):
        req = json.loads(request.data)
        reply = dict(binding, request_id=req['request_id'], raw_actions=raw.tolist(),
            request_payload_sha256=hashlib.sha256(request.data).hexdigest(),
            rgb_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), rgb_shape=list(pixels.shape),
            state_f32_sha256=hashlib.sha256(state.astype('<f4').tobytes()).hexdigest(),
            raw_action_f32_sha256=hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest())
        if mutate[0] == 'binding':
            reply['pair_admission_sha256'] = '6' * 64
        elif mutate[0] == 'sub_float32':
            reply['raw_actions'][0][0] = 20.00000001
        elif mutate[0] == 'alias_type':
            reply['checkpoint_step'] = float(reply['checkpoint_step'])
        elif mutate[0] == 'nested_type':
            reply['checkpoint_binding'] = dict(reply['checkpoint_binding'],
                optimizer_step=float(reply['checkpoint_binding']['optimizer_step']))
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(client.urllib.request, 'urlopen', response)
    actual, _ = instance.predict(pixels, state, 'FREE', 'R4')
    np.testing.assert_array_equal(actual, raw)
    assert instance.count == 1
    mutate[0] = 'binding'
    with pytest.raises(ValueError, match='Policy binding changed'):
        instance.predict(pixels, state, 'FREE', 'R4')
    assert instance.count == 1
    mutate[0] = 'sub_float32'
    with pytest.raises(ValueError, match='exact model float32'):
        instance.predict(pixels, state, 'FREE', 'R4')
    assert instance.count == 1
    for invalid_type in ('alias_type', 'nested_type'):
        mutate[0] = invalid_type
        with pytest.raises(ValueError):
            instance.predict(pixels, state, 'FREE', 'R4')
        assert instance.count == 1


def test_negative_admission_rejects_before_model_import(monkeypatch):
    module = SimpleNamespace(verify_pair_admission=lambda *_: dict(candidate_admitted=False, service_eligible=False))
    monkeypatch.setitem(sys.modules, 'sim.auscultation_pi05_v8.pair_admission', module)
    import sim.auscultation_pi05_v8 as package
    monkeypatch.setattr(package, 'pair_admission', module, raising=False)
    # No JAX or model import is needed to reject a failed controlled experiment.
    with pytest.raises(ValueError, match='Positive reconstructed B admission'):
        actor.load_actor('/synthetic/admission.json', '4' * 64)


@pytest.mark.parametrize('port', ['1@elsewhere.invalid', '8830', True, 80, 65536])
def test_invalid_client_port_rejects_before_any_request(monkeypatch, tmp_path, port):
    config = tmp_path / 'private.json'
    config.write_text(json.dumps(dict(client_port=port, token='synthetic-token-not-a-real-credential')))
    monkeypatch.setattr(client.urllib.request, 'urlopen', lambda *_args, **_kwargs: pytest.fail('no network request permitted'))
    with pytest.raises(ValueError, match='loopback client port'):
        client.PolicyClient(config)


def test_actor_reads_only_admitted_metadata_bytes(tmp_path):
    path = tmp_path / 'metadata.json'
    original = b'{"checkpoint":"admitted"}\n'
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    assert actor.read_bound_json(path, expected) == {'checkpoint': 'admitted'}
    path.write_text('{"checkpoint":"replaced"}\n')
    with pytest.raises(ValueError, match='changed before parsing'):
        actor.read_bound_json(path, expected)


def test_source_snapshot_rejects_path_escape_before_mkdir(tmp_path):
    from sim.auscultation_pi05_v8.policy_service import write_source_snapshot
    output = tmp_path / 'snapshot'
    with pytest.raises(ValueError):
        write_source_snapshot(output, {'v8/../private.py': 'a' * 64})
    assert not output.exists()

"""V8 integration regressions; synthetic tests are not policy task acceptance."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).parents[1]
OLD = HERE.parent / 'auscultation_pi05_v7'


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def native(name):
    if sys.platform == 'darwin':
        pytest.skip('Inherited physical native imports run in the cloud CPU snapshot')
    return importlib.import_module('sim.auscultation_pi05_v8.' + name)


class IdentityOnly(ast.NodeTransformer):
    """Reverse only enumerated V8 identity changes for physical AST comparison."""
    def visit_ImportFrom(self, node):
        old = {'runtime_contract': 'coverage_pilot_bridge_runtime_contract',
               'policy_client': 'coverage_pilot_policy_client'}
        if node.module in old and node.level == 1:
            node.module = old[node.module]
        prefix = 'sim.auscultation_pi05_v7.'
        if node.module and node.module.startswith(prefix):
            node.module = node.module[len(prefix):]
            node.level = 1
        return node

    def visit_Constant(self, node):
        values = {'wam_v8_horizon_pair_headroom_replay_v1': 'wam_v7_coverage_pilot_bridge_headroom_replay_v1',
                  'Actual V8 pair client type differs': 'Actual coverage_pilot client type differs',
                  'pi05_v8': 'pi05_v7'}
        if isinstance(node.value, str) and node.value in values:
            node.value = values[node.value]
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'args' and node.attr == 'runtime_admission':
            node.attr = 'pilot_admission'
        return node

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == 'pair_runtime_binding':
            node.arg = 'pilot_runtime_binding'
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if ast.unparse(node) == "package_path('v7') / name":
            return ast.parse('Path(__file__).with_name(name)', mode='eval').body
        return node


def equivalent(original, adapted):
    assert ast.dump(original) == ast.dump(IdentityOnly().visit(copy.deepcopy(adapted)))


def test_complete_numerical_replay_and_sensor_reconstruction_unchanged():
    old = functions(OLD / 'audit_coverage_pilot_bridge_headroom_actions.py')
    new = functions(HERE / 'audit_actions.py')
    for name in ('require', 'vector', 'close', 'digest', 'f32_digest', 'valid_hash', 'request_digest',
                 'verify_sensor', 'inherited_audit_source_paths', 'audit_original_gates',
                 'verify_runtime_evidence', 'verify_inference_snapshot', 'verify_registration', '_audit', 'audit_records'):
        equivalent(old[name], new[name])
    equivalent(old['_audit_episode_replay'], new['_audit_episode_replay'])


def test_complete_physical_acceptance_and_runner_unchanged():
    equivalent(functions(OLD / 'assess_coverage_pilot_bridge_headroom_trials.py')['inherited_check_documents'],
               functions(HERE / 'assess_trials.py')['inherited_check_documents'])
    original = functions(OLD / 'run_coverage_pilot_bridge_headroom_isaac.py')
    adapted = functions(HERE / 'run_headroom_isaac.py')
    for name in ('capture', 'physical_classes', 'validate_sequence_bundles'):
        equivalent(original[name], adapted[name])
    main = adapted['main']
    admission = next(i for i, n in enumerate(main.body) if any(isinstance(x, ast.Call) and
        isinstance(x.func, ast.Name) and x.func.id == 'verify_pair_inputs' for x in ast.walk(n)))
    kit = next(i for i, n in enumerate(main.body) if isinstance(n, ast.ImportFrom) and n.module == 'isaacsim')
    assert admission < kit


def test_truthful_source_contract_and_unchanged_execution(monkeypatch):
    c = native('runtime_contract')
    monkeypatch.setattr(c.B, 'source_contract', lambda: {'SYNTHETIC': True})
    value = c.runtime_connection_contract()
    assert len(value['connection_source_sha256']) == 5
    assert len(value['training_source_sha256']) == 24
    assert len(value['inference_source_sha256']) == 5
    assert value['executed_runner_module'] == 'sim.auscultation_pi05_v8.run_headroom_isaac'
    assert value['controller_class'] == 'sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController'
    assert value['headroom_implementation']['runtime_execution_contract']['jerk_rad_s3'] == [7.99999] * 7
    assert value['headroom_implementation']['original_whole_stream_gate']['joint_jerk_rad_s3'] == 8.
    controller, trajectory = native('run_headroom_isaac').physical_classes()
    assert controller.__module__ == 'sim.auscultation_pi05_v7.headroom_learned_controller'
    assert trajectory.__module__ == 'sim.auscultation_pi05_v7.headroom_execution'


def query_fixture(monkeypatch, tmp_path):
    a = native('audit_actions')
    from sim.auscultation_pi05_v8 import action_encoding as codec
    from sim.auscultation_pi05_v8.tests.test_action_encoding import episodes
    stats = codec.fit_stats(episodes(), codec.B)
    normalization = {'residual_actions': stats}
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'normalization': normalization}))
    checkpoint = dict(checkpoint_path=str(tmp_path), metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    binding = dict(normalization_sha256=a.E.digest(normalization), checkpoint_metadata_sha256='a' * 64,
                   pair_admission_sha256='b' * 64, training_source_sha256={'SYNTHETIC': 'c' * 64})
    admission = {'checkpoint_binding': checkpoint}
    monkeypatch.setattr(a.C, 'verify_policy_against_admission', lambda actual, _: None)
    state = np.arange(41, dtype=np.float32) / 100
    normalized = np.linspace(-3., 3., 70, dtype=np.float32).reshape(10, 7)
    raw = codec.decode_chunk(normalized, state[:7], 'FREE', stats, mode=codec.B)
    receipt = dict(binding, request_id='synthetic-1', observation_anchor_q=state[:7].tolist(),
                   model_normalized_residual=normalized.tolist(), raw_actions=raw.tolist())
    learned = [dict(phase='FREE', policy_state=state.tolist(), inference=receipt, raw_actions=raw.tolist())]
    server = [copy.deepcopy(receipt)]
    return a, learned, server, binding, admission


def test_exact_codec_replay_uses_actual_measured_anchor_and_every_horizon(monkeypatch, tmp_path):
    a, learned, server, binding, admission = query_fixture(monkeypatch, tmp_path)
    assert a.verify_experiment_requests(learned, server, binding, admission) == 1
    learned[0]['raw_actions'][9][6] += 1e-12
    with pytest.raises(ValueError, match='exact finite float32'):
        a.verify_experiment_requests(learned, server, binding, admission)


@pytest.mark.parametrize('kind', ['anchor', 'normalized', 'raw', 'duplicate', 'missing', 'metadata'])
def test_codec_replay_rejects_wrong_decode_or_changed_evidence(monkeypatch, tmp_path, kind):
    a, learned, server, binding, admission = query_fixture(monkeypatch, tmp_path)
    if kind == 'anchor':
        learned[0]['inference']['observation_anchor_q'][0] = .5
    elif kind == 'normalized':
        server[0]['model_normalized_residual'][0][0] = .5
    elif kind == 'raw':
        learned[0]['raw_actions'][8][1] = .5
    elif kind == 'duplicate':
        server.append(copy.deepcopy(server[0]))
    elif kind == 'missing':
        server.clear()
    else:
        (tmp_path / 'metadata.json').write_text('{"normalization":{}}')
    with pytest.raises(ValueError):
        a.verify_experiment_requests(learned, server, binding, admission)


def test_assessor_requires_pair_bridge_and_codec_replay():
    m = native('assess_trials')
    value = dict(schema=m.SCHEMA, passed=True, errors=[], pair_queries_checked=1,
        pair_binding_verified=True, pair_admission_verified=True, connection_source_snapshot_verified=True,
        pair_request_bindings_verified=True, admission_bridge_verified=True, codec_decode_verified=True,
        inherited_action_audit=dict(schema='wam_v8_horizon_pair_headroom_replay_v1', passed=True))
    assert m.verify_outer_audit(value) == value['inherited_action_audit']
    for key in ('pair_binding_verified', 'pair_admission_verified', 'admission_bridge_verified', 'codec_decode_verified'):
        with pytest.raises(ValueError):
            m.verify_outer_audit(dict(value, **{key: False}))


def test_snapshot_also_binds_unused_declared_objects(monkeypatch, tmp_path):
    a = native('audit_actions')
    folder = tmp_path / 'input_snapshot'
    objects = folder / 'objects'
    objects.mkdir(parents=True)
    files = {}
    def store(key, data):
        raw = json.dumps(data, sort_keys=True).encode()
        identity = hashlib.sha256(raw).hexdigest()
        (objects / identity).write_bytes(raw)
        files[key] = {'sha256': identity}
        return identity
    pair_sha = store('source/pi05_v8/pair_admission_file', {'SYNTHETIC_pair': True})
    manifest_sha = store('source/pi05_v8/pair_manifest', {'SYNTHETIC_manifest': True})
    admission = dict(pair_admission_sha256=pair_sha, manifest_sha256=manifest_sha)
    store('source/pi05_v8/pair_selected_admission', admission)
    receipt = {'SYNTHETIC_receipt': True}
    store('source/pi05_v8/pair_admission_bridge_receipt', receipt)
    connection = {'SYNTHETIC_connection': True}
    connection_sha = store('source/pi05_v8/pair_connection_contract', connection)
    unused = store('unused/source', {'SYNTHETIC_unused': 'must still be bound'})
    path = folder / 'manifest.json'
    path.write_text(json.dumps({'files': files}))
    outcome = dict(input_snapshot='input_snapshot/manifest.json', input_snapshot_manifest_sha256=a.E.file_sha(path),
                   admission_bridge_receipt=receipt, runtime_connection_contract_file_sha256=connection_sha)
    monkeypatch.setattr(a.C, 'training_source_paths', lambda: {})
    monkeypatch.setattr(a.C, 'connection_source_paths', lambda: {})
    monkeypatch.setattr(a, 'pair_inference_sources', lambda: {})
    monkeypatch.setattr(a.C, 'runtime_connection_contract', lambda: connection)
    monkeypatch.setattr(a.C.B, 'verify_receipt', lambda *_: True)
    artifacts = {}
    a.verify_connection_snapshot(tmp_path, outcome, admission, artifacts)
    assert artifacts[str(objects / unused)] == unused
    (objects / unused).write_text('corrupted unused object')
    with pytest.raises(ValueError, match='Snapshot object changed'):
        a.verify_connection_snapshot(tmp_path, outcome, admission, {})


def test_ten_singles_and_one_real_sequence_required(monkeypatch):
    m = native('assess_trials')
    cases = [dict(case_id=p, points=[p], scope='full', role='nominal_single_point') for p in m.POINTS]
    cases.append(dict(case_id='sequence', points=list(m.POINTS), scope='full', role='nominal_continuous_sequence'))
    plan = dict(schema='wam_v8_horizon_pair_headroom_acceptance_plan_v1', checkpoint_metadata_sha256='a' * 64,
                pair_admission_sha256='b' * 64, cases=cases)
    monkeypatch.setattr(m, 'assess_case', lambda record, checkpoint:
                        dict(passed=True, errors=[], pair_admission_sha256='b' * 64))
    result = m.assess(plan)
    assert result['all_declared_tests_passed'] and not result['unseen_body_pose_motion_validated']
    for change in ('missing', 'reordered', 'contact'):
        bad = copy.deepcopy(plan)
        if change == 'missing':
            bad['cases'].pop(0)
        elif change == 'reordered':
            bad['cases'][-1]['points'].reverse()
        else:
            bad['cases'][0]['scope'] = 'contact'
        with pytest.raises(ValueError):
            m.assess(bad)


@pytest.mark.parametrize('name', ['audit_actions', 'assess_trials'])
def test_cancellation_propagates_after_owned_bridge_cleanup(tmp_path, monkeypatch, name):
    m = native(name)
    receipt = {'SYNTHETIC_cancelled': True}
    admission = dict(pair_admission_path='/SYNTHETIC/pair', pair_admission_sha256='a' * 64)
    for file, value in [('outcome.json', {'pair_runtime_binding': admission}), ('collection_result.json', {}),
                        ('learned_actions_audit.json', {'pair_runtime_binding': admission}), ('observed_supervisor_audit.json', {})]:
        (tmp_path / file).write_text(json.dumps(value))
    def interrupted(*args):
        raise m.C.B.AdmissionInterrupted('SYNTHETIC cancellation', receipt)
    monkeypatch.setattr(m.C, 'selected_admission', interrupted)
    with pytest.raises(m.C.B.AdmissionInterrupted) as caught:
        if name == 'audit_actions':
            m.audit_episode(tmp_path, expected_checkpoint_metadata_sha256='b' * 64)
        else:
            monkeypatch.setattr(m, 'verify_outer_audit', lambda actions: {})
            m.assess_case(dict(case_id='SYNTHETIC', directory=str(tmp_path), points=['R4'], scope='full'), 'b' * 64)
    assert caught.value.receipt is receipt

import json

from sim.auscultation_training.run_checked_evaluation import digest_file, verify_result


def artifacts(tmp_path, accepted):
    outcome = {'success': accepted, 'requested_points': ['R1'],
               'points': [{'target': 'R1', 'success': accepted}],
               'policy': {'checkpoint_sha256': 'a' * 64},
               'dataset_role': 'student_evaluation', 'expert_demonstration': False}
    (tmp_path / 'outcome.json').write_text(json.dumps(outcome))
    (tmp_path / 'control_commands.json').write_text('[]')
    (tmp_path / 'expert_trace.jsonl').write_text('{}\n')
    hashes = {name: digest_file(tmp_path / name) for name in
              ('outcome.json', 'control_commands.json', 'expert_trace.jsonl')}
    (tmp_path / 'independent_audit.json').write_text(json.dumps({'accepted': accepted, 'artifact_sha256': hashes}))
    (tmp_path / 'evaluation_summary.json').write_text(json.dumps(
        {'closed_loop_accepted': accepted, 'checkpoint_sha256': 'a' * 64}))


def test_native_zero_exit_cannot_promote_rejected_episode(tmp_path):
    artifacts(tmp_path, False)
    assert not verify_result(tmp_path, 'a' * 64, ['R1'], 0)['closed_loop_accepted']


def test_matching_complete_evidence_accepts_but_changed_trace_rejects(tmp_path):
    artifacts(tmp_path, True)
    assert verify_result(tmp_path, 'a' * 64, ['R1'], 0)['closed_loop_accepted']
    (tmp_path / 'expert_trace.jsonl').write_text('changed')
    result = verify_result(tmp_path, 'a' * 64, ['R1'], 0)
    assert not result['closed_loop_accepted']
    assert 'audit_hash_mismatch:expert_trace.jsonl' in result['reasons']


def test_missing_points_checkpoint_or_process_success_rejects(tmp_path):
    artifacts(tmp_path, True)
    assert not verify_result(tmp_path, 'a' * 64, ['R1', 'L1'], 0)['closed_loop_accepted']
    assert not verify_result(tmp_path, 'b' * 64, ['R1'], 0)['closed_loop_accepted']
    assert not verify_result(tmp_path, 'a' * 64, ['R1'], -9)['closed_loop_accepted']


def test_outcome_replaced_after_read_cannot_pass(tmp_path, monkeypatch):
    from pathlib import Path
    artifacts(tmp_path, True)
    original = Path.read_bytes
    def replaced(path):
        data = original(path)
        if path == tmp_path / 'outcome.json':
            outcome = json.loads(data)
            outcome['success'] = False
            path.write_text(json.dumps(outcome))
        return data
    monkeypatch.setattr(Path, 'read_bytes', replaced)
    result = verify_result(tmp_path, 'a' * 64, ['R1'], 0)
    assert not result['closed_loop_accepted']
    assert 'evidence_changed:outcome.json' in result['reasons']

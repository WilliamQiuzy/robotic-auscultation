"""Run Isaac in a child process and return status from verified saved evidence.

Kit fast shutdown can terminate Python with status zero before its return
statement. This parent never imports Isaac and treats child exit zero alone
as insufficient. Only the child created here is terminated on timeout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_result(output, checkpoint_sha256, points, child_returncode):
    output = Path(output)
    errors = []
    fingerprints = {}
    try:
        documents = {}
        for name in ('evaluation_summary.json', 'outcome.json', 'independent_audit.json'):
            data = (output / name).read_bytes()
            fingerprints[name] = hashlib.sha256(data).hexdigest()
            documents[name] = json.loads(data)
            if not isinstance(documents[name], dict):
                raise ValueError('invalid evidence object: ' + name)
        summary = documents['evaluation_summary.json']
        outcome = documents['outcome.json']
        audit = documents['independent_audit.json']
        if child_returncode != 0:
            errors.append('child_process_failed')
        if summary.get('closed_loop_accepted') is not True:
            errors.append('evaluation_rejected')
        if outcome.get('success') is not True or audit.get('accepted') is not True:
            errors.append('controller_or_independent_audit_rejected')
        if outcome.get('requested_points') != list(points):
            errors.append('requested_points_mismatch')
        if ([point.get('target') for point in outcome.get('points', [])] != list(points)
                or any(point.get('success') is not True for point in outcome.get('points', []))):
            errors.append('incomplete_point_sequence')
        if (summary.get('checkpoint_sha256') != checkpoint_sha256
                or outcome.get('policy', {}).get('checkpoint_sha256') != checkpoint_sha256):
            errors.append('checkpoint_mismatch')
        if outcome.get('dataset_role') != 'student_evaluation' or outcome.get('expert_demonstration') is not False:
            errors.append('incorrect_student_role')
        for name in ('outcome.json', 'control_commands.json', 'expert_trace.jsonl'):
            if name not in fingerprints:
                fingerprints[name] = digest_file(output / name)
            if fingerprints[name] != audit.get('artifact_sha256', {}).get(name):
                errors.append('audit_hash_mismatch:' + name)
        for name, expected in fingerprints.items():
            if digest_file(output / name) != expected:
                errors.append('evidence_changed:' + name)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors.append('missing_or_invalid_evidence:' + str(error))
    return {'closed_loop_accepted': not errors, 'reasons': errors,
            'child_returncode': child_returncode, 'output': str(output),
            'checkpoint_sha256': checkpoint_sha256, 'requested_points': list(points),
            'evidence_sha256': fingerprints}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('v1', 'v2', 'v3'), default='v2')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', nargs='+', default=['R1'])
    parser.add_argument('--contact-mode', choices=('hybrid_normal', 'pure_bc'), default='hybrid_normal')
    parser.add_argument('--speed-scale', type=float, default=None)
    parser.add_argument('--timeout-s', type=float, default=1800.)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_s <= 7200:
        parser.error('timeout-s must be in (0, 7200]')
    output, model = args.output.resolve(), args.model_dir.resolve()
    if output.exists():
        parser.error('output must be a new directory')
    manifest_path = model / 'run_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        parser.error('model manifest fingerprint mismatch')
    manifest = json.loads(manifest_bytes)
    checkpoint_sha = manifest['checkpoint_sha256']
    if digest_file(model / 'policy.pt') != checkpoint_sha:
        parser.error('checkpoint fingerprint mismatch')
    module = 'sim.auscultation_training.evaluate_bc' + ('' if args.version == 'v1' else '_' + args.version)
    command = [args.python, '-m', module, '--model-dir', str(model),
               '--manifest-sha256', args.manifest_sha256, '--output', str(output),
               '--points', *args.points, '--contact-mode', args.contact_mode]
    if args.speed_scale is not None:
        command += ['--speed-scale', str(args.speed_scale)]
    log = (args.log or output.with_suffix('.log')).resolve()
    timed_out = False
    with log.open('x') as stream:
        try:
            child = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout_s)
            returncode = child.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    result = verify_result(output, checkpoint_sha, args.points, returncode)
    if timed_out:
        result['closed_loop_accepted'] = False
        result['reasons'].append('child_timeout')
    result['log'] = str(log)
    for name, expected in result['evidence_sha256'].items():
        if not (output / name).is_file() or digest_file(output / name) != expected:
            result['closed_loop_accepted'] = False
            result['reasons'].append('evidence_changed_before_report:' + name)
    report = output / 'runner_summary.json' if output.is_dir() else log.with_suffix('.runner.json')
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['closed_loop_accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Hash-bound accepted inputs and unchanged mechanical gates for skill V1."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .config import (ROOT, PROJECT, TARGETS, BASELINE, SELECTED_PLANS, RUNTIME,
                     CAMERA, frozen_configuration, validate_config)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_path(project, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'unsafe pinned path')
    path = (project / relative).resolve()
    require(project in path.parents and '..' not in Path(relative).parts, 'unsafe pinned path: ' + relative)
    return path


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def verify_contract(contract=None, *, project=PROJECT):
    """Verify exact pinned bytes, allowing only a different project root on Linux."""
    if contract is None or isinstance(contract, (str, Path)):
        contract = read(contract or ROOT / 'contract.json')
    project = Path(project).resolve()
    require(contract.get('schema') == 'auscultation_skill_contract_v1', 'unexpected skill contract')
    require(contract.get('target_order') == list(TARGETS), 'fixed target order changed')
    validate_config(contract.get('configuration'))
    files = contract.get('fixed_files')
    required = contract.get('required_files')
    require(isinstance(files, dict) and bool(files) and isinstance(required, list)
            and len(required) == len(set(required)) and set(files) == set(required), 'required pin inventory changed')
    plans = contract.get('selected_plan_sha256', {})
    require(set(plans) == set(TARGETS) and all(_digest(x) for x in plans.values()), 'selected plan pins changed')
    for relative, expected in files.items():
        path = _safe_path(project, relative)
        require(_digest(expected) and path.is_file() and sha(path) == expected, 'pinned input changed: ' + relative)
    selected = contract.get('selected_plans')
    if selected is not None:
        directory = _safe_path(project, selected)
        for name, digest in plans.items():
            relative = str((directory / ('planned_paths_' + name + '.json')).relative_to(project))
            require(files.get(relative) == digest, 'selected plan differs from fixed pin: ' + name)
    return contract


def validate_outcome_contract(outcome, contract, points, *, speed_scale=None):
    require(isinstance(points, list) and bool(points) and len(points) == len(set(points))
            and all(point in TARGETS for point in points), 'invalid requested points')
    require(outcome.get('success') is True, 'physics episode did not complete')
    speed = outcome.get('speed_scale')
    require(type(speed) in (float, int) and math.isfinite(speed) and .7 <= speed <= 1., 'invalid speed scale')
    expected_phase = dict(contract['baseline_phase_config'])
    reference_speed = contract.get('reference_speed_scale', 1.)
    for key in ('approach_timeout_s', 'contact_timeout_s', 'retract_timeout_s', 'episode_timeout_s'):
        if key in expected_phase:
            expected_phase[key] = expected_phase[key] * reference_speed / speed
    require(outcome.get('phase_config') == expected_phase, 'mechanical phase configuration changed')
    require(outcome.get('executor_config') == contract['configuration']['executor'],
            'executed executor configuration changed')
    policy = outcome.get('actual_contact_policy', {})
    require(policy.get('action_scaling') == contract['configuration']['action_scaling'],
            'executed action scaling changed')
    config = contract['configuration']['executor']
    require(policy.get('desired_normal_velocity_bounds_m_s') ==
            [-config['normal_inward_speed_limit_m_s'] * speed, config['normal_speed_limit_m_s']],
            'executed directional action bounds changed')
    require(outcome.get('requested_points') == points, 'requested point order changed')
    actual = outcome.get('points', [])
    require([point.get('target') for point in actual] == points
            and all(point.get('success') is True for point in actual), 'incomplete point inventory')
    if speed_scale is not None:
        require(speed == speed_scale, 'requested speed scale changed')
    return True


def create_contract(destination=ROOT / 'contract.json'):
    """Pin accepted materialized inputs and all parent source/scene snapshot bytes."""
    from sim.upper_pair_posture_20260924 import contract as parent
    from sim.upper_pair_posture_20260924.run import verify_selection
    destination = Path(destination)
    require(not destination.exists(), 'contract destination must be new')
    parent_path = parent.ROOT / 'contract.json'
    selected = parent.validate_materialized_set(SELECTED_PLANS, read(parent_path), contract_path=parent_path)
    verify_selection(SELECTED_PLANS, parent_path)
    snapshot_path = BASELINE / 'input_snapshot/manifest.json'
    snapshot = read(snapshot_path)
    outcome = read(BASELINE / 'outcome.json')
    require(outcome.get('success') is True and outcome.get('requested_points') == list(TARGETS),
            'accepted baseline must be complete full10')
    files = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        require(PROJECT in path.parents, 'input outside project')
        relative = str(path.relative_to(PROJECT))
        actual = sha(path)
        require(expected is None or actual == expected, 'baseline source changed: ' + relative)
        require(relative not in files or files[relative] == actual, 'pin changed during capture')
        files[relative] = actual
    for label, entry in snapshot['files'].items():
        source = entry['source_path']
        require('/sim/' in source, 'unsupported baseline source root: ' + label)
        path = PROJECT / ('sim/' + source.split('/sim/', 1)[1])
        # Archived AppleDouble sidecars are non-runtime tar metadata. Bind their
        # captured bytes without requiring platform-specific sidecars on Mac.
        if path.name.startswith('._'):
            pin(snapshot_path.parent / 'objects' / entry['sha256'], entry['sha256'])
        else:
            pin(path, entry['sha256'])
        if label.endswith('/plan'):
            require(entry['sha256'] == selected[label.split('/')[0]], 'selected plan differs from executed baseline')
    for path in (parent_path, SELECTED_PLANS / 'selection.json', CAMERA, snapshot_path,
                 BASELINE / 'outcome.json', ROOT / 'config.py',
                 PROJECT / 'sim/expert_quality_20260924/quality_profile.json'):
        pin(path)
    # The old semantic contract additionally binds canonical source inputs and posture limits.
    for relative, expected in read(parent_path)['fixed_files'].items():
        pin(PROJECT / relative, expected)
    contract = {'schema': 'auscultation_skill_contract_v1', 'target_order': list(TARGETS),
                'selected_plans': str(SELECTED_PLANS.relative_to(PROJECT)),
                'baseline_episode': str(BASELINE.relative_to(PROJECT)),
                'runtime_directory': str(RUNTIME.relative_to(PROJECT)),
                'camera': str(CAMERA.relative_to(PROJECT)),
                'parent_contract': str(parent_path.relative_to(PROJECT)),
                'parent_contract_sha256': sha(parent_path),
                'selected_plan_sha256': selected,
                'baseline_phase_config': outcome['phase_config'],
                'reference_speed_scale': outcome['speed_scale'],
                'configuration': frozen_configuration(),
                'fixed_files': dict(sorted(files.items())), 'required_files': sorted(files),
                'scope': 'Fixed accepted scene/camera/poses and mechanical gates; only versioned contact executor and actor change. No acoustic, clinical or hardware acceptance.'}
    verify_contract(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', type=Path)
    parser.add_argument('--contract', type=Path, default=ROOT / 'contract.json')
    args = parser.parse_args()
    result = create_contract(args.create) if args.create else verify_contract(args.contract)
    print(json.dumps({'status': 'passed', 'fixed_file_count': len(result['fixed_files'])}))


if __name__ == '__main__':
    main()

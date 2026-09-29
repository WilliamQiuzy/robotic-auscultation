"""Recompute contact-mechanics acceptance from a saved 240 Hz trace.

This audit does not infer acoustic coupling from a front-ring force. Reported
success and hold counters are never sufficient without the underlying samples.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .phase_manager import PhaseConfig
from .kinematics import OpenArmKinematics


def audit_rows(rows, outcome, require_hold=True, require_release=True):
    config = asdict(PhaseConfig())
    config.update(outcome.get('phase_config', {}))
    reasons = []
    if len(rows) < 2:
        return {'accepted': False, 'reasons': ['insufficient_trace']}
    keys = ['time_s', 'hall_N', 'compression_m', 'gap_m', 'tangent_m',
            'angle_deg', 'tcp_speed_m_s', 'diaphragm_N', 'lip_N', 'contact_rim_N',
            'carriage_other_N', 'frame_N', 'forbidden_N']
    try:
        a = {key: np.asarray([r[key] for r in rows], dtype=float) for key in keys}
        q = np.asarray([r['q'] for r in rows], dtype=float)
        command = np.asarray([r['q_command'] for r in rows], dtype=float)
    except (ValueError, KeyError, TypeError):
        return {'accepted': False, 'reasons': ['malformed_trace']}
    if not all(np.isfinite(x).all() for x in list(a.values()) + [q, command]):
        return {'accepted': False, 'reasons': ['nonfinite_trace']}
    dt = 1/240
    if not np.allclose(np.diff(a['time_s']), dt, rtol=0, atol=1e-7):
        reasons.append('timestamp_gap')
    kin = OpenArmKinematics()
    for label, values, tolerance in [('joint', q, .002), ('command', command, 1e-8)]:
        if values.shape != (len(rows), 7):
            reasons.append(label + '_shape')
        elif np.any(values < kin.lower_limits - tolerance) or np.any(values > kin.upper_limits + tolerance):
            reasons.append(label + '_limit')
    front = a['diaphragm_N'] + a['lip_N'] + a['contact_rim_N']
    total = front + a['carriage_other_N'] + a['frame_N'] + a['forbidden_N']
    for key in ['carriage_other_N', 'frame_N', 'forbidden_N']:
        if a[key].max() > .5:
            reasons.append(key + '_contact')
    if total.max() > 6.:
        reasons.append('physical_force_limit')
    if a['hall_N'].max() >= config['hard_force_limit_N']:
        reasons.append('hall_force_limit')
    if a['compression_m'].max() >= config['max_compression_m']:
        reasons.append('stroke_limit')
    # Independently filter the physical measurement; no stored audit flag used.
    filtered = np.zeros(len(rows)); value = 0.
    for i, force in enumerate(front):
        value += dt/(.02+dt)*(force-value)
        filtered[i] = value
    good = ((np.abs(a['hall_N']-config['target_force_N']) <= config['force_tolerance_N'])
            & (a['compression_m'] >= config['min_contact_compression_m'])
            & (a['compression_m'] <= config['max_compression_m'])
            & (np.abs(a['gap_m']) <= config['contact_normal_tolerance_m'])
            & (a['tangent_m'] <= config['hold_tangential_tolerance_m'])
            & (a['angle_deg'] <= config['angle_tolerance_deg'])
            & (a['tcp_speed_m_s'] <= config['hold_speed_limit_m_s'])
            & (filtered > .15)
            & np.asarray([r['phase'] == 'HOLD' for r in rows]))
    longest = run = 0
    for value in good:
        run = run + 1 if value else 0
        longest = max(longest, run)
    if require_hold and longest*dt < config['hold_duration_s']-2*dt:
        reasons.append('continuous_contact_hold_too_short')
    # Require released final contact; sequence files should be audited per point.
    release = int(np.ceil(config['release_confirm_duration_s']/dt))
    if require_release and (len(rows) < release or np.any(a['hall_N'][-release:] >= config['release_force_limit_N'])
            or np.any(a['gap_m'][-release:] < config['retract_clearance_m'])
            or np.any(front[-release:] > .15)):
        reasons.append('release_not_verified')
    if not outcome.get('success'):
        reasons.append('controller_did_not_complete')
    return {'accepted': not reasons, 'reasons': reasons,
            'scope': 'simulated contact mechanics; acoustic coupling and clinical adequacy unverified',
            'continuous_front_contact_hold_s': longest*dt,
            'max_physical_force_N': float(total.max()),
            'max_hall_N': float(a['hall_N'].max()),
            'max_compression_m': float(a['compression_m'].max()),
            'max_forbidden_contact_N': float(a['forbidden_N'].max()),
            'max_carriage_other_N': float(a['carriage_other_N'].max()),
            'hold_membrane_force_max_N': float(a['diaphragm_N'][good].max()) if good.any() else None,
            'hold_center_gap_range_m': [float(a['gap_m'][good].min()), float(a['gap_m'][good].max())] if good.any() else None,
            'physics_steps': len(rows)}


def audit_sequence(rows, outcome):
    """Every requested point needs its own complete contact and release."""
    overall = audit_rows(rows, outcome, require_hold=False, require_release=False)
    reports, reasons = [], list(overall['reasons'])
    points = outcome.get('points', [])
    names = [point.get('target') for point in points]
    if not names or names != outcome.get('requested_points') or len(set(names)) != len(names):
        reasons.append('requested_point_coverage')
    previous_end = 0
    for point in points:
        start, contact_end, end = [point.get(key) for key in
            ['start_physics_row','contact_end_physics_row','end_physics_row']]
        if (any(not isinstance(v,int) or isinstance(v,bool) for v in [start,contact_end,end])
                or start != previous_end or not 0 <= start < contact_end <= end <= len(rows)):
            reasons.append('invalid_point_interval')
            continue
        previous_end = end
        if any(row.get('target') != point['target'] for row in rows[start:end]):
            reasons.append('trace_point_identity')
        report = audit_rows(rows[start:contact_end], point)
        report['target'] = point['target']
        reports.append(report)
        if not report['accepted']:
            reasons.append(point['target']+':'+','.join(report['reasons']))
        if not point.get('training_accepted'):
            reasons.append(point['target']+':controller_rejected')
    if previous_end != len(rows):
        reasons.append('uncovered_trace_rows')
    return {'accepted': not reasons, 'reasons': reasons, 'points': reports,
            'whole_sequence': overall, 'scope': overall.get('scope')}


def audit_directory(path):
    artifacts = {name: (path/name).read_bytes() for name in
                 ['expert_trace.jsonl','control_commands.json','outcome.json']}
    rows = [json.loads(line) for line in artifacts['expert_trace.jsonl'].splitlines()]
    outcome = json.loads(artifacts['outcome.json'])
    report = audit_sequence(rows,outcome) if 'points' in outcome else audit_rows(rows,outcome)
    report['mechanical_accepted'] = report['accepted']
    from .dataset import validate_control_trace
    try:
        report['control_trace_binding'] = validate_control_trace(json.loads(artifacts['control_commands.json']),rows)
    except (KeyError, TypeError, ValueError) as error:
        report['accepted'] = False
        report['reasons'].append('control_trace_binding:'+str(error))
    report['artifact_sha256'] = {name: hashlib.sha256(data).hexdigest() for name,data in artifacts.items()}
    if 'points' in outcome and 'input_snapshot' not in outcome:
        report['accepted'] = False
        report['reasons'].append('sequence_missing_input_snapshot')
    if 'input_snapshot' in outcome:
        from .controller import verify_snapshot
        try:
            snapshot = path/Path(outcome['input_snapshot']).parent
            manifest_sha = hashlib.sha256((snapshot/'manifest.json').read_bytes()).hexdigest()
            if manifest_sha != outcome['input_snapshot_manifest_sha256']:
                raise ValueError('snapshot_manifest_mismatch')
            manifest = verify_snapshot(snapshot)
            expected = manifest['files']['source/collision_regeneration']['sha256']
            if outcome.get('collision_regeneration_sha256') != expected:
                raise ValueError('geometry_source_binding_mismatch')
        except (KeyError, OSError, ValueError) as error:
            report['accepted'] = False
            report['reasons'].append('provenance:'+str(error))
    for name, data in artifacts.items():
        if (path/name).read_bytes() != data:
            report['accepted'] = False
            report['reasons'].append('artifact_changed_during_audit:'+name)
    (path/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    report = audit_directory(args.episode)
    print(json.dumps(report, indent=2))
    if not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

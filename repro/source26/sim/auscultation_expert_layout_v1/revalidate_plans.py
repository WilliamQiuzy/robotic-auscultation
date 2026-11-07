"""Bind saved sampled paths to current URDF and regenerated collision geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from .scene import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_geometry_binding(root=ROOT):
    path = root/'results/collision_regeneration_verification.json'
    data = json.loads(path.read_text())
    if data.get('equal_exported_arrays') is not True or data.get('equal_metadata') is not True:
        raise ValueError('Collision regeneration did not match')
    for name, expected in data['existing_outputs_sha256'].items():
        if Path(name).name != name or sha(root/'results'/name) != expected:
            raise ValueError('Collision export has changed: '+name)
    for name, expected in data['source_files_relative'].items():
        candidate = (root.parent.parent/name).resolve()
        if not candidate.is_relative_to(root.parent.parent.resolve()) or sha(candidate) != expected:
            raise ValueError('Collision source has changed: '+name)
    return sha(path)


def revalidate(path, kin, checker, binding):
    urdf_sha = kin.revalidation_urdf_sha256
    if sha(DEFAULT_URDF_PATH) != urdf_sha:
        raise ValueError('URDF changed after kinematics construction')
    original = path.read_bytes()
    plan = json.loads(original)
    row = plan['points'][0]
    if not row['success']:
        raise ValueError('No successful route: '+row['name'])
    provenance = plan['provenance']
    for label in ['audit','targets','runtime','collision_metadata','collision_arrays']:
        name = provenance[label+'_file']
        source = (ROOT if label == 'runtime' else ROOT/'results')/name
        if Path(name).name != name or sha(source) != provenance[label+'_sha256']:
            raise ValueError('Plan input changed: '+label)
    base = np.asarray(plan['world_from_left_base'])
    transfer, approach = np.asarray(row['transfer_q']), np.asarray(row['approach_q'])
    if not np.allclose(transfer[0], 0.) or not np.allclose(transfer[-1], approach[0]):
        raise ValueError('Disconnected home/approach route')
    checks = []
    for route in [transfer, approach]:
        for start, end in zip(route, route[1:]):
            result = checker.edge(start, end)
            if not result['clear']:
                raise ValueError('Current geometry rejected route: '+str(result))
            checks.append(result)
    pre = base@kin.tcp(transfer[-1])
    expected_pre = np.asarray(row['position_world']) + np.asarray(row['command_normal_world'])*row['free_approach_start_gap_m']
    loaded = base@kin.tcp(row['loaded_q'], compression=row['loaded_compression_m'])
    position_error = np.linalg.norm(loaded[:3,3]-row['expected_contact_center_world'])
    angle_error = np.linalg.norm(Rotation.from_matrix(loaded[:3,:3].T@np.asarray(row['command_rotation_world'])).as_rotvec())
    if position_error > 2e-5 or angle_error > 2e-4 or np.linalg.norm(pre[:3,3]-expected_pre) > 2e-5:
        raise ValueError('Saved pose no longer agrees with current kinematics')
    contact = checker.check(row['loaded_q'], clearance=.0003, allow_patient_contact=True,
                            compression=row['loaded_compression_m'])
    if not contact['clear']:
        raise ValueError('Loaded endpoint rejected by current geometry')
    if path.read_bytes() != original:
        raise ValueError('Plan changed during revalidation')
    if sha(DEFAULT_URDF_PATH) != urdf_sha:
        raise ValueError('URDF changed during path checks')
    provenance.update(kinematics_urdf_file=DEFAULT_URDF_PATH.name, kinematics_urdf_sha256=urdf_sha)
    plan['revalidation'] = {'accepted': True, 'input_plan_sha256': hashlib.sha256(original).hexdigest(),
        'collision_regeneration_sha256': binding, 'edge_count': len(checks),
        'checked_samples': sum(r['samples'] for r in checks), 'loaded_check': contact,
        'loaded_position_error_m': float(position_error), 'loaded_angle_error_rad': float(angle_error),
        'scope': 'Current URDF, sampled free paths, separate loaded endpoint and regenerated collision arrays; physical execution still required.'}
    path.with_suffix('.before_revalidation.json').write_bytes(original)
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print('REVALIDATED',row['name'],len(checks),'edges',flush=True)


def main():
    from .collision import CollisionChecker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',nargs='+',required=True)
    args = parser.parse_args()
    binding = validate_geometry_binding()
    base = np.asarray(json.loads((ROOT/'runtime_inspection.json').read_text())['world_from_left_base'])
    urdf_sha = sha(DEFAULT_URDF_PATH)
    kin = OpenArmKinematics()
    if sha(DEFAULT_URDF_PATH) != urdf_sha:
        raise ValueError('URDF changed during kinematics construction')
    kin.revalidation_urdf_sha256 = urdf_sha
    checker = CollisionChecker(kin,base)
    for point in args.points:
        revalidate(ROOT/'results'/('planned_paths_'+point+'.json'),kin,checker,binding)
    if validate_geometry_binding() != binding:
        raise ValueError('Geometry source changed during revalidation')


if __name__ == '__main__':
    main()

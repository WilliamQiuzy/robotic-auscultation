"""Independently recheck saved new-layout routes before the physics runner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .collision import CollisionChecker
from .controller import validate_plan
from .kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from .posture import PostureChecker, PostureLimits
from .revalidate_plans import revalidate, validate_geometry_binding
from .scene import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    report_path = directory/'planning_report.json'
    report = json.loads(report_path.read_text())
    assert report['success'] is True
    for name, expected in report['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    binding = validate_geometry_binding()
    base = np.asarray(json.loads((ROOT/'runtime_inspection.json').read_text())['world_from_left_base'])
    kin = OpenArmKinematics()
    kin.revalidation_urdf_sha256 = sha(DEFAULT_URDF_PATH)
    limits = PostureLimits.from_dict(json.loads((directory/'posture_limits.json').read_text()), kin)
    checker = PostureChecker(CollisionChecker(kin, base, directory=directory), base, limits)
    points = [s+str(i) for i in range(1,6) for s in ('R','L')]
    summaries = []
    for point in points:
        path = directory/('planned_paths_'+point+'.json')
        assert not path.with_suffix('.before_revalidation.json').exists()
        bundle = validate_plan(path, directory, ROOT)
        assert bundle.row['posture_limits'] == limits.to_dict()
        revalidate(path, kin, checker, binding)
        checked = validate_plan(path, directory, ROOT)
        summaries.append({'point': point, **checked.plan['revalidation'], 'plan_sha256': sha(path)})
    assert validate_geometry_binding() == binding
    result = {'accepted': True, 'points': summaries, 'planning_report_sha256': sha(report_path),
              'scope': 'Fresh geometry-bound path/posture checks; actual physics and contact still required.'}
    (directory/'preflight_revalidation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('NEW_LAYOUT_REVALIDATED', len(summaries), flush=True)


if __name__ == '__main__':
    main()

"""Require actual GPU FK agreement before using the WAM cuRobo bridge."""
import json
from pathlib import Path

from .curobo_backend import EXPECTED_COMMIT
from .planner_service import sha
from .robot_bridge import load_bridge, validate_backend_fk, FK_LINKS


def expected_assets(robot,world,hash_file=sha):
    paths=[robot['robot_cfg']['kinematics']['urdf_path']]
    spheres=robot['robot_cfg']['kinematics'].get('collision_spheres')
    if isinstance(spheres,str):paths.append(spheres)
    paths.extend(mesh['file_path'] for mesh in world.get('mesh',{}).values())
    return {path:hash_file(path) for path in paths}


def verify_assets(metadata,assets):
    if not assets or metadata.get('asset_sha256')!=assets:
        raise ValueError('backend actual asset hashes differ from captured bridge')


def verify_fk_pair(request, result, expected_robot, joint_vectors, request_sha, backend_sha, assets=None):
    metadata=result.get('metadata',{})
    if (request.get('operation')!='fk' or request.get('robot_config')!=expected_robot
            or request.get('joint_vectors')!=joint_vectors or request.get('link_names')!=list(FK_LINKS)):
        raise ValueError('FK request differs from captured robot or reference vectors')
    if (metadata.get('curobo_commit')!=EXPECTED_COMMIT
            or metadata.get('curobo_source_dirty') not in (False,'')
            or metadata.get('request_sha256')!=request_sha
            or metadata.get('backend_source_sha256')!=backend_sha):
        raise ValueError('FK backend source/request binding mismatch')
    if assets is not None:verify_assets(metadata,assets)
    return validate_backend_fk(joint_vectors,result)


def validate_preflight(bridge_dir,request_path,result_path):
    robot,_,_=load_bridge(bridge_dir)
    vectors=json.loads((Path(bridge_dir)/'fk_reference.json').read_text())['joint_vectors']
    request=json.loads(Path(request_path).read_text());result=json.loads(Path(result_path).read_text())
    report=verify_fk_pair(request,result,robot,vectors,sha(request_path),
                         sha(Path(__file__).with_name('curobo_backend.py')),expected_assets(robot,{}))
    return dict(report,request_sha256=sha(request_path),result_sha256=sha(result_path),
                bridge_manifest_sha256=sha(Path(bridge_dir)/'bridge_manifest.json'))

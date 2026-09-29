import copy
import numpy as np
import pytest

from sim.curobo_expert_v1.preflight import expected_assets,verify_assets,verify_fk_pair
from sim.curobo_expert_v1.curobo_backend import EXPECTED_COMMIT
from sim.curobo_expert_v1.robot_bridge import FK_LINKS,reference_fk
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics


def fixture():
    kin=OpenArmKinematics();vectors=[((kin.lower_limits+kin.upper_limits)/2).tolist()]
    robot={'robot_cfg':{'kinematics':{'urdf_path':'/bridge/robot.urdf'}}}
    request={'operation':'fk','robot_config':robot,'joint_vectors':vectors,'link_names':list(FK_LINKS)}
    ref=reference_fk(vectors)
    metadata={'joint_names':list(kin.joint_names),'pose_frame':'robot_base','quaternion_order':'wxyz',
        'curobo_commit':EXPECTED_COMMIT,'curobo_source_dirty':False,'request_sha256':'request',
        'backend_source_sha256':'backend','asset_sha256':{'/bridge/robot.urdf':'urdf'}}
    result={'success':True,'metadata':metadata,
        'link_poses':{name:[ref['poses'][0][i]] for i,name in enumerate(ref['link_names'])}}
    return request,result,robot,vectors,metadata['asset_sha256'].copy()


def test_valid_fk_and_assets():
    req,res,robot,vectors,assets=fixture()
    assert verify_fk_pair(req,res,robot,vectors,'request','backend',assets)['status']=='passed'


@pytest.mark.parametrize('mutation',['asset','request_hash','source','frame','robot','joint_order','pose'])
def test_fk_rejects_mismatched_evidence(mutation):
    req,res,robot,vectors,assets=fixture();req=copy.deepcopy(req)
    if mutation=='asset':res['metadata']['asset_sha256']['/bridge/robot.urdf']='different'
    elif mutation=='request_hash':res['metadata']['request_sha256']='different'
    elif mutation=='source':res['metadata']['backend_source_sha256']='different'
    elif mutation=='frame':res['metadata']['pose_frame']='world'
    elif mutation=='robot':req['robot_config']['robot_cfg']['kinematics']['urdf_path']='other'
    elif mutation=='joint_order':res['metadata']['joint_names'].reverse()
    elif mutation=='pose':res['link_poses'][FK_LINKS[-1]][0][0]+=.01
    with pytest.raises(ValueError):verify_fk_pair(req,res,robot,vectors,'request','backend',assets)


def test_world_assets_require_complete_exact_mapping():
    robot={'robot_cfg':{'kinematics':{'urdf_path':'a','collision_spheres':[]}}}
    world={'mesh':{'obstacle':{'file_path':'b'}}}
    assets=expected_assets(robot,world,lambda path:path+'hash')
    verify_assets({'asset_sha256':{'a':'ahash','b':'bhash'}},assets)
    with pytest.raises(ValueError):verify_assets({'asset_sha256':{'a':'ahash'}},assets)

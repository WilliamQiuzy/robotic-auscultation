import json
from pathlib import Path
import numpy as np
import pytest

from sim.ausc_policy.approved_camera import load_approved_camera, local_camera_transform

CAMERA=Path('sim/camera_framing_20260923/recommended_camera.json').read_bytes()


def test_only_exact_approved_bytes_are_accepted():
    assert load_approved_camera(CAMERA)['downward_pitch_delta_deg']==7.
    changed=json.loads(CAMERA);changed['world_from_camera_usd'][2][3]-=.05
    with pytest.raises(ValueError,match='approved'):
        load_approved_camera(json.dumps(changed).encode())


def test_parent_relative_transform_restores_exact_world_pose():
    desired=np.array(load_approved_camera(CAMERA)['world_from_camera_usd'])
    parent=np.eye(4);parent[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]];parent[:3,3]=[.1,-.2,.4]
    local=local_camera_transform(desired,parent)
    np.testing.assert_allclose(parent@local,desired,atol=1e-12,rtol=0)


@pytest.mark.parametrize('malformed',[np.zeros((4,4)),np.diag([2.,1.,1.,1.]),np.full((4,4),np.nan)])
def test_nonrigid_transform_rejected(malformed):
    with pytest.raises(ValueError,match='rigid'):
        local_camera_transform(np.eye(4),malformed)


@pytest.mark.parametrize('physical_parent',[False,True])
def test_static_attribute_values_do_not_hide_physical_parent(physical_parent):
    from types import SimpleNamespace
    from sim.ausc_policy.approved_camera import _require_static_ancestors
    api=object()
    parent=SimpleNamespace(IsPseudoRoot=lambda:False,HasAPI=lambda value:value is api and physical_parent,
        GetAttributes=lambda:[],GetParent=lambda:None)
    camera=SimpleNamespace(IsPseudoRoot=lambda:False,HasAPI=lambda value:False,
        GetAttributes=lambda:[],GetParent=lambda:parent)
    if physical_parent:
        with pytest.raises(ValueError,match='physical parent'):
            _require_static_ancestors(camera,api)
    else:_require_static_ancestors(camera,api)

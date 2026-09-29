import numpy as np
import pytest
from sim.ausc_policy.capture_probe import declared_cases, perturb_pose


def test_declared_grid_has_exact_35_distinct_single_axis_nodes():
    cases=declared_cases()
    assert len(cases)==35
    assert len({c['id'] for c in cases})==35
    assert sum(c['kind']=='translation' for c in cases)==16
    assert sum(c['kind']=='rotation' for c in cases)==18
    assert all(c['axis'] in (0,1,2) for c in cases if c['kind']!='center')


def test_translation_is_in_declared_tool_tangents_and_outward_normal():
    pose=np.eye(4);normal=np.array([0.,0.,-1.])
    cases=declared_cases()
    case=next(c for c in cases if c['kind']=='translation' and c['axis']==2 and c['value']>0)
    result=perturb_pose(pose,normal,case)
    np.testing.assert_allclose(result[:3,3],normal*case['value'])
    np.testing.assert_array_equal(result[:3,:3],pose[:3,:3])
    np.testing.assert_array_equal(pose,np.eye(4))


def test_rotation_changes_only_orientation_and_keeps_proper_rotation():
    case=next(c for c in declared_cases() if c['kind']=='rotation')
    result=perturb_pose(np.eye(4),np.array([0.,0.,-1.]),case)
    np.testing.assert_array_equal(result[:3,3],[0.,0.,0.])
    np.testing.assert_allclose(result[:3,:3].T@result[:3,:3],np.eye(3),atol=1e-14)
    assert np.linalg.det(result[:3,:3])==pytest.approx(1.)


def test_unapproved_node_or_inconsistent_normal_is_rejected():
    with pytest.raises(ValueError): perturb_pose(np.eye(4),[0,0,-1],dict(id='bad',kind='translation',axis=0,value=.1))
    with pytest.raises(ValueError): perturb_pose(np.eye(4),[1,0,0],declared_cases()[0])

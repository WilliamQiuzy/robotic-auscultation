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


@pytest.fixture
def geometry_binding(tmp_path):
    import json, hashlib
    node='t2_+0.00200'; root=tmp_path/'geometry';(root/'output').mkdir(parents=True)
    case=dict(case={'id':node},q=[0.]*7,ik_accurate=True,free_geometry_passed=True,
              direct_entry_geometry_passed=True,full_curve_geometry_passed=True,training_dataset_eligible=False)
    owner=dict(completed=True,gpu_released=True,stage=dict(process_group_drained=True,exit_code=0),
               config=dict(experiment='capture_probe',point='R4',core_assets={'scene':'frozen'}))
    complete=dict(completed=True,case_count=35,cases=[case])
    def save():
        digest={}
        for relative,value in [('result.json',owner),('output/COMPLETE.json',complete),('output/'+node+'.json',case)]:
            data=json.dumps(value).encode();(root/relative).write_bytes(data);digest[relative]=hashlib.sha256(data).hexdigest()
        return dict(case_id='entry-normal-plus',case_ids=['entry-normal-plus'],core_assets={'scene':'frozen'},
                    geometry_probe=dict(path=str(root),owner_sha256=digest['result.json'],
                    complete_sha256=digest['output/COMPLETE.json'],cases={'entry-normal-plus':dict(node=node,sha256=digest['output/'+node+'.json'])}))
    return root,case,owner,save


def test_actual_component_requires_bound_completed_safe_node(geometry_binding):
    from sim.ausc_policy.capture_probe import load_geometry_case
    _,case,_,save=geometry_binding
    assert load_geometry_case(save())==case


@pytest.mark.parametrize('defect',['bytes','not_drained','curve_refused','wrong_assets','wrong_node','nan_q'])
def test_actual_component_refuses_unqualified_or_mismatched_node(geometry_binding,defect):
    from sim.ausc_policy.capture_probe import load_geometry_case
    root,case,owner,save=geometry_binding
    if defect=='not_drained': owner['stage']['process_group_drained']=False
    if defect=='curve_refused': case['full_curve_geometry_passed']=False
    if defect=='wrong_node': case['case']['id']='center'
    if defect=='nan_q': case['q'][2]=float('nan')
    config=save()
    if defect=='bytes': (root/'output/t2_+0.00200.json').write_text('{}')
    if defect=='wrong_assets': config['core_assets']={'scene':'different'}
    with pytest.raises(ValueError): load_geometry_case(config)

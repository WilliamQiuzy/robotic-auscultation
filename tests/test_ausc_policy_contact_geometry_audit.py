from copy import deepcopy
import numpy as np
import pytest

from sim.ausc_policy.contact_geometry_audit import audit_contacts, sampled_hold_poses


def sample():
    return dict(phase='HOLD', time_s=8/240,
        diaphragm_N=0., lip_N=4., contact_rim_N=0., carriage_other_N=0., frame_N=0., forbidden_N=0.,
        contact_points=[dict(colliders=['/World/Patient/shirt',
            '/World/Stethoscope/Carriage/Colliders/RetentionLip/part0'], impulse_Ns=[0., 4/240, 0.])])


def test_nonzero_hall_or_lip_cannot_be_counted_as_diaphragm_contact():
    row=sample();row['hall_N']=4.;old=deepcopy(row)
    result=audit_contacts([row])
    assert result['component_force_peak_N']['diaphragm']==0
    assert result['component_force_peak_N']['lip']==4.
    assert result['diaphragm_contact_points']==0
    assert row==old


def test_misclassified_or_missing_contact_points_are_detected():
    row=sample();row['contact_points'][0]['colliders'][1]='/World/Stethoscope/Carriage/Colliders/Diaphragm'
    with pytest.raises(ValueError,match='component'): audit_contacts([row])
    row=sample();row['contact_points']=[]
    with pytest.raises(ValueError,match='component'): audit_contacts([row])


def test_actual_pose_selection_uses_saved_physics_indices_not_array_position():
    rows=[dict(sample(),phase='FREE',time_s=(i+1)/240) for i in range(24)]
    rows[15]['phase']='HOLD'
    arrays=dict(paths=np.array(['/World/Stethoscope/Carriage']),row_indices=np.array([7,15,23]),
        poses=np.zeros((3,1,7)))
    arrays['poses'][:,:,3]=1.
    result=sampled_hold_poses(rows,arrays)
    assert [(r[0],r[1]) for r in result]==[(15,0)]
    arrays['row_indices'][1]=16
    with pytest.raises(ValueError):sampled_hold_poses(rows,arrays)


def test_missing_or_invalid_actual_rigid_pose_cannot_fall_back_to_fk():
    row=sample();arrays=dict(paths=np.array(['/World/Stethoscope/Fixed']),
        row_indices=np.array([7]),poses=np.zeros((1,1,7)))
    with pytest.raises(ValueError):sampled_hold_poses([row]*8,arrays)
    arrays['paths'][0]='/World/Stethoscope/Carriage'
    with pytest.raises(ValueError):sampled_hold_poses([row]*8,arrays)

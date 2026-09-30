from copy import deepcopy

import pytest

from sim.ausc_policy import geometry_plan


CONTACT_SURFACE='membrane_and_device_retainer_v2'


def candidate_report():
    fit=dict(passed=True,contact_surface=CONTACT_SURFACE,contact_face_collision=False,
             distances_m=dict(fixed=.01,clamp=.005,shell=.0015,contact_face=.00002))
    candidate=dict(passed=True,contact_surface=CONTACT_SURFACE,normal_world=[0,0,1],
                   rotation_world=[[1,0,0],[0,-1,0],[0,0,-1]],surface_gap_m=.00002,
                   normal_deviation_from_surface_deg=0.,gap_axis='command_normal',fit=fit)
    return dict(contact_surface=CONTACT_SURFACE,candidates=[candidate])


def test_geometry_planning_audit_preserves_versioned_contact_face_receipt():
    report=candidate_report();before=deepcopy(report)
    audit=geometry_plan.fit_audit('L2',.0025,report)
    assert audit['contact_surface']==CONTACT_SURFACE
    assert audit['candidates'][0]['contact_surface']==CONTACT_SURFACE
    assert audit['candidates'][0]['first_contact']==report['candidates'][0]
    assert audit['candidates'][0]['finite_head_fit']==report['candidates'][0]['fit']
    assert report==before


@pytest.mark.parametrize('layer',['summary','first_contact','fit'])
def test_geometry_planning_cannot_relabel_unversioned_receipt(layer):
    report=candidate_report()
    target=(report if layer=='summary' else report['candidates'][0] if layer=='first_contact'
            else report['candidates'][0]['fit'])
    target.pop('contact_surface');before=deepcopy(report)
    with pytest.raises(ValueError,match='contact.surface'):
        geometry_plan.fit_audit('L2',.0025,report)
    assert report==before

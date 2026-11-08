"""User-confirmed device retainer contact, distinct from printed clamp contact."""
from types import SimpleNamespace

import pytest
import numpy as np

from sim.ausc_policy.contact_collision import pair_clearance
from sim.ausc_policy.contact_geometry_audit import group
from sim.ausc_policy.geometry_epoch import qualify_contact
from sim.ausc_policy.fresh_expert import FreshExpert


ROOT = '/World/Stethoscope/Carriage/Colliders/'
PATIENT = '/World/Patient/Body'
RET = ROOT + 'DeviceContactRing/sector_00'


def rows(membrane=0., retainer=4., **extra):
    row = dict(diaphragm_N=membrane, retainer_N=retainer, lip_N=0.,
               contact_rim_N=0., carriage_other_N=0., frame_N=0., forbidden_N=0.,
               compression_m=.0025, hall_N=4.)
    row['retainer_front_audit']=dict(passed=True,force_N=retainer,
                                   force_matches_ledger=True,off_front_force_N=0.)
    row.update(extra)
    return [dict(row) for _ in range(720)]


def test_device_retainer_may_touch_patient_in_contact_mode_only():
    for index in range(32):
        path = ROOT + 'DeviceContactRing/sector_' + str(index).zfill(2)
        for a, b in ((path, PATIENT), (PATIENT, path)):
            assert pair_clearance(a, b, .003, True) is None
            assert pair_clearance(a, b, .003, False) == .003
        assert pair_clearance(path, '/World/Table', .003, True) == .003


def test_nearby_names_and_printed_parts_do_not_acquire_contact_permission():
    for name in ('DeviceContactRing/sector_32', 'DeviceContactRing/sector_0',
                 'DeviceContactRingOther/sector_00', 'DeviceContactRing/sector_00/other',
                 'RetentionLip/sector_00', 'ContactRim/sector_00', 'ClampWall/sector_00'):
        assert pair_clearance(ROOT + name, PATIENT, .0001, True) == .003
    for name in ('DeviceEdge', 'DeviceCone', 'DeviceDial'):
        assert pair_clearance(ROOT + name, PATIENT, .003, True) == .0003


def test_retainer_force_is_separate_from_membrane_and_forbidden_clamp():
    assert group([RET, PATIENT]) == 'retainer'
    assert group([PATIENT, RET]) == 'retainer'
    assert group([ROOT+'Diaphragm', PATIENT]) == 'diaphragm'
    assert group([ROOT+'RetentionLip/sector_00', PATIENT]) == 'lip'
    assert group([ROOT+'ContactRim/sector_00', PATIENT]) == 'contact_rim'
    assert group([RET, '/World/Table']) == 'forbidden'


@pytest.mark.parametrize('membrane,retainer', [(0.,4.), (1.5,2.5), (4.,0.)])
def test_load_on_real_contact_components_passes_without_relabeling(membrane,retainer):
    report=qualify_contact(rows(membrane,retainer))
    assert report['passed']
    assert report['continuous_contact_hold_s'] == pytest.approx(3.)
    assert report['maxima']['diaphragm_N'] == membrane
    assert report['maxima']['retainer_N'] == retainer
    assert report['training_eligible'] is False


@pytest.mark.parametrize('name', ['lip_N','contact_rim_N','carriage_other_N','frame_N','forbidden_N'])
def test_printed_clamp_and_other_forbidden_loads_still_fail(name):
    assert not qualify_contact(rows(**{name:1e-8}))['passed']


def test_retainer_is_included_in_original_force_stop():
    from sim.auscultation_expert_layout_v1.controller import ExpertController
    expert=FreshExpert.__new__(FreshExpert)
    expert.env=SimpleNamespace(last_contacts={'retainer':6.01})
    expert.config=SimpleNamespace(hard_force_limit_N=5.5)
    violations,total=ExpertController._violations(expert,dict(compression_m=.0025,hall_N=4.))
    assert total==pytest.approx(6.01)
    assert 'physical force limit' in violations


def test_front_evidence_excludes_printed_parts_but_total_counts_them():
    from sim.auscultation_expert_layout_v1.controller import ExpertController
    expert=FreshExpert.__new__(FreshExpert)
    expert.env=SimpleNamespace(last_contacts={'retainer':4.,'lip':2.01})
    expert.config=SimpleNamespace(hard_force_limit_N=5.5)
    violations,total=ExpertController._violations(expert,dict(compression_m=.0025,hall_N=4.))
    assert total==pytest.approx(6.01)
    assert 'physical force limit' in violations
    assert sum(expert.env.last_contacts.get(k,0.) for k in expert.front_contact_groups)==4.


def test_runtime_classifier_and_ledger_group_match_audit():
    from sim.ausc_policy.fresh_expert import FreshContactEnv
    from sim.auscultation_expert_layout_v1.env import IsaacContactEnv
    env=FreshContactEnv.__new__(FreshContactEnv)
    legacy=IsaacContactEnv.__new__(IsaacContactEnv)
    assert 'retainer' in env.contact_groups
    for pair in ([RET,PATIENT], [PATIENT,RET], [ROOT+'RetentionLip/sector_00',PATIENT],
                 [ROOT+'DeviceEdge',PATIENT], [RET,'/World/Table']):
        assert env.contact_group(*pair)==group(pair)
    assert legacy.contact_group(RET,PATIENT)=='carriage_other'


def test_membrane_lookalike_is_forbidden_in_new_runtime_and_raw_audit():
    from sim.ausc_policy.fresh_expert import FreshContactEnv
    from sim.auscultation_expert_layout_v1.env import IsaacContactEnv
    from sim.ausc_policy.contact_geometry_audit import audit_contacts
    from sim.ausc_policy.retainer_contact import CONTACT_SURFACE
    env=FreshContactEnv.__new__(FreshContactEnv)
    legacy=IsaacContactEnv.__new__(IsaacContactEnv)
    false_b=ROOT+'DeviceContactRing/Diaphragm'
    for pair in ((false_b,PATIENT),(PATIENT,false_b)):
        assert env.contact_group(*pair)=='carriage_other'
        assert legacy.contact_group(*pair)=='diaphragm'
    data=rows(membrane=0.,retainer=0.,carriage_other_N=4.)
    for row in data:
        row.update(phase='HOLD',contact_surface=CONTACT_SURFACE,
                   contact_points=[dict(colliders=[false_b,PATIENT],impulse_Ns=[0.,0.,4/240])])
    audit=audit_contacts(data)
    assert audit['component_force_peak_N']['carriage_other']==4.
    assert audit['component_force_peak_N']['diaphragm']==0.
    assert not qualify_contact(data)['passed']


def test_incomplete_or_nonfinite_new_force_evidence_does_not_pass():
    missing=rows()
    del missing[0]['retainer_N']
    try:
        result=qualify_contact(missing)
    except (KeyError,ValueError):
        pass
    else:
        assert not result['passed']
    for bad in (float('nan'),float('inf'),-1.):
        assert not qualify_contact(rows(retainer=bad))['passed']
    assert not qualify_contact(rows(retainer=6.01))['passed']
    assert not qualify_contact(rows(compression_m=.0061))['passed']
    assert not qualify_contact(rows(hall_N=5.5))['passed']


def test_actual_contact_points_restrict_retainer_permission_to_front():
    from sim.ausc_policy import retainer_contact as module
    audit=getattr(module,'audit_retainer_front',None)
    assert callable(audit), 'Native contact locations must distinguish front from side'
    point=dict(colliders=[RET,PATIENT],position_world=[.022,0.,-.047],
               impulse_Ns=[0.,0.,4/240])
    result=audit([point],np.eye(4),4.)
    assert result['passed'] and result['force_N']==pytest.approx(4.)
    # Same retainer collider, but a side/back contact may not be accepted.
    point['position_world']=[.023,0.,-.046]
    assert not audit([point],np.eye(4),4.)['passed']
    # A 4N ledger with omitted raw points must not pass.
    assert not audit([],np.eye(4),4.)['passed']
    point['position_world']=[.022,0.,-.047]
    world=np.eye(4);world[:3,3]=[1.,2.,3.]
    point['position_world']=(world[:3,:3]@point['position_world']+world[:3,3]).tolist()
    assert audit([point],world,4.)['passed']


def test_independent_force_acceptance_requires_retainer_surface_audit():
    for kind in ('missing','off_front','unmatched','wrong_force'):
        data=rows()
        if kind=='missing':del data[0]['retainer_front_audit']
        elif kind=='off_front':data[0]['retainer_front_audit']['passed']=False
        elif kind=='unmatched':data[0]['retainer_front_audit']['force_matches_ledger']=False
        else:data[0]['retainer_front_audit']['force_N']=0.
        assert not qualify_contact(data)['passed']

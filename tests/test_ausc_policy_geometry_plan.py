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


def test_plane_options_forwards_optional_margin_without_changing_runtime_gate():
    target=dict(contact_plane_radius_m=.023,contact_plane_normal_world=[0.,0.,1.])
    assert geometry_plan.plane_options({},target,None)=={}
    assert geometry_plan.plane_options({},target,{'identity':'p000446'})==dict(
        plane_normal=[0.,0.,1.],plane_tilt_limit_deg=3.)
    config=dict(planning_plane_margin_deg=.25)
    options=geometry_plan.plane_options(config,target,{'identity':'p000446'})
    assert options==dict(plane_normal=[0.,0.,1.],plane_tilt_limit_deg=3.,planning_plane_margin_deg=.25)


@pytest.mark.parametrize('config,patient',[
    (dict(planning_plane_margin_deg=.25),None),
    (dict(planning_plane_margin_deg=None),{'identity':'p000446'}),
    (dict(planning_plane_margin_deg=.25,plane_tilt_limit_deg=10.),{'identity':'p000446'}),
    (dict(planning_plane_margin_deg=.25,fit_refinement_diagnostic={}),{'identity':'p000446'}),
])
def test_geometry_plan_cannot_silently_ignore_margin(config,patient):
    with pytest.raises(ValueError,match='planning.*margin'):
        geometry_plan.plane_options(config,dict(contact_plane_radius_m=.023,contact_plane_normal_world=[0.,0.,1.]),patient)


def test_audit_preserves_margin_with_original_contact_receipt():
    report=candidate_report();policy=dict(schema='m2_planning_plane_margin_v1',margin_deg=.25,candidate_tilt_limit_deg=2.75)
    report['candidates'][0]['planning_plane_policy']=policy
    audit=geometry_plan.fit_audit('R3',.0025,report)
    assert audit['candidates'][0]['first_contact']['planning_plane_policy']==policy


@pytest.mark.parametrize('has_fit',[True,False])
def test_margin_reaches_real_candidate_search_and_plan_receipts(tmp_path,monkeypatch,has_fit):
    """Run orchestration with an analytical fit oracle and stubbed robot planner."""
    import json,sys
    from types import SimpleNamespace
    from sim.ausc_policy import finite_head
    from test_ausc_policy_geometry_ik_hints import fixture,run_geometry_plan,write,sha
    data=fixture(tmp_path);fit=object.__new__(finite_head.FiniteHeadFit)
    output,_=run_geometry_plan(data,monkeypatch,with_hints=False)
    data.target.update(contact_plane_normal_world=[0.,0.,1.],contact_plane_radius_m=.023)
    receipt=data.source_hint.parent/'patient.json'
    write(receipt,dict(identity='p000446',asset_sha256='c'*64,identity_split_sha256='d'*64,
                      non_patient_scene_unchanged=True,targets=[data.target]))
    data.config.update(patient_receipt=dict(file=str(receipt),sha256=sha(receipt)),planning_plane_margin_deg=.25)
    write(data.source_hint.parent/'config.json',data.config)
    def contact(*args):
        if not has_fit:return dict(passed=False,reason='missing_contact_bracket',fit_evaluations=1)
        return dict(candidate_report()['candidates'][0],fit_evaluations=1)
    fit.first_contact=contact
    monkeypatch.setattr(finite_head,'FiniteHeadFit',lambda *args:fit)
    calls=[]
    def plan(target,audit,base,kin,checker,limits,**kwargs):
        calls.append(audit)
        assert checker.check([0.]*7)['clear']
        assert kwargs['seeds_per_candidate']==4 and kwargs['max_routes']==2
        assert {row['first_contact']['plane_tilt_deg'] for row in audit['candidates']}=={0.,1.5,2.,2.5,2.75}
        assert all(row['finite_head_fit']['passed'] for row in audit['candidates'])
        return dict(success=True,route_score=1.,selected_fit=audit['candidates'][-1])
    monkeypatch.setitem(sys.modules,'sim.auscultation_expert_layout_v1.plan_low_elbow',SimpleNamespace(plan_target=plan))
    geometry_plan.main()
    complete=json.loads((output/'COMPLETE.json').read_text())
    fits=json.loads((output/'L3_fits.json').read_text())
    assert complete['planning_plane_policy']==fits['planning_plane_policy']
    assert complete['targets'][0]['geometry_path_found'] is has_fit
    assert len(calls)==int(has_fit) and not complete['physics_executed']
    if has_fit:
        selected=json.loads((output/'L3_fresh_plan.json').read_text())
        first=selected['selected_fit']['first_contact']
        assert first['planning_plane_policy']==complete['planning_plane_policy']
        assert first['plane_tilt_deg']==2.75 and first['plane_tilt_limit_deg']==3.
        assert selected['patient_provenance']['binding']==data.config['patient_receipt']
    else:
        assert not (output/'L3_fresh_plan.json').exists()

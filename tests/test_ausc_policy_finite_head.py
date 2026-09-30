import numpy as np
import pytest
from sim.ausc_policy import geometry_epoch
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics


def test_component_frames_match_actual_132mm_kinematics_under_compression():
    from sim.ausc_policy import finite_head
    kin = OpenArmKinematics(contact_frame=geometry_epoch.contact_frame)
    q = np.array([.2,-.5,.2,1.1,-.3,.1,.2])
    for compression in (0.,.0025,.006):
        wrist = kin.fk(q); tcp = kin.tcp(q,compression)
        fixed, moving = finite_head.component_frames(tcp[:3,3],wrist[:3,:3],compression)
        adapter=np.diag([1.,-1.,-1.,1.]);adapter[2,3]=.085
        slider=np.eye(4);slider[2,3]=compression
        np.testing.assert_allclose(fixed,wrist@adapter,atol=1e-12)
        np.testing.assert_allclose(moving,wrist@adapter@slider,atol=1e-12)


def test_contact_frames_reject_invalid_physical_domain():
    from sim.ausc_policy import finite_head
    for compression in (-.001,.0061,float('nan')):
        with pytest.raises(ValueError):finite_head.component_frames(np.zeros(3),np.eye(3),compression)
    with pytest.raises(ValueError):finite_head.component_frames(np.zeros(3),np.zeros((3,3)),0.)


def test_candidate_family_reaches_diagonals_without_expanding_original_square():
    from sim.ausc_policy.finite_head import FiniteHeadFit
    fit=object.__new__(FiniteHeadFit)
    fit.first_contact=lambda position,normal,rotation,compression: dict(
        passed=True, center_world_m=(np.asarray(position)+.00002*normal).tolist(),
        surface_gap_m=.00002, contact_surface_gap_m=0.,
        diaphragm_residual_distance_m=.00002, first_contact_search_resolution_m=.000001,
        contact_bracket_width_m=.000001, contact_numeric_tolerance_m=1e-9,
        raw_anchor_world_m=np.asarray(position).tolist(), fit=dict(passed=True),
        fit_evaluations=1)
    normal=np.array([.3,.4,np.sqrt(.75)])
    x=np.cross(normal,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(normal,x)
    result=fit.candidates(np.zeros(3),normal)
    assert {35.,40.}.issubset({c['normal_deviation_from_surface_deg'] for c in result['candidates']})
    for c in result['candidates']:
        n=np.array(c['normal_world']);uv=np.array([n@x,n@y])/(n@normal)
        assert np.max(abs(uv))<=.6+1e-12


def plane_fit(*, tilt_touch_m=0., blocked_below_m=None, never_touch=False,
              contact_roundoff=False):
    """Analytical plane/disc oracle; tests search only, not native FCL."""
    from sim.ausc_policy.finite_head import FiniteHeadFit
    fit=object.__new__(FiniteHeadFit)
    queries=[]
    def query(center, rotation, compression):
        gap=float(center[2])-tilt_touch_m
        distance=(gap+.001 if never_touch else max(gap,0.))
        collision=gap<=0 and not never_touch and not contact_roundoff
        if contact_roundoff and gap==0:distance=1e-17
        distances=dict(fixed=.01,clamp=.01,shell=.001,diaphragm=distance)
        if blocked_below_m is not None and center[2]<blocked_below_m:
            distances['clamp']=.0025
        queries.append((np.asarray(center).copy(),compression))
        return dict(passed=False,distances_m=distances,diaphragm_collision=collision,
                    full_robot_collision_checked=False,physics_executed=False)
    fit.check=query
    return fit,queries


@pytest.mark.parametrize('roundoff', [False,True])
def test_first_contact_plane_replaces_millimetre_gap_with_measured_near_contact(roundoff):
    fit,queries=plane_fit(contact_roundoff=roundoff)
    anchor=np.zeros(3);before=anchor.copy()
    result=fit.first_contact(anchor,[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert result['passed']
    assert 1e-5<=result['diaphragm_residual_distance_m']<=3e-5
    assert result['surface_gap_m']<.00003
    assert abs(result['contact_surface_gap_m'])<=result['first_contact_search_resolution_m']
    assert result['contact_bracket_width_m']<=result['first_contact_search_resolution_m']
    np.testing.assert_array_equal(anchor,before)
    np.testing.assert_array_equal(result['raw_anchor_world_m'],anchor)
    np.testing.assert_allclose(result['center_world_m'],queries[-1][0],atol=0,rtol=0)
    assert all(q[1]==.0025 for q in queries)


def test_tilted_finite_disc_stops_at_first_edge_contact_not_raw_anchor():
    from scipy.spatial.transform import Rotation
    angle=np.deg2rad(20.)
    radius=.01
    touch=radius*np.sin(angle)
    fit,_=plane_fit(tilt_touch_m=touch)
    rotation=Rotation.from_euler('y',angle).as_matrix()@np.diag([1.,-1.,-1.])
    result=fit.first_contact(np.zeros(3),[0,0,1],rotation,.003)
    assert result['passed']
    assert abs(result['contact_surface_gap_m']-radius*np.tan(angle))<=1e-6
    assert result['surface_gap_m']>radius*np.tan(angle)
    assert 1e-5<=result['diaphragm_residual_distance_m']<=3e-5
    np.testing.assert_allclose(result['center_world_m'],-rotation[:,2]*result['surface_gap_m'])
    assert result['gap_axis']=='command_normal'
    assert result['raw_anchor_world_m']==[0.,0.,0.]


def test_lip_obstruction_cannot_be_accepted_by_changing_shell_threshold():
    fit,_=plane_fit(blocked_below_m=.0002)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed']
    assert result['reason']=='noncontact_obstruction'
    assert result['rejected_fit']['distances_m']['clamp']==.0025


def test_missing_contact_bracket_is_rejected_even_with_old_1_5mm_fit():
    fit,_=plane_fit(never_touch=True)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed']
    assert result['reason']=='missing_contact_bracket'


@pytest.mark.parametrize('kwargs', [
    dict(position=[np.nan,0,0]),dict(local_normal=[0,0,0]),
    dict(local_normal=[0,0,np.nan]),dict(rotation=np.zeros((3,3))),
    dict(compression=-.001),dict(compression=.0061),
])
def test_first_contact_rejects_bad_inputs(kwargs):
    fit,_=plane_fit()
    inputs=dict(position=np.zeros(3),local_normal=[0,0,1],rotation=np.diag([1.,-1.,-1.]),compression=.0025)
    inputs.update(kwargs)
    with pytest.raises(ValueError):fit.first_contact(**inputs)


def test_first_contact_rejects_nonfinite_distance():
    fit,_=plane_fit()
    fit.check=lambda *args:dict(distances_m=dict(fixed=.01,clamp=.01,shell=.001,diaphragm=np.nan),
                               diaphragm_collision=False)
    with pytest.raises(ValueError,match='distance'):
        fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)


def test_first_contact_finds_thin_surface_even_when_both_search_endpoints_are_clear():
    fit,_=plane_fit()
    def thin_surface(center,rotation,compression):
        z=float(center[2]);lower=.003;upper=.0031
        inside=lower<=z<=upper
        return dict(distances_m=dict(fixed=.01,clamp=.01,shell=.001,
            diaphragm=0. if inside else min(abs(z-lower),abs(z-upper))),diaphragm_collision=inside)
    fit.check=thin_surface
    assert not thin_surface([0,0,0],None,None)['diaphragm_collision']
    assert not thin_surface([0,0,.005],None,None)['diaphragm_collision']
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert result['passed']
    assert abs(result['contact_surface_gap_m']-.0031)<=1e-6
    assert result['surface_gap_m']>.0031


@pytest.mark.parametrize('part,clearance', [('fixed',.002999),('shell',.000299)])
def test_first_contact_never_lowers_noncontact_thresholds(part,clearance):
    fit,_=plane_fit()
    original=fit.check
    def blocked(*args):
        result=original(*args);result['distances_m'][part]=clearance
        return result
    fit.check=blocked
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'


def test_final_near_contact_position_rechecks_all_noncontact_parts():
    fit,_=plane_fit()
    original=fit.check;counts={}
    def changing_final(*args):
        result=original(*args)
        key=tuple(args[0]);counts[key]=counts.get(key,0)+1
        d=result['distances_m']['diaphragm']
        if 1e-5<=d<=3e-5 and counts[key]>1:
            result['distances_m']['clamp']=.002
        return result
    fit.check=changing_final
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'
    assert result['rejected_fit']['distances_m']['clamp']==.002


def test_already_touching_outer_endpoint_has_no_valid_outside_bracket():
    fit,_=plane_fit(tilt_touch_m=.0051)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='outer_endpoint_not_clear'


@pytest.mark.parametrize('roundoff', [-1e-16,0.,1e-16])
def test_contact_bracket_handles_equal_noncontact_boundaries_without_relaxing_final_gate(roundoff):
    fit,_=plane_fit(contact_roundoff=True)
    original=fit.check
    def boundary_roundoff(*args):
        result=original(*args);gap=float(args[0][2])
        result['distances_m'].update(fixed=.003+gap+roundoff,clamp=.003+gap+roundoff,
                                    shell=.0003+gap+roundoff)
        return result
    fit.check=boundary_roundoff
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert result['passed']
    assert result['fit']['distances_m']['fixed']>=.003
    assert result['fit']['distances_m']['clamp']>=.003
    assert result['fit']['distances_m']['shell']>=.0003
    assert result['bracket_noncontact_numeric_tolerance_m']==1e-9
    assert result['first_contact_search_min_step_m']==1e-9


def test_contact_boundary_violation_larger_than_one_nm_is_rejected():
    fit,_=plane_fit();original=fit.check
    def blocked_boundary(*args):
        result=original(*args)
        result['distances_m']['shell']=.0003+float(args[0][2])-3e-9
        return result
    fit.check=blocked_boundary
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'


def test_final_near_contact_has_no_one_nm_clearance_exemption():
    fit,_=plane_fit();original=fit.check;counts={}
    def final_roundoff(*args):
        result=original(*args);key=tuple(args[0]);counts[key]=counts.get(key,0)+1
        if 1e-5<=result['distances_m']['diaphragm']<=3e-5 and counts[key]>1:
            result['distances_m']['shell']=.0003-1e-16
        return result
    fit.check=final_roundoff
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'

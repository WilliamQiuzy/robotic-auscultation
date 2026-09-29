import copy
import numpy as np
import pytest
from sim.auscultation_pi05_v7.numerical_command_gate import compare_command,DT


def actual_fixture():
    q=np.zeros((4,7));q[:,3]=[1.171404415658409,1.1716842574510222,1.171962995227978,1.1722412076929807]
    profile=dict(p=[1.171122891146435],v=[.06781410549334305],a=[-.13025796851787588],j=[8.],start_s=0.,end_s=.02440373732307942)
    certificate=dict(schema='wam_constant_jerk_command_certificate_v1',samples=[dict(physics_row=4788+i,calculation_id=379,
        trajectory_time_s=t,command=q[i].tolist()) for i,t in enumerate([DT,2*DT,3*DT,4*DT])],
        axis_sections={'3':profile})
    return q,certificate


def test_actual_rejected_stencil_has_separate_numerical_pass():
    q,c=actual_fixture();r=compare_command(q[:3],q[3],velocity=.175,acceleration=1.,certificate=c)
    assert r['raw_passed'] is False and r['passed'] is True
    assert r['raw_peaks']['jerk']==8.000000011179509
    assert r['intervened'] is True and r['proofs']['3']['position_errors_within_half_ulp'] is True
    assert r['proofs']['3']['analytic_jerk_rad_s3']==8.
    assert r['jerk_representation_budget_rad_s3'][3]<1.23e-8


def test_old_gate_passes_without_any_exception_certificate():
    r=compare_command(np.zeros((3,7)),np.zeros(7),velocity=.175,acceleration=1.)
    assert r['passed'] and r['raw_passed'] and not r['intervened']


@pytest.mark.parametrize('mutation',[None,'different_query','different_row','different_clock','different_q','cross_segment'])
def test_missing_or_inconsistent_certificate_fails_closed(mutation):
    q,c=actual_fixture()
    if mutation is None:c=None
    elif mutation=='different_query':c['samples'][0]['calculation_id']=378
    elif mutation=='different_row':c['samples'][0]['physics_row']-=1
    elif mutation=='different_clock':c['samples'][0]['trajectory_time_s']+=1e-8
    elif mutation=='different_q':c['samples'][0]['command'][3]+=1e-9
    elif mutation=='cross_segment':c['axis_sections']['3']['end_s']=2.5*DT
    r=compare_command(q[:3],q[3],velocity=.175,acceleration=1.,certificate=c)
    assert r['passed'] is False and r['raw_passed'] is False


def test_true_analytic_jerk_excess_cannot_hide_inside_representation_budget():
    q,c=actual_fixture();c['axis_sections']['3']['j']=[8.+1e-10]
    r=compare_command(q[:3],q[3],velocity=.175,acceleration=1.,certificate=c)
    assert not r['passed'];assert 'analytic jerk' in r['reason']


def test_clear_q_jerk_excess_rejects_without_clipping():
    q,c=actual_fixture();q[-1,3]+=1e-10;c['samples'][-1]['command']=q[-1].tolist()
    original=q.copy();r=compare_command(q[:3],q[3],velocity=.175,acceleration=1.,certificate=c)
    assert not r['passed'];np.testing.assert_array_equal(q,original)


def test_velocity_and_acceleration_gate_are_never_overridden():
    q,c=actual_fixture()
    assert not compare_command(q[:3],q[3],velocity=.001,acceleration=1.,certificate=c)['passed']
    assert not compare_command(q[:3],q[3],velocity=.175,acceleration=.001,certificate=c)['passed']


@pytest.mark.parametrize('history,command',[(np.zeros((2,7)),np.zeros(7)),(np.full((3,7),np.nan),np.zeros(7)),(np.zeros((3,7)),np.full(7,np.inf))])
def test_malformed_or_nonfinite_history_raises(history,command):
    with pytest.raises(ValueError):compare_command(history,command,velocity=.175,acceleration=1.)


def test_q_stencil_accurate_arithmetic_is_not_itself_a_repair():
    from fractions import Fraction as F
    q,_=actual_fixture();values=[F.from_float(float(x)) for x in q[:,3]]
    jerk=(-values[0]+3*values[1]-3*values[2]+values[3])/F(1,240)**3
    assert jerk>F(8)+F(1,10**8)

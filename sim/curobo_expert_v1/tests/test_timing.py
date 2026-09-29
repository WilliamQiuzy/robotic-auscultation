import json
from pathlib import Path
import numpy as np
import pytest
from sim.curobo_expert_v1.timing import smooth_commands,condition_path
from sim.expert_quality_20260924.candidate_planner import discrete_dynamic_peaks
from sim.expert_quality_20260924.candidate_controller import CandidateController


def test_curved_path_obeys_limits_including_stationary_endpoints():
    path=np.zeros((5,7));path[:,0]=[0,.2,.3,.35,.6];path[:,1]=[0,.1,.4,.3,.1]
    commands,r=smooth_commands(path,return_report=True)
    np.testing.assert_array_equal(commands[-1],path[-1])
    peaks=discrete_dynamic_peaks(path,commands)
    assert peaks==r['peaks']
    assert peaks['velocity_rad_s']<=.25 and peaks['acceleration_rad_s2']<=1 and peaks['jerk_rad_s3']<=1.5


def test_stationary_and_duplicate_knots_are_defined():
    path=np.zeros((3,7));commands,r=smooth_commands(path,return_report=True)
    assert commands.shape==(240,7) and np.all(commands==0) and r['peaks']['jerk_rad_s3']==0


def test_budget_exhaustion_never_returns_unbounded_commands():
    with pytest.raises(RuntimeError):smooth_commands(np.array([np.zeros(7),np.ones(7)]),maximum_seconds=.05)


def test_real_r1_curve_avoids_polyline_knot_retiming_pathology():
    root=Path(__file__).resolve().parents[2]/'auscultation_expert_layout_v1'
    row=json.loads((root/'results/expert_v2/plans/planned_paths_R1.json').read_text())['points'][0]
    route,index=CandidateController._canonical(row)
    commands,r=smooth_commands(route[index:],return_report=True)
    assert r['duration_s']<60  # prior polyline needed 201.2 s at identical jerk bound
    assert r['peaks']['jerk_rad_s3']<=1.5
    # Geometry is a proposal only; this test deliberately does not claim clearance.


def test_dense_float32_quantization_does_not_create_slow_artificial_knots():
    q=np.linspace(np.ones(7),np.ones(7)*1.3,5000).astype(np.float32).astype(float)
    kept,proof=condition_path(q)
    assert len(kept)==2 and proof['retained_indices']==[0,4999]
    _,r=smooth_commands(q,return_report=True)
    assert r['duration_s']<5 and r['peaks']['jerk_rad_s3']<=1.5


def test_conditioning_preserves_real_bend_and_every_removed_sample_bound():
    a=np.zeros(7);b=np.ones(7)*.1;c=b.copy();c[2]=-.3
    path=np.vstack([np.linspace(a,b,99),np.linspace(b,c,113)[1:]])
    q,p=condition_path(path)
    np.testing.assert_array_equal(q,np.vstack([a,b,c]))
    assert p['retained_indices']==[0,98,len(path)-1]

"""Known geometry is an independent contact-report convention oracle."""
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def module():return importlib.import_module('sim.ausc_policy.contact_semantics_oracle')


def test_matrix_contains_front_side_back_gap_motion_rigid_and_creation_controls():
    cases=module().cases()
    assert len(cases)==72 and len({c['id'] for c in cases})==72
    assert {c['feature'] for c in cases}=={'front','side','back'}
    assert {c['gap_m'] for c in cases}=={0.,.00015,.00035}
    assert {c['speed_m_s'] for c in cases}=={0.,.02}
    assert {c['transformed'] for c in cases}=={False,True}
    assert {c['reverse_creation'] for c in cases}=={False,True}
    assert max(c['gap_m'] for c in cases)<.001+.00003


@pytest.mark.parametrize('feature',['front','side','back'])
def test_known_support_face_contains_exact_wedge_vertices(feature):
    from sim.ausc_policy.retainer_geometry import ring_sector_meshes
    vertices,_=ring_sector_meshes()[0]
    frame=module().support_face(feature)
    n=np.asarray(frame['outward_local']);s=frame['support_m']
    distances=vertices@n-s
    assert distances.max()==pytest.approx(0.,abs=1e-14)
    assert np.count_nonzero(np.abs(distances)<1e-12)==4
    assert np.dot(n,frame['center_local'])==pytest.approx(s)


@pytest.mark.parametrize('ring_index',[0,1])
def test_oracle_compares_hypotheses_without_promoting_or_changing_points(ring_index):
    from scipy.spatial.transform import Rotation
    m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',[19,27,31],degrees=True).as_matrix()
    m[:3,3]=[.2,.3,1.]
    face=module().support_face('front');n=m[:3,:3]@np.asarray(face['outward_local'])
    center=m[:3,:3]@np.asarray(face['center_local'])+m[:3,3]
    plane=center+.00035*n
    reported=plane.copy();normal=-n if ring_index==0 else n
    row=module().point_residuals(reported,normal,.00035,ring_index,m,plane,n,face)
    expected='1.0' if ring_index==0 else '-1.0'
    assert abs(row['plate_raw_residual_m'])<1e-14
    assert set(row['plate_face_residual_m'])==set(row['ring_face_residual_m'])
    assert abs(row['plate_face_residual_m'][expected])==pytest.approx(.00035)
    assert abs(row['ring_face_residual_m'][expected])<1e-14
    assert abs(row['ring_face_residual_m']['0.0'])==pytest.approx(.00035)
    assert row['evidence_only'] and not row['production_rule_authorized']
    np.testing.assert_array_equal(reported,plane)


@pytest.mark.parametrize('feature',['front','side','back'])
def test_plate_normal_matches_authored_float32_triangle_winding(feature):
    axis=np.asarray(module().support_face(feature)['outward_local'])
    fixture=module().plate_fixture(np.array([.2,.3,1.]),axis)
    points=np.asarray(fixture['points'])
    normal=np.cross(points[2]-points[0],points[1]-points[0])
    normal/=np.linalg.norm(normal)
    np.testing.assert_allclose(normal,fixture['plate_normal_world'],atol=1e-12)
    assert np.dot(normal,axis)<-.999999


def test_missing_controls_are_unavailable_and_never_semantically_complete():
    result=module().evidence_status(module().cases(),[])
    assert not result['evidence_complete']
    assert result['evidence_status']=='unavailable'
    assert len(result['cases'])==72
    assert all(r['control_status']=='unavailable_no_contacts' for r in result['cases'])
    assert result['native_pair_reversal_status']=='unavailable'


def test_missing_pose_or_normal_is_explicit_even_with_positive_contact():
    c=module().cases()[0]
    rows=[dict(case=c['id'],execution_stage='controlled_step',force_N=1.,
        retainer_pair_index=0,unavailable=['normal_world','pre'],by_pose={})]
    result=module().evidence_status([c],rows)
    assert not result['evidence_complete']
    assert result['cases'][0]['control_status']=='unavailable_missing_fields'
    assert result['cases'][0]['unavailable_counts']=={'normal_world':1,'pre':1}


def test_zero_impulse_geometry_is_available_and_pair_reversal_is_observed():
    c=module().cases()[0]
    rows=[dict(case=c['id'],execution_stage='controlled_step',force_N=0.,
        retainer_pair_index=i,unavailable=[],by_pose={}) for i in [0,1]]
    result=module().evidence_status([c],rows)
    assert result['evidence_complete']
    assert result['cases'][0]['positive_contacts']==0
    assert result['positive_impulse_cases']==0
    assert result['native_pair_reversal_status']=='observed'
    rows[0]['unavailable']=['normal_world']
    result=module().evidence_status([c],rows)
    assert not result['evidence_complete']
    assert result['cases'][0]['unavailable_counts']=={'normal_world':1}


@pytest.mark.parametrize('failure_stage',['run','close','sdk_zero_exit',None])
def test_failure_preserves_partial_artifacts_and_complete_waits_for_close(tmp_path,monkeypatch,failure_stage):
    m=module();output=tmp_path/'output';config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(experiment='contact_semantics_oracle')))
    monkeypatch.setattr(sys,'argv',['oracle','--config',str(config),'--output',str(output)])
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    closed=[]
    class App:
        def __init__(self,settings):assert settings['fast_shutdown'] is False
        def close(self):
            assert not (output/'COMPLETE.json').exists()
            closed.append(True)
            if failure_stage=='close':raise RuntimeError('close failed')
    monkeypatch.setitem(sys.modules,'isaacsim',SimpleNamespace(SimulationApp=App))
    def fake_run(folder):
        folder.mkdir();(folder/'native_journal.jsonl').write_text('{"kind":"contact_callback"}\n')
        if failure_stage=='run':raise RuntimeError('step failed after contact')
        if failure_stage=='sdk_zero_exit':raise SystemExit(0)
        return dict(execution_completed=True,completed=False,evidence_complete=False)
    monkeypatch.setattr(m,'run',fake_run)
    if failure_stage:
        with pytest.raises(RuntimeError):m.main()
        failed=json.loads((output/'FAILED.json').read_text())
        assert not failed['execution_completed'] and not failed['completed']
        assert failed['artifacts']['native_journal.jsonl']['bytes']>0
        assert not (output/'COMPLETE.json').exists()
    else:
        m.main();complete=json.loads((output/'COMPLETE.json').read_text())
        assert complete['execution_completed']
        assert complete['runtime_shutdown']['app_close_returned']
    assert closed==[True]

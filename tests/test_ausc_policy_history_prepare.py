import pytest
from sim.ausc_policy.history_prepare import history_frame_indices


def frames():
    return [dict(index=i,time_s=(i+1)*.05,target='R4',state=[0.]*32) for i in range(30)]


def test_causal_history_uses_only_actual_same_target_camera_frames():
    assert history_frame_indices(frames(),20)==[10,15,18,19,20]


@pytest.mark.parametrize('defect',['early','time_gap','target_change','index_change'])
def test_no_history_padding_retiming_or_cross_target_reuse(defect):
    rows=frames();origin=20
    if defect=='early':origin=5
    if defect=='time_gap':rows[15]['time_s']+=.001
    if defect=='target_change':rows[18]['target']='L4'
    if defect=='index_change':rows[19]['index']=18
    with pytest.raises(ValueError):history_frame_indices(rows,origin)

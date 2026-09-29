"""The lower staging constraint has literal world-height/normal-clearance meaning."""
import numpy as np
import pytest
import time
import json
from types import SimpleNamespace

from sim.auscultation_training.plan_low_elbow import lower_staging, default_limits, path_score, candidate_indices, checked_transfer
from sim.auscultation_training.kinematics import OpenArmKinematics
from sim.auscultation_training import plan_low_elbow


@pytest.mark.parametrize("nz", [-.6, 0., .6])
def test_staging_has_exact_vertical_drop_and_outward_clearance(nz):
    normal = np.array([0., -np.sqrt(1.-nz*nz), nz])
    pre = np.array([.1, .8, 1.4])
    base = np.eye(4); base[:3,3] = [.2,.3,.4]
    target=pre-normal*.06
    world, local = lower_staging({"pre_world":pre,"normal_world":normal,"free_end_world":target+normal*.015},base)
    assert world[2] == pytest.approx(pre[2]-.06)
    assert np.dot(world-target, normal) >= .035-1e-12
    np.testing.assert_allclose(base[:3,:3]@local+base[:3,3],world)


def test_staging_rejects_degenerate_vertical_normal():
    with pytest.raises(ValueError,match="near-vertical"):
        lower_staging({"pre_world":np.ones(3),"normal_world":np.array([0.,0.,1.])},np.eye(4))


def test_default_envelope_contains_home_and_excludes_proximal_half_turns():
    kin=OpenArmKinematics();base=np.eye(4)
    limits=default_limits(kin,base)
    assert limits.joint_lower_rad[0] == pytest.approx(np.deg2rad(-140.))
    assert limits.joint_lower_rad[1] == pytest.approx(np.deg2rad(-140.))
    assert np.all(np.asarray(limits.joint_lower_rad)<=0)
    assert np.all(np.asarray(limits.joint_upper_rad)>=0)
    expected=(base@kin.link_transforms(np.zeros(7))["openarm_left_link1"])[2,3]
    assert limits.max_elbow_z_m == expected


def test_scoring_penalizes_a_redundant_joint_excursion():
    kin=OpenArmKinematics();base=np.eye(4)
    start=np.zeros(7);goal=np.array([0.,0.,0.,.4,0.,0.,0.])
    detour=np.array([.5,0.,0.,.2,0.,0.,0.])
    assert path_score([start,goal],kin,base)<path_score([start,detour,goal],kin,base)


def test_bounded_candidates_cover_tail_of_large_audit_without_duplicates():
    indices=candidate_indices(484,60)
    assert len(indices)==len(set(indices))==60
    assert indices[0]==0 and 483 in indices[:24]
    assert sorted(candidate_indices(10,60))==list(range(10))


def test_transfer_deadline_stops_before_any_new_collision_sample():
    kin=OpenArmKinematics();base=np.eye(4);limits=default_limits(kin,base)
    def check(*args,**kwargs):
        pytest.fail("expired transfer search must not perform collision work")
    path, report=checked_transfer(SimpleNamespace(check=check),limits,np.zeros(7),np.zeros(7),
                                  kin,base,np.random.default_rng(1),time.monotonic()-1.)
    assert path is None and report["reason"]=="time_budget_exhausted"


def test_direct_transfer_samples_all_joint_interpolation_before_acceptance():
    kin=OpenArmKinematics();base=np.eye(4);limits=default_limits(kin,base);seen=[]
    def check(q,**kwargs):
        seen.append(q.copy());return {"clear":True,"minimum_distance_m":.003}
    home=np.zeros(7);goal=home.copy();goal[3]=.12
    path, report=checked_transfer(SimpleNamespace(check=check),limits,home,goal,
                                  kin,base,np.random.default_rng(1),time.monotonic()+3.)
    assert len(path)==2 and report["reason"]=="direct_checked"
    assert len(seen)>=9
    np.testing.assert_allclose(seen[0],home);np.testing.assert_allclose(seen[-1],goal)


def test_hint_from_another_audit_is_rejected_before_any_ik(tmp_path,monkeypatch):
    hints=tmp_path/'hints.json'
    hints.write_text(json.dumps({'R1':[{'candidate_index':0,'loaded_q':[0.]*7,'audit_sha256':'0'*64}]}))
    monkeypatch.setattr(plan_low_elbow,'plan_target',lambda *args,**kwargs:pytest.fail('unbound hint reached IK'))
    with pytest.raises(ValueError,match='hint audit source differs'):
        plan_low_elbow.main(['--points','R1','--hints',str(hints),'--output-dir',str(tmp_path/'plans')])

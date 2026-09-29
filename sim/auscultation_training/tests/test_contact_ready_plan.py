"""Regression for a loaded-safe plan whose prepose violates contact posture."""
import hashlib
import json
import numpy as np
import pytest

from sim.auscultation_training.controller import validate_contact_ready_route, validate_plan
from sim.auscultation_training.kinematics import OpenArmKinematics, DEFAULT_URDF_PATH
from sim.auscultation_training.posture import PostureLimits
from sim.auscultation_training.tests.test_controller import fixture_plan


BASE = np.array([[1., 0., 0., 0.],
                 [0., -3.5745606228321236e-6, .9999999999936114, .3309999994933605],
                 [0., -.9999999999936114, -3.5745606228321236e-6, 1.376228643993393],
                 [0., 0., 0., 1.]])
PRE = [1.1844562862516155, .10365817541713009, -1.250368741466552,
       2.281299985977783, -.8385123604523895, .6434833046708357, -.08659697376744688]
LOADED = [1.1743152732835647, .000508082505952345, -1.3033869468422703,
          2.204951937954263, -.8583749238240931, .6436918303104692, -.03897499798209649]


def fixture():
    kin = OpenArmKinematics()
    limits = PostureLimits(1.451228420583354, tuple(kin.lower_limits), tuple(kin.upper_limits),
                          max_elbow_above_shoulder_m=.075, min_contact_elbow_below_wrist_m=.03)
    row = {"posture_limits": limits.to_dict(), "approach_q": [PRE, LOADED], "loaded_q": LOADED}
    return kin, limits, row


def test_loaded_success_does_not_qualify_unsafe_prepose():
    kin, limits, row = fixture()
    assert limits.evaluate(kin, BASE, LOADED, contact=True)["clear"]
    assert limits.evaluate(kin, BASE, PRE, contact=True)["elbow_below_wrist_m"] == pytest.approx(.026274, abs=1e-6)
    with pytest.raises(ValueError, match="contact_elbow_below_wrist"):
        validate_contact_ready_route(row, BASE, kin)


def test_plan_preflight_rejects_contact_unsafe_prepose(tmp_path):
    path = fixture_plan(tmp_path)
    plan = json.loads(path.read_text())
    _, _, row = fixture()
    plan["world_from_left_base"] = BASE.tolist()
    plan["points"][0].update(row)
    plan["points"][0]["transfer_q"] = [[0.]*7, PRE]
    replacements = {"runtime": json.dumps({"world_from_left_base": BASE.tolist()}).encode(),
                    "kinematics_urdf": DEFAULT_URDF_PATH.read_bytes()}
    for label, content in replacements.items():
        (tmp_path/plan["provenance"][label+"_file"]).write_bytes(content)
        plan["provenance"][label+"_sha256"] = hashlib.sha256(content).hexdigest()
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="contact_elbow_below_wrist"):
        validate_plan(path, tmp_path, tmp_path)


def test_safe_endpoints_do_not_hide_unsafe_approach_waypoint():
    kin, _, row = fixture()
    row["approach_q"] = [LOADED, PRE, LOADED]
    with pytest.raises(ValueError, match="contact_elbow_below_wrist"):
        validate_contact_ready_route(row, BASE, kin)


def test_contact_ready_posture_is_accepted():
    kin, _, row = fixture()
    row["approach_q"] = [LOADED, LOADED]
    validate_contact_ready_route(row, BASE, kin)


@pytest.mark.parametrize("loaded", [[], [float("nan")]*7, [0.]*6])
def test_malformed_loaded_pose_is_rejected(loaded):
    kin, _, row = fixture()
    row["loaded_q"] = loaded
    with pytest.raises(ValueError, match="invalid contact-ready route"):
        validate_contact_ready_route(row, BASE, kin)

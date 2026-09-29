"""Independently recompute every selected accepted pose and source binding."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .evaluate_layouts import ROOT, HERE, PLANS, sha
from sim.auscultation_training.kinematics import OpenArmKinematics
from sim.auscultation_training.posture import PostureLimits


def main():
    kin = OpenArmKinematics()
    initial = json.loads((HERE / "layout_ik_review.json").read_text())
    base = np.asarray(initial["world_from_left_base"])
    limits = PostureLimits.from_dict(initial["posture_limits"], kin)
    count = 0
    max_position_error = max_rotation_error = 0.
    summaries = []
    for filename in ["layout_ik_review.json", "layout_ik_refinement.json", "layout_ik_factorial.json"]:
        document = json.loads((HERE / filename).read_text())
        assert document["source_unchanged"]
        assert all(sha(ROOT/p)==h for p,h in document["source_sha256"].items())
        if "initial_report_sha256" in document:
            assert document["initial_report_sha256"] == sha(HERE / "layout_ik_review.json")
        for variant in document["variants"]:
            accepted = []
            failures = []
            for point in variant["points"]:
                row = json.loads((PLANS/f"planned_paths_{point['name']}.json").read_text())["points"][0]
                assert np.allclose(point["target_world_m"],np.asarray(row["position_world"])+variant["offset_world_m"],rtol=0,atol=1e-12)
                for pose_name,pose in point["poses"].items():
                    if not pose["accepted"]:
                        failures.append(point["name"]+":"+pose_name); continue
                    selected=pose["selected"]
                    q=np.asarray(selected["q_rad"])
                    actual=base@kin.tcp(q,pose["compression_m"])
                    pe=float(np.linalg.norm(actual[:3,3]-pose["target_tcp_world_m"]))
                    re=float(np.linalg.norm(Rotation.from_matrix(actual[:3,:3]@np.asarray(pose["rotation_world"]).T).as_rotvec()))
                    assert pe<=2e-5 and re<=2e-4
                    assert limits.evaluate(kin,base,q,contact=pose_name!="staging")["clear"]
                    links=kin._chain(q)[0]
                    elbow_z=(base@links["openarm_left_link4"])[2,3]
                    wrist_z=(base@links["openarm_left_link7"])[2,3]
                    assert elbow_z<=limits.max_elbow_z_m-.005+1e-8
                    if pose_name!="staging":
                        assert wrist_z-elbow_z>=limits.min_contact_elbow_below_wrist_m+.005-1e-8
                    margin=float(np.rad2deg(np.min(np.r_[q-kin.lower_limits,kin.upper_limits-q])))
                    assert abs(margin-selected["min_urdf_joint_margin_deg"])<=1e-9
                    accepted.append((margin,point["name"],pose_name))
                    max_position_error=max(max_position_error,pe)
                    max_rotation_error=max(max_rotation_error,re)
                    count+=1
            summaries.append({"name":variant["name"],"offset_world_m":variant["offset_world_m"],
                "accepted_pose_count":len(accepted),"all_three_poses_accepted_points":variant["all_three_poses_accepted_points"],
                "worst_urdf_margin_deg_point_pose":min(accepted) if accepted else None,"failed_poses":failures})
    report={"accepted":True,"scope":"Recomputed FK, position/rotation residual, unchanged tool pose translation, URDF and posture constraints, source SHA256. No collision/path/dynamics validation.",
        "selected_accepted_poses_checked":count,"max_position_error_m":max_position_error,"max_orientation_error_rad":max_rotation_error,
        "input_sha256":{n:sha(HERE/n) for n in ["evaluate_layouts.py","refine_layouts.py","factorial_control.py","layout_ik_review.json","layout_ik_refinement.json","layout_ik_factorial.json","verify_review.py"]},"variants":summaries}
    (HERE/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="variants"},indent=2))


if __name__=="__main__":
    main()

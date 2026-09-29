"""Follow-up static IK: gentler seat changes and bounded alternate L3 seeds."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .evaluate_layouts import ROOT, HERE, MODULE, PLANS, NAMES, sha, metrics
from sim.auscultation_training.kinematics import OpenArmKinematics
from sim.auscultation_training.pose_ik import IKBudgetExpired, solve_posture_ik
from sim.auscultation_training.posture import PostureLimits


def main():
    began = time.monotonic()
    deadline = began + 150.
    original = json.loads((HERE / "layout_ik_review.json").read_text())
    assert all(sha(ROOT/p)==h for p,h in original["source_sha256"].items())
    kin = OpenArmKinematics()
    base = np.asarray(original["world_from_left_base"])
    limits = PostureLimits.from_dict(original["posture_limits"], kin)
    rng = np.random.default_rng(20260923)
    variants = [("farther_5cm_lower_2cm", [0., .05, -.02], NAMES, 2),
                ("farther_10cm_lower_2cm", [0., .10, -.02], NAMES, 2),
                ("farther_5cm_only", [0., .05, 0.], NAMES, 2),
                ("farther_10cm_lower_3cm", [0., .10, -.03], NAMES, 2),
                ("farther_5cm_lower_5cm_L3_extended", [0., .05, -.05], ["L3"], 14),
                ("farther_10cm_lower_5cm_L3_extended", [0., .10, -.05], ["L3"], 14)]
    out = {"schema": "layout_static_ik_refinement_v1", "scope": original["scope"],
        "assumptions": {**original["assumptions"], "selection": "largest min URDF joint margin among accepted deterministic seeds", "global_wall_budget_s": 150.,
            "stool_seat_and_body_vertical_translation_m": "varies by scenario", "seed_detail": "first two seeds identical to initial screen; further seeds uniform within existing posture envelope, random seed20260923"},
        "initial_report_sha256": sha(HERE / "layout_ik_review.json"),
        "source_sha256": {**original["source_sha256"], str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "variants": []}
    for name, offset, names, seed_count in variants:
        delta = np.asarray(offset)
        variant = {"name":name, "offset_world_m":offset, "seeds_per_pose":seed_count, "points":[]}
        for point_name in names:
            row = json.loads((PLANS/f"planned_paths_{point_name}.json").read_text())["points"][0]
            point = {"name":point_name, "target_world_m":(np.asarray(row["position_world"])+delta).tolist(), "poses":{}}
            for pose_name, original_q, compression, contact in [
                    ("staging",row["lower_staging_q"],0.,False),
                    ("precontact",row["approach_q"][0],0.,True),
                    ("loaded",row["loaded_q"],row["loaded_compression_m"],True)]:
                q0 = np.asarray(original_q)
                tcp = kin.tcp(q0,compression)
                pos = tcp[:3,3]+base[:3,:3].T@delta
                seeds = [q0,q0+np.asarray([-.12,-.18,.08,-.10,.08,-.03,-.06])]
                seeds += [rng.uniform(limits.joint_lower_rad,limits.joint_upper_rad) for _ in range(seed_count-2)]
                attempts=[]
                for si, seed in enumerate(seeds):
                    if time.monotonic()>=deadline:
                        attempts.append({"seed_index":si,"accepted":False,"budget_expired":True}); break
                    try:
                        q,ok,pe,re=solve_posture_ik(kin,base,limits,pos,tcp[:3,:3],seed,compression=compression,
                            maxiter=120,deadline=min(deadline,time.monotonic()+2.),contact=contact)
                        attempts.append({"seed_index":si,"accepted":bool(ok),"position_error_m":pe,
                            "orientation_error_rad":re,**metrics(kin,base,limits,q,contact)})
                    except IKBudgetExpired:
                        attempts.append({"seed_index":si,"accepted":False,"budget_expired":True})
                accepted=[x for x in attempts if x["accepted"]]
                candidates=accepted or [x for x in attempts if "position_error_m" in x]
                selected=(max(candidates,key=lambda x:x["min_urdf_joint_margin_deg"]) if accepted else
                    min(candidates,key=lambda x:x["position_error_m"]+.1*x["orientation_error_rad"])) if candidates else None
                point["poses"][pose_name]={"accepted":bool(accepted),"compression_m":compression,
                    "target_tcp_world_m":(base[:3,:3]@pos+base[:3,3]).tolist(),"rotation_world":(base[:3,:3]@tcp[:3,:3]).tolist(),
                    "attempts":attempts,"selected":selected}
            point["all_three_poses_accepted"]=all(x["accepted"] for x in point["poses"].values())
            variant["points"].append(point)
        variant["all_three_poses_accepted_points"]=sum(p["all_three_poses_accepted"] for p in variant["points"])
        variant["accepted_pose_count"]=sum(s["accepted"] for p in variant["points"] for s in p["poses"].values())
        accepted=[s["selected"] for p in variant["points"] for s in p["poses"].values() if s["accepted"]]
        variant["min_urdf_joint_margin_deg_for_accepted_poses"]=min(s["min_urdf_joint_margin_deg"] for s in accepted) if accepted else None
        out["variants"].append(variant)
        print(json.dumps({k:v for k,v in variant.items() if k!="points"}),flush=True)
        (HERE/"layout_ik_refinement.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    out["elapsed_seconds"]=time.monotonic()-began
    out["source_unchanged"]=all(sha(ROOT/p)==h for p,h in out["source_sha256"].items())
    assert out["source_unchanged"]
    (HERE/"layout_ik_refinement.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()

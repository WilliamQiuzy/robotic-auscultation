"""GPU acceptance: official Franka path plus deliberately colliding start.

This demonstrates library execution only. It is not a WAM or Isaac acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path("/home/william/wam-sim/eval_curobo/curobo/src/curobo/content")
    cfg = yaml.safe_load((source / "configs/robot/franka.yml").read_text())
    k = cfg["robot_cfg"]["kinematics"]
    k["urdf_path"] = str(source / "assets" / k["urdf_path"])
    k["asset_root_path"] = str(source / "assets" / k["asset_root_path"])
    k["collision_spheres"] = str(source / "configs/robot" / k["collision_spheres"])
    start = [0., -1.3, 0., -2.5, 0., 1., 0.]
    end = [0.3, -1.3, 0., -2.5, 0., 1., 0.]
    common = {"robot_config": cfg, "start7": start, "end7": end,
              "options": {"max_attempts": 4, "timeout_s": 15.0}}
    cases = {
        "free_motion": {**common, "world": {"cuboid": {"distant_obstacle": {
            "dims": [0.2, 0.2, 0.2], "pose": [3., 3., 3., 1., 0., 0., 0.]}}}},
        "blocked_start": {**common, "world": {"cuboid": {"enclosing_obstacle": {
            "dims": [3., 3., 3.], "pose": [0., 0., 0.5, 1., 0., 0., 0.]}}}},
    }
    summary = {"scope": "official_franka_library_probe_not_WAM_acceptance", "cases": {}}
    for name, req in cases.items():
        request = args.output / f"{name}.request.json"
        output = args.output / f"{name}.result.json"
        request.write_text(json.dumps(req, indent=2) + "\n")
        with (args.output / f"{name}.log").open("w") as log:
            proc = subprocess.run([sys.executable, "-m", "sim.curobo_expert_v1.curobo_backend",
                                   "--request", str(request), "--output", str(output)],
                                  stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(output.read_text()) if output.exists() else {}
        summary["cases"][name] = {"returncode": proc.returncode, "success": result.get("success"),
                                 "failure": result.get("failure"), "metadata": result.get("metadata")}
    success = summary["cases"]["free_motion"]["success"] is True
    rejection = summary["cases"]["blocked_start"]["success"] is False
    blocked = summary["cases"]["blocked_start"].get("metadata") or {}
    rejection = rejection and "WORLD_COLLISION" in blocked.get("planner_status", "")
    summary["accepted"] = success and rejection
    (args.output / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"accepted": summary["accepted"], "output": str(args.output)}))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Versioned local-waypoint conditioning derived from immutable expert traces.

Only checked planner joint waypoints, their validity, and requested speed are
added. No demonstration time/index, future action, contact truth, or reward is
an actor input. Archived 240 Hz commands must exactly verify the assigned
ordered quintic segments; annotations never confer new physical acceptance.
"""
import argparse
from itertools import groupby
import json
from pathlib import Path
import tempfile

import numpy as np

from . import dataset as v1
from .controller import CONTROL_DT, CONTROL_STEPS, DT, timing_variation, verify_snapshot

ACTION_LIMIT_RAD, HISTORY_STEPS = v1.ACTION_LIMIT_RAD, v1.HISTORY_STEPS
sha256 = v1.sha256
JOINT_SUBPHASES = ("initial_home", "transfer", "return_bridge", "return_home")
FRAME_FEATURES = v1.FRAME_FEATURES + tuple("local_waypoint_error_"+str(i) for i in range(7)) + (
    "local_waypoint_valid", "requested_speed_scale")


def training_statistics(values):
    """Fit training rows with float64 accumulation and unit constant scales.

    Repeated float32 addition can invent variation in a constant speed column.
    Detect constants from the input values themselves, independent of moments.
    """
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    constant = np.all(values == values[0], axis=0)
    mean = np.where(constant, values[0], mean)
    std = np.where(constant | (std < 1e-6), 1., std)
    return mean, std


def encode_frame(row):
    base = v1.encode_frame(row)
    valid = row.get("local_waypoint_valid")
    if type(valid) is not bool or valid != (row["subphase"] in JOINT_SUBPHASES):
        raise ValueError("local waypoint validity disagrees with supervised stage")
    scale = timing_variation(row.get("speed_scale"))["speed_scale"]
    if valid:
        goal = v1._array(row.get("local_waypoint_q"), (7,), "local waypoint")
        error = goal - v1._array(row["observation"]["q"], (7,), "observed q")
    else:
        if row.get("local_waypoint_q") is not None:
            raise ValueError("absent local waypoint must be null")
        error = np.zeros(7)
    return np.r_[base, error, float(valid), scale].astype(np.float32)


class ObservationHistory(v1.ObservationHistory):
    def push(self, row):
        frame = encode_frame(row)
        if not isinstance(row.get("target"), str) or not row["target"]:
            raise ValueError("missing target name for history boundary")
        if row["target"] != self.target:
            self.frames = [frame.copy() for _ in range(HISTORY_STEPS)]
            self.target = row["target"]
        else:
            self.frames = (self.frames + [frame])[-HISTORY_STEPS:]
        return np.stack(self.frames)


def assign_waypoints(rows, plans, trace):
    """Return new annotated rows; verify every assigned 240 Hz command.

    Segment length is reconstructed only to verify source labels. Neither it
    nor its index/progress is exposed to the actor or used by live supervision.
    """
    annotated = [dict(row, local_waypoint_valid=False, local_waypoint_q=None) for row in rows]
    segments = []
    for (target, phase), group in groupby(enumerate(rows), lambda x: (x[1]["target"], x[1]["subphase"])):
        entries = list(group)
        if phase not in JOINT_SUBPHASES:
            continue
        plan = plans.get(target, {})
        path = np.asarray(plan.get("transfer_q"), dtype=float)
        if (plan.get("name") != target or plan.get("success") is not True or path.ndim != 2
                or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all()
                or not np.allclose(path[0], 0., atol=1e-9, rtol=0)):
            raise ValueError("invalid archived checked waypoint path")
        goals = {"initial_home": np.zeros((1,7)), "transfer": path[1:],
                 "return_bridge": path[-1:], "return_home": path[-2::-1]}[phase]
        offset = 0
        for goal in goals:
            if offset >= len(entries):
                raise ValueError("missing archived waypoint segment")
            index, first = entries[offset]
            scale = timing_variation(first.get("speed_scale"))["speed_scale"]
            start = v1._array(first.get("previous_q_command"), (7,), "segment start")
            seconds = max(1 / scale, 1.875*np.max(np.abs(goal-start))/(.25*scale))
            blocks = int(np.ceil(seconds / CONTROL_DT))
            part = entries[offset:offset+blocks]
            if len(part) != blocks:
                raise ValueError("incomplete archived waypoint segment")
            n = blocks * CONTROL_STEPS
            u = np.arange(1, n+1) / n
            commands = start + (10*u**3-15*u**4+6*u**5)[:, None]*(goal-start)
            for local, (row_index, row) in enumerate(part):
                a, b = local*CONTROL_STEPS, (local+1)*CONTROL_STEPS
                previous = start if a == 0 else commands[a-1]
                physics_start = row.get("physics_row", -1)
                physics_end = row.get("physics_row_end_exclusive", -1)
                if (row.get("speed_scale") != scale or physics_end-physics_start != CONTROL_STEPS
                        or physics_start != first["physics_row"]+a
                        or not np.isclose(row.get("action_duration_s", 0), CONTROL_DT, atol=1e-10, rtol=0)
                        or not np.allclose(row["previous_q_command"], previous, atol=1e-9, rtol=0)
                        or not np.allclose(row["q_command"], commands[b-1], atol=1e-9, rtol=0)):
                    raise ValueError("archived quintic waypoint endpoint verification failed")
                observed = np.asarray([r["q_command"] for r in trace[physics_start:physics_end]], float)
                if observed.shape != (CONTROL_STEPS,7) or not np.allclose(observed, commands[a:b], atol=1e-9, rtol=0):
                    raise ValueError("archived quintic physics interior verification failed")
                annotated[row_index].update(local_waypoint_valid=True, local_waypoint_q=goal.tolist())
            segments.append({"target": target, "subphase": phase, "first_control_row": index,
                "control_row_end_exclusive": index+blocks, "destination_q": goal.tolist(),
                "speed_scale": scale, "verified_physics_steps": n})
            offset += blocks
        if offset != len(entries):
            raise ValueError("unassigned archived waypoint commands")
    for row in annotated:
        encode_frame(row)
    return annotated, segments


def annotate_episode(directory, expected_hashes):
    directory = Path(directory)
    data = {name: (directory/name).read_bytes() for name in v1.ARTIFACT_NAMES}
    if any(sha256(data[name]) != expected_hashes[name] for name in data):
        raise ValueError("expert source changed during annotation")
    outcome = json.loads(data["outcome.json"])
    snapshot = directory / "input_snapshot"
    snapshot_bytes = (snapshot/"manifest.json").read_bytes()
    if sha256(snapshot_bytes) != outcome.get("input_snapshot_manifest_sha256"):
        raise ValueError("expert snapshot manifest fingerprint mismatch")
    manifest = verify_snapshot(snapshot)
    plans, plan_hashes = {}, {}
    rows = json.loads(data["control_commands.json"])
    for target in dict.fromkeys(row["target"] for row in rows):
        digest = manifest["files"][target+"/plan"]["sha256"]
        document = json.loads((snapshot/"objects"/digest).read_bytes())
        matches = [p for p in document["points"] if p.get("name") == target]
        if len(matches) != 1:
            raise ValueError("ambiguous archived target plan")
        plans[target], plan_hashes[target] = matches[0], digest
    trace = [json.loads(line) for line in data["expert_trace.jsonl"].splitlines()]
    annotated, segments = assign_waypoints(rows, plans, trace)
    verify_snapshot(snapshot)
    if (snapshot/"manifest.json").read_bytes() != snapshot_bytes:
        raise ValueError("expert snapshot changed during annotation")
    sidecar = {"schema": "auscultation_waypoint_annotations_v2",
        "scope": "derived planner-input annotations only; original independently accepted actions unchanged",
        "source_artifact_sha256": expected_hashes,
        "source_snapshot_manifest_sha256": sha256(snapshot_bytes), "plan_sha256": plan_hashes,
        "verification": "ordered quintic endpoint and every 240 Hz command; tolerance 1e-9 rad",
        "segments": segments,
        "rows": [{"control_row": i, "target": r["target"], "local_waypoint_valid": r["local_waypoint_valid"],
                  "local_waypoint_q": r["local_waypoint_q"], "speed_scale": r["speed_scale"]}
                 for i,r in enumerate(annotated)]}
    return annotated, sidecar


def build_dataset(directories, output, split_seed=0, evaluation_fraction=.25):
    """Reuse v1 acceptance, trace binding, whole-episode splitting, not its files."""
    with tempfile.TemporaryDirectory(prefix="ausc_v2_validate_") as temporary:
        source = Path(temporary)/"validated_v1"
        base = v1.build_dataset(directories, source, split_seed, evaluation_fraction)
        with np.load(source/"dataset.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        histories, sidecars = [], []
        for episode in base["episodes"]:
            annotated, sidecar = annotate_episode(episode["source"], episode["artifact_sha256"])
            history = ObservationHistory()
            histories.append(np.stack([history.push(row) for row in annotated]))
            sidecars.append(sidecar)
            episode["speed_scales"] = sorted(set(row["speed_scale"] for row in annotated))
        arrays["observations"] = np.concatenate(histories)
    training = arrays["observations"][arrays["split"] == 0].reshape(int((arrays["split"] == 0).sum()), -1)
    mean, std = training_statistics(training)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"dataset.npz", **arrays)
    for i, sidecar in enumerate(sidecars):
        name = "waypoint_annotations_"+str(i)+".json"
        data = (json.dumps(sidecar, separators=(",", ":"))+"\n").encode()
        (output/name).write_bytes(data)
        base["episodes"][i].update(waypoint_annotation_file=name, waypoint_annotation_sha256=sha256(data))
    base.update(schema="auscultation_bc_dataset_v2", frame_features=list(FRAME_FEATURES),
        waypoint_conditioning="checked local joint waypoint minus observed joint state, validity, and requested speed; no time/index input",
        generalization_limit="Only accepted frozen demonstrations; initial closed-loop evaluation must use a training speed. Unseen speed action error is not speed generalization success.",
        scaler={"fit_split":"train_only", "mean":mean.tolist(), "std":std.tolist()},
        arrays_sha256=sha256((output/"dataset.npz").read_bytes()))
    (output/"manifest.json").write_text(json.dumps(base, indent=2)+"\n")
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--evaluation-fraction", type=float, default=.25)
    args = parser.parse_args(argv)
    manifest = build_dataset(args.episodes, args.output, args.split_seed, args.evaluation_fraction)
    print(json.dumps({"samples":manifest["sample_count"], "frame_features":len(FRAME_FEATURES), "output":str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

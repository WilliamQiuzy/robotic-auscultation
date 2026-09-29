"""Fail-closed import of independently accepted simulation expert episodes.

Only proprioception, RGB-D-derived goals, Hall estimates, and observable stages
enter the actor. Physics contact truth is reserved for the independent audit.
The dataset is a mechanics experiment; it contains no acoustic labels/rewards.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

HISTORY_STEPS = 4
ACTION_LIMIT_RAD = .0125
CONTROL_DT_S = .05
PHYSICS_STEPS = 12
PHASES = ("APPROACH", "CONTACT", "HOLD", "RETRACT", "DONE", "ABORT")
SUBPHASES = ("initial_home", "settle", "transfer", "align", "contact_cycle",
             "withdraw_to_prepose", "withdraw_settle", "return_bridge", "return_home", "home_settle")
ARTIFACT_NAMES = ("expert_trace.jsonl", "control_commands.json", "outcome.json")
FRAME_FEATURES = tuple(
    ["q_"+str(i) for i in range(7)] + ["dq_"+str(i) for i in range(7)]
    + ["tcp_base_"+axis for axis in "xyz"] + ["goal_relative_base_"+axis for axis in "xyz"]
    + ["normal_base_"+axis for axis in "xyz"] + ["hall_N", "compression_m"]
    + ["previous_q_command_"+str(i) for i in range(7)]
    + [name+"_r"+str(row)+"c"+str(col) for name in ("tcp", "target")
       for col in range(2) for row in range(3)]
    + ["phase_"+p for p in PHASES] + ["subphase_"+p for p in SUBPHASES])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid " + name) from error
    if result.shape != shape:
        raise ValueError("invalid shape for " + name)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite " + name)
    return result


def encode_frame(row):
    """Strict feature allowlist, shared by offline data and future online actor.

    Rotations use their first two columns (a continuous six-dimensional pose
    representation). Unknown extra keys, including audit forces, are ignored.
    Missing required observations are errors, never zero-filled measurements.
    """
    observation = row.get("observation", {})
    required = ("q", "dq", "tcp_base_m", "goal_relative_base_m", "normal_base",
                "hall_N", "compression_m", "tcp_rotation_base", "target_rotation_base")
    for key in required:
        if key not in observation:
            raise ValueError("missing observation " + key)
    if row.get("phase") not in PHASES or row.get("subphase") not in SUBPHASES:
        raise ValueError("unknown observable stage")
    parts = [_array(observation[key], (size,), key) for key, size in
             (("q", 7), ("dq", 7), ("tcp_base_m", 3), ("goal_relative_base_m", 3), ("normal_base", 3))]
    if not np.isclose(np.linalg.norm(parts[-1]), 1., atol=1e-5):
        raise ValueError("normal_base is not a unit vector")
    parts += [_array([observation["hall_N"], observation["compression_m"]], (2,), "Hall/compression"),
              _array(row.get("previous_q_command"), (7,), "previous_q_command")]
    for key in ("tcp_rotation_base", "target_rotation_base"):
        rotation = _array(observation[key], (3, 3), key)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
            raise ValueError("invalid rotation " + key)
        parts.append(rotation[:, :2].T.reshape(-1))
    parts += [np.array([row["phase"] == x for x in PHASES], float),
              np.array([row["subphase"] == x for x in SUBPHASES], float)]
    return np.concatenate(parts).astype(np.float32)


class ObservationHistory:
    """Four observed frames; repeat the first frame at episode/target start."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = []
        self.target = None

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


@dataclass
class Episode:
    episode_id: str
    group_id: str
    scenario_id: str
    seed: int
    source: str
    artifact_sha256: dict
    observations: np.ndarray
    actions: np.ndarray
    source_interpolations: list


def validate_control_trace(controls, trace):
    """Bind every action block to the exact physics samples it labels.

    This pure validation helper is shared with the independent episode audit.
    It raises ValueError on a mismatch; it does not assert contact acceptance.
    The first command is compared with the recorded initial observation because
    no pre-step physics sample exists. Later observations are compared with the
    preceding physics row in matching frames only, never guessed transforms.
    """
    if not isinstance(controls, list) or not controls or not all(isinstance(row, dict) for row in controls):
        raise ValueError("empty or malformed control episode")
    if not isinstance(trace, list) or not trace or not all(isinstance(row, dict) for row in trace):
        raise ValueError("empty or malformed physics trace")
    dt = CONTROL_DT_S / PHYSICS_STEPS
    times = np.asarray([float(_array(row.get("time_s"), (), "physics timestamp")) for row in trace])
    if not np.allclose(np.diff(times), dt, atol=1e-7, rtol=0):
        raise ValueError("physics timestamp spacing mismatch")
    cursor = 0
    for block in controls:
        start, end = block.get("physics_row"), block.get("physics_row_end_exclusive")
        if (type(start) is not int or type(end) is not int or start != cursor
                or end - start != PHYSICS_STEPS or end > len(trace)):
            raise ValueError("control coverage has a hole, overlap, or invalid bounds")
        cursor = end
    if cursor != len(trace):
        raise ValueError("control blocks do not cover the entire physics trace")
    # Observation key, possible same-frame trace keys, required shape.
    fields = (("q", ("q",), (7,)), ("dq", ("dq",), (7,)),
              ("hall_N", ("hall_N",), ()), ("compression_m", ("compression_m",), ()),
              ("tcp_base_m", ("tcp_base_m", "tcp_base"), (3,)),
              ("tcp_rotation_base", ("tcp_rotation_base", "rotation_base"), (3, 3)),
              ("tcp_world", ("tcp_world",), (3,)),
              ("rotation_world", ("rotation_world",), (3, 3)),
              ("axis_world", ("axis_world",), (3,)))
    comparisons = {key: 0 for key, _, _ in fields}
    for block in controls:
        start, end = block["physics_row"], block["physics_row_end_exclusive"]
        if block.get("command_block_complete") is not True or block.get("executed_physics_steps") != PHYSICS_STEPS:
            raise ValueError("incomplete command block")
        if not np.isclose(block.get("action_duration_s", 0), CONTROL_DT_S, atol=1e-10, rtol=0):
            raise ValueError("unsupported action duration")
        target = block.get("target")
        if not isinstance(target, str) or not target or any(row.get("target") != target for row in trace[start:end]):
            raise ValueError("control target differs from physics block target")
        if (block.get('phase') != trace[start].get('phase')
                or any(row.get('subphase') != block.get('subphase') for row in trace[start:end])):
            raise ValueError('control stage differs from physics trace')
        command = _array(block.get("q_command"), (7,), "q_command")
        endpoint = _array(trace[end-1].get("q_command"), (7,), "physics q_command")
        if not np.allclose(command, endpoint, atol=1e-8, rtol=0):
            raise ValueError("control command endpoint differs from physics trace")
        prior = _array(block.get("previous_q_command"), (7,), "previous_q_command")
        observation = block.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("missing control observation")
        if start:
            expected_prior = _array(trace[start-1].get("q_command"), (7,), "preceding physics command")
            if not np.allclose(prior, expected_prior, atol=1e-8, rtol=0):
                raise ValueError("prior command continuity differs from physics trace")
            expected_time = times[start-1]
            preceding = trace[start-1]
            for key, aliases, shape in fields:
                raw_key = next((alias for alias in aliases if alias in preceding), None)
                # Core recorded state must be bound when present in the trace.
                # Optional world-pose observations are compared only if supplied.
                optional_world_field = key in ("tcp_world", "rotation_world", "axis_world")
                if raw_key is not None and (key in observation or not optional_world_field):
                    observed = _array(observation.get(key), shape, "observation " + key)
                    actual = _array(preceding[raw_key], shape, "preceding physics " + key)
                    if not np.allclose(observed, actual, atol=1e-7, rtol=0):
                        raise ValueError("control observation " + key + " differs from preceding physics state")
                    comparisons[key] += 1
        else:
            initial = _array(observation.get("q"), (7,), "initial observed q")
            if not np.allclose(prior, initial, atol=1e-8, rtol=0):
                raise ValueError("initial prior command differs from initial observed q")
            expected_time = times[0] - dt
        timestamp = float(_array(block.get("time_s"), (), "control timestamp"))
        if not np.isclose(timestamp, expected_time, atol=1e-7, rtol=0):
            raise ValueError("control timestamp differs from preceding physics observation")
        action = _array(block.get("action_joint_delta_rad"), (7,), "action_joint_delta_rad")
        if np.max(np.abs(action)) > ACTION_LIMIT_RAD + 1e-9:
            raise ValueError("joint action exceeds bound")
        if not np.allclose(command-prior, action, atol=1e-8, rtol=0):
            raise ValueError("action does not equal traced command increment")
    actor_fields = [key for key, _, _ in fields if key not in ("tcp_world", "rotation_world", "axis_world")]
    return {"control_blocks": len(controls), "physics_rows_covered": len(trace),
            "state_field_comparisons": comparisons,
            "unverified_state_fields": [key for key in actor_fields if comparisons[key] < len(controls)-1 or len(controls) == 1],
            "initial_state_binding": "prior command equals recorded initial q; no independent pre-step physics sample",
            "pose_binding": "matching coordinate frames only; no world/base transform inferred"}


def load_episode(directory):
    """Read exact bytes once and require a hash-bound independent acceptance."""
    directory = Path(directory)
    try:
        artifacts = {name: (directory / name).read_bytes() for name in ARTIFACT_NAMES}
        audit_bytes = (directory / "independent_audit.json").read_bytes()
        audit = json.loads(audit_bytes)
    except (OSError, ValueError) as error:
        raise ValueError("missing or invalid independent audit/artifacts: " + str(directory)) from error
    if audit.get("accepted") is not True:
        raise ValueError("independent audit did not accept episode")
    hashes = {name: sha256(data) for name, data in artifacts.items()}
    if any(audit.get("artifact_sha256", {}).get(name) != digest for name, digest in hashes.items()):
        raise ValueError("independent audit artifact fingerprint mismatch")
    hashes["independent_audit.json"] = sha256(audit_bytes)
    outcome = json.loads(artifacts["outcome.json"])
    if (outcome.get('dataset_role','expert_demonstration') != 'expert_demonstration'
            or outcome.get('expert_demonstration',True) is not True
            or outcome.get('controller_type','analytic_expert') == 'learned_bc'):
        raise ValueError('student evaluation is not an expert demonstration')
    if outcome.get("success") is not True or outcome.get("training_accepted") is not True:
        raise ValueError("controller did not accept entire episode")
    points = outcome.get("points", [])
    if not points or any(p.get("success") is not True or p.get("training_accepted") is not True for p in points):
        raise ValueError("controller did not accept every point")
    scenario, seed = outcome.get("scenario_id"), outcome.get("seed")
    if not isinstance(scenario, str) or not scenario.strip() or type(seed) is not int:
        raise ValueError("scenario_id and integer seed required for grouped split")
    if not np.isclose(outcome.get("control_dt_s", 0), CONTROL_DT_S, atol=1e-10):
        raise ValueError("unsupported controller duration")
    rows = json.loads(artifacts["control_commands.json"])
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty control episode")
    try:
        trace = [json.loads(line) for line in artifacts["expert_trace.jsonl"].splitlines()]
    except (ValueError, TypeError) as error:
        raise ValueError("malformed hashed physics trace") from error
    validate_control_trace(rows, trace)
    history, observations, actions = ObservationHistory(), [], []
    interpolations = set()
    previous = None
    for index, row in enumerate(rows):
        if row.get("recovery") is not False or row.get("training_accepted") is not True:
            raise ValueError("recovery or rejected row in accepted episode")
        if row.get("command_block_complete") is not True or row.get("executed_physics_steps") != PHYSICS_STEPS:
            raise ValueError("incomplete command block")
        if not np.isclose(row.get("action_duration_s", 0), CONTROL_DT_S, atol=1e-10):
            raise ValueError("unsupported action duration")
        if row.get("physics_row_end_exclusive", -1) - row.get("physics_row", -1) != PHYSICS_STEPS:
            raise ValueError("invalid physics block bounds")
        timestamp = _array(row.get("time_s"), (), "time_s")
        prior = _array(row.get("previous_q_command"), (7,), "previous_q_command")
        action = _array(row.get("action_joint_delta_rad"), (7,), "action_joint_delta_rad")
        command = _array(row.get("q_command"), (7,), "q_command")
        if np.max(np.abs(action)) > ACTION_LIMIT_RAD + 1e-9:
            raise ValueError("joint action exceeds bound")
        if previous is not None:
            if (not np.allclose(prior, previous["q_command"], atol=1e-8, rtol=0)
                    or not np.isclose(float(timestamp)-previous["time_s"], CONTROL_DT_S, atol=1e-7, rtol=0)
                    or row["physics_row"] != previous["physics_row_end_exclusive"]):
                raise ValueError("control continuity violation")
        elif row["physics_row"] != 0:
            raise ValueError("episode does not start at first physics row")
        if not np.allclose(command - prior, action, atol=1e-8, rtol=0):
            raise ValueError("action does not equal command increment")
        interpolation = row.get("action_interpolation")
        if not isinstance(interpolation, str) or not interpolation:
            raise ValueError("missing action interpolation provenance")
        interpolations.add(interpolation)
        observations.append(history.push(row))
        actions.append(action)
        previous = row
    group_id = sha256(json.dumps([scenario, seed], separators=(",", ":")).encode())
    return Episode(hashes["expert_trace.jsonl"], group_id, scenario, seed, str(directory.resolve()),
                   hashes, np.stack(observations), np.asarray(actions, dtype=np.float32), sorted(interpolations))


def build_dataset(directories, output, split_seed=0, evaluation_fraction=.25):
    """Split whole scenario/seed groups before computing any training scaler."""
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must lie strictly between zero and one")
    episodes = [load_episode(path) for path in directories]
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode trace")
    actor_groups = {}
    for episode in episodes:
        actor_digest = sha256(episode.observations.tobytes() + episode.actions.tobytes())
        actor_groups.setdefault(actor_digest, set()).add(episode.group_id)
    if any(len(groups) > 1 for groups in actor_groups.values()):
        raise ValueError("identical actor trajectory appears under different scenario/seed groups")
    groups = sorted({e.group_id for e in episodes})
    if len(groups) < 2:
        raise ValueError("at least two scenario/seed groups required for held-out data")
    shuffled = np.random.default_rng(split_seed).permutation(groups)
    evaluation_count = min(len(groups)-1, max(1, int(np.ceil(len(groups)*evaluation_fraction))))
    held_out = set(shuffled[:evaluation_count])
    observations = np.concatenate([e.observations for e in episodes])
    actions = np.concatenate([e.actions for e in episodes])
    indices = np.concatenate([np.full(len(e.actions), i, np.int32) for i, e in enumerate(episodes)])
    split = np.concatenate([np.full(len(e.actions), int(e.group_id in held_out), np.uint8) for e in episodes])
    train = observations[split == 0].reshape(int((split == 0).sum()), -1)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-6, 1., std)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "dataset.npz", observations=observations, actions=actions,
                        episode_index=indices, split=split)
    manifest = {"schema": "auscultation_bc_dataset_v1", "scope": "simulation mechanics only; no audio or reward",
        "history_steps": HISTORY_STEPS, "history_padding": "repeat first frame; reset per episode and target change",
        "frame_features": list(FRAME_FEATURES), "feature_order": "oldest to newest frame, then frame_features",
        "action": "7 joint command increments relative to previous_q_command", "action_limit_rad": ACTION_LIMIT_RAD,
        "control_dt_s": CONTROL_DT_S, "physics_steps_per_action": PHYSICS_STEPS,
        "action_interpolation": {"expert": sorted({s for e in episodes for s in e.source_interpolations}),
            "student": "linear over 12 physics steps",
            "limitation": "BC labels match block endpoints; expert quintic transfer interior differs from student linear interpolation. Closed-loop validation is required."},
        "split_seed": split_seed, "evaluation_fraction": evaluation_fraction,
        "split_unit": "whole episodes grouped by scenario_id and seed",
        "generalization_limit": "A different provenance seed alone does not establish a varied scene or trajectory.",
        "scaler": {"fit_split": "train_only", "mean": mean.tolist(), "std": std.tolist()},
        "episodes": [{"episode_id": e.episode_id, "group_id": e.group_id, "scenario_id": e.scenario_id,
            "seed": e.seed, "source": e.source, "artifact_sha256": e.artifact_sha256,
            "samples": len(e.actions), "split": "evaluation" if e.group_id in held_out else "train"} for e in episodes],
        "sample_count": len(actions), "arrays_sha256": sha256((output / "dataset.npz").read_bytes())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--evaluation-fraction", type=float, default=.25)
    args = parser.parse_args(argv)
    manifest = build_dataset(args.episodes, args.output, args.split_seed, args.evaluation_fraction)
    print(json.dumps({"samples": manifest["sample_count"], "episodes": len(manifest["episodes"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

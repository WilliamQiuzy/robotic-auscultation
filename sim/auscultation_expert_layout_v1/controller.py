"""Import-safe expert orchestration for simulation contact mechanics only.

The actor uses joint/TCP/Hall observations. PhysX force labels are independent
acceptance evidence and are never actor inputs. A passed run establishes the
configured mechanical experiment, not acoustic coupling or clinical validity.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

from .phase_manager import ContactObservation, Phase, PhaseConfig, PhaseManager
from .targets import _base_transform, load_targets

DT = 1 / 240
CONTROL_STEPS = 12
CONTROL_DT = DT * CONTROL_STEPS
FRONT_GROUPS = ("diaphragm", "lip", "contact_rim")
FORBIDDEN_GROUPS = ("frame", "forbidden", "carriage_other")
INPUT_LABELS = ("audit", "targets", "runtime", "collision_metadata", "collision_arrays", "kinematics_urdf")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timing_variation(speed_scale):
    """Describe a real timing change; a random-seed label alone is not one."""
    if (isinstance(speed_scale, bool) or not isinstance(speed_scale, Real)
            or not np.isfinite(speed_scale) or not .7 <= speed_scale <= 1.):
        raise ValueError("speed_scale must be a finite number in [0.7, 1.0]")
    scale = float(speed_scale)
    return {"kind": "deterministic_motion_timing", "speed_scale": scale,
            "joint_command_speed_limit_rad_s": .25 * scale,
            "minimum_transfer_segment_duration_s": 1. / scale,
            "free_approach_far_speed_m_s": .012 * scale,
            "free_approach_near_speed_m_s": .002 * scale,
            "retraction_speed_m_s": .01 * scale,
            "fault_retraction_speed_m_s": .005 * scale,
            "travel_timeout_multiplier": 1. / scale,
            "force_feedback_gain_scale": 1.,
            "unchanged": ["target coordinates", "checked geometric paths", "force setpoint and band",
                          "force and stroke limits", "continuous contact hold duration",
                          "contact settle duration", "release confirmation duration", "physics/control rates"],
            "scope": "timing variation only; no geometric, visual, or contact-material randomization"}


@dataclass(frozen=True)
class PlanBundle:
    plan: dict
    row: dict
    target: object
    inputs: dict
    plan_path: Path


def validate_contact_ready_route(row, base, kinematics):
    """CONTACT begins at the prepose, so its complete route needs the gate."""
    from .posture import PostureLimits
    limits = PostureLimits.from_dict(row["posture_limits"], kinematics)
    approach = np.asarray(row.get("approach_q"), float)
    loaded = np.asarray(row.get("loaded_q"), float)
    if (approach.ndim != 2 or approach.shape[1] != 7 or len(approach) < 2
            or loaded.shape != (7,) or not np.isfinite(approach).all()
            or not np.isfinite(loaded).all()):
        raise ValueError("invalid contact-ready route")
    route = np.vstack([approach, loaded])
    for edge, (start, end) in enumerate(zip(route[:-1], route[1:])):
        count = max(1, int(np.ceil(np.max(np.abs(end-start)) / .015)))
        for fraction in np.linspace(0., 1., count+1):
            result = limits.evaluate(kinematics, base, start+fraction*(end-start), contact=True)
            if not result["clear"]:
                raise ValueError("prepose/contact-route posture rejected at edge " + str(edge)
                                 + ": " + ", ".join(result["posture_violations"]))


def validate_plan(plan_path, input_dir, runtime_dir):
    """Read exact inputs once, check every provenance hash and pose identity."""
    plan_path, input_dir, runtime_dir = map(Path, (plan_path, input_dir, runtime_dir))
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    inputs = {"plan": (plan_path, plan_bytes)}
    provenance = plan.get("provenance", {})
    for label in INPUT_LABELS:
        name = provenance.get(label + "_file")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("missing or unsafe provenance filename: " + label)
        path = (runtime_dir if label in ("runtime", "kinematics_urdf") else input_dir) / name
        data = path.read_bytes()
        if digest(data) != provenance.get(label + "_sha256"):
            raise ValueError("plan fingerprint mismatch: " + label)
        inputs[label] = (path, data)
    if plan.get("targets_file") != provenance["targets_file"] or plan.get("audit_file") != provenance["audit_file"]:
        raise ValueError("plan source metadata disagrees with provenance")
    if len(plan.get("points", [])) != 1:
        raise ValueError("expected exactly one point in each plan")
    row = plan["points"][0]
    if row.get("success") is not True:
        raise ValueError("plan has no successful sampled geometric route")
    pre_gap = row.get('free_approach_start_gap_m', .06)
    if not isinstance(pre_gap, (int,float)) or not np.isfinite(pre_gap) or not .025 <= pre_gap <= .06:
        raise ValueError('unsupported checked prepose gap')
    base = _base_transform(plan["world_from_left_base"])
    runtime_base = _base_transform(json.loads(inputs["runtime"][1])["world_from_left_base"])
    if not np.allclose(base, runtime_base, atol=1e-9, rtol=0):
        raise ValueError("plan base differs from recorded runtime base")
    targets_path = inputs["targets"][0]
    target = next((t for t in load_targets(base, targets_path) if t.name == row.get("name")), None)
    if targets_path.read_bytes() != inputs["targets"][1]:
        raise ValueError("targets changed during validation")
    if target is None or not np.allclose(row.get("position_world"), target.position_world, atol=1e-9, rtol=0):
        raise ValueError("plan target coordinates differ from detector source")
    if not np.allclose(row.get("normal_world"), target.normal_world, atol=1e-9, rtol=0):
        raise ValueError("plan target normal differs from source")
    audit = json.loads(inputs["audit"][1])
    if audit.get("name") != target.name or audit.get("compression_m") != row.get("loaded_compression_m"):
        raise ValueError("audit source metadata differs from plan")
    index = row.get("selected_audit_candidate_index")
    if not isinstance(index, int) or not 0 <= index < len(audit.get("candidates", [])):
        raise ValueError("selected audit candidate is absent")
    candidate = audit["candidates"][index]
    if row.get("selected_fit") != candidate:
        raise ValueError("selected fit differs from hashed audit")
    normal = np.asarray(row["command_normal_world"], float)
    rotation = np.asarray(row["command_rotation_world"], float)
    if (normal.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(normal).all() or not np.isfinite(rotation).all()
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)
            or not np.allclose(rotation[:, 2], -normal, atol=1e-6)
            or not np.allclose(normal, candidate["normal_world"], atol=1e-9)
            or not np.allclose(rotation, candidate["wrist_rotation_world"], atol=1e-9)):
        raise ValueError("invalid command pose or mismatched audit pose")
    transfer = np.asarray(row.get("transfer_q"), float)
    approach = np.asarray(row.get("approach_q"), float)
    if (transfer.ndim != 2 or transfer.shape[1] != 7 or len(transfer) < 2
            or not np.isfinite(transfer).all() or not np.allclose(transfer[0], 0., atol=1e-9)
            or approach.ndim != 2 or approach.shape[1] != 7 or len(approach) < 2
            or not np.isfinite(approach).all() or not np.allclose(transfer[-1], approach[0], atol=1e-9)):
        raise ValueError("invalid home/transfer/approach route")
    if row.get("geometric_clearance_m", 0) < .003 or row.get("joint_edge_sample_step_rad", 1) > .015:
        raise ValueError("plan sampled clearance criteria are weaker than required")
    if "posture_limits" in row:
        # Use the captured, hash-verified URDF, before importing Isaac or moving
        # any joint. A valid loaded endpoint alone cannot qualify the prepose.
        import tempfile
        from .kinematics import OpenArmKinematics
        with tempfile.TemporaryDirectory(prefix="ausc_preflight_urdf_") as temporary:
            pinned = Path(temporary) / "kinematics.urdf"
            pinned.write_bytes(inputs["kinematics_urdf"][1])
            validate_contact_ready_route(row, base, OpenArmKinematics(pinned))
    return PlanBundle(plan, row, target, inputs, plan_path)


def create_snapshot(bundles, destination, source_paths):
    """Create an exclusive content-addressed, read-only input snapshot.

    Current source hashes establish run reproducibility. They do not prove an
    old exported collider file was generated from those sources; that requires
    the separate collision-regeneration verification manifest.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    objects = destination / "objects"
    objects.mkdir()
    manifest = {"schema": 1, "files": {}, "points": [], "source_binding":
                "Current source snapshot; require independent collider regeneration verification."}
    def add(key, path, data):
        sha = digest(data)
        object_path = objects / sha
        if not object_path.exists():
            object_path.write_bytes(data)
            object_path.chmod(0o444)
        manifest["files"][key] = {"source_path": str(path.resolve()), "sha256": sha, "bytes": len(data)}
    for bundle in bundles:
        name = bundle.target.name
        manifest["points"].append(name)
        for label, (path, data) in bundle.inputs.items():
            add(name + "/" + label, path, data)
    for key, path in source_paths.items():
        path = Path(path)
        add("source/" + key, path, path.read_bytes())
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o444)
    return destination


def verify_snapshot(snapshot, verify_sources=False):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for key, item in manifest["files"].items():
        if digest((snapshot / "objects" / item["sha256"]).read_bytes()) != item["sha256"]:
            raise ValueError("snapshot fingerprint mismatch: " + key)
        if verify_sources and digest(Path(item["source_path"]).read_bytes()) != item["sha256"]:
            raise ValueError("source changed since snapshot: " + key)
    return manifest


class EpisodeRejected(RuntimeError):
    pass


class ExpertController:
    # Current-geometry subclasses explicitly select their true front interface.
    # Historical controllers retain their original group definitions.
    front_contact_groups = FRONT_GROUPS
    total_contact_groups = FRONT_GROUPS + FORBIDDEN_GROUPS
    """One environment, continuously integrated motion, bounded actor commands."""
    def __init__(self, env, kinematics, checker, config=None, speed_scale=1.):
        self.variation = timing_variation(speed_scale)
        self.speed_scale = self.variation["speed_scale"]
        self.env, self.kin, self.checker = env, kinematics, checker
        base_config = config or PhaseConfig(contact_normal_tolerance_m=.006,
            approach_timeout_s=120, episode_timeout_s=180, max_compression_m=.006,
            hold_tangential_tolerance_m=.003)
        # Longer travel must not consume the contact/withdrawal watchdog merely
        # because the planned approach is slower. Dwell and hold watchdog stay fixed.
        self.config = replace(base_config, **{name: getattr(base_config, name) / self.speed_scale
            for name in ("approach_timeout_s", "contact_timeout_s", "retract_timeout_s", "episode_timeout_s")})
        self.control_rows = []
        self.q = np.asarray(env.state()["q"], float).copy()
        self.target = None
        self.manager = None
        self.joint_speed_limit = self.variation["joint_command_speed_limit_rad_s"]

    def metrics(self, state):
        error = state["tcp_world"] - self.target.position_world
        gap = float(error @ self.normal)
        return {"gap_m": gap, "tangent_m": float(np.linalg.norm(error - gap * self.normal)),
                "angle_deg": float(np.rad2deg(np.arccos(np.clip(state["axis_world"] @ -self.normal, -1, 1))))}

    def _violations(self, state):
        contacts = self.env.last_contacts
        violations = [key + " contact" for key in FORBIDDEN_GROUPS if contacts.get(key, 0.) > .5]
        total = sum(contacts.get(key, 0.) for key in self.total_contact_groups)
        if not np.isfinite(total) or total > 6.:
            violations.append("physical force limit")
        if not np.isfinite(state["compression_m"]) or state["compression_m"] > .006:
            violations.append("physical stroke limit")
        if not np.isfinite(state["hall_N"]) or state["hall_N"] >= self.config.hard_force_limit_N:
            violations.append("observable force limit")
        return violations, total

    def _tick(self, command, subphase, active=False, recovery=False):
        phase = "RECOVERY" if recovery else self.manager.state.phase.value
        state = self.env.step(command, phase, self.target.name, {"subphase": subphase})
        self.q = np.asarray(command).copy()
        metrics = self.metrics(state)
        violations, total = self._violations(state)
        face = sum(self.env.last_contacts.get(key, 0.) for key in self.front_contact_groups)
        alpha = DT / (.02 + DT)
        self.face_filtered += alpha * (face - self.face_filtered)
        self.total_filtered += alpha * (total - self.total_filtered)
        if active:
            obs = ContactObservation(state["hall_N"], max(0., state["compression_m"]),
                metrics["gap_m"], metrics["tangent_m"], metrics["angle_deg"],
                self.env.rows[-1]["tcp_speed_m_s"], tcp_position_m=tuple(state["tcp_world"]))
            previous = self.manager.state
            current = self.manager.update(obs, DT)
            front_contact = self.face_filtered > .15 and self.face_filtered > .1 * max(self.total_filtered, .01)
            if previous.phase == Phase.HOLD and current.hold_elapsed > previous.hold_elapsed and front_contact:
                self.audit_hold += DT
            elif current.phase == Phase.HOLD:
                self.audit_hold = 0.
            self.best_audit_hold = max(self.best_audit_hold, self.audit_hold)
        self.env.rows[-1].update(metrics, hold_elapsed_s=self.manager.state.hold_elapsed,
            audit_hold_s=self.audit_hold, front_contact_filtered_N=self.face_filtered,
            total_filtered_N=self.total_filtered, physical_violations=violations,
            recovery=recovery, training_accepted=False)
        if violations and not recovery:
            raise EpisodeRejected("; ".join(violations))
        if recovery:
            baseline, baseline_total, baseline_stroke = self.recovery_baseline
            if (any(self.env.last_contacts.get(key, 0.) > max(.5, baseline[key] + .5) for key in FORBIDDEN_GROUPS)
                    or total > max(6., baseline_total + 1.)
                    or state["compression_m"] > max(.006, baseline_stroke + .00025)):
                raise EpisodeRejected("recovery contact worsened; simulation stopped")
        return state

    def _block(self, commands, subphase, active=False, recovery=False, gap=None):
        commands = np.asarray(commands, float)
        if commands.ndim != 2 or commands.shape[1] != 7 or not np.isfinite(commands).all():
            raise EpisodeRejected("invalid actor command")
        if np.any(commands < self.kin.lower_limits) or np.any(commands > self.kin.upper_limits):
            raise EpisodeRejected("actor command outside joint limits")
        deltas = np.diff(np.vstack([self.q, commands]), axis=0)
        if np.max(np.abs(deltas)) > self.joint_speed_limit * DT + 1e-9:
            raise EpisodeRejected("actor joint command speed limit")
        state = self.env.state()
        start = self.q.copy()
        row = {"physics_row": len(self.env.rows), "time_s": state["time_s"],
            "speed_scale": self.speed_scale,
            "target": self.target.name, "phase": self.manager.state.phase.value,
            "subphase": subphase, "recovery": recovery, "training_accepted": False,
            "observation": {"q": state["q"].tolist(), "dq": state["dq"].tolist(),
                "tcp_base_m": state["tcp_base"].tolist(),
                "goal_relative_base_m": (self.target.position_base - state["tcp_base"]).tolist(),
                "normal_base": self.normal_base.tolist(), "hall_N": float(state["hall_N"]),
                "tcp_rotation_base": (self.env.base[:3,:3].T @ state["rotation_world"]).tolist(),
                "target_rotation_base": self.rotation_base.tolist(),
                "compression_m": float(state["compression_m"])},
            "previous_q_command": start.tolist(), "q_command": commands[-1].tolist(),
            "action_joint_delta_rad": (commands[-1] - start).tolist(),
            "action_duration_s": len(commands) * DT, "command_gap_m": gap,
            "action_interpolation": "physics-rate command rows are authoritative"}
        self.control_rows.append(row)
        try:
            for command in commands:
                state = self._tick(command, subphase, active=active, recovery=recovery)
        finally:
            row["physics_row_end_exclusive"] = len(self.env.rows)
            row["executed_physics_steps"] = len(self.env.rows) - row["physics_row"]
            row["command_block_complete"] = row["executed_physics_steps"] == len(commands)
        return state

    def _hold(self, seconds, subphase):
        for _ in range(int(np.ceil(seconds / CONTROL_DT))):
            self._block(np.repeat(self.q[None, :], CONTROL_STEPS, axis=0), subphase)

    def _joint_segment(self, destination, subphase):
        start = self.q.copy()
        destination = np.asarray(destination, float)
        seconds = max(1. / self.speed_scale, 1.875 * np.max(np.abs(destination - start)) / self.joint_speed_limit)
        steps = int(np.ceil(seconds / CONTROL_DT)) * CONTROL_STEPS
        for offset in range(0, steps, CONTROL_STEPS):
            u = np.arange(offset + 1, offset + CONTROL_STEPS + 1) / steps
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            self._block(start + blend[:, None] * (destination - start), subphase)

    def _pose_block(self, position_base, rotation_base, subphase, active=False, recovery=False, gap=None):
        solution = self.kin.solve_ik(position_base, orientation_base=rotation_base,
            seed=np.clip(self.q, self.kin.lower_limits, self.kin.upper_limits), max_starts=1, max_nfev=80,
            posture_weight=.003)
        if not solution.success:
            raise EpisodeRejected("pose IK failed in " + subphase)
        commands = self.q + np.arange(1, CONTROL_STEPS + 1)[:, None] / CONTROL_STEPS * (solution.q - self.q)
        return self._block(commands, subphase, active=active, recovery=recovery, gap=gap)

    def _contact_cycle(self):
        gap = self.pre_gap
        for _ in range(int(np.ceil(80 / self.speed_scale / CONTROL_DT))):
            state = self.env.state()
            phase = self.manager.state.phase
            if phase == Phase.RETRACT:
                speed = .01 * self.speed_scale
            elif state["hall_N"] > .3 or state["compression_m"] > .00015:
                speed = float(np.clip((state["hall_N"] - self.config.target_force_N) * .001, -.002, .002))
            else:
                speed = (-.012 if gap > .020 else -.002) * self.speed_scale
            gap += speed * CONTROL_DT
            if gap < -.01:
                raise EpisodeRejected("contact search exceeded displacement budget")
            self._pose_block(self.target.position_base + self.normal_base * gap, self.rotation_base,
                             "contact_cycle", active=True, gap=gap)
            if self.manager.state.completed:
                break
        if not self.manager.state.success:
            raise EpisodeRejected("state machine failed: " + str(self.manager.state.failure_reason))
        if self.best_audit_hold < self.config.hold_duration_s - 2 * DT:
            raise EpisodeRejected("observable hold lacked independent front contact")

    def _move_to_prepose(self, row):
        """Default expert route: dynamically reached home to checked prepose."""
        if np.max(np.abs(self.env.state()["q"])) > .03 or np.max(np.abs(self.q)) > .03:
            raise EpisodeRejected("point must start at dynamically reached zero home")
        transfer = np.asarray(row["transfer_q"], float)
        if not row.get("success") or not np.allclose(transfer[0], 0., atol=1e-9):
            raise EpisodeRejected("missing checked zero-home route")
        if np.max(np.abs(self.q)) > 1e-9:
            if not self.checker.edge(self.q, np.zeros(7))["clear"]:
                raise EpisodeRejected("initial home bridge has no sampled clearance")
            self._joint_segment(np.zeros(7), "initial_home")
        self._hold(1., "settle")
        for destination in transfer[1:]:
            self._joint_segment(destination, "transfer")
        self._hold(1., "align")
        return transfer

    def _withdraw_to_prepose(self, transfer):
        """Release along the retained target frame before choosing onward motion."""
        # Continue along the retained target frame; no point-to-point teleport.
        # Commanded zero-compression gap differs from measured gap while loaded.
        position = self.kin_position_from_command()
        gap = float((position - self.target.position_base) @ self.normal_base)
        for _ in range(int(np.ceil(6 / self.speed_scale / CONTROL_DT))):
            gap = min(self.pre_gap, gap + .01 * self.speed_scale * CONTROL_DT)
            self._pose_block(self.target.position_base + self.normal_base * gap, self.rotation_base,
                             "withdraw_to_prepose", gap=gap)
            if gap >= self.pre_gap - 1e-12:
                break
        self._hold(.2, "withdraw_settle")
        state = self.env.state()
        if self.metrics(state)["gap_m"] < self.pre_gap - .002 or state["hall_N"] >= .2:
            raise EpisodeRejected("prepose withdrawal not verified")

    def _return_home(self, transfer):
        self._withdraw_to_prepose(transfer)
        # Redundant IK can change the elbow at the same TCP. Check that bridge.
        check = self.checker.edge(self.q, transfer[-1])
        self.bridge_checks.append(check)
        if not check["clear"]:
            raise EpisodeRejected("return bridge has no sampled clearance")
        self._joint_segment(transfer[-1], "return_bridge")
        for destination in reversed(transfer[:-1]):
            self._joint_segment(destination, "return_home")
        self._hold(.5, "home_settle")
        if np.max(np.abs(self.env.state()["q"])) > .03:
            raise EpisodeRejected("dynamic home tracking error")

    def kin_position_from_command(self):
        return self.kin.tcp(self.q)[:3, 3]

    def _recover(self):
        state = self.env.state()
        metric = self.metrics(state)
        result = {"attempted": False, "released": False, "bounded_duration_s": 20. / self.speed_scale,
                  "scope": "best-effort simulated outward withdrawal; episode remains rejected"}
        # The target normal is locally meaningful only inside its approach tube.
        if metric["tangent_m"] > .02 or not -.02 <= metric["gap_m"] <= .08:
            result["reason"] = "outside verified local target corridor; simulation stopped"
            return result
        result["attempted"] = True
        rotation = self.env.base[:3, :3].T @ state["rotation_world"]
        position = state["tcp_base"].copy() - self.normal_base * max(0., state["compression_m"])
        baseline = {key: self.env.last_contacts.get(key, 0.) for key in FORBIDDEN_GROUPS}
        baseline_total = self._violations(state)[1]
        baseline_stroke = state["compression_m"]
        self.recovery_baseline = baseline, baseline_total, baseline_stroke
        release_time = 0.
        try:
            for _ in range(int(np.ceil(20 / self.speed_scale / CONTROL_DT))):
                position += self.normal_base * .005 * self.speed_scale * CONTROL_DT
                state = self._pose_block(position, rotation, "fault_retract", recovery=True)
                metric = self.metrics(state)
                _, total = self._violations(state)
                released = metric["gap_m"] >= self.pre_gap and state["hall_N"] < .2 and total < .2
                release_time = release_time + CONTROL_DT if released else 0.
                if release_time >= self.config.release_confirm_duration_s:
                    result.update(released=True, reason="outward withdrawal and release observed")
                    break
                if metric["gap_m"] > min(.085, self.pre_gap + .025):
                    raise EpisodeRejected("recovery displacement budget exhausted")
            else:
                result["reason"] = "recovery timeout; simulation stopped"
        except Exception as error:
            result["reason"] = str(error)
        result["final_gap_m"] = self.metrics(self.env.state())["gap_m"]
        return result

    def run(self, points):
        outcome = {"success": False, "training_accepted": False, "points": [],
            "scope": "simulation contact mechanics only; no acoustic or clinical success",
            "phase_config": asdict(self.config), "physics_dt_s": DT,
            "speed_scale": self.speed_scale, "variation": self.variation,
            "control_dt_s": CONTROL_DT, "action": "bounded joint-command increments from prior command",
            "resets_between_points": 0, "joint_speed_limit_rad_s": self.joint_speed_limit}
        if not points:
            outcome["failure_reason"] = "empty sequence"
            return outcome
        for _, row in points:
            pre_gap = row.get('free_approach_start_gap_m', .06)
            if not isinstance(pre_gap, (int,float)) or not np.isfinite(pre_gap) or not .025 <= pre_gap <= .06:
                outcome['failure_reason'] = 'unsupported checked prepose gap'
                return outcome
        self.sequence_length = len(points)
        for self.sequence_index, (target, row) in enumerate(points):
            self.target = target
            self.normal = np.asarray(row["command_normal_world"], float)
            self.normal_base = self.env.base[:3, :3].T @ self.normal
            self.rotation_base = self.env.base[:3, :3].T @ np.asarray(row["command_rotation_world"], float)
            self.pre_gap = float(row.get("free_approach_start_gap_m", .06))
            if not .025 <= self.pre_gap <= .06:
                outcome["failure_reason"] = "unsupported checked prepose gap"
                return outcome
            point_config = replace(self.config, precontact_gap_m=min(.04, self.pre_gap),
                                   retract_clearance_m=min(.04, self.pre_gap - .005))
            self.manager = PhaseManager(point_config)
            self.face_filtered = self.total_filtered = self.audit_hold = self.best_audit_hold = 0.
            self.bridge_checks = []
            point = {"target": target.name, "success": False, "training_accepted": False,
                     "world_target_m": target.position_world.tolist(), "command_normal_world": self.normal.tolist(),
                     "start_physics_row": len(self.env.rows), "point_provenance": target.provenance,
                     "phase_config": asdict(point_config), "checked_prepose_gap_m": self.pre_gap}
            outcome["points"].append(point)
            motion_started = False
            print('POINT_START', target.name, 'physics_time_s', self.env.state()['time_s'], flush=True)
            try:
                transfer = self._move_to_prepose(row)
                motion_started = True
                self._contact_cycle()
                point["contact_end_physics_row"] = len(self.env.rows)
                self._return_home(transfer)
                point.update(success=True, training_accepted=True)
            except Exception as error:
                point["failure_reason"] = str(error)
                point["recovery"] = self._recover() if motion_started or len(self.env.rows) > point["start_physics_row"] else {
                    "attempted": False, "released": False, "reason": "pre-motion validation failed"}
                outcome["failure_reason"] = target.name + ": " + str(error)
            point.update(end_physics_row=len(self.env.rows), final_phase=self.manager.state.phase.value,
                verified_front_contact_hold_s=self.best_audit_hold,
                observed_hold_s=self.manager.state.hold_elapsed, return_bridge_checks=self.bridge_checks)
            print('POINT_RESULT', target.name, point['success'], point.get('failure_reason'),
                  'hold_s', self.best_audit_hold, flush=True)
            if not point["success"]:
                break
        outcome["success"] = len(outcome["points"]) == len(points) and all(p["success"] for p in outcome["points"])
        outcome["training_accepted"] = outcome["success"]
        for point in outcome["points"]:
            point["training_accepted"] = outcome["training_accepted"] and point["success"]
        # A failed continuous sequence contributes no accepted demonstrations.
        if outcome["training_accepted"]:
            for row in self.control_rows:
                row["training_accepted"] = not row["recovery"]
            for row in self.env.rows:
                row["training_accepted"] = not row.get("recovery", False)
        outcome["physics_steps"] = len(self.env.rows)
        return outcome

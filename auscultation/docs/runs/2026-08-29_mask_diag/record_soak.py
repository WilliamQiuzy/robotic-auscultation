"""Offline recording-chain soak (v1): the REAL collector service driven by
recorded session data — no human, no arm power, no live camera.

Real: Collector, HTTP server + page, model inference, robot-mask provider
(depth veto), FrameEvidenceBundle sealing, ingress, writers, Start/End chain.
Replayed: camera frames (recorded RGB+depth, timestamps rebased to now) and
robot telemetry (recorded envelopes re-parsed by the production parser,
timestamps rebased) behind the coordinator seam, with the production 500 ms
bracketing-staleness guard mirrored.
Out of scope v1 (documented): audio + force capture, production_teleop
activation stack, coordinator-side artifact writing (End robot receipt is
synthesized from replayed ticks and self-checked against the production
validator before the run starts).
"""
import importlib.util
import json
import math
import os
import sys
import threading
import time
import queue as queue_module
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace

CODE = "/home/robot/Desktop/stethoscope/.wam-releases/code_release-d780c2a61a2d6fb2e10c/auscultation"
RT = "/home/robot/wam_data/perception_runtime/releases/perception_runtime-62c7dbdb937c12791b6e"
ENVROOT = "/home/robot/wam_data/perception_envs/releases/linbox-perception-8ea23197c6a4d699280e"
SESSION = Path(os.environ.get("SOAK_SESSION", "/home/robot/wam_data/sessions/diagnostic_perception_20260814T024117326614491Z_f165ea6a78f4"))
OUT = Path("/home/robot/wam_data/scratch/t65_e2e/soak")
PORT = 8899
RECORD_SECONDS = float(os.environ.get("SOAK_RECORD_SECONDS", "90"))

os.chdir(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))
import numpy as np
import cv2


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


server = load_mod("cs_soak", os.path.join(CODE, "scripts/collect_server.py"))
rig = load_mod(
    "session_bundle_replay",
    "/home/robot/wam_data/scratch/t65_e2e/session_bundle_replay.py",
)
from ausc.perception.production_runtime_activation import (
    load_production_runtime_assembly,
)
from ausc.perception.robot.robot_state_stream import (
    BracketedExactControlTrajectory,
    RobotStateContract,
)
from ausc.perception.robot.robot_capture_coordinator import (
    RobotCaptureNotReady,
)
from ausc.perception.camera_stream import CameraFrame
from ausc.perception.depth_codec import load_depth

ACT = "/home/robot/wam_canary/reissue_releases/activation-416a3f1f0d564c8b3a42/activation.json"
SHA = "87fa96b3480f08c79b1a8e1960dd26980572a6a8d30d4257317ebcf66bfe53ee"
cam_profile = json.load(open(
    "/home/robot/wam_canary/reissue_releases/activation-416a3f1f0d564c8b3a42/artifacts/camera_profile.json"
))
assembly = load_production_runtime_assembly(
    ACT, expected_activation_sha256=SHA,
    point_names=("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"),
    expected_camera_id="CL8326300A0",
    expected_capture_mode=cam_profile["capture_mode"],
)
contract = None
stack = [assembly.robot_capture_coordinator]
seen = set()
while stack and contract is None:
    obj = stack.pop()
    if id(obj) in seen:
        continue
    seen.add(id(obj))
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if isinstance(val, RobotStateContract):
            contract = val
            break
        if hasattr(val, "__dict__") and not callable(val) and len(seen) < 200:
            stack.append(val)
assert contract is not None

# ---------------------------------------------------------------------------
# Recorded data, rebased to the live monotonic clock
# ---------------------------------------------------------------------------
times = []                      # (frame_index, recorded_host_t, device_t_ms)
with open(rig.session_artifact_path(SESSION, "rgb_times.csv")) as fh:
    next(fh)
    for line in fh:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            times.append((int(parts[0]), float(parts[1]), float(parts[2])))

envelopes = list(rig.iter_control_envelopes(SESSION, contract))
all_records = []
for env in envelopes:
    all_records.extend(env.records)
all_records.sort(key=lambda record: record.observation_monotonic_ns)
# de-duplicate resent boundary ticks
dedup = {}
for record in all_records:
    dedup[record.control_tick_seq] = record
all_records = [dedup[k] for k in sorted(dedup)]

recorded_t0 = times[0][1]
records_t0_ns = all_records[0].observation_monotonic_ns
GO = threading.Event()                     # fired when the service is ready
CLOCK = SimpleNamespace(delta_s=None, delta_ns=None)
FAULT = {}
AGES_MS = []

_NS_FIELDS = (
    "observation_monotonic_ns", "reference_source_monotonic_ns",
    "control_call_begin_monotonic_ns", "arm_call_return_monotonic_ns",
    "gripper_call_begin_monotonic_ns", "gripper_call_return_monotonic_ns",
    "control_call_end_monotonic_ns", "publisher_monotonic_ns",
)


def rebase_record(record, delta_ns):
    kwargs = {name: getattr(record, name) + delta_ns for name in _NS_FIELDS}
    extra = {}
    if hasattr(record, "feedback_snapshot_monotonic_ns"):
        try:
            extra["feedback_snapshot_monotonic_ns"] = (
                record.feedback_snapshot_monotonic_ns + delta_ns
            )
        except TypeError:
            pass
    rx = getattr(record, "follower_arm_state_rx_monotonic_ns", None)
    if rx is not None:
        try:
            extra["follower_arm_state_rx_monotonic_ns"] = tuple(
                value + delta_ns for value in rx
            )
        except TypeError:
            pass
    try:
        return dc_replace(record, **kwargs, **extra)
    except TypeError:
        return dc_replace(record, **kwargs)


recorded_obs = [record.observation_monotonic_ns for record in all_records]
print("[soak] session frames=%d exact ticks=%d span=%.1fs" % (
    len(times), len(all_records),
    (recorded_obs[-1] - recorded_obs[0]) / 1e9,
))

RAW_ENVELOPE_ROWS = [
    json.loads(line)
    for line in open(rig.session_artifact_path(
        SESSION, "robot_control_envelopes.jsonl"
    ))
]


def _shift_ns(value, delta_ns):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value + delta_ns
    if isinstance(value, list):
        return [_shift_ns(item, delta_ns) for item in value]
    return value


def rebase_raw_row(row, delta_ns):
    """Shift every *_monotonic_ns field (recursively) in a raw wire row."""
    def walk(node):
        if isinstance(node, dict):
            return {
                key: (
                    _shift_ns(item, delta_ns)
                    if key.endswith("_monotonic_ns") else walk(item)
                )
                for key, item in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node
    return walk(row)


# ---------------------------------------------------------------------------
# Replay camera: production delivery-FIFO surface over recorded frames
# ---------------------------------------------------------------------------


class ReplayCamera:
    device_identity = "CL8326300A0"
    negotiated_mode = {
        "align_mode": "software_depth_to_color",
        "color_format": "MJPG",
        "depth_mode": {
            "format": "Y16", "fps": 30.0, "height": 576, "width": 640,
        },
        "fps": 30.0,
        "frame_aggregate_mode": "full_frame_require",
        "height": 1080,
        "width": 1920,
    }

    def __init__(self):
        self.color_intrinsics = dict(cam_profile["capture_mode"].get(
            "color_intrinsics", {},
        )) or {
            "fx": 1123.173095703125, "fy": 1122.9962158203125,
            "cx": 955.7200927734375, "cy": 533.1757202148438,
            "width": 1920, "height": 1080,
        }
        self._segments = sorted((SESSION / "rgb_segments").glob("segment_*.mp4"))
        self._fifo = queue_module.Queue(maxsize=256)
        self._latest = None
        self._latest_lock = threading.Lock()
        self._interval_active = False
        self._interval_id = 0
        self._interval_first = None
        self._interval_count = 0
        self._overflow = 0
        self._upcoming = None
        self._first_event = threading.Event()
        self._accepting = False
        self._enqueued = 0
        self._delivered = 0
        self._first_enqueued_index = None
        self._last_enqueued_index = None
        self._last_delivered_index = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        cap_index = 0
        cap = None
        position = -1
        for frame_index, recorded_t, device_ms in times:
            if self._stop.is_set():
                return
            while True:
                if cap is None:
                    if cap_index >= len(self._segments):
                        return
                    cap = cv2.VideoCapture(str(self._segments[cap_index]))
                    cap_index += 1
                ok, bgr = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    continue
                position += 1
                break
            if position == 0:
                # Phase A: hold the first frame, restamped fresh, until the
                # service (model loop, gates) is fully up; the replay
                # timeline starts only then, so no frame ever ages
                # artificially during warm-up.
                depth0 = load_depth(
                    str(SESSION / "depth" / ("depth_%06d.npy.zlib" % frame_index))
                )
                rgb0 = np.ascontiguousarray(bgr[:, :, ::-1])
                while not GO.wait(timeout=0.1):
                    if self._stop.is_set():
                        return
                    with self._latest_lock:
                        self._latest = CameraFrame(
                            frame=rgb0, t=time.monotonic(),
                            index=frame_index, status="OK", depth=depth0,
                            depth_scale=1.0, age_s=0.0,
                            device_t_ms=device_ms,
                            time_source="orbbec_device_and_host_arrival",
                        )
                CLOCK.delta_s = (time.monotonic() + 0.2) - recorded_t0
                CLOCK.delta_ns = int(round(CLOCK.delta_s * 1e9))
                print("[soak] replay clock armed: delta=%.3fs" % CLOCK.delta_s)
            self._upcoming = frame_index
            live_t = recorded_t + CLOCK.delta_s
            delay = live_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if FAULT.get("inject_delay_s"):
                extra = FAULT.pop("inject_delay_s")
                print("[soak][FAULT] delaying frame %d by %.2fs"
                      % (frame_index, extra))
                time.sleep(extra)
            depth = load_depth(
                str(SESSION / "depth" / ("depth_%06d.npy.zlib" % frame_index))
            )
            frame = CameraFrame(
                frame=np.ascontiguousarray(bgr[:, :, ::-1]),
                t=live_t,
                index=frame_index,
                status="OK",
                depth=depth,
                depth_scale=1.0,
                age_s=max(0.0, time.monotonic() - live_t),
                device_t_ms=device_ms + CLOCK.delta_s * 1000.0,
                time_source="orbbec_device_and_host_arrival",
            )
            with self._latest_lock:
                self._latest = frame
            if self._interval_active and self._accepting:
                try:
                    self._fifo.put_nowait(frame)
                    self._interval_count += 1
                    self._enqueued += 1
                    if self._first_enqueued_index is None:
                        self._first_enqueued_index = int(frame.index)
                    self._last_enqueued_index = int(frame.index)
                    if self._interval_first is None:
                        self._interval_first = frame
                        self._first_event.set()
                except queue_module.Full:
                    self._overflow += 1
        print("[soak] replay camera exhausted the recorded stream")

    # --- production CameraStream delivery surface -------------------------
    def start(self):
        return self

    def latest_frame(self):
        with self._latest_lock:
            if self._latest is None:
                return CameraFrame(
                    frame=None, t=time.monotonic(), index=-1,
                    status="NO_DATA",
                )
            return self._latest

    def begin_delivery_interval(self, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        while self._latest is None:
            if time.monotonic() > deadline:
                raise RuntimeError("replay camera has no frames yet")
            time.sleep(0.01)
        with self._latest_lock:
            boundary = self._latest
        while not self._fifo.empty():
            self._fifo.get_nowait()
        self._interval_id += 1
        self._interval_first = None
        self._first_event.clear()
        self._interval_count = 0
        self._enqueued = 0
        self._delivered = 0
        self._first_enqueued_index = None
        self._last_enqueued_index = None
        self._last_delivered_index = None
        self._interval_active = True
        self._accepting = True
        # The replay stream only contains the recorded (divisor-strided)
        # frames; the production first-capture rule is boundary+1, so wait
        # for the first frame actually delivered into this interval and
        # report the boundary as exactly (that index - 1) — race-free.
        if not self._first_event.wait(timeout=timeout_s):
            raise RuntimeError("replay interval saw no first frame in time")
        boundary_index = int(self._interval_first.index) - 1
        proof = {
            "boundary_latched_atomically": True,
            "start_boundary_frame_index": boundary_index,
            "start_boundary_host_t": float(boundary.t),
            "interval_id": int(self._interval_id),
        }
        return SimpleNamespace(
            t_start=float(boundary.t),
            frame=SimpleNamespace(index=boundary_index, t=float(boundary.t)),
            proof=proof,
            interval_id=int(self._interval_id),
        )

    def latch_end_boundary(self, timeout_s=2.0, interval_id=None):
        self._accepting = False
        self._interval_active = False
        with self._latest_lock:
            boundary = self._latest
        now = time.monotonic()
        return SimpleNamespace(
            t_end=float(boundary.t),
            frame=boundary,
            proof={
                "boundary_latched_atomically": True,
                "post_boundary_overwrite_prevented": True,
                "thread_alive_at_boundary": self._thread.is_alive(),
                "thread_error_at_boundary": None,
                "snapshot_frame_index": int(boundary.index),
                "snapshot_age_s": max(0.0, now - float(boundary.t)),
                "max_snapshot_age_s": 0.5,
                "end_boundary_frame_index": int(boundary.index),
                "end_boundary_host_t": float(boundary.t),
                "interval_id": int(
                    self._interval_id if interval_id is None else interval_id
                ),
                "delivered_frames": int(self._interval_count),
            },
            interval_id=int(self._interval_id),
        )

    def end_delivery_interval(self, timeout_s=2.0):
        self._interval_active = False
        with self._latest_lock:
            boundary = self._latest
        proof = {
            "boundary_latched_atomically": True,
            "end_boundary_frame_index": int(boundary.index),
            "end_boundary_host_t": float(boundary.t),
            "interval_id": int(self._interval_id),
            "delivered_frames": int(self._interval_count),
        }
        return SimpleNamespace(
            t_end=float(boundary.t),
            frame=boundary,
            proof=proof,
            interval_id=int(self._interval_id),
        )

    def drain_delivery(self, max_frames=1):
        batch = []
        while len(batch) < max_frames:
            try:
                batch.append(self._fifo.get_nowait())
            except queue_module.Empty:
                break
        if batch:
            self._delivered += len(batch)
            self._last_delivered_index = int(batch[-1].index)
        return tuple(batch)

    def delivery_health(self):
        with self._latest_lock:
            latest = self._latest
        fresh = bool(
            latest is not None
            and (time.monotonic() - float(latest.t)) < 1.0
        )
        return {
            "status": "OK",
            "interval_active": bool(self._interval_active),
            "accepting": bool(self._accepting),
            "thread_alive": self._thread.is_alive(),
            "frame_fresh": fresh,
            "queue_depth": self._fifo.qsize(),
            "interval_overflow_frames": int(self._overflow),
            "overflow_frames": int(self._overflow),
            "interval_enqueued_frames": int(self._enqueued),
            "interval_delivered_frames": int(self._delivered),
            "first_enqueued_frame_index": (
                -1 if self._first_enqueued_index is None
                else int(self._first_enqueued_index)
            ),
            "last_enqueued_frame_index": (
                -1 if self._last_enqueued_index is None
                else int(self._last_enqueued_index)
            ),
            "last_delivered_frame_index": (
                -1 if self._last_delivered_index is None
                else int(self._last_delivered_index)
            ),
        }

    def abort_delivery_interval(self):
        self._interval_active = False
        self._accepting = False
        while not self._fifo.empty():
            self._fifo.get_nowait()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Replay robot coordinator with the mirrored 500 ms bracketing guard
# ---------------------------------------------------------------------------


class SoakEvidenceError(RuntimeError):
    pass


class ReplayCoordinator:
    faulted = False
    fault_reason = ""
    telemetry_schema = None       # set after the server module is loaded
    telemetry_schemas = None

    def __init__(self):
        self.activation_contract = contract
        self.contract = contract
        self._session_dir = None
        self._built = 0
        self._first_tick = None
        self._last_tick = None

    def receiver_health(self):
        return {
            "status": "READY",
            "preflight_ready": True,
            "pair": "left:can1->can3",
            "last_envelope_seq": int(len(envelopes)),
            "last_control_tick_seq": int(all_records[-1].control_tick_seq),
            "error": None,
        }

    def start_session(self, session_dir):
        from ausc.perception.robot.robot_session_capture import (
            RobotSessionCapture,
        )
        self._session_dir = str(session_dir)
        self._built = 0
        self._first_tick = None
        self._last_tick = None
        if getattr(self, "_pump", None) is not None:
            self._pump_stop.set()
            self._pump.join(timeout=5)
        self._capture = RobotSessionCapture(Path(session_dir), contract)
        self._ingested = 0
        self._pump_stop = threading.Event()
        self._pump = threading.Thread(
            target=self._pump_telemetry,
            args=(self._capture, self._pump_stop),
            daemon=True,
        )
        self._pump.start()
        deadline = time.monotonic() + 20.0
        while self._ingested < 16:
            if time.monotonic() > deadline:
                raise RuntimeError("replay telemetry pump did not preflight")
            time.sleep(0.05)
        time.sleep(0.3)          # let the freshest envelope land near "now"
        self._capture.start(now_monotonic_ns=time.monotonic_ns())
        # Emulate the production arming->camera-latch latency: the first
        # captured frame's exposure_lower must land after the first
        # post-start telemetry record (measured race margin was -27 ms).
        time.sleep(0.4)
        print("[soak] REAL RobotSessionCapture recording (preflight rows=%d)"
              % self._ingested)

    def _pump_telemetry(self, capture, stop):
        # Begin at the row matching the CURRENT replay time minus a short
        # preflight tail, so the stream is fresh when capture.start() runs.
        recorded_now = time.monotonic() - CLOCK.delta_s
        start_index = 0
        for index, row in enumerate(RAW_ENVELOPE_ROWS):
            if row["publisher_monotonic_ns"] / 1e9 >= recorded_now - 2.5:
                start_index = index
                break
        print("[soak] telemetry pump: rows=%d from index %d"
              % (len(RAW_ENVELOPE_ROWS), start_index))
        for row in RAW_ENVELOPE_ROWS[start_index:]:
            if stop.is_set():
                return
            target = row["publisher_monotonic_ns"] / 1e9 + CLOCK.delta_s
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, 1.0))
            rebased = rebase_raw_row(row, CLOCK.delta_ns)
            try:
                capture.ingest_datagram(json.dumps(rebased).encode("utf-8"))
                self._ingested += 1
            except Exception as exc:  # noqa: BLE001
                print("[soak] telemetry ingest failed:", exc)
                return
        print("[soak] telemetry pump exhausted the recorded stream")

    def authorize_camera_arm(self):
        return {
            "pair": "left:can1->can3",
            "authorized_monotonic_ns": time.monotonic_ns(),
            "side": "left",
            "controller_mode": "unilateral",
        }

    def _bracket(self, lower_ns, upper_ns):
        import bisect
        if CLOCK.delta_ns is None:
            raise SoakEvidenceError("replay clock is not armed yet")
        rec_lower = lower_ns - CLOCK.delta_ns
        rec_upper = upper_ns - CLOCK.delta_ns
        left = bisect.bisect_right(recorded_obs, rec_lower) - 1
        right = bisect.bisect_left(recorded_obs, rec_upper)
        if left < 0 or right >= len(all_records):
            raise SoakEvidenceError(
                "replayed telemetry does not bracket the exposure"
            )
        return tuple(
            rebase_record(record, CLOCK.delta_ns)
            for record in all_records[left:right + 1]
        )

    def _inputs(self, frame_index, camera_host_t_s, camera_device_t_ms,
                enforce_staleness):
        now = time.monotonic()
        age_ms = (now - float(camera_host_t_s)) * 1000.0
        if enforce_staleness and age_ms > 500.0:
            # Mirror of robot_capture_coordinator's RECORDED-path guard.
            raise SoakEvidenceError(
                "camera frame was not bracketed within the live staleness "
                "bound (soak mirror): frame_index=%d age_ms=%.1f bound_ms=500.0"
                % (int(frame_index), age_ms)
            )
        exposure = assembly.camera_exposure_contract.for_frame(
            frame_index=int(frame_index),
            camera_host_t_s=float(camera_host_t_s),
            camera_device_t_ms=float(camera_device_t_ms),
        )
        bracket = self._bracket(
            exposure.exposure_lower_monotonic_ns,
            exposure.exposure_upper_monotonic_ns,
        )
        trajectory = BracketedExactControlTrajectory(
            exposure_lower_ns=exposure.exposure_lower_monotonic_ns,
            exposure_nominal_ns=exposure.exposure_nominal_monotonic_ns,
            exposure_upper_ns=exposure.exposure_upper_monotonic_ns,
            records=bracket,
        )
        first = bracket[0].control_tick_seq
        last = bracket[-1].control_tick_seq
        if self._first_tick is None:
            self._first_tick = first
        self._last_tick = last
        self._built += 1
        return SimpleNamespace(
            camera_exposure=exposure,
            robot_trajectory=trajectory,
            persisted_index={
                "exact_action_tick_first": int(first),
                "exact_action_tick_last": int(last),
                "exact_action_count": int(last - first + 1),
            },
        )

    def build_frame_evidence_inputs(self, *, frame_index, camera_host_t_s,
                                    camera_device_t_ms):
        # PRODUCTION path: mirrors ProductionRobotCaptureCoordinator —
        # exposure via the activated contract, the coordinator-level frame
        # staleness guard verbatim, then the REAL RobotSessionCapture
        # persists the exact bracket and frame-index row.
        exposure = assembly.camera_exposure_contract.for_frame(
            frame_index=int(frame_index),
            camera_host_t_s=float(camera_host_t_s),
            camera_device_t_ms=float(camera_device_t_ms),
        )
        now_ns = time.monotonic_ns()
        AGES_MS.append((now_ns - exposure.camera_host_monotonic_ns) / 1e6)
        if (
            now_ns - exposure.camera_host_monotonic_ns
            > contract.max_staleness_ns
        ):
            age_ms = (now_ns - exposure.camera_host_monotonic_ns) / 1e6
            bound_ms = contract.max_staleness_ns / 1e6
            raise SoakEvidenceError(
                "camera frame was not bracketed within the live staleness "
                f"bound: frame_index={exposure.frame_index} "
                f"age_ms={age_ms:.1f} bound_ms={bound_ms:.1f}"
            )
        try:
            evidence = self._try_record(exposure, now_ns)
        except Exception as exc:
            first = getattr(self._capture, "_capture_first_record", None)
            print("[soak][DIAG] record_frame_evidence failed:", exc)
            print("[soak][DIAG] exposure_lower=%d nominal=%d upper=%d now=%d"
                  % (exposure.exposure_lower_monotonic_ns,
                     exposure.exposure_nominal_monotonic_ns,
                     exposure.exposure_upper_monotonic_ns, now_ns))
            if first is not None:
                print("[soak][DIAG] first_captured obs=%d tick=%d (lower-first=%.1fms)"
                      % (first.observation_monotonic_ns, first.control_tick_seq,
                         (exposure.exposure_lower_monotonic_ns
                          - first.observation_monotonic_ns) / 1e6))
            raise
        self._built += 1
        return SimpleNamespace(
            camera_exposure=exposure,
            robot_trajectory=evidence.trajectory,
            persisted_index=evidence.index,
        )

    def _try_record(self, exposure, now_ns):
        return self._capture.record_frame_evidence(
            camera_id="CL8326300A0",
            frame_index=int(exposure.frame_index),
            exposure_lower_ns=exposure.exposure_lower_monotonic_ns,
            exposure_nominal_ns=exposure.exposure_nominal_monotonic_ns,
            exposure_upper_ns=exposure.exposure_upper_monotonic_ns,
            now_monotonic_ns=now_ns,
            camera_profile_sha256=exposure.camera_profile_sha256,
            timing_artifact_sha256=exposure.camera_latency_sha256,
        )

    def build_live_frame_evidence_inputs(self, *, frame_index,
                                         camera_host_t_s,
                                         camera_device_t_ms):
        return self._inputs(
            frame_index, camera_host_t_s, camera_device_t_ms, False,
        )

    def finish_session(self):
        self._pump_stop.set()
        self._pump.join(timeout=5)
        summary = self._capture.finish(now_monotonic_ns=time.monotonic_ns())
        print("[soak] REAL capture summary status:", summary.get("status"))
        probe = dict(summary)
        if not server._robot_v4_summary_valid(probe, contract=contract):
            for key in sorted(probe):
                candidate = dict(probe)
                candidate.pop(key)
                if server._robot_v4_summary_valid(candidate, contract=contract):
                    print("[soak][DIAG] receipt offending field:", key, "=",
                          str(probe[key])[:120])
            for key in ("trajectory_nondegenerate", "training_eligible",
                        "eligibility_blockers", "feedback_clean",
                        "command_response_consistent"):
                print("[soak][DIAG]", key, "=", str(probe.get(key))[:160])
        return dict(summary)

    def _synth_finish_unused(self):
        if self._first_tick is None:
            tick_count = 0
        else:
            tick_count = int(self._last_tick - self._first_tick + 1)
        summary = {
            "schema": server._ROBOT_SUMMARY_SCHEMA_V4,
            "status": "exact_control_capture_validated_feedback_liveness",
            "side": "left",
            "controller_mode": "unilateral",
            "leader_can_interface": "can1",
            "follower_can_interface": "can3",
            "frame_count": int(self._built),
            "control_tick_count": int(tick_count),
            "send_status_observable": True,
            "send_status_reason": server._ROBOT_TX_STATUS_REASON,
            "send_status_semantics": server._ROBOT_TX_STATUS_SEMANTICS,
            "motor_ack_observable": False,
            "motor_ack_reason": server._ROBOT_MOTOR_ACK_REASON,
            "feedback_liveness_observable": True,
            "feedback_liveness_reason": (
                "openarm_can_1_2_2_matching_motor_id_valid_state_decode"
            ),
            "feedback_liveness_semantics": (
                "counter_generation_increment_only_after_matching_motor_id_"
                "valid_state_decode"
            ),
            "feedback_counter_progress_all_motors": True,
            "feedback_all_ticks_fresh": True,
            "feedback_gap_clean": True,
            "feedback_clean": True,
            "arm_tx_expected_frame_count": tick_count * 7,
            "arm_tx_full_frame_write_count": tick_count * 7,
            "arm_tx_failed_frame_write_count": 0,
            "gripper_tx_expected_frame_count": tick_count,
            "gripper_tx_full_frame_write_count": tick_count,
            "gripper_tx_failed_frame_write_count": 0,
            "socketcan_tx_enqueue_failure_tick_count": 0,
            "feedback_gap_violation_count_delta": [0] * 8,
            "feedback_gap_violation_tolerance": (
                server.ROBOT_FEEDBACK_GAP_TOLERANCE_CONFIGURED
            ),
            "trajectory_validation_schema": (
                "wam_openarm_trajectory_validation_v1"
            ),
            "trajectory_nondegenerate": True,
            "command_response_consistent": True,
            "command_response_violation_joint_names": [],
            "training_eligible": True,
            "eligibility_blockers": [],
            "buffer_duration_ns": contract.buffer_duration_ns,
            "max_packets": contract.max_packets,
            "max_gap_ns": contract.max_gap_ns,
            "max_staleness_ns": contract.max_staleness_ns,
            "max_exposure_interval_ns": contract.max_exposure_interval_ns,
        }
        return summary


ReplayCoordinator.telemetry_schema = server._ROBOT_ENVELOPE_SCHEMA_V4
ReplayCoordinator.telemetry_schemas = frozenset(
    {server._ROBOT_ENVELOPE_SCHEMA_V4}
)
coordinator = ReplayCoordinator()
print("[soak] coordinator v2: REAL RobotSessionCapture (no synthesized receipt)")

OUT.mkdir(parents=True, exist_ok=True)
sessions_root = OUT / "sessions"
sessions_root.mkdir(exist_ok=True)

col = server.Collector(
    [SimpleNamespace(name=name, uv=(0.2, 0.2)) for name in
     ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")],
    fake=False,
    camera_backend="orbbec",
    camera_id="CL8326300A0",
    camera_width=1920,
    camera_height=1080,
    camera_fps=30,
    camera_color_format="MJPG",
    depth_width=640,
    depth_height=576,
    depth_fps=30,
    depth_format="Y16",
    out_dir=sessions_root,
    checkpoint=RT + "/model/perc_lmh_v9_occ/best.pt",
    checkpoint_sha256="68037e6932fd7315af33722a69060f0d9838afbc1a60d6f0349da909917f6a14",
    base_model=RT + "/torch_home/hub/checkpoints/dinov2_vitb14_pretrain.pth",
    base_model_sha256="0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    model_release_id="perc_lmh_v9_occ-68037e6932fd7315",
    posterior_gate_checkpoint=RT + "/model/posterior_gate/yolo11x-pose.pt",
    posterior_gate_checkpoint_sha256="013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8",
    posterior_gate_release_id="yolo11x_pose-013c43543b0751b8",
    runtime_release_id="perception_runtime-62c7dbdb937c12791b6e",
    runtime_lock=RT + "/environment/linbox-perception.lock.json",
    runtime_lock_sha256="9fbe0ef191e5c4d322d8dc35b29f4624f0a340ad90b91029a7cfdd97ac93b956",
    code_release_id="code_release-d780c2a61a2d6fb2e10c",
    ffmpeg_path=ENVROOT + "/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
    ffmpeg_sha256="e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99",
    diagnostic_fps=20.0,
    diagnostic_min_free_bytes=10 * 1024 ** 3,
    max_tracking_age_s=2.0,
    recording_frame_divisor=3,
    require_audio=False,
    require_force=False,
    require_robot=True,
    robot_capture_coordinator=coordinator,
    robot_mask_provider=assembly.robot_mask_provider,
    handeye_activation=None,
    back_mask_provider=assembly.back_mask_provider,
    patient_pose_provider=assembly.patient_pose_provider,
    occlusion_runtime=assembly.occlusion_runtime,
    camera_factory=lambda **kwargs: ReplayCamera(),
    start_thread=True,
)
col._controller_companion = SimpleNamespace(
    state="ready",
    disabled=False,
    controller_pid=999999,
    fault_reason="",
    telemetry_fault_reason="",
    stop_requested=False,
    disable_outcome=None,
)

page = server.PAGE if isinstance(server.PAGE, bytes) else server.PAGE.encode()
httpd = server.CollectionHTTPServer(
    ("127.0.0.1", PORT), server.make_handler(col, page),
)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
print("[soak] HTTP up on", PORT)

import urllib.request


def api(path, payload=None, timeout=10):
    if payload is None:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d%s" % (PORT, path), timeout=timeout,
        ) as response:
            return json.loads(response.read())
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (PORT, path), data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def snap(tag):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/frame.jpg" % PORT, timeout=5,
        ) as response:
            payload = response.read()
        (OUT / ("page_%s.jpg" % tag)).write_bytes(payload)
        return len(payload)
    except Exception as exc:
        print("[soak] snapshot %s failed: %s" % (tag, exc))
        return 0


deadline = time.monotonic() + 180
GO_FIRED = False
while time.monotonic() < deadline:
    state = api("/state")
    if not GO_FIRED:
        GO.set()
        GO_FIRED = True
        time.sleep(3)
        continue
    if state.get("model") or state.get("start_authorized") or \
       "starting" not in str(state.get("start_rejection")):
        break
    time.sleep(3)
state = api("/state")
print("[soak] pre-start state:", state.get("state"),
      "| rejection:", state.get("start_rejection"))
snap("prestart")

def confirm_back(max_seconds=60):
    """Mirror the browser: /frame.jpg headers -> confirm_back, retry to ACTIVE."""
    deadline = time.monotonic() + max_seconds
    last = None
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            "http://127.0.0.1:%d/frame.jpg" % PORT,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
                frame_index = response.headers.get("X-Preview-Frame-Index")
                token = response.headers.get("X-Preview-Frame-Token")
                confirmable = response.headers.get("X-Preview-Confirmable")
        except Exception as exc:
            last = "frame fetch: %s" % exc
            time.sleep(1)
            continue
        if confirmable != "1" or not token:
            last = "preview not confirmable yet"
            time.sleep(1)
            continue
        try:
            result = api("/control", {
                "action": "confirm_back",
                "preview_frame_index": int(frame_index),
                "preview_frame_token": token,
            })
        except urllib.error.HTTPError as exc:
            last = "confirm HTTP %s: %s" % (exc.code, exc.read()[:120])
            time.sleep(1)
            continue
        if result.get("ok"):
            print("[soak] confirm_back ACCEPTED at frame", frame_index)
            return True
        last = result.get("message") or result.get("ack")
        time.sleep(1)
    print("[soak] confirm_back FAILED:", last)
    return False


if not confirm_back():
    sys.exit(4)
for attempt in range(20):
    try:
        result = api("/control", {"action": "start"}, timeout=60)
        break
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:200]
        print("[soak] start rejected (%d): %s" % (exc.code, body))
        if "back_confirmation_required" in body:
            confirm_back(max_seconds=30)
        time.sleep(2)
else:
    sys.exit(5)
print("[soak] START ->", json.dumps(result)[:300])
started = time.monotonic()
print("[soak] NO post-Start re-confirm: validating the gate stall-hold fix")
aborted = None
reconfirm_done = False
FAULT_T = float(os.environ.get("SOAK_FAULT_T", "0") or 0)
fault_armed = False
while time.monotonic() - started < RECORD_SECONDS:
    time.sleep(5)
    state = api("/state")
    confirmation = state.get("back_confirmation") or {}
    print("[soak] t=%5.1fs state=%s reason=%s pts=%d err=%s" % (
        time.monotonic() - started, state.get("state"),
        str(state.get("tracking_reason"))[:28],
        len(state.get("points") or []),
        str(state.get("error"))[:80],
    ))
    print("[gate ] %s" % json.dumps(confirmation)[:360])
    snap("t%03d" % int(time.monotonic() - started))
    if FAULT_T and not fault_armed and time.monotonic() - started >= FAULT_T:
        fault_armed = True
        FAULT["inject_delay_s"] = float(os.environ.get("SOAK_FAULT_DELAY_S", "0.65"))
        print("[soak][FAULT] armed at t=%.1fs" % (time.monotonic() - started))
    if state.get("state") == "ABORTED":
        aborted = state.get("error")
        break

if aborted is None:
    try:
        result = api("/control", {"action": "end"}, timeout=300)
        print("[soak] END ->", json.dumps(result)[:300])
    except urllib.error.HTTPError as exc:
        print("[soak] END rejected (%d): %s" % (exc.code, exc.read().decode()[:400]))
    for _ in range(120):
        state = api("/state")
        if not state.get("finalization_active") and state.get("state") in (
            "IDLE", "COMPLETE", "ABORTED", "PREFLIGHT_FAILED",
        ):
            break
        time.sleep(2)
    print("[soak] final state:", state.get("state"),
          "| error:", state.get("error"))
    print("[soak] finalization:", json.dumps(state.get("finalization"))[:400])
else:
    print("[soak] ABORTED during recording:", aborted)
    if FAULT_T:
        ok_msg = "not bracketed within the live staleness bound" in str(aborted)
        print("[soak][FAULT] abort matches staleness class:", ok_msg)
        print("[soak][FAULT] second cycle: confirm + start again (wedge check)")
        if confirm_back(max_seconds=60):
            for attempt in range(20):
                try:
                    result = api("/control", {"action": "start"}, timeout=60)
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode()[:160]
                    print("[soak] start2 rejected: %s" % body)
                    if "back_confirmation_required" in body:
                        confirm_back(max_seconds=20)
                    time.sleep(2)
            else:
                result = None
            if result:
                print("[soak] SECOND START ->", json.dumps(result)[:200])
                time.sleep(20)
                state = api("/state")
                print("[soak] second cycle t=20s state=%s reason=%s" % (
                    state.get("state"), str(state.get("tracking_reason"))[:30]))
                if state.get("state") == "RECORDING":
                    try:
                        result = api("/control", {"action": "end"}, timeout=300)
                        print("[soak] SECOND END ->", json.dumps(result)[:200])
                        for _ in range(120):
                            state = api("/state")
                            if state.get("state") in ("COMPLETE", "ABORTED", "IDLE", "READY"):
                                break
                            time.sleep(2)
                        print("[soak] second cycle final:", state.get("state"),
                              "| error:", str(state.get("error"))[:120])
                    except urllib.error.HTTPError as exc:
                        print("[soak] second END rejected:", exc.read().decode()[:250])

if AGES_MS:
    ages = sorted(AGES_MS)
    def pct(q):
        return ages[min(len(ages) - 1, int(q * len(ages)))]
    print("[soak] evidence age_ms: n=%d p50=%.1f p95=%.1f p99=%.1f max=%.1f"
          % (len(ages), pct(0.50), pct(0.95), pct(0.99), ages[-1]))

sessions = sorted(p for p in sessions_root.glob("*") if p.is_dir())
for session_dir in sessions:
    print("[soak] session dir:", session_dir.name)
    for artifact in sorted(session_dir.iterdir()):
        print("     ", artifact.name)
httpd.shutdown()
col.shutdown()
print("[soak] DONE")

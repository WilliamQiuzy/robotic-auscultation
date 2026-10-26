"""End-to-end integration fuzz of the DEPLOYMENT chain: real NormalsDetector-stub →
real ModelBasedTracker → real AuscultationController + safety state machine, driven over a
synthetic sequence with injected adversarial events (occlusion bursts, depth-NaN bursts, dropout).

This tests the COMPOSITION (where contract bugs between components hide), not units. The core
invariant asserted end-to-end: whenever the arm SERVOS, the commanded point that tick was a
genuinely valid target (measured + TRACKING + finite world frame + low uncertainty). And the
tracker never emits a non-finite xyz_world, and never crashes on any injected event.
"""
import os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from ausc.perception.tracking.replay import make_synthetic_sequence, SyntheticDetector  # noqa: E402
from ausc.perception.tracking.model_tracker import ModelBasedTracker, ModelConfig  # noqa: E402
from ausc.perception.robot.controller import AuscultationController  # noqa: E402
from ausc.perception.robot.safety import SafetyConfig, State  # noqa: E402

# The controller refuses to run without limits derived from a force calibration, and latches ESTOP
# on a missing force reading, so the chain gets a calibrated config and a fresh no-contact reading.
_FORCE_CALIBRATION = SimpleNamespace(calibrated=True, max_screened_force_n=4.6, hard_limit_force_n=5.5)
_NO_CONTACT = SimpleNamespace(force_n=0.0, status="WARN", age_s=0.0, valid=True)


class MockArm:
    def __init__(self):
        self.p = np.zeros(3); self.servo_calls = 0; self.hold_calls = 0

    def tcp_xyz(self):
        return self.p.copy()

    def servo_toward(self, target_xyz, max_step_m):
        self.servo_calls += 1
        d = np.asarray(target_xyz, float) - self.p; n = np.linalg.norm(d)
        self.p = self.p + (d if n <= max_step_m else d / n * max_step_m)
        return self.p.copy()

    def hold(self):
        self.hold_calls += 1

    def retract(self, direction_base, max_step_m):
        d = np.asarray(direction_base, float)
        self.p = self.p + d / np.linalg.norm(d) * max_step_m
        return self.p.copy()


def _run_chain(perturb, n_frames=120, target="R3"):
    frames, intr = make_synthetic_sequence(n_frames=n_frames, seed=1)
    det = SyntheticDetector(intr, noise_px=1.0)
    extr = np.eye(4)                                   # xyz_world = camera frame (finite stand-in)
    tr = ModelBasedTracker(det, intr, ModelConfig(init_frames=8), extrinsics=extr)
    arm = MockArm()
    ctrl = AuscultationController(arm, SafetyConfig.from_force_calibration(_FORCE_CALIBRATION))
    maxstd = ctrl.cfg.max_position_std_m
    commanded = False
    violations = []
    for i, fr in enumerate(frames):
        depth = fr.depth
        ev = perturb(i)
        if ev == "drop":
            det._gt_full = None                       # detector returns None (full occlusion)
        else:
            det.set_truth(fr.gt_pixels)
            if ev == "nan":
                depth = fr.depth.astype(np.float32).copy()
                depth[100:300, 200:420] = np.nan      # inject a NaN depth band
        out = tr.process(fr.rgb, depth, t=fr.t, depth_scale=fr.depth_scale)
        # tracker output sanity EVERY tick (no crash, 10 pts, world finite-or-None)
        assert len(out.points) == 10
        for q in out.points:
            assert q.xyz_world is None or np.all(np.isfinite(q.xyz_world))
            assert np.isfinite(q.position_std)
        if out.initialized and not commanded:
            ctrl.command(target); commanded = True
        if not commanded:
            continue
        prev = arm.servo_calls
        ctrl.tick(out.points, perception_t=out.t, force_sample=_NO_CONTACT)
        if arm.servo_calls > prev:                    # arm MOVED → commanded point must be valid
            p = next((q for q in out.points if q.name == target), None)
            ok = (p is not None and p.measured and p.status == "TRACKING"
                  and p.xyz_world is not None and np.all(np.isfinite(p.xyz_world))
                  and np.isfinite(p.position_std) and p.position_std <= maxstd)
            if not ok:
                snap = None if p is None else (p.measured, p.status, float(p.position_std))
                violations.append((i, snap))
    return arm, violations


def test_chain_never_servos_on_degraded_target_under_perturbation():
    def perturb(i):
        if 30 <= i < 45:
            return "drop"                              # occlusion burst
        if i in (60, 61, 62, 80, 95):
            return "nan"                               # depth-NaN bursts
        if 100 <= i < 110:
            return "drop"                              # second dropout
        return "ok"
    arm, violations = _run_chain(perturb)
    assert violations == [], f"arm servoed on a degraded target at: {violations[:5]}"
    assert arm.servo_calls > 0                         # but it DID move on good frames (non-trivial)


def test_chain_clean_sequence_reaches_tracking_and_moves():
    arm, violations = _run_chain(lambda i: "ok")
    assert violations == []
    assert arm.servo_calls > 0


def test_chain_full_blackout_after_init_never_servos_forever():
    """Detector dies permanently after init: points coast → LOST; the arm must stop servoing
    (no indefinite motion on stale/extrapolated data)."""
    def perturb(i):
        return "drop" if i >= 20 else "ok"
    arm, violations = _run_chain(perturb, n_frames=80)
    assert violations == []
    # after a long blackout the controller must not still be servoing on coasting/LOST points
    # (the last servo, if any, was during the good prefix)


def test_chain_depth_nan_every_frame_no_crash():
    """Pathological: NaN depth band on EVERY frame must never crash and never emit a NaN world pt."""
    arm, violations = _run_chain(lambda i: "nan", n_frames=60)
    assert violations == []

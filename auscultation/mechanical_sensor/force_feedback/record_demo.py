"""Real-robot imitation-learning demonstration recorder, force-aware.

LingBot-VA (the WAM imitation pipeline) only *loads* LeRobot datasets -- it has
no real-robot recorder. This scaffold IS that recorder, with the contact-force
sensor fully wired in as a learned observation. Cameras / robot state / teleop
are behind small pluggable interfaces (a MockRobot runs end-to-end with no
hardware) so you drop in the OpenArm + RealSense + teleop when the rig is up.

Per-frame LeRobot structure written here (matches wan_va configs' obs keys plus
the new force key):

    observation.images.cam_high     uint8  (H, W, 3)
    observation.state               float32 (state_dim,)
    observation.contact_force_n     float32 (1,)        <-- learned force channel
    action                          float32 (action_dim,)

Run a quick mock episode (no hardware, real force if a sensor is connected):

    python record_demo.py --repo-id ausc/openarm_demo --episodes 1 --steps 60 \
        --port /dev/cu.usbmodem101            # omit --port to mock the force too
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from force_sensor import ForceCalibration, ForceSensor, STATUS_NO_DATA  # noqa: E402

# --- observation / action shapes (edit to match the OpenArm rig) -------------
CAM_KEYS = ["observation.images.cam_high"]
IMG_HW = (240, 320)
STATE_DIM = 14          # e.g. 7-DoF arm + gripper, or OpenArm joint vector
ACTION_DIM = 14
FORCE_KEY = "observation.contact_force_n"


# --- pluggable rig interface -------------------------------------------------
class Robot(Protocol):
    """Implement this against the OpenArm SDK + cameras + teleop device."""

    def get_images(self) -> dict[str, np.ndarray]: ...   # cam_key -> HxWx3 uint8
    def get_state(self) -> np.ndarray: ...               # (STATE_DIM,) float32
    def get_teleop_action(self) -> np.ndarray: ...       # (ACTION_DIM,) float32
    def send_action(self, action: np.ndarray) -> None: ...
    def close(self) -> None: ...


@dataclass
class MockRobot:
    """Runs the whole pipeline with no hardware so the recorder is testable."""

    _t: int = 0

    def get_images(self) -> dict[str, np.ndarray]:
        h, w = IMG_HW
        return {k: np.zeros((h, w, 3), dtype=np.uint8) for k in CAM_KEYS}

    def get_state(self) -> np.ndarray:
        return np.full(STATE_DIM, 0.01 * self._t, dtype=np.float32)

    def get_teleop_action(self) -> np.ndarray:
        self._t += 1
        return np.full(ACTION_DIM, 0.01 * self._t, dtype=np.float32)

    def send_action(self, action: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass


# --- dataset writer (real LeRobot if installed, else a local fallback) --------
def make_dataset(repo_id: str, fps: int):
    features = {
        **{k: {"dtype": "image", "shape": (*IMG_HW, 3), "names": ["h", "w", "c"]} for k in CAM_KEYS},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        FORCE_KEY: {"dtype": "float32", "shape": (1,), "names": ["contact_force_n"]},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features), "lerobot"
    except Exception:
        return _LocalWriter(repo_id, fps, features), "local-fallback"


@dataclass
class _LocalWriter:
    """Minimal stand-in that records the same per-frame dicts to .npz when the
    lerobot package is not installed -- keeps this scaffold runnable anywhere."""

    repo_id: str
    fps: int
    features: dict
    _frames: list[dict] = field(default_factory=list)

    def add_frame(self, frame: dict, task: str | None = None) -> None:
        self._frames.append({k: np.asarray(v) for k, v in frame.items()})

    def save_episode(self) -> None:
        out = Path("../generated/force_feedback") / f"{self.repo_id.replace('/', '_')}_ep.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, n_frames=len(self._frames), force=[f[FORCE_KEY] for f in self._frames])
        print(f"[local-fallback] {len(self._frames)} frames -> {out} (install `lerobot` for real datasets)")
        self._frames.clear()


# --- the record loop (this is the integration point) -------------------------
def record_episode(dataset, robot: Robot, force: ForceSensor | None, steps: int, fps: int, task: str) -> None:
    period = 1.0 / fps
    for _ in range(steps):
        t0 = time.monotonic()
        images = robot.get_images()
        state = robot.get_state()
        action = robot.get_teleop_action()
        # >>> the whole point: sample contact force and record it every step <<<
        force_n = force.read().force_n if force is not None else 0.0
        frame = {
            **images,
            "observation.state": state.astype(np.float32),
            FORCE_KEY: np.array([force_n], dtype=np.float32),
            "action": action.astype(np.float32),
        }
        dataset.add_frame(frame, task=task)
        robot.send_action(action)
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
    dataset.save_episode()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Force-aware LeRobot demonstration recorder")
    ap.add_argument("--repo-id", default="ausc/openarm_demo")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--task", default="auscultate the chest")
    ap.add_argument("--port", help="force sensor serial port; omit to mock the force channel")
    ap.add_argument("--calibration", default=str(Path(__file__).with_name("force_feedback_calibration_veo6.json")))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    robot: Robot = MockRobot()  # TODO: replace with OpenArmRobot(...) when the rig is up
    dataset, backend = make_dataset(args.repo_id, args.fps)
    print(f"[recorder] backend={backend} repo={args.repo_id} fps={args.fps} force={'real' if args.port else 'mock'}")

    force_cm = (
        ForceSensor(args.port, ForceCalibration.from_json(args.calibration))
        if args.port else None
    )
    try:
        force = force_cm.start() if force_cm else None
        if force is not None and force.read().status == STATUS_NO_DATA:
            time.sleep(0.3)  # let the first sample arrive
        for ep in range(args.episodes):
            print(f"[recorder] episode {ep + 1}/{args.episodes} ...")
            record_episode(dataset, robot, force, args.steps, args.fps, args.task)
    finally:
        if force_cm:
            force_cm.stop()
        robot.close()
    print("[recorder] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

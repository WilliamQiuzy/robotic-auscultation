"""Phase-1 MuJoCo session generator — REAL contact physics (not scripted forces).

Loads the OpenArm-left MJCF (mechanical_sensor/sim/openarm_mjcf.py), IK-drives the stethoscope tip
to press into a breathing 'back' surface, steps the MuJoCo contact solver, and records a session in
record_session.py format. Unlike the Phase-0 generator the CONTACT FORCE EMERGES from the tip
penetrating the surface (Kelvin-Voigt solver contact) and the follower joints are the physically
achieved qpos (leader = commanded IK targets = action). Runs in the arm64 mujoco venv (~/envs/mjx).

    ~/envs/mjx/bin/python scripts/gen_mujoco_session.py --seconds 20 --out data/sessions/session_mj0001
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import wave

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "mechanical_sensor", "generated", "mujoco", "openarm_left_scene.xml")
SR = 16000
Q_HOME = np.array([0.0, -0.8, 0.0, 1.2, 0.0, 0.3, 0.0])
BREATH_HZ = 0.25
F_TARGET = 4.0            # N — the auscultation contact-force setpoint (design: 4 N target / 8 N limit)
PRESS_GAIN = 1.3e-5       # m per N error per control step (SLOW loop → breathing shows in the force)
PRESS_MAX = 0.06          # cap how far past home the tip may press (safety)
BREATH_A = 0.009          # breathing slide amplitude (m) — surface rise/fall modulates the force


def _ik(m, dk, target, tip_id, q_seed, iters=12, lam=0.05, maxstep=0.02):
    q = q_seed.copy()
    lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        dk.qpos[:] = 0.0; dk.qpos[:7] = q
        mujoco.mj_forward(m, dk)
        dx = np.asarray(target) - dk.site(tip_id).xpos
        nrm = np.linalg.norm(dx)
        if nrm > maxstep:
            dx = dx / nrm * maxstep
        mujoco.mj_jacSite(m, dk, jacp, None, tip_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), dx)
        q = np.clip(q + dq, lo, hi)
    return q


def _tip_force(m, d, disk_gid) -> float:
    f = np.zeros(6); tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if disk_gid in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f); tot += abs(f[0])
    return tot


def domain_randomize(m, rng):
    """Per-session appearance randomization (physics untouched) for vision sim-to-real: lighting,
    skin tone, arm color, background, camera pose. A vision encoder trained across these is robust
    to the sim→real gap (docs/audio_reward_design.md Phase-1 next step)."""
    def gid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    for i in range(m.nlight):                                   # lighting position + color
        m.light_pos[i] = [rng.uniform(-0.5, 0.8), rng.uniform(-0.9, -0.1), rng.uniform(0.4, 0.9)]
        m.light_diffuse[i] = rng.uniform(0.5, 1.0, 3)
    bg = gid("back_surf")
    if bg >= 0:                                                 # skin tone
        m.geom_rgba[bg] = [rng.uniform(0.55, 0.95), rng.uniform(0.38, 0.68), rng.uniform(0.33, 0.58), 1]
    fg = gid("floor")
    if fg >= 0:                                                 # background
        m.geom_rgba[fg] = [*rng.uniform(0.1, 0.55, 3), 1]
    for i in range(m.ngeom):                                    # arm limb color jitter (grey-ish)
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("openarm_left") or (m.geom_rgba[i][0] > 0.6 and i not in (bg, fg)):
            g = rng.uniform(0.55, 0.85); m.geom_rgba[i] = [g, g + rng.uniform(-0.05, 0.05), g, 1]
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "clinician")
    if cid >= 0:                                                # camera viewpoint jitter
        m.cam_pos[cid] = m.cam_pos[cid] + rng.uniform(-0.06, 0.06, 3)
        m.cam_fovy[cid] = float(np.clip(m.cam_fovy[cid] + rng.uniform(-6, 6), 30, 75))


def _write_wav(path, y):
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def _clean_clip(seconds):
    import soundfile as sf
    cands = sorted(glob.glob(os.path.join(ROOT, "data", "audio", "_samples", "hf_lung_v1", "*.wav")))
    n = int(seconds * SR)
    if not cands:
        return (0.05 * np.random.randn(n)).astype(np.float32)
    y, sr = sf.read(cands[0]); y = y.mean(1) if y.ndim > 1 else y
    if sr != SR:                                            # simple linear resample (no scipy dep)
        y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * SR / sr)), np.arange(len(y)), y)
    y = np.tile(y, int(np.ceil(n / len(y))))[:n]
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.6).astype(np.float32)


def generate(out_dir, seconds=20.0, control_hz=50.0, seed=0, randomize=True, f_target=F_TARGET):
    import imageio.v2 as imageio
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(SCENE)
    if randomize:
        domain_randomize(m, rng)                               # vision sim-to-real (appearance only)
    d = mujoco.MjData(m); dk = mujoco.MjData(m)
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    disk_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "stetho_disk")
    breath_dof = 7
    d.qpos[:7] = Q_HOME; d.ctrl[:7] = Q_HOME
    mujoco.mj_forward(m, d)
    p_home = d.site("tip").xpos.copy()

    phys_hz = 1.0 / m.opt.timestep
    n_phys = int(seconds * phys_hz)
    dec_ctrl = int(round(phys_hz / control_hz))          # steps per control log
    dec_force = int(round(phys_hz / 100.0))
    dec_cam = int(round(phys_hz / 25.0))
    t0 = 1000.0
    approach, retract = 0.30 * seconds, 0.85 * seconds
    renderer = mujoco.Renderer(m, 180, 240)

    robot_rows, force_rows, obs_rows, rgb_times, frames = [], [], [], [], []
    press = 0.0
    for k in range(n_phys):
        t = k / phys_hz
        ramp = min(np.clip(t / approach, 0, 1), np.clip((seconds - t) / (seconds - retract), 0, 1))
        fN = _tip_force(m, d, disk_gid)                              # measured emergent contact force
        press = float(np.clip(press + PRESS_GAIN * (f_target * ramp - fN), 0.0, PRESS_MAX))  # force loop
        target = p_home + np.array([0.0, press, 0.0])               # tip position that holds ~F_TARGET
        d.ctrl[breath_dof] = BREATH_A * (0.5 - 0.5 * np.cos(2 * np.pi * BREATH_HZ * t))  # inhale rises
        d.ctrl[:7] = _ik(m, dk, target, tip_id, d.qpos[:7].copy())
        mujoco.mj_step(m, d)
        tt = t0 + t
        if k % dec_force == 0:
            force_rows.append((tt, int((1.6 - 0.05 * fN) / 3.3 * 65535), 1.6 - 0.05 * fN, fN * 0.4, fN))
        if k % dec_ctrl == 0:
            robot_rows.append((tt, d.qpos[:7].copy(), d.ctrl[:7].copy()))
            obs_rows.append((tt, fN))
        if k % dec_cam == 0:
            renderer.update_scene(d, camera="clinician"); frames.append(renderer.render()); rgb_times.append((len(frames) - 1, tt))
    renderer.close()

    # write robot / force / obs / rgb / audio / manifest (record_session format)
    with open(os.path.join(out_dir, "robot.csv"), "w") as fp:
        fp.write("t," + ",".join(f"q{i+1}" for i in range(7)) + "," + ",".join(f"a{i+1}" for i in range(7)) + "\n")
        for tt, q, a in robot_rows:
            fp.write(f"{tt:.4f}," + ",".join(f"{v:.5f}" for v in q) + "," + ",".join(f"{v:.5f}" for v in a) + "\n")
    with open(os.path.join(out_dir, "force.csv"), "w") as fp:
        fp.write("t,raw,hall_v,displacement_mm,force_n,status\n")
        for tt, raw, hv, dmm, fN in force_rows:
            fp.write(f"{tt:.4f},{raw},{hv:.4f},{dmm:.3f},{fN:.3f},OK\n")
    imageio.mimwrite(os.path.join(out_dir, "rgb.mp4"), frames, fps=25, macro_block_size=1)
    with open(os.path.join(out_dir, "rgb_times.csv"), "w") as fp:
        fp.write("frame_index,t\n"); fp.writelines(f"{i},{t:.4f}\n" for i, t in rgb_times)
    # audio: clean lung clip, degraded where contact force is out of the 3-5 N band
    y = _clean_clip(seconds)
    ft = np.array([r[0] - t0 for r in force_rows]); fN = np.array([r[4] for r in force_rows])
    good = np.interp(np.arange(len(y)) / SR, ft, ((fN >= 3) & (fN <= 5)).astype(float))
    y = y * (0.3 + 0.7 * good) + rng.normal(0, 1, len(y)).astype(np.float32) * 0.25 * (1 - good)
    _write_wav(os.path.join(out_dir, "audio.wav"), y / (np.max(np.abs(y)) + 1e-9) * 0.9)
    with open(os.path.join(out_dir, "obs.csv"), "w") as fp:
        fp.write("t,force_n,force_status,audio_rms_db,audio_peak_db,cam_index,cam_t\n")
        for i, (tt, fN_) in enumerate(obs_rows):
            fp.write(f"{tt:.4f},{fN_:.3f},OK,-30.0,-12.0,{i},{tt:.4f}\n")
    manifest = {
        "t0_monotonic": t0, "t0_wall": "MUJOCO", "duration_s": seconds, "control_hz": control_hz,
        "streams": ["audio", "cam", "force", "robot"], "synthetic": True, "engine": "mujoco",
        "audio": {"sr": SR, "t0_monotonic": t0, "device": "mujoco", "channel": 1},
        "force": {"port": "mujoco-contact", "calibration": "emergent", "n_rows_hint": f"{len(force_rows)}"},
        "camera": {"backend": "mujoco-render", "index": 0, "n_rgb_frames": len(frames),
                   "n_depth_frames": 0, "has_depth": False},
        "robot": {"arm": "openarm_left", "n_joints": 7, "state": "follower qpos", "action": "IK leader targets"},
        "n_obs": len(obs_rows), "health": {"audio": {"status": "OK"}, "force": {"status": "OK"}},
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    peak = max((r[4] for r in force_rows), default=0.0)
    inband = np.mean([(3 <= r[4] <= 5) for r in force_rows]) if force_rows else 0.0
    print(f"[mj] {out_dir}: {len(robot_rows)} steps, {len(force_rows)} force rows, {len(frames)} frames | "
          f"emergent contact peak={peak:.2f}N in-band[3-5]={inband*100:.0f}%")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-randomize", action="store_true", help="disable per-session domain randomization")
    ap.add_argument("--f-target", type=float, default=F_TARGET,
                    help="contact-force setpoint (N); low (<3) → under-press → gate=False → low reward")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sessions", "session_mj0001"))
    a = ap.parse_args()
    generate(a.out, seconds=a.seconds, control_hz=a.control_hz, seed=a.seed,
             randomize=not a.no_randomize, f_target=a.f_target)

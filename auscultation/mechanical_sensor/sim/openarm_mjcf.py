"""Generate a self-contained MuJoCo MJCF of the OpenArm LEFT arm (7-DOF) from the URDF's concrete
kinematics + inertials — no external meshes (they aren't in-repo; visual meshes are cosmetic for
contact physics, and self-containment is the point). Links become visual capsules; only the
stethoscope tip has a contact geom; a breathing-jointed 'back' surface sits within reach.

    python mechanical_sensor/sim/openarm_mjcf.py            # writes generated/mujoco/openarm_left_scene.xml

Runs in the arm64 mujoco venv (~/envs/mjx). Used by scripts/gen_mujoco_session.py (Phase 1).
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # auscultation/
URDF = os.path.join(ROOT, "ausc", "perception", "robot", "openarm_v10.urdf")
OUT = os.path.join(ROOT, "mechanical_sensor", "generated", "mujoco", "openarm_left_scene.xml")
ARM = "openarm_left"
Q_HOME = np.array([0.0, -0.8, 0.0, 1.2, 0.0, 0.3, 0.0])


def _rpy_to_quat(r: float, p: float, y: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)   # (w,x,y,z)


def _f(s: str | None, d: str = "0 0 0") -> list[float]:
    return [float(x) for x in (s or d).split()]


def _chain():
    """Parse the URDF left-arm chain: ordered [(joint_name, child_link, origin_xyz, rpy, axis, lo, hi)]
    plus per-link inertials and the tcp offset."""
    r = ET.parse(URDF).getroot()
    jmap = {j.get("name"): j for j in r.findall("joint")}
    links = {l.get("name"): l for l in r.findall("link")}
    chain = []
    for i in range(1, 8):
        j = jmap[f"{ARM}_joint{i}"]
        o = j.find("origin"); ax = j.find("axis"); lim = j.find("limit")
        chain.append({
            "joint": f"{ARM}_joint{i}", "child": j.find("child").get("link"),
            "xyz": _f(o.get("xyz")), "rpy": _f(o.get("rpy")), "axis": _f(ax.get("xyz"), "0 0 1"),
            "lo": float(lim.get("lower")), "hi": float(lim.get("upper")),
        })
    tcpj = jmap[f"{ARM}_hand_tcp_joint"]
    tcp_xyz = _f(tcpj.find("origin").get("xyz"))

    def inert(link_name):
        l = links.get(link_name); ine = l.find("inertial") if l is not None else None
        if ine is None:
            return None
        m = float(ine.find("mass").get("value"))
        com = _f(ine.find("origin").get("xyz")) if ine.find("origin") is not None else [0, 0, 0]
        I = ine.find("inertia")
        full = [float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        return {"mass": max(m, 1e-4), "com": com, "full": full}
    return chain, tcp_xyz, inert


def _arm_body_xml(chain, tcp_xyz, inert) -> str:
    """Nested MJCF bodies link1..link7 (+ tip) as a string, from the chain."""
    parts = []
    # figure out each link's child offset (for the visual capsule fromto)
    for i, seg in enumerate(chain):
        nxt = chain[i + 1]["xyz"] if i + 1 < len(chain) else tcp_xyz
        parts.append((seg, nxt))
    lines, indent = [], "      "
    close = []
    for depth, (seg, nxt) in enumerate(parts):
        w, x, y, z = _rpy_to_quat(*seg["rpy"])
        pad = indent + "  " * depth
        lines.append(f'{pad}<body name="{seg["child"]}" pos="{seg["xyz"][0]:.6f} {seg["xyz"][1]:.6f} '
                     f'{seg["xyz"][2]:.6f}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">')
        lines.append(f'{pad}  <joint name="{seg["joint"]}" type="hinge" '
                     f'axis="{seg["axis"][0]:.0f} {seg["axis"][1]:.0f} {seg["axis"][2]:.0f}" '
                     f'range="{seg["lo"]:.4f} {seg["hi"]:.4f}" damping="5.0"/>')
        ic = inert(seg["child"])
        if ic:
            lines.append(f'{pad}  <inertial pos="{ic["com"][0]:.5f} {ic["com"][1]:.5f} {ic["com"][2]:.5f}" '
                         f'mass="{ic["mass"]:.5f}" fullinertia="{" ".join(f"{v:.7f}" for v in ic["full"])}"/>')
        ln = math.sqrt(sum(c * c for c in nxt))
        if ln < 0.02:                                       # child ~coincident: sphere, not a capsule
            lines.append(f'{pad}  <geom type="sphere" size="0.02" '
                         f'rgba="0.7 0.72 0.75 1" contype="0" conaffinity="0"/>')
        else:
            lines.append(f'{pad}  <geom type="capsule" fromto="0 0 0 {nxt[0]:.5f} {nxt[1]:.5f} {nxt[2]:.5f}" '
                         f'size="{min(0.022, ln/2):.4f}" rgba="0.7 0.72 0.75 1" contype="0" conaffinity="0"/>')
        close.append(pad)
    # stethoscope tip at the tcp offset of the last link: a contact disk
    pad = indent + "  " * len(parts)
    lines.append(f'{pad}<body name="stetho_tip" pos="{tcp_xyz[0]:.6f} {tcp_xyz[1]:.6f} {tcp_xyz[2]:.6f}">')
    lines.append(f'{pad}  <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>')
    lines.append(f'{pad}  <geom name="stetho_disk" type="cylinder" size="0.026 0.00125" '
                 f'rgba="0.05 0.05 0.05 1" contype="1" conaffinity="1" '
                 f'solref="0.02 1" solimp="0.9 0.95 0.001"/>')
    lines.append(f'{pad}  <site name="tip" pos="0 0 0" size="0.004"/>')
    lines.append(f'{pad}</body>')
    for pad in reversed(close):
        lines.append(f'{pad}</body>')
    return "\n".join(lines)


def build(out: str = OUT) -> str:
    chain, tcp_xyz, inert = _chain()
    arm_xml = _arm_body_xml(chain, tcp_xyz, inert)
    base = chain[0]  # link1 hangs under the world-fixed base; put base at origin
    acts = "\n".join(
        f'    <position name="act_{s["joint"]}" joint="{s["joint"]}" kp="55" '
        f'ctrlrange="{s["lo"]:.4f} {s["hi"]:.4f}"/>' for s in chain)
    # provisional scene to FK the home tip so we can place the back surface within reach
    prov = _SCENE.format(arm=arm_xml, back="", acts=acts, breath="")
    import mujoco
    m = mujoco.MjModel.from_xml_string(prov)
    d = mujoco.MjData(m)
    d.qpos[:7] = Q_HOME
    mujoco.mj_forward(m, d)
    tip = d.site("tip").xpos.copy()
    # place a soft 'back' box just beyond the tip along the tip's approach (world -y here), breathing on y
    back_pos = tip + np.array([0.0, 0.03, 0.0])
    back = (f'    <body name="back" pos="{back_pos[0]:.4f} {back_pos[1]:.4f} {back_pos[2]:.4f}">\n'
            f'      <joint name="breath" type="slide" axis="0 -1 0" range="-0.02 0.02" damping="2"/>\n'
            f'      <geom name="back_surf" type="box" size="0.12 0.01 0.12" rgba="0.80 0.52 0.48 1" '
            f'contype="1" conaffinity="1" solref="0.15 1" solimp="0.2 0.7 0.02"/>\n'  # soft tissue
            f'    </body>')
    breath = '    <position name="act_breath" joint="breath" kp="60" ctrlrange="-0.02 0.02"/>'
    xml = _SCENE.format(arm=arm_xml, back=back, acts=acts + "\n" + breath, breath="")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(xml)
    return out


_SCENE = """<mujoco model="openarm_left_auscultation">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="320" offheight="240"/></visual>
  <default><geom friction="0.9 0.02 0.002" condim="3"/></default>
  <worldbody>
    <light pos="0.3 -0.4 0.6" dir="-0.3 0.4 -0.6"/>
    <camera name="clinician" pos="0.55 -0.35 0.45" xyaxes="0.5 0.8 0 -0.4 0.25 0.85"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1" rgba="0.3 0.3 0.32 1" contype="0" conaffinity="0"/>
    <body name="{ARM_BASE}" pos="0 0 0.35">
{arm}
    </body>
{back}
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>""".replace("{ARM_BASE}", ARM + "_base")


if __name__ == "__main__":
    p = build()
    import mujoco
    m = mujoco.MjModel.from_xml_path(p)
    d = mujoco.MjData(m)
    d.qpos[:7] = Q_HOME
    mujoco.mj_forward(m, d)
    print(f"MJCF ok: {p}")
    print(f"  nq={m.nq} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    print(f"  home tip xpos = {d.site('tip').xpos.round(3)}  (arm base at z=0.35)")
    print(f"  back geom present: {'back_surf' in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]}")

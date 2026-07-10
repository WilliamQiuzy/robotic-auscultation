"""Mock teleop — emit a robot.csv exactly like the fixed unilateral_control REC_CSV, but with no arm.

Writes `$REC_CSV` (default robot.csv) for `seconds`, ~100 Hz, in the format
`t,q1..q7,a1..a7,grip` where t = ABSOLUTE time.monotonic() (the same CLOCK_MONOTONIC the real teleop
now uses and record_session.py stamps on), q = follower (obs), a = leader (action, slightly ahead +
moving). Lets `record_teleop_session.sh` be dress-rehearsed end-to-end (sensors + robot.csv → one
aligned session) before trusting the real hardware.

    REC_CSV=/path/robot.csv python scripts/mock_teleop.py 10
"""
import math
import os
import sys
import time

Q_HOME = [0.0, -0.8, 0.0, 1.2, 0.0, 0.3, 0.0]
Q_REACH = [0.15, 0.10, -0.10, 1.55, 0.05, 0.95, -0.05]


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    path = os.environ.get("REC_CSV", "robot.csv")
    with open(path, "w") as f:
        f.write("t,q1,q2,q3,q4,q5,q6,q7,a1,a2,a3,a4,a5,a6,a7,grip\n")
        t0 = time.monotonic(); t_end = t0 + secs
        print(f"[mock_teleop] logging obs+action (absolute monotonic) -> {path} for {secs}s", flush=True)
        while True:
            now = time.monotonic()
            if now >= t_end:
                break
            ph = 0.5 * (1 - math.cos(2 * math.pi * (now - t0) / secs))     # smooth 0→1→0 reach
            a = [h + ph * (r - h) for h, r in zip(Q_HOME, Q_REACH)]         # leader action (moving)
            q = [av - 0.01 for av in a]                                     # follower obs lags leader
            f.write(f"{now:.4f}," + ",".join(f"{v:.5f}" for v in q) + ","
                    + ",".join(f"{v:.5f}" for v in a) + ",0.08\n")
            f.flush()
            time.sleep(0.01)                                               # ~100 Hz


if __name__ == "__main__":
    main()

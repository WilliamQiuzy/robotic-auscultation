"""End-to-end acceptance test for the IL sim-data pipeline — locks in the Phase-0/1 invariants.

Runs the whole chain on a fresh procedural (Phase-0) session — synthetic session → LeRobot
conversion (proxy Q) → reward-weighted BC — and asserts everything Phases 0/1 established. Uses the
pure-numpy generator + proxy Q (no MuJoCo/AST), so it protects the pipeline against regressions
without heavy/native deps.

    python scripts/test_il_pipeline.py            # prints PASS/FAIL, exit 0/1
    pytest scripts/test_il_pipeline.py            # also works as a pytest

Needs the dev deps (cv2, soundfile, pandas, torch) present in the anaconda env.
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # auscultation/
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ausc"), os.path.join(ROOT, "ausc", "audio")]


def test_il_pipeline_e2e():
    import numpy as np
    import pandas as pd

    import gen_synthetic_session as gen
    import session_to_lerobot as conv
    import train_bc_smoke as bc
    from reward_schema import load_segments, validate

    with tempfile.TemporaryDirectory() as tmp:
        sess = gen.generate(os.path.join(tmp, "session_test"), seconds=16.0, control_hz=50.0, seed=0)
        out = os.path.join(tmp, "lerobot_test")
        info = conv.convert(sess, out)                                   # proxy Q (fast, no AST)

        # 1) no training-ready field is missing (robot state/action + audio present)
        assert info["MISSING"] == [], f"MISSING fields: {info['MISSING']}"

        # 2) segments: at least one good-contact segment with reward>0, all validate
        segs = load_segments(os.path.join(out, "segments.jsonl"))
        for s in segs:
            validate(s)
        assert any(s.contact.gate and (s.reward or 0) > 0 for s in segs), "no gate=True reward>0 segment"

        # 3) frames: every frame joins a valid segment, carries a weight, contiguous, 7-DOF state+action
        df = pd.read_parquet(os.path.join(out, "frames.parquet"))
        assert set(df.segment_index) == set(range(len(segs))), "frames do not cover all segments"
        assert df["reward.weight"].notna().all(), "some frames missing reward.weight"
        assert (df.frame_index.to_numpy() == np.arange(len(df))).all(), "frame_index not contiguous"
        st = np.array(df["observation.state"].tolist()); ac = np.array(df["action"].tolist())
        assert st.shape[1] == 7 and ac.shape[1] == 7, "state/action not 7-DOF"

        # 4) reward weight concentrates in the good-contact window
        g = df["observation.contact_gate"]
        assert df.loc[g, "reward.weight"].mean() > df.loc[~g, "reward.weight"].mean(), \
            "reward weight not concentrated in contact"

        # 5) trainable: reward-weighted BC converges
        m = bc.smoke(out, epochs=150)
        assert m["drop_x"] > 10, f"BC did not converge (only {m['drop_x']:.0f}x)"

    return True


if __name__ == "__main__":
    try:
        test_il_pipeline_e2e()
        print("IL PIPELINE E2E: PASS")
        sys.exit(0)
    except AssertionError as e:
        print(f"IL PIPELINE E2E: FAIL — {e}")
        sys.exit(1)

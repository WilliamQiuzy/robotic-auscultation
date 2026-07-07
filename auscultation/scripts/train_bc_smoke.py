#!/usr/bin/env python3
"""Reward-weighted BC smoke test — the 'is the data trainable?' acceptance check (Phase 0).

Loads a converted session's frames.parquet (from session_to_lerobot.py), trains a tiny MLP
`observation.state (+contact force) -> action` with per-frame `reward.weight` as sample weights
(filtered/reward-weighted BC, docs/audio_reward_design.md §3), and reports that the weighted loss
converges. This validates the training MECHANICS end-to-end (data loads, weights apply, loss drops);
it is NOT a policy-quality eval — the synthetic data is trivially learnable by construction.

    python scripts/train_bc_smoke.py data/lerobot/session_sim0001

Needs: pandas, torch.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def smoke(lerobot_dir: str, epochs: int = 300) -> dict:
    import torch
    import torch.nn as nn

    df = pd.read_parquet(os.path.join(lerobot_dir, "frames.parquet"))
    if "action" not in df.columns:
        raise SystemExit("frames have no `action` — this session lacks robot data (teleop unwired).")
    state = np.array(df["observation.state"].tolist(), np.float32)
    force = df["observation.contact_force_n"].fillna(0).to_numpy(np.float32)[:, None]
    X = torch.tensor(np.concatenate([state, force], 1))            # obs = joints + contact force
    y = torch.tensor(np.array(df["action"].tolist(), np.float32))  # action = leader target joints
    w = torch.tensor(df["reward.weight"].fillna(0).to_numpy(np.float32))

    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    l0 = None
    for ep in range(epochs):
        opt.zero_grad()
        per = ((net(X) - y) ** 2).mean(1)                         # per-sample MSE
        loss = (w * per).sum() / (w.sum() + 1e-8)                 # reward-weighted BC
        loss.backward(); opt.step()
        if ep == 0:
            l0 = float(loss)
    mask = (w > 0.05).numpy()
    rmse = float((((net(X).detach().numpy() - y.numpy())[mask]) ** 2).mean() ** 0.5)
    return {"loss_start": l0, "loss_end": float(loss), "drop_x": l0 / (float(loss) + 1e-12),
            "contact_frames": int(mask.sum()), "contact_action_rmse_rad": rmse}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lerobot_dir")
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    m = smoke(a.lerobot_dir, a.epochs)
    print(f"reward-weighted BC: loss {m['loss_start']:.5f} -> {m['loss_end']:.6f} "
          f"({m['drop_x']:.0f}x)  contact-frame RMSE {m['contact_action_rmse_rad']:.4f} rad "
          f"(n={m['contact_frames']})")
    print("TRAINABLE ✓ (training mechanics validated; not a policy-quality eval)")

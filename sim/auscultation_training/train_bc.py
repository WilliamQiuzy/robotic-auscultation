"""Train a small CPU-only supervised BC baseline from an accepted dataset.

Reports held-out action error only. Training loss is neither closed-loop
success nor a contact-safety/acoustic result. No simulator or reward is used.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import ACTION_LIMIT_RAD, FRAME_FEATURES, HISTORY_STEPS, sha256
from .policy import BoundedJointPolicy


def fit_supervised(policy, observations, actions, epochs=50, batch_size=256, learning_rate=.001, seed=0):
    """Fixed-epoch optimizer; held-out actions never select weights or scalers."""
    if epochs <= 0 or batch_size <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive training settings required")
    x = torch.as_tensor(observations, dtype=torch.float32, device="cpu")
    y = torch.as_tensor(actions, dtype=torch.float32, device="cpu")
    if len(x) == 0 or y.shape != (len(x), 7) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("invalid supervised arrays")
    if torch.max(torch.abs(y)).item() > ACTION_LIMIT_RAD + 1e-8:
        raise ValueError("supervised action exceeds bound")
    policy.to("cpu").train()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            indices = permutation[start:start+batch_size]
            optimizer.zero_grad()
            loss = torch.mean(((policy(x[indices]) - y[indices]) / ACTION_LIMIT_RAD) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
                raise ValueError('nonfinite trained parameter')
        with torch.no_grad():
            total, count = 0., 0
            for start in range(0, len(x), batch_size):
                error = (policy(x[start:start+batch_size])-y[start:start+batch_size])/ACTION_LIMIT_RAD
                total += torch.sum(error**2).item()
                count += error.numel()
            if not np.isfinite(total/count):
                raise ValueError('nonfinite final training loss')
            losses.append(total/count)
    return losses


def action_metrics(policy, observations, actions, batch_size=1024):
    policy.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            predictions.append(policy(torch.as_tensor(observations[start:start+batch_size], dtype=torch.float32)).numpy())
    error = np.concatenate(predictions) - actions
    return {"sample_count": len(actions), "mae_rad": float(np.abs(error).mean()),
            "rmse_rad": float(np.sqrt(np.mean(error**2))),
            "per_joint_mae_rad": np.abs(error).mean(axis=0).tolist(),
            "max_abs_error_rad": float(np.max(np.abs(error))),
            "scope": "held-out action prediction only; no closed-loop success measured"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    # This entry point is deliberately bounded for local CPU debugging.
    if not 1 <= args.epochs <= 500 or not 1 <= args.batch_size <= 4096 or not 1 <= args.threads <= 4:
        parser.error("CPU debug limits: epochs 1..500, batch size 1..4096, threads 1..4")
    manifest_bytes = (args.dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    arrays_bytes = (args.dataset / "dataset.npz").read_bytes()
    if sha256(arrays_bytes) != manifest.get("arrays_sha256"):
        raise ValueError("dataset array fingerprint mismatch")
    if (manifest.get("schema") != "auscultation_bc_dataset_v1"
            or manifest.get("frame_features") != list(FRAME_FEATURES)
            or manifest.get("history_steps") != HISTORY_STEPS
            or manifest.get("action_limit_rad") != ACTION_LIMIT_RAD
            or manifest.get("scaler", {}).get("fit_split") != "train_only"):
        raise ValueError("unsupported dataset observation/action schema")
    # Use the validated bytes, preventing a concurrent file replacement.
    import io
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as data:
        x, y, split = data["observations"], data["actions"], data["split"]
        episode_indices = data["episode_index"]
    if x.shape != (len(y), HISTORY_STEPS, len(FRAME_FEATURES)) or y.shape != (len(x), 7) or split.shape != (len(x),):
        raise ValueError("invalid dataset array shapes")
    if len(x) > 200000:
        raise ValueError("dataset exceeds 200000-sample CPU debug budget")
    if not np.isin(split, [0, 1]).all() or not (split == 0).any() or not (split == 1).any():
        raise ValueError("missing train/evaluation split")
    if episode_indices.shape != split.shape:
        raise ValueError("missing episode indices")
    for index, item in enumerate(manifest["episodes"]):
        mask = episode_indices == index
        expected = 0 if item["split"] == "train" else 1
        if not mask.any() or not np.all(split[mask] == expected) or mask.sum() != item["samples"]:
            raise ValueError("episode split mismatch")
    groups = {}
    for item in manifest["episodes"]:
        groups.setdefault(item["group_id"], set()).add(item["split"])
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError("scenario/seed group leakage")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite dataset")
    training_values = x[split == 0].reshape(int((split == 0).sum()), -1)
    expected_mean, expected_std = training_values.mean(axis=0), training_values.std(axis=0)
    expected_std = np.where(expected_std < 1e-6, 1., expected_std)
    if (np.asarray(manifest["scaler"]["mean"]).shape != expected_mean.shape
            or np.asarray(manifest["scaler"]["std"]).shape != expected_std.shape
            or not np.allclose(manifest["scaler"]["mean"], expected_mean, atol=1e-7, rtol=0)
            or not np.allclose(manifest["scaler"]["std"], expected_std, atol=1e-7, rtol=0)):
        raise ValueError("scaler differs from training-only statistics")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    policy = BoundedJointPolicy(HISTORY_STEPS*len(FRAME_FEATURES), mean=manifest["scaler"]["mean"], std=manifest["scaler"]["std"])
    losses = fit_supervised(policy, x[split == 0], y[split == 0], args.epochs, args.batch_size, args.learning_rate, args.seed)
    metrics = action_metrics(policy, x[split == 1], y[split == 1])
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = {"schema": "auscultation_bc_policy_v1", "state_dict": policy.state_dict(),
        "input_dim": policy.input_dim, "hidden": list(policy.hidden), "history_steps": HISTORY_STEPS,
        "frame_features": list(FRAME_FEATURES), "action_limit_rad": ACTION_LIMIT_RAD,
        "dataset_manifest_sha256": sha256(manifest_bytes)}
    torch.save(checkpoint, args.output / "policy.pt")
    report = {"schema": "auscultation_bc_training_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "seed": args.seed, "threads": args.threads,
        "scope": "supervised simulation-mechanics CPU debug baseline; no rewards or audio",
        "architecture": {"type": "shared_goal_conditioned_MLP", "input_dim": policy.input_dim,
            "hidden": list(policy.hidden), "output_dim": 7, "output": "tanh times 0.0125 rad"},
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "Adam", "loss": "mean squared joint delta error divided by action_limit_rad squared",
        "training_loss": losses, "evaluation_action_metrics": metrics, "closed_loop_evaluated": False,
        "scaler": manifest["scaler"], "action_interpolation": manifest["action_interpolation"],
        "train_episode_ids": [e["episode_id"] for e in manifest["episodes"] if e["split"] == "train"],
        "evaluation_episode_ids": [e["episode_id"] for e in manifest["episodes"] if e["split"] == "evaluation"],
        "dataset_manifest_sha256": sha256(manifest_bytes), "dataset_arrays_sha256": sha256(arrays_bytes),
        "checkpoint_sha256": sha256((args.output / "policy.pt").read_bytes()),
        "code_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()) for name in ("dataset.py", "policy.py", "train_bc.py")},
        "runtime": {"numpy": np.__version__, "torch": torch.__version__}}
    (args.output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "final_training_loss": losses[-1], "held_out_action_mae_rad": metrics["mae_rad"], "closed_loop_evaluated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

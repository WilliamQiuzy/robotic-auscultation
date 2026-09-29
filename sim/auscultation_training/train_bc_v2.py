"""Bounded CPU BC training for verified local-waypoint observations, version 2.

Uses the frozen v1 optimizer and MLP implementation with a versioned encoder.
No held-out selection, reward, simulator, or acoustic feedback is introduced.
"""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
import torch

from .dataset_v2 import ACTION_LIMIT_RAD, FRAME_FEATURES, HISTORY_STEPS, sha256, training_statistics
from .policy import BoundedJointPolicy
from .train_bc import fit_supervised, action_metrics

MODEL_CODE_FILES = ("dataset.py", "dataset_v2.py", "policy.py", "train_bc.py", "train_bc_v2.py")


def verify_annotation_features(observations, sidecar):
    """Bind every history frame's new features to the verified sidecar labels."""
    rows = sidecar.get("rows", [])
    if len(rows) != len(observations):
        raise ValueError("waypoint annotation sample count mismatch")
    target, first = None, 0
    for index, row in enumerate(rows):
        if row.get("control_row") != index or not isinstance(row.get("target"), str) or not row["target"]:
            raise ValueError("invalid waypoint annotation row")
        if row["target"] != target:
            target, first = row["target"], index
        for frame, previous in enumerate(range(index-HISTORY_STEPS+1,index+1)):
            label = rows[max(first,previous)]
            valid = label.get("local_waypoint_valid")
            if type(valid) is not bool:
                raise ValueError("invalid waypoint validity label")
            if valid:
                goal = np.asarray(label.get("local_waypoint_q"), float)
                if goal.shape != (7,) or not np.isfinite(goal).all():
                    raise ValueError("invalid waypoint joint label")
                error = goal-observations[index,frame,:7]
            else:
                if label.get("local_waypoint_q") is not None:
                    raise ValueError("absent waypoint has non-null label")
                error = np.zeros(7)
            scale = label.get("speed_scale")
            if type(scale) not in (int,float) or not .7<=scale<=1:
                raise ValueError("invalid waypoint speed label")
            expected = np.r_[error,float(valid),scale]
            if not np.allclose(observations[index,frame,-9:],expected,atol=5e-7,rtol=0):
                raise ValueError("waypoint conditioning differs from bound annotation")


def load_training_data(directory):
    directory = Path(directory)
    manifest_bytes = (directory/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    arrays_bytes = (directory/"dataset.npz").read_bytes()
    if sha256(arrays_bytes) != manifest.get("arrays_sha256"):
        raise ValueError("dataset array fingerprint mismatch")
    if (manifest.get("schema") != "auscultation_bc_dataset_v2"
            or manifest.get("frame_features") != list(FRAME_FEATURES)
            or manifest.get("history_steps") != HISTORY_STEPS
            or manifest.get("action_limit_rad") != ACTION_LIMIT_RAD
            or manifest.get("scaler", {}).get("fit_split") != "train_only"):
        raise ValueError("unsupported v2 dataset schema")
    sidecars = []
    for episode in manifest.get("episodes", []):
        name = episode.get("waypoint_annotation_file", "")
        if not name or Path(name).name != name:
            raise ValueError("invalid waypoint annotation path")
        data = (directory/name).read_bytes()
        if sha256(data) != episode.get("waypoint_annotation_sha256"):
            raise ValueError("waypoint annotation fingerprint mismatch")
        sidecar = json.loads(data)
        if sidecar.get("schema") != "auscultation_waypoint_annotations_v2":
            raise ValueError("unsupported waypoint annotation schema")
        if sidecar.get("source_artifact_sha256") != episode.get("artifact_sha256"):
            raise ValueError("waypoint annotation source binding mismatch")
        sidecars.append(sidecar)
    with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
        arrays = {name:archive[name].copy() for name in archive.files}
    x, y, split, indices = [arrays[name] for name in ("observations", "actions", "split", "episode_index")]
    if (x.shape != (len(y),HISTORY_STEPS,len(FRAME_FEATURES)) or y.shape != (len(x),7)
            or split.shape != (len(x),) or indices.shape != split.shape
            or not 0 < len(x) <= 200000 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.max(np.abs(y)) > ACTION_LIMIT_RAD+1e-8):
        raise ValueError("invalid bounded v2 supervised arrays")
    if not np.isin(split,[0,1]).all() or not (split==0).any() or not (split==1).any():
        raise ValueError("missing train/evaluation split")
    episodes = manifest.get("episodes", [])
    if not episodes or not np.isin(indices,range(len(episodes))).all():
        raise ValueError("invalid episode indices")
    groups = {}
    for index, episode in enumerate(episodes):
        mask = indices==index
        if episode.get("split") not in ("train","evaluation"):
            raise ValueError("invalid episode split")
        if not mask.any() or mask.sum()!=episode["samples"] or not np.all(split[mask]==int(episode["split"]=="evaluation")):
            raise ValueError("episode split mismatch")
        verify_annotation_features(x[mask], sidecars[index])
        groups.setdefault(episode["group_id"],set()).add(episode["split"])
        speeds = episode.get("speed_scales", [])
        if not speeds or any(type(s) not in (int,float) or not .7 <= s <= 1 for s in speeds):
            raise ValueError("invalid episode speed metadata")
        verified_speeds = sorted({row["speed_scale"] for row in sidecars[index]["rows"]})
        if speeds != verified_speeds:
            raise ValueError("episode speed metadata differs from verified annotation rows")
    if any(len(values)!=1 for values in groups.values()):
        raise ValueError("scenario/seed group leakage")
    values = x[split==0].reshape(int((split==0).sum()), -1)
    mean, std = training_statistics(values)
    for key, expected in (("mean",mean),("std",std)):
        actual = np.asarray(manifest["scaler"].get(key), float)
        if actual.shape != expected.shape or not np.allclose(actual,expected,atol=1e-7,rtol=0):
            raise ValueError("scaler differs from training-only statistics")
    return manifest, manifest_bytes, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1<=args.epochs<=500 or not 1<=args.batch_size<=4096 or not 1<=args.threads<=4:
        parser.error("CPU debug limits: epochs 1..500, batch size 1..4096, threads 1..4")
    if args.output.exists():
        raise ValueError("output already exists; frozen models are immutable")
    manifest, manifest_bytes, arrays = load_training_data(args.dataset)
    x,y,split = arrays["observations"],arrays["actions"],arrays["split"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    policy = BoundedJointPolicy(HISTORY_STEPS*len(FRAME_FEATURES), mean=manifest["scaler"]["mean"], std=manifest["scaler"]["std"])
    losses = fit_supervised(policy,x[split==0],y[split==0],args.epochs,args.batch_size,args.learning_rate,args.seed)
    metrics = action_metrics(policy,x[split==1],y[split==1])
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint = {"schema":"auscultation_bc_policy_v2", "state_dict":policy.state_dict(),
        "input_dim":policy.input_dim, "hidden":list(policy.hidden), "history_steps":HISTORY_STEPS,
        "frame_features":list(FRAME_FEATURES), "action_limit_rad":ACTION_LIMIT_RAD,
        "dataset_manifest_sha256":sha256(manifest_bytes)}
    torch.save(checkpoint,args.output/"policy.pt")
    report = {"schema":"auscultation_bc_training_v2", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "device":"cpu", "seed":args.seed, "threads":args.threads,
        "scope":"supervised simulation waypoint/contact BC; no rewards, audio, or closed-loop result",
        "architecture":{"type":"shared_waypoint_conditioned_MLP", "input_dim":policy.input_dim,
            "hidden":list(policy.hidden), "output_dim":7, "output":"tanh times 0.0125 rad"},
        "epochs":args.epochs, "batch_size":args.batch_size, "learning_rate":args.learning_rate,
        "optimizer":"Adam", "loss":"mean squared joint delta error divided by action_limit_rad squared",
        "training_loss":losses, "evaluation_action_metrics":metrics, "closed_loop_evaluated":False,
        "scaler":manifest["scaler"], "action_interpolation":manifest["action_interpolation"],
        "train_episode_ids":[e["episode_id"] for e in manifest["episodes"] if e["split"]=="train"],
        "evaluation_episode_ids":[e["episode_id"] for e in manifest["episodes"] if e["split"]=="evaluation"],
        "training_speed_scales":sorted({s for e in manifest["episodes"] if e["split"]=="train" for s in e["speed_scales"]}),
        "evaluation_speed_scales":sorted({s for e in manifest["episodes"] if e["split"]=="evaluation" for s in e["speed_scales"]}),
        "speed_generalization_claim":False, "dataset_manifest_sha256":sha256(manifest_bytes),
        "dataset_arrays_sha256":manifest["arrays_sha256"],
        "waypoint_annotation_sha256":{e["waypoint_annotation_file"]:e["waypoint_annotation_sha256"] for e in manifest["episodes"]},
        "checkpoint_sha256":sha256((args.output/"policy.pt").read_bytes()),
        "code_sha256":{name:sha256(Path(__file__).with_name(name).read_bytes()) for name in MODEL_CODE_FILES},
        "runtime":{"numpy":np.__version__,"torch":torch.__version__}}
    (args.output/"run_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(args.output), "final_training_loss":losses[-1],
        "held_out_action_mae_rad":metrics["mae_rad"], "closed_loop_evaluated":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

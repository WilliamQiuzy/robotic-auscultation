"""Independent, blind Robometer video review; never a robot controller or RL reward.

The numerical audit remains authoritative for force, collisions, timing and joint
motion. A model's uncalibrated task-progress score cannot certify those properties.
Heavy model dependencies are imported only by the explicit ``score`` command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MODEL_ID = "robometer/Robometer-4B"
MODEL_REVISION = "beef63bc914c5c189329d49c6d712d96d632aa34"
SOURCE_COMMIT = "352d160389daa964788de1ec933d1925f3a6de4f"
TASK = (
    "Move the stethoscope on the robot's left arm toward the indicated point on "
    "the seated person's back, make contact, hold it steadily, and withdraw. "
    "Use a direct, smooth arm movement and avoid unnecessary elbow lifting or twisting."
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_path(parent: Path, relative: str, digest: str) -> Path:
    path = (parent / relative).resolve()
    require(path.is_file(), "missing bound file: " + relative)
    require(len(digest) == 64 and sha256(path) == digest, "file hash mismatch: " + relative)
    return path


def validate_timing(clip: dict, fps: float, physics_dt: float, pose_dt: float) -> None:
    frames = clip["frames"]
    require(math.isfinite(fps) and fps > 0, "invalid fps")
    require(len(frames) >= 2 and clip["frame_count"] == len(frames), "frame count mismatch")
    require(abs(clip["duration_s"] - len(frames) / fps) < 1e-7, "duration is not 1x")
    origin = frames[0]["simulation_interval_end_s"] - 1 / fps
    for index, frame in enumerate(frames):
        require(frame["frame_index"] == index, "noncontiguous frame index")
        values = [frame[k] for k in ("video_time_s", "simulation_interval_end_s", "measurement_time_s", "pose_time_s")]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "nonfinite time")
        require(abs(frame["video_time_s"] - index / fps) < 1e-7, "video time mismatch")
        require(abs(frame["simulation_interval_end_s"] - origin - (index + 1) / fps) < 1e-7, "retimed video rejected")
        age = frame["simulation_interval_end_s"] - frame["measurement_time_s"]
        require(-1e-7 <= age <= physics_dt + 1e-7, "measurement clock mismatch")
        pose_age = frame["measurement_time_s"] - frame["pose_time_s"]
        require(-1e-7 <= pose_age <= pose_dt + 1e-7, "stale recorded pose")
        require(type(frame["measurement_row"]) is int and type(frame["pose_row"]) is int, "row indices must be integers")
        require(0 <= frame["pose_row"] <= frame["measurement_row"], "invalid source row")
        if index:
            require(frame["measurement_row"] > frames[index - 1]["measurement_row"], "nonmonotonic recorded rows")


def verify_trace_rows(path: Path, clips: list[dict]) -> None:
    """Bind sidecar times to the authoritative 240 Hz trace, not self-reported time."""
    expected: dict[int, list[float]] = {}
    for clip in clips:
        for frame in clip["frames"]:
            expected.setdefault(frame["measurement_row"], []).append(frame["measurement_time_s"])
            expected.setdefault(frame["pose_row"], []).append(frame["pose_time_s"])
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index in expected:
                actual = json.loads(line)["time_s"]
                require(all(abs(actual - value) < 1e-8 for value in expected.pop(index)), "trace clock binding mismatch")
            if not expected:
                break
    require(not expected, "frame references absent trace rows")


def validate_review(path: Path, project: Path = PROJECT) -> tuple[dict, list[dict]]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == "blind_recorded_review_v1", "unsupported review schema")
    require(manifest.get("playback_speed") == 1., "only original-clock 1x clips allowed")
    require(manifest.get("no_acceptance_or_model_scores_in_pixels") is True, "blind pixels required")
    require(manifest.get("no_interpolation") is True, "actual recorded poses required")
    require(manifest.get("source_files_unchanged") is True, "render source verification missing")
    sources = {name: checked_path(project, name, digest) for name, digest in manifest["source_sha256"].items()}
    traces = [p for p in sources.values() if p.name == "expert_trace.jsonl"]
    require(len(traces) == 1, "exactly one authoritative trace required")
    clips = manifest["clips"]
    require(bool(clips), "no review clips")
    require(len({c["clip_id"] for c in clips}) == len(clips), "duplicate clip id")
    validated = []
    for index, clip in enumerate(clips):
        require(clip["view"] in ("posture", "robotcam"), "unknown view")
        validate_timing(clip, manifest["fps"], manifest["source_physics_dt_s"], manifest["source_pose_dt_s"])
        video = checked_path(path.parent, clip["video"], clip["video_sha256"])
        validated.append({"blind_id": f"C{index + 1:03d}", "video_path": str(video),
                          "video_sha256": clip["video_sha256"], "clip": clip})
    verify_trace_rows(traces[0], clips)
    return manifest, validated


def sample_indices(frame_count: int, maximum: int = 8) -> list[int]:
    require(type(maximum) is int and 2 <= maximum <= 32, "sample count must be 2..32")
    require(frame_count >= 2, "video too short")
    size = min(frame_count, maximum)
    return [round(i * (frame_count - 1) / (size - 1)) for i in range(size)]


def phase_anchor_indices(frames: list[dict], maximum: int = 8) -> list[int]:
    """Include each contact stage's midpoint; metadata selects pixels, never labels the judge."""
    require(5 <= maximum <= 32, "phase sampling needs at least five frames")
    anchors = {0, len(frames) - 1}
    for phase in ("CONTACT", "HOLD", "RETRACT"):
        candidates = [i for i, frame in enumerate(frames) if frame.get("phase") == phase]
        if candidates:
            anchors.add(candidates[len(candidates) // 2])
    for index in sample_indices(len(frames), maximum):
        if len(anchors) >= min(maximum, len(frames)):
            break
        anchors.add(index)
    for index in range(len(frames)):
        if len(anchors) >= min(maximum, len(frames)):
            break
        anchors.add(index)
    return sorted(anchors)


def decode_samples(item: dict, fps: float, resolution: list[int], maximum: int, sampling: str = "uniform"):
    import cv2
    import numpy as np
    clip = item["clip"]
    indices = (phase_anchor_indices(clip["frames"], maximum) if sampling == "phase_anchors"
               else sample_indices(clip["frame_count"], maximum))
    capture = cv2.VideoCapture(item["video_path"])
    require(capture.isOpened(), "video decoder did not open")
    try:
        require(abs(capture.get(cv2.CAP_PROP_FPS) - fps) < 1e-4, "encoded fps mismatch")
        require(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == clip["frame_count"], "encoded frame count mismatch")
        require([round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))] == resolution, "encoded dimensions mismatch")
        samples = []
        wanted = set(indices)
        # Sequential decode confirms complete media, not only the header metadata.
        count = 0
        while True:
            ok, pixels = capture.read()
            if not ok:
                break
            if count in wanted:
                samples.append(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            count += 1
        require(count == clip["frame_count"], "incomplete encoded stream")
        require(len(samples) == len(indices), "missing sampled frames")
        result = np.stack(samples)
        return result, {"policy": sampling, "frame_indices": indices,
                        "measurement_time_s": [clip["frames"][i]["measurement_time_s"] for i in indices],
                        "sampled_phases_for_audit_only": [clip["frames"][i].get("phase") for i in indices],
                        "decoded_rgb_sha256": hashlib.sha256(result.tobytes()).hexdigest()}
    finally:
        capture.release()


def pair_consistency(probability_ab: float, probability_ba: float, tolerance: float = .15) -> dict:
    """Probabilities refer to first versus second; swapped results should complement."""
    require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                for p in (probability_ab, probability_ba)), "invalid preference probability")
    residual = abs(probability_ab + probability_ba - 1.)
    return {"p_first_ab": probability_ab, "p_first_ba": probability_ba,
            "swap_complement_error": residual, "order_consistent": residual <= tolerance,
            "symmetrized_p_a": (probability_ab + 1. - probability_ba) / 2.,
            "scope": "task preference only; not force, safety, motion quality or clinical validation"}


def decision(hard_status: str, model_ran: bool, calibration_passed: bool = False,
             calibrated_visual_status: str = "unknown") -> dict:
    require(hard_status in ("pass", "fail", "unknown"), "invalid hard status")
    require(calibrated_visual_status in ("pass", "fail", "unknown"), "invalid visual status")
    if hard_status == "fail":
        return {"status": "fail", "reason": "hard_audit_failure_cannot_be_overridden"}
    if not model_ran or not calibration_passed:
        return {"status": "unknown", "reason": "model_not_run_or_domain_calibration_incomplete"}
    if calibrated_visual_status == "fail":
        return {"status": "fail", "reason": "calibrated_visual_review_failed"}
    if hard_status != "pass" or calibrated_visual_status != "pass":
        return {"status": "unknown", "reason": "required_review_incomplete"}
    return {"status": "pass", "reason": "bound_hard_and_calibrated_visual_reviews_passed"}


def verify_weight_keys(checkpoint_keys: set, model_keys: set) -> list[str]:
    # The official released checkpoint retains this old, unused head; the pinned
    # current RBM code implements exactly progress, preference, and success heads.
    legacy = {f"similarity_head.{layer}.{part}" for layer in (0, 1, 4) for part in ("weight", "bias")}
    missing, extra = model_keys - checkpoint_keys, checkpoint_keys - model_keys
    require(not missing and extra <= legacy,
            "checkpoint/model key mismatch: missing=" + str(sorted(missing)) + "; extra=" + str(sorted(extra)))
    return sorted(extra)


class RobometerJudge:
    """Official frozen model, ordinary Torch branch; inference only, no logging service."""
    def __init__(self, weights: Path, base: Path, device: str):
        import torch
        import yaml
        from dataclasses import fields
        from robometer.configs.experiment_configs import ExperimentConfig
        from robometer.utils.setup_utils import setup_model_and_processor, setup_batch_collator
        config = yaml.safe_load((weights / "config.yaml").read_text())
        config = {k: v for k, v in config.items() if k in {f.name for f in fields(ExperimentConfig)}}
        self.config = ExperimentConfig(**config)
        require(not self.config.model.use_peft, "unexpected adapter checkpoint")
        # The released checkpoint stores a training-time Unsloth setting. Select
        # the official ordinary Torch path explicitly; trained weights are unchanged.
        self.config.model.use_unsloth = False
        # Only resolve the same pinned base locally; all trained RBM weights are then loaded.
        self.config.model.base_model_id = str(base.resolve())
        self.tokenizer, processor, self.model = setup_model_and_processor(self.config.model, str(weights.resolve()))
        from safetensors import safe_open
        mapping = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]
        state = self.model.state_dict()
        unused_keys = verify_weight_keys(set(mapping), set(state))
        for shard in sorted(set(mapping.values())):
            with safe_open(str(weights / shard), framework="pt", device="cpu") as loaded:
                for key in loaded.keys():
                    if key in unused_keys:
                        continue
                    expected = loaded.get_tensor(key)
                    require(torch.equal(state[key].detach().cpu(), expected.to(state[key].dtype)),
                            "loaded tensor differs from frozen checkpoint: " + key)
        self.weight_verification = {"checkpoint_keys": len(mapping), "verified_model_keys": len(state),
            "legacy_unused_checkpoint_keys": unused_keys, "all_loaded_tensors_exact_after_dtype_conversion": True}
        self.device = torch.device(device)
        self.model.to(self.device).eval().requires_grad_(False)
        self.collator = setup_batch_collator(processor, self.tokenizer, self.config, is_eval=True)

    def _trajectory(self, frames, identity: str):
        from robometer.data.dataset_types import Trajectory
        return Trajectory(frames=frames, frames_shape=tuple(frames.shape), task=TASK,
                          id=identity, metadata={"subsequence_length": len(frames)}, video_embeddings=None)

    def _infer(self, sample, kind: str) -> dict:
        import torch
        from robometer.evals.eval_server import compute_batch_outputs
        batch = self.collator([sample])[kind + "_inputs"]
        # Labels are collator placeholders, never human quality labels or source audit outcomes.
        batch = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
        with torch.inference_mode():
            result = compute_batch_outputs(self.model, self.tokenizer, batch, sample_type=kind,
                is_discrete_mode=self.config.loss.progress_loss_type.lower() == "discrete",
                num_bins=self.config.loss.progress_discrete_bins or self.config.model.progress_discrete_bins)
        # Reject nonfinite head output rather than serialize NaN as an unexplained score.
        json.dumps(result, allow_nan=False)
        return result

    def single(self, frames, identity: str) -> dict:
        from robometer.data.dataset_types import ProgressSample
        return self._infer(ProgressSample(trajectory=self._trajectory(frames, identity)), "progress")

    def pair(self, first, second) -> dict:
        from robometer.data.dataset_types import PreferenceSample
        return self._infer(PreferenceSample(chosen_trajectory=self._trajectory(first, "A"),
                            rejected_trajectory=self._trajectory(second, "B")), "preference")


def probability(result: dict) -> float:
    values = result.get("prediction_probs")
    require(isinstance(values, list) and len(values) == 1, "missing single pair probability")
    value = values[0]
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    require(type(value) in (int, float), "preference probability is not scalar")
    return float(value)


def write_new(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def model_binding(weights: Path, base: Path, source: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(commit == SOURCE_COMMIT, "unreviewed official source revision")
    def files(root):
        return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                if p.is_file() and ".cache" not in p.parts and p.suffix in (".json", ".yaml", ".safetensors", ".txt", ".jinja")}
    require(bool(list(weights.glob("*.safetensors"))), "model weights absent")
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_commit": commit,
            "model_files_sha256": files(weights), "base_files_sha256": files(base),
            "base_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "runtime_overrides": {"use_unsloth": False, "base_model_id": "pinned local Qwen3-VL-4B-Instruct", "reason": "official standard Torch inference branch, unchanged trained weights"},
            "source_python_sha256": {str(p.relative_to(source)): sha256(p) for p in sorted((source / "robometer").rglob("*.py"))},
            "adapter_sha256": sha256(Path(__file__)), "task": TASK,
            "protocol_sha256": sha256(ROOT / "judge_protocol.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "score"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--sampling", choices=("uniform", "phase_anchors"), default="phase_anchors")
    parser.add_argument("--calibration-controls", action="store_true")
    parser.add_argument("--clip-ids", nargs="+")
    parser.add_argument("--negative", choices=("frozen", "truncated"), default="frozen")
    parser.add_argument("--reference-manifest", type=Path,
                        help="Optional independently bound reference clips with identical point/view IDs")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists; results are immutable")
    manifest_digest = sha256(args.manifest)
    manifest, clips = validate_review(args.manifest, args.project)
    if args.clip_ids:
        require(set(args.clip_ids) <= {c["clip"]["clip_id"] for c in clips}, "requested clip missing")
        clips = [c for c in clips if c["clip"]["clip_id"] in args.clip_ids]
    report = {"schema": "wam_visual_judge_result_v1", "input_manifest_sha256": manifest_digest,
              "model_ran": False, "training_or_rl_reward": False, "input_source_sha256": manifest["source_sha256"],
              "clips": [], "pairs": [], "calibration": {"status": "not_run"},
              "decision": decision("unknown", False), "hard_audit_status": "unknown"}
    decoded = {}
    for item in clips:
        frames, sample = decode_samples(item, manifest["fps"], manifest["resolution"], args.sample_frames, args.sampling)
        decoded[item["blind_id"]] = frames
        report["clips"].append({"blind_id": item["blind_id"], "source_clip_id": item["clip"]["clip_id"],
              "view": item["clip"]["view"], "video_sha256": item["video_sha256"], "sampling": sample})
    reference_decoded = {}
    if args.reference_manifest:
        reference_digest = sha256(args.reference_manifest)
        reference_manifest, reference_clips = validate_review(args.reference_manifest, args.project)
        references = {c["clip"]["clip_id"]: c for c in reference_clips}
        report["reference"] = {"manifest_sha256": reference_digest,
                               "source_sha256": reference_manifest["source_sha256"], "clips": []}
        for item in clips:
            original = item["clip"]
            require(original["clip_id"] in references, "matching point/view absent in reference")
            reference = references[original["clip_id"]]
            require(reference["clip"]["target"] == original["target"] and reference["clip"]["view"] == original["view"],
                    "reference comparison target/view mismatch")
            frames, sample = decode_samples(reference, reference_manifest["fps"], reference_manifest["resolution"],
                                            args.sample_frames, args.sampling)
            reference_decoded[item["blind_id"]] = frames
            report["reference"]["clips"].append({"candidate_blind_id": item["blind_id"],
                  "video_sha256": reference["video_sha256"], "sampling": sample})
    if args.command == "score":
        require(all(p is not None and p.is_dir() for p in (args.weights, args.base, args.source)), "local pinned model/base/source required")
        import os
        import sys
        os.environ.update(WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(args.source.resolve()))
        report["model_binding"] = model_binding(args.weights, args.base, args.source)
        snapshot = args.output.parent / ("visual_judge_" + report["model_binding"]["adapter_sha256"] + ".py")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            with snapshot.open("xb") as stream:
                stream.write(Path(__file__).read_bytes())
        require(sha256(snapshot) == report["model_binding"]["adapter_sha256"], "adapter snapshot mismatch")
        report["model_binding"]["adapter_snapshot"] = snapshot.name
        protocol_snapshot = args.output.parent / ("judge_protocol_" + report["model_binding"]["protocol_sha256"] + ".json")
        if not protocol_snapshot.exists():
            with protocol_snapshot.open("xb") as stream:
                stream.write((ROOT / "judge_protocol.json").read_bytes())
        require(sha256(protocol_snapshot) == report["model_binding"]["protocol_sha256"], "protocol snapshot mismatch")
        report["model_binding"]["protocol_snapshot"] = protocol_snapshot.name
        judge = RobometerJudge(args.weights, args.base, args.device)
        report["model_binding"]["loaded_weight_verification"] = judge.weight_verification
        for clip in report["clips"]:
            identity = clip["blind_id"]
            clip["raw_model_outputs"] = judge.single(decoded[identity], identity)
            print("SCORED", identity, flush=True)
            if args.reference_manifest:
                reference = reference_decoded[identity]
                ab = judge.pair(decoded[identity], reference)
                ba = judge.pair(reference, decoded[identity])
                report["pairs"].append({"pair_kind": "candidate_vs_bound_reference", "source_blind_id": identity,
                    "synthetic_control": False, "raw_ab": ab, "raw_ba": ba,
                    "consistency": pair_consistency(probability(ab), probability(ba))})
            if args.calibration_controls:
                import numpy as np
                # Derived diagnostic controls, never mislabeled as executed failure episodes.
                negative = (np.repeat(decoded[identity][:1], len(decoded[identity]), axis=0)
                            if args.negative == "frozen" else decoded[identity][:max(2, len(decoded[identity]) // 2)])
                ab = judge.pair(decoded[identity], negative)
                ba = judge.pair(negative, decoded[identity])
                report["pairs"].append({"pair_kind": "exploratory_real_vs_" + args.negative, "source_blind_id": identity,
                    "synthetic_control": True, "control_frame_count": len(negative),
                    "control_rgb_sha256": hashlib.sha256(negative.tobytes()).hexdigest(),
                    "raw_ab": ab, "raw_ba": ba, "consistency": pair_consistency(probability(ab), probability(ba))})
        report["model_ran"] = True
        report["calibration"] = {"status": "exploratory_only" if args.calibration_controls else "not_run",
              "domain_calibration_passed": False,
              "negative_type": args.negative if args.calibration_controls else None,
              "reason": (("frozen-first-frame controls test no-motion sensitivity only" if args.negative == "frozen" else
                          "truncated sampled-prefix controls test progress sensitivity with a frame-count confound")
                         if args.calibration_controls else "no diagnostic controls requested") +
                        "; physical collision/twist/force defects require separately labeled clips"}
        report["decision"] = decision("unknown", True, False)
        # Re-check media, all bound physics sources and the input sidecar after inference.
        validate_review(args.manifest, args.project)
        require(sha256(args.manifest) == manifest_digest, "input manifest changed during model scoring")
        if args.reference_manifest:
            validate_review(args.reference_manifest, args.project)
            require(sha256(args.reference_manifest) == reference_digest, "reference changed during model scoring")
    write_new(args.output, report)
    print(json.dumps({"output": str(args.output), "model_ran": report["model_ran"], "decision": report["decision"]}))


if __name__ == "__main__":
    main()

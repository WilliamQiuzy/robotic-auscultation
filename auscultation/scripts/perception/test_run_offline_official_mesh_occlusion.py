from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import mujoco
import numpy as np
import pytest

from ausc.perception.tracking import offline_official_mesh_experiment as core_experiment
from ausc.perception.tracking.offline_mesh_simulation import (
    GroundTruthFrame,
    ObservationFrame,
    POLICY_NAMES,
    RequiredStratum,
    evaluate_occlusion_policies,
    score_occlusion_policy_metrics,
)
from scripts.perception import run_offline_official_mesh_occlusion as runner


REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_RAW_ROOT = REPO_ROOT / "docs/runs/2026-07-15-offline-official-mesh-occlusion/raw"

STATUS = "OFFLINE_METHOD_SIMULATION_ONLY"
SCOPE = "offline_method_simulation_only"
RUNTIME_MANIFEST_SHA256 = (
    "7e3307a163e8b9e6c3c32cb038f442171d0d23794182cffb1aa1c1cb983ef346"
)
ASSET_MANIFEST_SHA256 = (
    "05e66c94cd8b36ff79bef7bfcaa73d176e621120ac6398e808ad5df938a0cf35"
)
CONVERSION_MANIFEST_SHA256 = (
    "b5e826af2539bceafea1d638503362a2e8811038ad238e29bb9c263a652c3202"
)
OPENARM_COMMIT = "846a82d8baf64efdbb7f55c7f64b9b004939061b"
OPENARM_URDF_SHA256 = "63caf1116a844c7eca25e91f0acf025c2798cc4795cf727b23433eacde80ca12"
NVIDIA_HUMAN_USD_SHA256 = (
    "2c65e15587a311c5d14cdbd6a1408e0c3d8f3bec69f483150f04ca4c8f4e3ccc"
)
POINT_NAMES = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")
OFFICIAL_ROBOT_GEOM_IDS = tuple(range(21))
OFFICIAL_HUMAN_GEOM_IDS = tuple(range(21, 28))
GEOM_OBJECT_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)
FRAME_TIMESTAMPS_S = (0.00, 0.50, 0.55)
GEOMETRY_ACCEPTED = (True, False, True)
COMPLETE_SCENARIO_ID = "R1__w0__occ0500ms"
REPORT_VIDEO_FRAME_INDEX = 2
OBSERVATION_SEED_BASE = 51020260715
PATIENT_MOTION_SEED_BASE = 51020260716
MAX_FREEZE_AGE_S = 0.40
MAX_FREEZE_ERROR_P95_M = 0.010
PATIENT_MOTION_BOUND_M_S = 0.013

FALSE_COMPLETION_GATES = {
    "isaac_simulation_complete": False,
    "physical_collision_contact_validated": False,
    "detector_in_loop_replay_complete": False,
    "hand_eye_calibration_complete": False,
    "force_safety_complete": False,
    "hardware_timing_sync_complete": False,
    "breathing_deformation_validated": False,
    "real_patient_motion_validated": False,
    "production_ready": False,
}

CSV_HEADER = (
    "scenario_id",
    "target_name",
    "wrist_index",
    "occlusion_duration_s",
    "frame_index",
    "timestamp_s",
    "event_name",
    "observation_seed",
    "noise_contract_sha256",
    "observation_stream_sha256",
    "geometry_accepted",
    "visible_point_count_report_only",
    "accepted_estimate_x_m",
    "accepted_estimate_y_m",
    "accepted_estimate_z_m",
    "accepted_error_p95_m",
    "patient_motion_bound_m_s",
    "policy_name",
    "policy_state",
    "episode_id",
    "last_valid_age_s",
    "predicted_error_p95_m",
    "reacquired",
    "deliberately_unsafe",
    "estimate_x_m",
    "estimate_y_m",
    "estimate_z_m",
    "valid",
    "ground_truth_x_m",
    "ground_truth_y_m",
    "ground_truth_z_m",
    "error_m",
    "raw_depth_path",
    "raw_segmentation_path",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _core_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _write_sums(root: Path) -> None:
    destination = root / "SHA256SUMS.txt"
    rows = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != destination
    ]
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _changing_frames(frame_count: int = len(FRAME_TIMESTAMPS_S)) -> np.ndarray:
    yy, xx = np.indices((360, 640), dtype=np.uint16)
    frames = []
    for index in range(frame_count):
        frame = np.empty((360, 640, 3), dtype=np.uint8)
        frame[..., 0] = ((xx + 31 * index) % 256).astype(np.uint8)
        frame[..., 1] = ((yy * 2 + 47 * index) % 256).astype(np.uint8)
        frame[..., 2] = (((xx // 3) + (yy // 2) + 59 * index) % 256).astype(np.uint8)
        frame[110:250, 220 + 12 * index : 280 + 12 * index] = (235, 55, 25)
        cv2.rectangle(frame, (0, 0), (639, 71), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            "OFFLINE METHOD SIM - NOT ISAAC / NOT PRODUCTION",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OpenArm {OPENARM_COMMIT[:12]} | direct | unsafe lock | bounded freeze",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
    return np.stack(frames)


def _encode_video(path: Path, frames: np.ndarray, *, codec: str = "libx264") -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AssertionError("the locked Mac test environment requires ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    codec_args = ["-c:v", codec]
    if codec == "libx264":
        codec_args += ["-preset", "ultrafast", "-crf", "18"]
    else:
        codec_args += ["-q:v", "2"]
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "640x360",
            "-framerate",
            "12",
            "-i",
            "pipe:0",
            "-an",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(path),
        ],
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
        check=True,
        capture_output=True,
        timeout=30,
    )


def _predeclared_matrix_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for point_name in POINT_NAMES:
        for wrist_index in (0, 1):
            for duration in (0.10, 0.25, 0.50, 1.00):
                milliseconds = int(round(duration * 1000.0))
                scenario_id = f"{point_name}__w{wrist_index}__occ{milliseconds:04d}ms"
                complete = scenario_id == COMPLETE_SCENARIO_ID
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "target_name": point_name,
                        "wrist_index": wrist_index,
                        "occlusion_duration_s": duration,
                        "status": "COMPLETE" if complete else "IK_UNREACHABLE",
                        "frame_count": len(FRAME_TIMESTAMPS_S) if complete else 0,
                        "observation_seed": OBSERVATION_SEED_BASE + 10 * index,
                        "patient_motion_seed": PATIENT_MOTION_SEED_BASE + index,
                    }
                )
                index += 1
    return rows


def _input_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    (
        core_contract,
        core_contract_sha256,
    ) = core_experiment.build_experiment_input_contract()
    core_value = dict(core_contract)
    core_value["decision_surface"] = dict(core_contract["decision_surface"])
    core_value = json.loads(_core_canonical_bytes(core_value))
    assert hashlib.sha256(_core_canonical_bytes(core_value)).hexdigest() == (
        core_contract_sha256
    )
    strata = []
    for record in records:
        complete = record["status"] == "COMPLETE"
        direction = core_experiment._patient_direction(record["patient_motion_seed"])
        frames = _frame_specs(record) if complete else []
        strata.append(
            {
                **record,
                "patient_motion_direction_world": direction.tolist(),
                "events": [
                    {
                        key: frame[key]
                        for key in (
                            "frame_index",
                            "event_name",
                            "timestamp_s",
                            "observation_seed",
                            "noise_contract_sha256",
                        )
                    }
                    for frame in frames
                ],
            }
        )
    return {
        "schema": "wam_offline_official_mesh_occlusion_input_contract_v2",
        "status": STATUS,
        "scope": SCOPE,
        "production_activation": False,
        "core_contract_sha256": core_contract_sha256,
        "core_contract": core_value,
        "identities": {
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
            "conversion_manifest_sha256": CONVERSION_MANIFEST_SHA256,
        },
        "policy_contract": {
            "policy_names": list(POLICY_NAMES),
            "max_freeze_age_s": MAX_FREEZE_AGE_S,
            "max_freeze_error_p95_m": MAX_FREEZE_ERROR_P95_M,
            "patient_motion_bound_m_s": PATIENT_MOTION_BOUND_M_S,
        },
        "matrix": {
            "point_names": list(POINT_NAMES),
            "wrist_indices": [0, 1],
            "occlusion_durations_s": [0.10, 0.25, 0.50, 1.00],
            "required_strata_count": 80,
        },
        "render_provenance": {
            "width": 640,
            "height": 360,
            "raw_depth_dtype": "float32",
            "raw_segmentation_dtype": "int32",
            "raw_segmentation_channels": 2,
            "segmentation_object_type": GEOM_OBJECT_TYPE,
            "robot_geom_ids": list(OFFICIAL_ROBOT_GEOM_IDS),
            "human_geom_ids": list(OFFICIAL_HUMAN_GEOM_IDS),
        },
        "strata": strata,
    }


def _frame_specs(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    direction = core_experiment._patient_direction(record["patient_motion_seed"])
    base_truth = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    for frame_index, (event_name, timestamp_s, accepted) in enumerate(
        zip(
            ("pre", "contact", "post"),
            FRAME_TIMESTAMPS_S,
            GEOMETRY_ACCEPTED,
            strict=True,
        )
    ):
        truth = (
            base_truth
            + direction * core_experiment.PATIENT_TRANSLATION_SPEED_M_S * timestamp_s
        )
        event_observation_seed = record["observation_seed"] + frame_index
        rows.append(
            {
                "frame_index": frame_index,
                "event_name": event_name,
                "timestamp_s": timestamp_s,
                "observation_seed": event_observation_seed,
                "noise_contract_sha256": core_experiment.make_predeclared_observation_noise(
                    event_observation_seed
                ).contract_sha256,
                "geometry_accepted": accepted,
                "visible_point_count_report_only": 8 if accepted else 4,
                "accepted_estimate_world_m": truth.tolist() if accepted else None,
                "accepted_error_p95_m": 0.002 if accepted else None,
                "patient_motion_bound_m_s": PATIENT_MOTION_BOUND_M_S,
                "ground_truth_world_m": truth.tolist(),
                "raw_depth_path": f"raw/frames/frame_{frame_index:05d}_depth_m.npy",
                "raw_segmentation_path": (
                    f"raw/frames/frame_{frame_index:05d}_segmentation.npy"
                ),
            }
        )
    return rows


def _observation_payload(
    input_contract_sha256: str, record: dict[str, Any], frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "wam_offline_sparse_observation_stream_v1",
        "input_contract_sha256": input_contract_sha256,
        "scenario_id": record["scenario_id"],
        "target_name": record["target_name"],
        "wrist_index": record["wrist_index"],
        "occlusion_duration_s": record["occlusion_duration_s"],
        "observation_seed": record["observation_seed"],
        "patient_motion_seed": record["patient_motion_seed"],
        "patient_motion_direction_world": core_experiment._patient_direction(
            record["patient_motion_seed"]
        ).tolist(),
        "execution_status": record["status"],
        "events": [
            {
                "frame_index": frame["frame_index"],
                "event_name": frame["event_name"],
                "timestamp_s": frame["timestamp_s"],
                "observation_seed": frame["observation_seed"],
                "noise_contract_sha256": frame["noise_contract_sha256"],
                "geometry_accepted": frame["geometry_accepted"],
                "accepted_estimate_world_m": frame["accepted_estimate_world_m"],
                "accepted_error_p95_m": frame["accepted_error_p95_m"],
                "visible_point_count_report_only": frame[
                    "visible_point_count_report_only"
                ],
                "target_ground_truth_world_m": frame["ground_truth_world_m"],
            }
            for frame in frames
        ],
    }


def _with_observation_hashes(
    records: list[dict[str, Any]], input_contract_sha256: str
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        frames = _frame_specs(record) if record["status"] == "COMPLETE" else []
        stream_sha = hashlib.sha256(
            _core_canonical_bytes(
                _observation_payload(input_contract_sha256, record, frames)
            )
        ).hexdigest()
        result.append({**record, "observation_stream_sha256": stream_sha})
    return result


def _policy_products(
    records: list[dict[str, Any]],
) -> tuple[dict[str, tuple[Any, ...]], list[GroundTruthFrame], list[RequiredStratum]]:
    complete = next(row for row in records if row["status"] == "COMPLETE")
    stream_sha = complete["observation_stream_sha256"]
    observations = []
    truths = []
    for frame in _frame_specs(complete):
        observations.append(
            ObservationFrame(
                scenario_id=complete["scenario_id"],
                target_name=complete["target_name"],
                wrist_index=complete["wrist_index"],
                frame_index=frame["frame_index"],
                timestamp_s=frame["timestamp_s"],
                geometry_accepted=frame["geometry_accepted"],
                accepted_estimate_world_m=frame["accepted_estimate_world_m"],
                accepted_error_p95_m=frame["accepted_error_p95_m"],
                patient_motion_bound_m_s=frame["patient_motion_bound_m_s"],
                visible_point_count_report_only=frame[
                    "visible_point_count_report_only"
                ],
                observation_stream_sha256=stream_sha,
            )
        )
        truths.append(
            GroundTruthFrame(
                scenario_id=complete["scenario_id"],
                target_name=complete["target_name"],
                wrist_index=complete["wrist_index"],
                frame_index=frame["frame_index"],
                timestamp_s=frame["timestamp_s"],
                observation_stream_sha256=stream_sha,
                target_world_m=frame["ground_truth_world_m"],
            )
        )
    required = [
        RequiredStratum(
            scenario_id=row["scenario_id"],
            target_name=row["target_name"],
            wrist_index=row["wrist_index"],
            observation_stream_sha256=row["observation_stream_sha256"],
            observation_seed=row["observation_seed"],
            patient_motion_seed=row["patient_motion_seed"],
            execution_status=row["status"],
            declared_frame_count=row["frame_count"],
        )
        for row in records
    ]
    outputs = evaluate_occlusion_policies(
        observations,
        max_freeze_age_s=MAX_FREEZE_AGE_S,
        max_freeze_error_p95_m=MAX_FREEZE_ERROR_P95_M,
    )
    return outputs, truths, required


def _optional_float(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _float(value: float) -> str:
    return repr(float(value))


def _csv_rows(
    records: list[dict[str, Any]],
    outputs: dict[str, tuple[Any, ...]],
    truths: list[GroundTruthFrame],
) -> list[dict[str, str]]:
    complete = next(row for row in records if row["status"] == "COMPLETE")
    specs = _frame_specs(complete)
    truth_by_frame = {row.frame_index: row for row in truths}
    rows: list[dict[str, str]] = []
    for frame_index, spec in enumerate(specs):
        truth = truth_by_frame[frame_index]
        for policy_name in POLICY_NAMES:
            output = outputs[policy_name][frame_index]
            estimate = output.estimate_world_m
            error = (
                None
                if estimate is None
                else float(np.linalg.norm(estimate - truth.target_world_m))
            )
            accepted = spec["accepted_estimate_world_m"]
            row = {
                "scenario_id": complete["scenario_id"],
                "target_name": complete["target_name"],
                "wrist_index": str(complete["wrist_index"]),
                "occlusion_duration_s": str(complete["occlusion_duration_s"]),
                "frame_index": str(frame_index),
                "timestamp_s": _float(spec["timestamp_s"]),
                "event_name": spec["event_name"],
                "observation_seed": str(spec["observation_seed"]),
                "noise_contract_sha256": spec["noise_contract_sha256"],
                "observation_stream_sha256": complete["observation_stream_sha256"],
                "geometry_accepted": ("true" if spec["geometry_accepted"] else "false"),
                "visible_point_count_report_only": str(
                    spec["visible_point_count_report_only"]
                ),
                "accepted_estimate_x_m": ""
                if accepted is None
                else _float(accepted[0]),
                "accepted_estimate_y_m": ""
                if accepted is None
                else _float(accepted[1]),
                "accepted_estimate_z_m": ""
                if accepted is None
                else _float(accepted[2]),
                "accepted_error_p95_m": _optional_float(spec["accepted_error_p95_m"]),
                "patient_motion_bound_m_s": _float(PATIENT_MOTION_BOUND_M_S),
                "policy_name": policy_name,
                "policy_state": output.state,
                "episode_id": output.episode_id or "",
                "last_valid_age_s": _optional_float(output.last_valid_age_s),
                "predicted_error_p95_m": _optional_float(output.predicted_error_p95_m),
                "reacquired": "true" if output.reacquired else "false",
                "deliberately_unsafe": (
                    "true" if output.deliberately_unsafe else "false"
                ),
                "estimate_x_m": "" if estimate is None else _float(estimate[0]),
                "estimate_y_m": "" if estimate is None else _float(estimate[1]),
                "estimate_z_m": "" if estimate is None else _float(estimate[2]),
                "valid": "true" if output.valid else "false",
                "ground_truth_x_m": _float(truth.target_world_m[0]),
                "ground_truth_y_m": _float(truth.target_world_m[1]),
                "ground_truth_z_m": _float(truth.target_world_m[2]),
                "error_m": _optional_float(error),
                "raw_depth_path": spec["raw_depth_path"],
                "raw_segmentation_path": spec["raw_segmentation_path"],
            }
            assert tuple(row) == CSV_HEADER
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_payload(
    scored: dict[str, dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    denominators = scored[POLICY_NAMES[0]]["denominators"]
    assert all(scored[name]["denominators"] == denominators for name in POLICY_NAMES)
    return {
        "schema": "wam_offline_official_mesh_occlusion_metrics_v2",
        "status": STATUS,
        "scope": SCOPE,
        "production_activation": False,
        "isaac_equivalence": False,
        "identities": {
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
            "conversion_manifest_sha256": CONVERSION_MANIFEST_SHA256,
        },
        "denominators": denominators,
        "policies": {
            policy_name: {
                key: scored[policy_name][key]
                for key in ("pooled_frame", "equal_weight_per_stratum", "per_stratum",)
            }
            for policy_name in POLICY_NAMES
        },
        "failed_strata": [
            row["scenario_id"] for row in records if row["status"] != "COMPLETE"
        ],
        "completion_gates": FALSE_COMPLETION_GATES,
    }


def _video_timeline(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = next(row for row in records if row["status"] == "COMPLETE")
    frames = []
    for video_index, spec in enumerate(_frame_specs(complete)):
        frames.append(
            {
                "video_frame_index": video_index,
                "scenario_id": complete["scenario_id"],
                "target_name": complete["target_name"],
                "wrist_index": complete["wrist_index"],
                "frame_index": spec["frame_index"],
                "timestamp_s": spec["timestamp_s"],
                "event_name": spec["event_name"],
                "observation_stream_sha256": complete["observation_stream_sha256"],
                "raw_depth_path": spec["raw_depth_path"],
                "raw_segmentation_path": spec["raw_segmentation_path"],
            }
        )
    return {
        "schema": "wam_offline_official_mesh_occlusion_video_timeline_v1",
        "status": STATUS,
        "video_path": "offline_official_mesh_occlusion_h264.mp4",
        "fps": 12,
        "width": 640,
        "height": 360,
        "frame_count": len(frames),
        "report_frame_video_frame_index": REPORT_VIDEO_FRAME_INDEX,
        "overlay_contract": {
            "region_xywh": [0, 0, 640, 72],
            "banner_text": "OFFLINE METHOD SIM — NOT ISAAC / NOT PRODUCTION",
            "robot_commit": OPENARM_COMMIT,
            "policy_names": list(POLICY_NAMES),
        },
        "frames": frames,
    }


def _build_release(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True)
    for name, expected in (
        ("runtime_manifest.json", RUNTIME_MANIFEST_SHA256),
        ("asset_manifest.json", ASSET_MANIFEST_SHA256),
        ("conversion_manifest.json", CONVERSION_MANIFEST_SHA256),
    ):
        source = FROZEN_RAW_ROOT / name
        assert source.is_file() and _sha256(source) == expected
        shutil.copyfile(source, raw / name)

    predeclared = _predeclared_matrix_records()
    input_contract = _input_contract(predeclared)
    _write_json(raw / "input_contract.json", input_contract)
    input_sha = _sha256(raw / "input_contract.json")
    core_input_sha = input_contract["core_contract_sha256"]
    records = _with_observation_hashes(predeclared, core_input_sha)
    outputs, truths, required = _policy_products(records)
    rows = _csv_rows(records, outputs, truths)
    _write_csv(raw / "per_frame.csv", rows)
    rerun_path = raw / "determinism_rerun/per_frame.csv"
    _write_csv(rerun_path, rows)

    (raw / "frames").mkdir()
    for frame_index in range(len(FRAME_TIMESTAMPS_S)):
        depth = np.full((360, 640), 2.0, dtype=np.float32)
        depth[90:280, 180:460] = np.float32(1.20 + 0.01 * frame_index)
        segmentation = np.full((360, 640, 2), -1, dtype=np.int32)
        segmentation[90:280, 180:460, 0] = 25
        segmentation[90:280, 180:460, 1] = GEOM_OBJECT_TYPE
        segmentation[130:230, 250 + 8 * frame_index : 330 + 8 * frame_index, 0,] = 1
        segmentation[
            130:230, 250 + 8 * frame_index : 330 + 8 * frame_index, 1,
        ] = GEOM_OBJECT_TYPE
        np.save(raw / f"frames/frame_{frame_index:05d}_depth_m.npy", depth)
        np.save(
            raw / f"frames/frame_{frame_index:05d}_segmentation.npy", segmentation,
        )

    frames = _changing_frames()
    _encode_video(root / "offline_official_mesh_occlusion_h264.mp4", frames)
    assert cv2.imwrite(
        str(root / "report_frame.png"),
        cv2.cvtColor(frames[REPORT_VIDEO_FRAME_INDEX], cv2.COLOR_RGB2BGR),
    )
    (root / "README.md").write_text(
        "# Offline official-mesh occlusion\n\n"
        "OFFLINE METHOD SIM — NOT ISAAC / NOT PRODUCTION.\n",
        encoding="utf-8",
    )
    timeline = _video_timeline(records)
    _write_json(raw / "video_timeline.json", timeline)

    first_sha = _sha256(raw / "per_frame.csv")
    rerun_sha = _sha256(rerun_path)
    identities = {
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
        "conversion_manifest_sha256": CONVERSION_MANIFEST_SHA256,
        "asset_release_id": "official_assets-05e66c94cd8b36ff79be",
        "conversion_release_id": "official_meshes-b5e826af2539bceafea1",
        "robot_repository_url": "https://github.com/enactic/openarm_description.git",
        "robot_commit": OPENARM_COMMIT,
        "robot_urdf_sha256": OPENARM_URDF_SHA256,
        "human_asset_url": (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
            "Assets/Isaac/5.1/Isaac/People/Characters/"
            "original_male_adult_medical_01/male_adult_medical_01.usd"
        ),
        "human_usd_sha256": NVIDIA_HUMAN_USD_SHA256,
    }
    experiment_manifest = {
        "schema": "wam_offline_official_mesh_occlusion_experiment_v2",
        "status": STATUS,
        "scope": SCOPE,
        "production_activation": False,
        "isaac_equivalence": False,
        "identities": identities,
        "render": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "width": 640,
            "height": 360,
            "fps": 12,
            "frame_count": len(frames),
            "undistorted_pinhole_only": True,
            "raw_depth_dtype": "float32",
            "raw_segmentation_dtype": "int32",
            "raw_segmentation_channels": 2,
            "segmentation_object_type": GEOM_OBJECT_TYPE,
            "robot_geom_ids": list(OFFICIAL_ROBOT_GEOM_IDS),
            "human_geom_ids": list(OFFICIAL_HUMAN_GEOM_IDS),
        },
        "matrix": {
            "point_names": list(POINT_NAMES),
            "wrist_indices": [0, 1],
            "occlusion_durations_s": [0.10, 0.25, 0.50, 1.00],
            "required_strata_count": 80,
            "records": records,
        },
        "policies": list(POLICY_NAMES),
        "artifacts": {
            "video_path": "offline_official_mesh_occlusion_h264.mp4",
            "report_frame_path": "report_frame.png",
            "metrics_path": "raw/metrics.json",
            "per_frame_csv_path": "raw/per_frame.csv",
            "determinism_rerun_csv_path": "raw/determinism_rerun/per_frame.csv",
            "input_contract_path": "raw/input_contract.json",
            "video_timeline_path": "raw/video_timeline.json",
            "raw_frame_count": len(FRAME_TIMESTAMPS_S),
        },
        "determinism": {
            "input_contract_sha256": input_sha,
            "core_input_contract_sha256": core_input_sha,
            "first_run_per_frame_sha256": first_sha,
            "same_input_rerun_per_frame_sha256": rerun_sha,
            "same_input_match": True,
        },
        "completion_gates": FALSE_COMPLETION_GATES,
    }
    _write_json(raw / "experiment_manifest.json", experiment_manifest)

    scored = score_occlusion_policy_metrics(outputs, truths, required_strata=required,)
    _write_json(raw / "metrics.json", _metric_payload(scored, records))
    _write_json(
        root / "ARTIFACT_COMPLETE.json",
        {
            "schema": "wam_offline_official_mesh_occlusion_artifact_completion_v2",
            "status": STATUS,
            "artifact_write_complete": True,
            "production_activation": False,
            "linbox_deployment_complete": False,
            "physical_validation_complete": False,
        },
    )
    _write_sums(root)
    return root


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite_csv_release(
    release: Path, mutation: Callable[[list[dict[str, str]]], None]
) -> None:
    csv_path = release / "raw/per_frame.csv"
    rows = _read_csv(csv_path)
    mutation(rows)
    _write_csv(csv_path, rows)
    rerun = release / "raw/determinism_rerun/per_frame.csv"
    _write_csv(rerun, rows)
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_sha = _sha256(csv_path)
    manifest["determinism"]["first_run_per_frame_sha256"] = changed_sha
    manifest["determinism"]["same_input_rerun_per_frame_sha256"] = changed_sha
    _write_json(manifest_path, manifest)
    _write_sums(release)


@pytest.fixture(scope="session")
def canonical_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_release(tmp_path_factory.mktemp("offline_release_v2") / "release")


@pytest.fixture
def release(tmp_path: Path, canonical_release: Path) -> Path:
    destination = tmp_path / "release"
    shutil.copytree(canonical_release, destination)
    return destination


def test_valid_v2_release_replays_policy_metrics_timeline_and_false_gates(
    release: Path,
) -> None:
    report = runner.verify_offline_release(release)

    assert report["status"] == STATUS
    assert report["production_activation"] is False
    assert report["isaac_equivalence"] is False
    assert report["identities"] == {
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
        "conversion_manifest_sha256": CONVERSION_MANIFEST_SHA256,
        "robot_commit": OPENARM_COMMIT,
        "robot_urdf_sha256": OPENARM_URDF_SHA256,
        "human_usd_sha256": NVIDIA_HUMAN_USD_SHA256,
    }
    assert report["video"] == {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 640,
        "height": 360,
        "fps": 12,
        "frame_count": len(FRAME_TIMESTAMPS_S),
        "nonblack": True,
        "changing": True,
        "timeline_bound": True,
        "report_frame_bound": True,
        "overlay_region_present": True,
    }
    assert report["raw_products"] == {
        "depth_frame_count": len(FRAME_TIMESTAMPS_S),
        "segmentation_frame_count": len(FRAME_TIMESTAMPS_S),
        "csv_render_frame_count": len(FRAME_TIMESTAMPS_S),
    }
    assert report["completion_gates"] == FALSE_COMPLETION_GATES
    assert report["sha256_coverage_complete"] is True
    assert report["same_input_rerun_match"] is True
    assert report["policy_replay_match"] is True
    assert report["metrics_recomputed_match"] is True
    assert report["input_contract_bound"] is True
    assert len(report["release_payload_sha256"]) == 64


@pytest.mark.parametrize(
    "relative_path",
    (
        "raw/runtime_manifest.json",
        "raw/asset_manifest.json",
        "raw/conversion_manifest.json",
    ),
)
def test_frozen_manifests_must_match_exact_verified_bytes(
    release: Path, relative_path: str
) -> None:
    path = release / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected_but_well_formed"] = True
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="identity|manifest.*SHA-256"):
        runner.verify_offline_release(release)


@pytest.mark.parametrize(
    "relative_path",
    (
        "raw/experiment_manifest.json",
        "raw/metrics.json",
        "raw/input_contract.json",
        "raw/video_timeline.json",
    ),
)
def test_generated_v2_json_schemas_are_closed(
    release: Path, relative_path: str
) -> None:
    path = release / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected_field"] = "must not be silently ignored"
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="schema|unexpected|keys"):
        runner.verify_offline_release(release)


def test_manifest_identity_cross_links_are_not_self_asserted(release: Path) -> None:
    path = release / "raw/experiment_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identities"]["robot_commit"] = "0" * 40
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="robot|identity|commit"):
        runner.verify_offline_release(release)


@pytest.mark.parametrize("gate", tuple(FALSE_COMPLETION_GATES))
def test_every_non_offline_completion_gate_remains_false(
    release: Path, gate: str
) -> None:
    for relative_path in ("raw/experiment_manifest.json", "raw/metrics.json"):
        path = release / relative_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["completion_gates"][gate] = True
        _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="completion gate|production"):
        runner.verify_offline_release(release)


def test_csv_header_and_policy_output_fields_are_exact(release: Path) -> None:
    path = release / "raw/per_frame.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    header[17], header[18] = header[18], header[17]
    path.write_text(
        ",".join(header) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )
    rerun = release / "raw/determinism_rerun/per_frame.csv"
    shutil.copyfile(path, rerun)
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_sha = _sha256(path)
    manifest["determinism"]["first_run_per_frame_sha256"] = changed_sha
    manifest["determinism"]["same_input_rerun_per_frame_sha256"] = changed_sha
    _write_json(manifest_path, manifest)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="CSV header"):
        runner.verify_offline_release(release)


@pytest.mark.parametrize(
    ("frame_index", "policy_name", "field", "value", "message"),
    (
        (
            1,
            "bounded_freeze_then_lost",
            "policy_state",
            "BOUNDED_PATIENT_FRAME_FREEZE",
            "bounded|replay|policy",
        ),
        (
            1,
            "bounded_freeze_then_lost",
            "last_valid_age_s",
            "0.500001",
            "age|replay|policy",
        ),
        (
            1,
            "bounded_freeze_then_lost",
            "predicted_error_p95_m",
            "0.008501",
            "p95|replay|policy",
        ),
        (
            1,
            "indefinite_world_lock_baseline",
            "deliberately_unsafe",
            "false",
            "unsafe|policy",
        ),
        (1, "direct_geometry_only", "episode_id", "1", "episode|policy"),
    ),
)
def test_csv_replays_exact_policyframeoutput_and_boundary_contract(
    release: Path,
    frame_index: int,
    policy_name: str,
    field: str,
    value: str,
    message: str,
) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(
            item
            for item in rows
            if item["frame_index"] == str(frame_index)
            and item["policy_name"] == policy_name
        )
        row[field] = value
        if field == "policy_state" and value == "BOUNDED_PATIENT_FRAME_FREEZE":
            row["valid"] = "true"
            row["episode_id"] = "E0001"
            row["estimate_x_m"] = "0.0"
            row["estimate_y_m"] = "0.0"
            row["estimate_z_m"] = "1.0"
            truth = np.asarray(
                [
                    float(row["ground_truth_x_m"]),
                    float(row["ground_truth_y_m"]),
                    float(row["ground_truth_z_m"]),
                ]
            )
            row["error_m"] = repr(float(np.linalg.norm(truth - [0.0, 0.0, 1.0])))

    _rewrite_csv_release(release, mutate)
    with pytest.raises(runner.OfflineReleaseError, match=message):
        runner.verify_offline_release(release)


def test_csv_rejects_noncanonical_boolean_and_lost_null_semantics(
    release: Path,
) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        row = next(item for item in rows if item["valid"] == "false")
        row["valid"] = "FALSE"
        row["estimate_x_m"] = "0.0"

    _rewrite_csv_release(release, mutate)
    with pytest.raises(runner.OfflineReleaseError, match="boolean|valid|CSV"):
        runner.verify_offline_release(release)


@pytest.mark.parametrize(
    "mutation", ("pooled", "equal_weight", "per_stratum", "denominator"),
)
def test_all_metrics_are_recomputed_exactly_from_csv(
    release: Path, mutation: str
) -> None:
    path = release / "raw/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "pooled":
        payload["policies"]["direct_geometry_only"]["pooled_frame"]["coverage"][
            "numerator"
        ] += 1
    elif mutation == "equal_weight":
        payload["policies"]["direct_geometry_only"][
            "equal_weight_per_stratum"
        ] = payload["policies"]["direct_geometry_only"]["pooled_frame"]
    elif mutation == "per_stratum":
        payload["policies"]["bounded_freeze_then_lost"]["per_stratum"][0][
            "observation_seed"
        ] += 1
    else:
        payload["denominators"]["failed_strata_count"] -= 1
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="metric|recomput|denominator"):
        runner.verify_offline_release(release)


def test_input_contract_binds_exact_official_geom_provenance(release: Path) -> None:
    input_path = release / "raw/input_contract.json"
    input_contract = json.loads(input_path.read_text(encoding="utf-8"))
    input_contract["render_provenance"]["robot_geom_ids"] = [1]
    _write_json(input_path, input_contract)
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["render"]["robot_geom_ids"] = [1]
    manifest["determinism"]["input_contract_sha256"] = _sha256(input_path)
    _write_json(manifest_path, manifest)
    _write_sums(release)
    with pytest.raises(
        runner.OfflineReleaseError, match="official|geom|provenance|input"
    ):
        runner.verify_offline_release(release)


def test_input_contract_predeclares_constant_patient_motion_bound(
    release: Path,
) -> None:
    input_path = release / "raw/input_contract.json"
    input_contract = json.loads(input_path.read_text(encoding="utf-8"))
    input_contract["policy_contract"]["patient_motion_bound_m_s"] = 0.030
    _write_json(input_path, input_contract)
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["determinism"]["input_contract_sha256"] = _sha256(input_path)
    _write_json(manifest_path, manifest)
    _write_sums(release)
    with pytest.raises(
        runner.OfflineReleaseError, match="motion bound|input|observation"
    ):
        runner.verify_offline_release(release)


def test_csv_rejects_per_frame_motion_bound_drift(release: Path) -> None:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["frame_index"] == "1":
                row["patient_motion_bound_m_s"] = "0.014"

    _rewrite_csv_release(release, mutate)
    with pytest.raises(runner.OfflineReleaseError, match="motion bound|input|policy"):
        runner.verify_offline_release(release)


def test_embedded_core_contract_is_not_self_asserted(release: Path) -> None:
    input_path = release / "raw/input_contract.json"
    input_contract = json.loads(input_path.read_text(encoding="utf-8"))
    input_contract["core_contract"]["patient_motion_bound_m_s"] = 0.014
    forged_core_sha = hashlib.sha256(
        _core_canonical_bytes(input_contract["core_contract"])
    ).hexdigest()
    input_contract["core_contract_sha256"] = forged_core_sha
    _write_json(input_path, input_contract)
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["determinism"]["input_contract_sha256"] = _sha256(input_path)
    manifest["determinism"]["core_input_contract_sha256"] = forged_core_sha
    _write_json(manifest_path, manifest)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="core|immutable|frozen"):
        runner.verify_offline_release(release)


def test_observation_hash_is_recomputed_not_self_asserted(release: Path) -> None:
    forged = "c" * 64

    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["observation_stream_sha256"] = forged

    _rewrite_csv_release(release, mutate)
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = next(
        row for row in manifest["matrix"]["records"] if row["status"] == "COMPLETE"
    )
    complete["observation_stream_sha256"] = forged
    _write_json(manifest_path, manifest)
    timeline_path = release / "raw/video_timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    for frame in timeline["frames"]:
        frame["observation_stream_sha256"] = forged
    _write_json(timeline_path, timeline)
    metrics_path = release / "raw/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for policy in metrics["policies"].values():
        for row in policy["per_stratum"]:
            if row["scenario_id"] == COMPLETE_SCENARIO_ID:
                row["observation_stream_sha256"] = forged
    _write_json(metrics_path, metrics)
    _write_sums(release)
    with pytest.raises(
        runner.OfflineReleaseError, match="observation.*SHA|hash|stream"
    ):
        runner.verify_offline_release(release)


def test_video_timeline_is_a_bijection_to_csv_and_raw_frames(release: Path) -> None:
    path = release / "raw/video_timeline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frames"].pop()
    payload["frame_count"] -= 1
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(
        runner.OfflineReleaseError, match="timeline|video.*raw|bijection|frame count"
    ):
        runner.verify_offline_release(release)


def test_video_timeline_rejects_boolean_frame_indices(release: Path) -> None:
    path = release / "raw/video_timeline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frames"][0]["video_frame_index"] = False
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="timeline|integer|index"):
        runner.verify_offline_release(release)


def test_report_frame_must_match_declared_decoded_video_frame(release: Path) -> None:
    replacement = _changing_frames()[0]
    assert cv2.imwrite(
        str(release / "report_frame.png"), cv2.cvtColor(replacement, cv2.COLOR_RGB2BGR)
    )
    _write_sums(release)
    with pytest.raises(
        runner.OfflineReleaseError, match="report frame|decoded video|match"
    ):
        runner.verify_offline_release(release)


def test_video_overlay_metadata_is_closed_and_identity_bound(release: Path) -> None:
    path = release / "raw/video_timeline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["overlay_contract"]["robot_commit"] = "0" * 40
    _write_json(path, payload)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="overlay|commit|identity"):
        runner.verify_offline_release(release)


@pytest.mark.parametrize(
    ("codec", "frames", "message"),
    (
        ("mpeg4", _changing_frames(), "H.264|codec"),
        (
            "libx264",
            np.zeros((len(FRAME_TIMESTAMPS_S), 360, 640, 3), dtype=np.uint8),
            "black|nonblack",
        ),
        (
            "libx264",
            np.repeat(_changing_frames(1), len(FRAME_TIMESTAMPS_S), axis=0),
            "changing|static",
        ),
        ("libx264", _changing_frames(len(FRAME_TIMESTAMPS_S) - 1), "frame count"),
    ),
)
def test_video_is_real_h264_yuv420p_nonblack_changing_and_exact_length(
    release: Path, codec: str, frames: np.ndarray, message: str
) -> None:
    _encode_video(
        release / "offline_official_mesh_occlusion_h264.mp4", frames, codec=codec
    )
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match=message):
        runner.verify_offline_release(release)


def test_raw_depth_and_two_channel_geom_segmentation_are_csv_bound(
    release: Path,
) -> None:
    path = release / "raw/frames/frame_00001_segmentation.npy"
    np.save(path, np.zeros((360, 640), dtype=np.int32))
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="segmentation|shape|channel"):
        runner.verify_offline_release(release)


def test_npz_disguised_as_npy_is_rejected_as_offline_release_error(
    release: Path,
) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, depth=np.ones((360, 640), dtype=np.float32))
    (release / "raw/frames/frame_00001_depth_m.npy").write_bytes(buffer.getvalue())
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="NPY|NPZ|NumPy|raw depth"):
        runner.verify_offline_release(release)


def test_raw_npy_byte_cap_is_enforced_before_array_decode(release: Path) -> None:
    path = release / "raw/frames/frame_00001_depth_m.npy"
    path.write_bytes(path.read_bytes() + b"x" * (runner.MAX_NPY_HEADER_BYTES + 1))
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="large|cap|NPY|byte"):
        runner.verify_offline_release(release)


@pytest.mark.parametrize(
    "constant_name", ("MAX_VIDEO_BYTES", "MAX_REPORT_FRAME_BYTES"),
)
def test_media_byte_caps_are_enforced_before_decode(
    release: Path, monkeypatch: pytest.MonkeyPatch, constant_name: str,
) -> None:
    monkeypatch.setattr(runner, constant_name, 1)
    with pytest.raises(runner.OfflineReleaseError, match="large|cap|size"):
        runner.verify_offline_release(release)


def test_missing_csv_referenced_raw_frame_fails_with_fresh_checksums(
    release: Path,
) -> None:
    (release / "raw/frames/frame_00002_depth_m.npy").unlink()
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="depth|referenced|missing"):
        runner.verify_offline_release(release)


def test_sha256s_covers_every_file_exactly_once_except_itself(release: Path) -> None:
    sums = release / "SHA256SUMS.txt"
    lines = sums.read_text(encoding="utf-8").splitlines()
    sums.write_text(
        "\n".join(
            line
            for line in lines
            if not line.endswith("  offline_official_mesh_occlusion_h264.mp4")
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.OfflineReleaseError, match="SHA256SUMS|coverage"):
        runner.verify_offline_release(release)


def test_unhashed_extra_raw_file_is_rejected(release: Path) -> None:
    (release / "raw/unhashed.npy").write_bytes(b"not covered")
    with pytest.raises(runner.OfflineReleaseError, match="SHA256SUMS|coverage"):
        runner.verify_offline_release(release)


def test_same_input_rerun_is_byte_exact_and_not_self_reported(release: Path) -> None:
    rerun = release / "raw/determinism_rerun/per_frame.csv"
    with rerun.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    manifest_path = release / "raw/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["determinism"]["same_input_rerun_per_frame_sha256"] = _sha256(rerun)
    manifest["determinism"]["same_input_match"] = True
    _write_json(manifest_path, manifest)
    _write_sums(release)
    with pytest.raises(runner.OfflineReleaseError, match="determin|same-input|rerun"):
        runner.verify_offline_release(release)


def test_atomic_publication_is_no_replace_and_requires_completion_marker(
    tmp_path: Path, canonical_release: Path
) -> None:
    staging_a = tmp_path / ".staging-a"
    staging_b = tmp_path / ".staging-b"
    shutil.copytree(canonical_release, staging_a)
    shutil.copytree(canonical_release, staging_b)
    first_digest = runner.release_payload_sha256(staging_a)
    assert first_digest == runner.release_payload_sha256(staging_b)

    destination = tmp_path / "published"
    report = runner.publish_staged_release(staging_a, destination)
    assert report["release_payload_sha256"] == first_digest
    assert destination.is_dir() and not staging_a.exists()
    destination_before = runner.release_payload_sha256(destination)
    with pytest.raises(FileExistsError, match="exist|replace"):
        runner.publish_staged_release(staging_b, destination)
    assert staging_b.is_dir()
    assert runner.release_payload_sha256(destination) == destination_before

    incomplete = tmp_path / ".staging-incomplete"
    shutil.copytree(canonical_release, incomplete)
    (incomplete / "ARTIFACT_COMPLETE.json").unlink()
    _write_sums(incomplete)
    refused = tmp_path / "must-not-appear"
    with pytest.raises(runner.OfflineReleaseError, match="completion|complete"):
        runner.publish_staged_release(incomplete, refused)
    assert incomplete.is_dir() and not refused.exists()


def test_publish_reverifies_exact_snapshot_after_fsync_mutation(
    tmp_path: Path, canonical_release: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".staging-mutated"
    destination = tmp_path / "must-not-publish"
    shutil.copytree(canonical_release, staging)
    original = runner._fsync_tree

    def mutate_after_fsync(root: Path) -> None:
        original(root)
        with (root / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("post-fsync mutation\n")
        _write_sums(root)

    monkeypatch.setattr(runner, "_fsync_tree", mutate_after_fsync)
    with pytest.raises(
        runner.OfflineReleaseError, match="changed|mutation|payload|snapshot"
    ):
        runner.publish_staged_release(staging, destination)
    assert staging.is_dir()
    assert not destination.exists()


def test_publish_postrename_verification_rolls_back_mutation(
    tmp_path: Path, canonical_release: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".staging-postrename"
    destination = tmp_path / "must-roll-back"
    shutil.copytree(canonical_release, staging)
    original = runner._rename_no_replace

    def mutate_after_rename(source: Path, target: Path) -> None:
        original(source, target)
        with (target / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("post-rename mutation\n")
        _write_sums(target)

    monkeypatch.setattr(runner, "_rename_no_replace", mutate_after_rename)
    with pytest.raises(
        runner.OfflineReleaseError, match="changed|mutation|payload|snapshot"
    ):
        runner.publish_staged_release(staging, destination)
    assert staging.is_dir()
    assert not destination.exists()

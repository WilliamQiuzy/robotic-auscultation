from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.perception.tracking import offline_official_mesh_experiment as experiment
from ausc.perception.tracking.offline_official_mesh_assets import canonical_json_bytes
from ausc.perception.tracking.official_openarm_experiment import RenderedVisibility
from scripts.perception import offline_official_mesh_release_builder as builder
from scripts.perception import run_offline_official_mesh_occlusion as verifier


def test_raw_paths_are_unique_and_canonical_for_all_240_sparse_events() -> None:
    paths: set[str] = set()
    for target_name in ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"):
        for wrist_index in (0, 1):
            for duration_s in (0.10, 0.25, 0.50, 1.00):
                scenario_id = (
                    f"{target_name}__w{wrist_index}__occ"
                    f"{int(round(duration_s * 1000.0)):04d}ms"
                )
                for frame_index, event_name in enumerate(("pre", "contact", "post")):
                    depth, segmentation = builder.raw_paths_for_event(
                        scenario_id=scenario_id,
                        frame_index=frame_index,
                        event_name=event_name,
                    )
                    for relative in (depth, segmentation):
                        assert relative.startswith(f"raw/frames/{scenario_id}/")
                        assert ".." not in Path(relative).parts
                        assert relative not in paths
                        paths.add(relative)

    assert len(paths) == 480


def test_dry_run_plans_all_official_branches_without_writing_or_compiling(
    tmp_path: Path,
) -> None:
    calls: list[tuple[bytes, tuple[object, ...]]] = []

    def fake_planner(
        urdf_raw: bytes, anchors: tuple[object, ...]
    ) -> tuple[SimpleNamespace, ...]:
        calls.append((urdf_raw, anchors))
        return tuple(
            SimpleNamespace(
                target_name=name,
                status="COMPLETE",
                branches=(SimpleNamespace(wrist_index=0), SimpleNamespace(wrist_index=1)),
            )
            for name in builder.POINT_NAMES
        )

    destination = tmp_path / "official_mesh_offline_release-v1"
    report = builder.dry_run_release(destination, planner=fake_planner)

    assert len(calls) == 1
    assert calls[0][0]
    assert len(calls[0][1]) == 10
    assert report.required_strata_count == 80
    assert report.official_target_count == 10
    assert report.official_wrist_branch_count == 20
    assert report.serialization_prototype_event_count == 240
    assert report.temporal_core_v2_integrated is False
    assert report.patient_motion_bound_m_s == 0.013
    assert report.scenes_compiled is False
    assert report.production_activation is False
    assert report.destination == destination
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_staged_build_refuses_an_existing_destination_before_planning(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing-release"
    destination.mkdir()
    planner_called = False

    def forbidden_planner(_raw: bytes, _anchors: tuple[object, ...]) -> tuple[object, ...]:
        nonlocal planner_called
        planner_called = True
        return ()

    with pytest.raises(builder.OfflineReleaseBuildError, match="exists|overwrite"):
        builder.build_release_staging(
            destination, planner=forbidden_planner, allow_test_only=True
        )

    assert planner_called is False
    assert list(destination.iterdir()) == []


def test_real_staging_is_blocked_until_temporal_core_v2_is_integrated(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "must-not-be-created"

    with pytest.raises(builder.OfflineReleaseBuildError, match="core v2|20 Hz"):
        builder.build_release_staging(destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_test_only_flag_cannot_unlock_real_scene_or_full_raw_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "must-not-render"
    monkeypatch.setattr(
        builder,
        "_staging_path",
        lambda _destination: pytest.fail("guard must run before staging allocation"),
    )

    with pytest.raises(builder.OfflineReleaseBuildError, match="injected|test-only"):
        builder.build_release_staging(destination, allow_test_only=True)

    assert not destination.exists()


def test_keyboard_interrupt_removes_any_completion_and_checksum_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "interrupted-release"

    def interrupted_copy(staging: Path) -> None:
        (staging / "ARTIFACT_COMPLETE.json").write_bytes(b"false marker")
        (staging / "SHA256SUMS.txt").write_bytes(b"false sums")
        raise KeyboardInterrupt

    monkeypatch.setattr(builder, "_copy_frozen_manifests", interrupted_copy)
    with pytest.raises(KeyboardInterrupt):
        builder.build_release_staging(
            destination,
            allow_test_only=True,
            scene_factory=lambda *_args, **_kwargs: object(),
            array_writer=lambda *_args, **_kwargs: None,
            video_sink_factory=lambda *_args, **_kwargs: object(),
            release_verifier=lambda _path: {},
        )

    staging = list(tmp_path.glob(f".{destination.name}.staging-*"))
    assert len(staging) == 1
    assert not (staging[0] / "ARTIFACT_COMPLETE.json").exists()
    assert not (staging[0] / "SHA256SUMS.txt").exists()
    warning = (staging[0] / "TEST_ONLY_SERIALIZATION_PROTOTYPE.txt").read_text()
    assert "TEST ONLY" in warning
    assert "NOT A COMPLETE ARTIFACT" in warning


def test_cli_dry_run_is_explicit_machine_readable_and_does_not_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "future-release"
    calls: list[Path] = []

    def fake_dry_runner(path: str | Path) -> builder.DryRunReport:
        target = Path(path)
        calls.append(target)
        return builder.DryRunReport(
            destination=target,
            required_strata_count=80,
            official_target_count=10,
            official_wrist_branch_count=20,
            serialization_prototype_event_count=240,
            temporal_core_v2_integrated=False,
            patient_motion_bound_m_s=0.013,
            scenes_compiled=False,
            production_activation=False,
        )

    exit_code = builder.main(
        ["--dry-run", "--destination", str(destination)],
        dry_runner=fake_dry_runner,
        build_runner=lambda *_args, **_kwargs: pytest.fail("build path must not run"),
    )

    assert exit_code == 0
    assert calls == [destination]
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "destination": str(destination),
        "patient_motion_bound_m_s": 0.013,
        "production_activation": False,
        "required_strata_count": 80,
        "scenes_compiled": False,
        "official_target_count": 10,
        "official_wrist_branch_count": 20,
        "serialization_prototype_event_count": 240,
        "status": "DRY_RUN_IK_ONLY_TEMPORAL_CORE_V2_PENDING",
        "temporal_core_v2_integrated": False,
    }
    assert not destination.exists()


def test_dry_run_rejects_changed_frozen_runtime_asset_or_conversion_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name in (
        "runtime_manifest.json",
        "asset_manifest.json",
        "conversion_manifest.json",
    ):
        (frozen / name).write_bytes(b"{}\n")
    monkeypatch.setattr(builder, "FROZEN_RAW_ROOT", frozen)
    planner_called = False

    def forbidden_planner(_raw: bytes, _anchors: tuple[object, ...]) -> tuple[object, ...]:
        nonlocal planner_called
        planner_called = True
        return ()

    with pytest.raises(builder.OfflineReleaseBuildError, match="identity changed"):
        builder.dry_run_release(tmp_path / "future", planner=forbidden_planner)

    assert planner_called is False


def _fake_motion_plans() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            target_name=name,
            status="COMPLETE",
            failure_reason=None,
            branches=tuple(
                SimpleNamespace(
                    wrist_index=wrist_index,
                    approach_joint_positions_rad=np.zeros(7, dtype=np.float64),
                    contact_joint_positions_rad=np.full(
                        7, 1.0 + 0.1 * wrist_index, dtype=np.float64
                    ),
                )
                for wrist_index in (0, 1)
            ),
        )
        for name in builder.POINT_NAMES
    )


def _geometry_points() -> np.ndarray:
    return np.asarray(
        [
            [-0.025, 0.0000, 1.0],
            [-0.015, 0.0010, 1.0],
            [-0.005, -0.0005, 1.0],
            [0.005, 0.0005, 1.0],
            [0.015, -0.0008, 1.0],
            [0.025, 0.0010, 1.0],
            [-0.12, -0.15, 1.0],
            [0.12, -0.15, 1.0],
            [-0.12, 0.15, 1.0],
            [0.12, 0.15, 1.0],
        ],
        dtype=np.float64,
    )


class _FakeScene:
    def __init__(self, camera_pose_name: str) -> None:
        self.camera_pose_name = camera_pose_name
        self.set_calls = 0
        self.capture_calls = 0
        self.render_calls = 0
        self.closed = False
        self.left_q = np.zeros(7, dtype=np.float64)
        self.patient = np.eye(4)
        self._depth = np.full((360, 640), 1.25, dtype=np.float32)
        self._segmentation = np.full((360, 640, 2), -1, dtype=np.int32)
        self._segmentation[80:280, 100:300, 0] = 25
        self._segmentation[80:280, 100:300, 1] = 5
        self._segmentation[130:230, 250:390, 0] = 1
        self._segmentation[130:230, 250:390, 1] = 5

    def __enter__(self) -> "_FakeScene":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def set_frame(self, left_q: np.ndarray, patient: np.ndarray) -> None:
        self.set_calls += 1
        self.left_q = np.asarray(left_q, dtype=np.float64).copy()
        self.patient = np.asarray(patient, dtype=np.float64).copy()

    def project_current_anchors(self) -> SimpleNamespace:
        return SimpleNamespace(anchor_camera_position_m=_geometry_points())

    def _products(self) -> SimpleNamespace:
        value = (17 * self.set_calls + (43 if self.camera_pose_name == "oblique" else 0)) % 255
        rgb = np.full((360, 640, 3), value, dtype=np.uint8)
        rgb[:, :, 1] = np.uint8((value + 50) % 255)
        return SimpleNamespace(
            rgb=rgb,
            depth_m=self._depth,
            segmentation=self._segmentation,
        )

    def capture_frame_evidence(self, timestamp_s: float) -> SimpleNamespace:
        self.capture_calls += 1
        return SimpleNamespace(
            timestamp_s=timestamp_s,
            left_joint_positions=self.left_q.copy(),
            world_from_patient=self.patient.copy(),
            products=self._products(),
        )

    def render(self) -> SimpleNamespace:
        self.render_calls += 1
        return self._products()


def _gate_frame_from_fake_evidence(
    evidence: SimpleNamespace,
    *,
    reference_anchor_camera_m: np.ndarray,
    target_name: str,
) -> experiment.RenderedGeometryGateFrame:
    contact = float(evidence.left_joint_positions[0]) > 0.5
    supported = set(range(6)) if contact else set(range(6, 10))
    visibility = tuple(
        RenderedVisibility(
            name=name,
            visible_fraction=0.95 if index in supported else 0.0,
            robot_front_fraction=0.0 if index in supported else 1.0,
            valid_depth_fraction=1.0,
            mask_edge_distance_px=20.0 if index in supported else 0.0,
        )
        for index, name in enumerate(builder.POINT_NAMES)
    )
    anchors = experiment.load_official_anchor_contract()
    anchor = anchors[builder.POINT_NAMES.index(target_name)]
    target_truth = (
        evidence.world_from_patient[:3, :3] @ anchor.position_m
        + evidence.world_from_patient[:3, 3]
    )
    points = _geometry_points()
    world_from_cv = np.eye(4)
    world_from_cv[:3, 3] = target_truth - points[builder.POINT_NAMES.index(target_name)]
    return experiment.RenderedGeometryGateFrame(
        point_names=builder.POINT_NAMES,
        reference_anchor_camera_m=reference_anchor_camera_m,
        actual_anchor_camera_m=points,
        rendered_depth_support_m=points[:, 2],
        visibility=visibility,
        target_name=target_name,
        world_from_cv_camera=world_from_cv,
        evidence_source="offline_frame_evidence_rendered_instance_mask_and_depth",
    )


@pytest.mark.skip(
    reason=(
        "obsolete 240-event serializer fixture; temporal core v2 uses 10420 "
        "20 Hz frames and the bound evidence-decision provider"
    )
)
def test_three_event_serializer_prototype_through_primary_and_oblique_scenes(
    tmp_path: Path,
) -> None:
    scenes: dict[str, _FakeScene] = {}
    array_writes: list[tuple[str, tuple[int, ...], str]] = []
    video_frames: list[tuple[tuple[int, ...], float]] = []
    built_experiments: list[experiment.InMemorySparseExperiment] = []

    def scene_factory(_release: Path, *, camera_pose_name: str) -> _FakeScene:
        scene = _FakeScene(camera_pose_name)
        scenes[camera_pose_name] = scene
        return scene

    def array_writer(path: Path, value: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        array_writes.append((path.as_posix(), value.shape, value.dtype.name))
        path.write_bytes(b"unit-array")

    class FakeVideoSink:
        def __init__(self, path: Path, *_args: object, **_kwargs: object) -> None:
            self.path = path

        def __enter__(self) -> "FakeVideoSink":
            return self

        def write(self, frame: np.ndarray) -> None:
            video_frames.append((frame.shape, float(np.std(frame[0:72]))))

        def __exit__(self, *_args: object) -> None:
            self.path.write_bytes(b"unit-h264")

    def tracked_experiment_builder(
        plans: tuple[object, ...],
        anchors: tuple[experiment.OfficialAnchor, ...],
        *,
        gate_provider: object,
    ) -> experiment.InMemorySparseExperiment:
        result = experiment.build_in_memory_sparse_experiment(
            plans, anchors, gate_provider=gate_provider
        )
        built_experiments.append(result)
        return result

    verified: list[Path] = []

    def schema_verifier(path: Path) -> dict[str, str]:
        root = Path(path)
        manifest = verifier._validate_manifest(root)
        contract, _outer_sha, core_sha = verifier._validate_input_contract(
            root, manifest
        )
        with (root / "raw/per_frame.csv").open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        groups: dict[tuple[str, int], list[dict[str, str]]] = {}
        for row in csv_rows:
            groups.setdefault((row["scenario_id"], int(row["frame_index"])), []).append(
                row
            )
        strata = {row["scenario_id"]: row for row in contract["strata"]}
        for record in manifest["matrix"]["records"]:
            payload = verifier._observation_stream_payload(
                core_input_contract_sha256=core_sha,
                record=record,
                stratum=strata[record["scenario_id"]],
                frame_groups=groups,
            )
            assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == (
                record["observation_stream_sha256"]
            )
        verifier._validate_csv_schedule_against_input(
            contract, {key: rows[0] for key, rows in groups.items()}
        )
        verified.append(root)
        return {"status": "ok"}

    destination = tmp_path / "offline-official-mesh-method-v1"
    result = builder.build_release_staging(
        destination,
        allow_test_only=True,
        conversion_release=builder.DEFAULT_CONVERSION_RELEASE,
        planner=lambda _raw, _anchors: _fake_motion_plans(),
        scene_factory=scene_factory,
        experiment_builder=tracked_experiment_builder,
        gate_frame_factory=_gate_frame_from_fake_evidence,
        array_writer=array_writer,
        video_sink_factory=FakeVideoSink,
        release_verifier=schema_verifier,
    )

    assert result.destination == destination
    assert result.staging_root.parent == destination.parent
    assert result.staging_root.name.startswith(f".{destination.name}.staging-")
    assert not destination.exists()
    assert verified == [result.staging_root]
    assert tuple(scenes) == ("primary", "oblique")
    assert scenes["primary"].capture_calls == 240
    assert scenes["oblique"].render_calls == 240
    assert scenes["primary"].closed and scenes["oblique"].closed
    assert len(built_experiments) == 1
    core = built_experiments[0]
    assert len(core.strata) == 80
    assert len(core.observations) == 240
    assert all(len(row.events) == 3 for row in core.strata)
    assert all(row.status == "COMPLETE" for row in core.strata)
    assert len(array_writes) == 480
    assert {shape for _path, shape, _dtype in array_writes} == {
        (360, 640),
        (360, 640, 2),
    }
    assert {dtype for _path, _shape, dtype in array_writes} == {"float32", "int32"}
    assert len({path for path, _shape, _dtype in array_writes}) == 480
    assert len(video_frames) == 240
    assert all(shape == (360, 640, 3) for shape, _std in video_frames)
    assert all(std > 8.0 for _shape, std in video_frames)

    release = result.staging_root
    input_contract = json.loads((release / "raw/input_contract.json").read_text())
    manifest = json.loads((release / "raw/experiment_manifest.json").read_text())
    timeline = json.loads((release / "raw/video_timeline.json").read_text())
    assert input_contract["core_contract_sha256"] == core.input_contract_sha256
    assert input_contract["core_contract"]["patient_motion_bound_m_s"] == 0.013
    assert len(input_contract["strata"]) == 80
    assert all(len(row["events"]) == 3 for row in input_contract["strata"])
    assert manifest["determinism"]["core_input_contract_sha256"] == (
        core.input_contract_sha256
    )
    assert manifest["artifacts"]["raw_frame_count"] == 240
    assert len(timeline["frames"]) == 240
    assert timeline["frame_count"] == 240
    assert {row["event_name"] for row in timeline["frames"]} == {
        "pre",
        "contact",
        "post",
    }
    assert not (release / "._scratch").exists()

    with (release / "raw/per_frame.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 720
    assert tuple(rows[0])[5:12] == (
        "timestamp_s",
        "event_name",
        "observation_seed",
        "noise_contract_sha256",
        "observation_stream_sha256",
        "geometry_accepted",
        "visible_point_count_report_only",
    )
    assert {row["event_name"] for row in rows} == {"pre", "contact", "post"}
    assert all(row["noise_contract_sha256"] for row in rows)
    assert all(row["patient_motion_bound_m_s"] == "0.013" for row in rows)
    assert len({(row["scenario_id"], row["frame_index"]) for row in rows}) == 240
    assert hashlib.sha256((release / "raw/per_frame.csv").read_bytes()).hexdigest() == (
        hashlib.sha256(
            (release / "raw/determinism_rerun/per_frame.csv").read_bytes()
        ).hexdigest()
    )
    assert not (release / "ARTIFACT_COMPLETE.json").exists()
    assert not (release / "SHA256SUMS.txt").exists()
    prototype_warning = (release / "TEST_ONLY_SERIALIZATION_PROTOTYPE.txt").read_text()
    assert "TEST ONLY" in prototype_warning
    assert "20 Hz temporal core v2" in prototype_warning
    assert "NOT A COMPLETE ARTIFACT" in prototype_warning

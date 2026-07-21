from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

from ausc.perception.tracking import offline_official_mesh_experiment as experiment
from ausc.perception.tracking.offline_mesh_simulation import ROBOT_WORLD_FROM_ROOT
from ausc.perception.tracking.offline_mesh_simulation import OfflineMeshScene
from ausc.perception.tracking.official_openarm_experiment import (
    POINT_NAMES,
    RenderedVisibility,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
OFFICIAL_URDF = (
    WORKSPACE_ROOT
    / ".wam-build/offline-official-openarm-assets/releases"
    / "official_assets-05e66c94cd8b36ff79be"
    / "openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)


@pytest.fixture(scope="module")
def official_motion_plans() -> tuple[experiment.TargetMotionPlan, ...]:
    return experiment.plan_official_ik_matrix(
        OFFICIAL_URDF.read_bytes(), experiment.load_official_anchor_contract()
    )


def test_frozen_ik_seed_recipe_is_exact_little_endian_float64() -> None:
    seeds = experiment.frozen_ik_seed_table(OFFICIAL_URDF.read_bytes())

    assert seeds.shape == (37, 7)
    assert seeds.dtype == np.float64
    assert not seeds.flags.writeable
    assert np.array_equal(seeds[0], experiment.HOME_JOINT_POSITIONS_RAD)
    assert hashlib.sha256(seeds.astype("<f8", copy=False).tobytes()).hexdigest() == (
        "c7c3407b5f86ccffdc4761b1dd9752105f2cd051710bed7f2560d7553cc9c8f5"
    )


def test_official_anchor_contract_keeps_exact_ten_labcoat_anchors() -> None:
    anchors = experiment.load_official_anchor_contract()

    assert tuple(anchor.name for anchor in anchors) == POINT_NAMES
    assert experiment.anchor_contract_sha256(anchors) == (
        "3ea89362672086acea6f7af5aaed5baa27def8a58fc07db59d1ddb69476f87d3"
    )
    assert len({anchor.triangle_id for anchor in anchors}) == 10
    for anchor in anchors:
        assert anchor.position_m.shape == (3,)
        assert anchor.normal.shape == (3,)
        assert anchor.barycentric.shape == (3,)
        assert not anchor.position_m.flags.writeable
        assert not anchor.normal.flags.writeable
        assert not anchor.barycentric.flags.writeable
        assert math.isclose(float(np.linalg.norm(anchor.normal)), 1.0, abs_tol=1e-12)
        assert math.isclose(float(anchor.barycentric.sum()), 1.0, abs_tol=1e-12)


def test_real_official_ik_plans_twenty_branches_and_validates_every_trajectory(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    anchors = experiment.load_official_anchor_contract()
    plans = official_motion_plans

    assert tuple(plan.target_name for plan in plans) == POINT_NAMES
    assert all(plan.status == "COMPLETE" for plan in plans)
    assert sum(len(plan.branches) for plan in plans) == 20
    for anchor, plan in zip(anchors, plans, strict=True):
        np.testing.assert_allclose(
            plan.contact_target_world_m,
            anchor.position_m + 0.03 * anchor.normal,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            plan.approach_target_world_m,
            anchor.position_m + 0.15 * anchor.normal,
            atol=1e-12,
            rtol=0.0,
        )
        contact_world = ROBOT_WORLD_FROM_ROOT @ np.append(
            plan.contact_target_robot_root_m, 1.0
        )
        approach_world = ROBOT_WORLD_FROM_ROOT @ np.append(
            plan.approach_target_robot_root_m, 1.0
        )
        np.testing.assert_allclose(contact_world[:3], plan.contact_target_world_m)
        np.testing.assert_allclose(approach_world[:3], plan.approach_target_world_m)
        assert tuple(branch.wrist_index for branch in plan.branches) == (0, 1)
        assert plan.branches[1].orientation_separation_from_first_rad >= math.radians(20.0)
        for branch in plan.branches:
            assert branch.status == "COMPLETE"
            assert branch.position_error_m <= 5.0e-4
            assert branch.trajectory_joint_positions.shape == (121, 7)
            assert branch.trajectory_time_s.shape == (121,)
            assert branch.trajectory_root_poses.shape == (121, 4, 4)
            np.testing.assert_allclose(
                branch.trajectory_joint_positions[[0, 30, 60, 90, 120]],
                np.vstack(
                    (
                        experiment.HOME_JOINT_POSITIONS_RAD,
                        branch.approach_joint_positions_rad,
                        branch.contact_joint_positions_rad,
                        branch.approach_joint_positions_rad,
                        experiment.HOME_JOINT_POSITIONS_RAD,
                    )
                ),
                atol=1.0e-15,
                rtol=0.0,
            )
            assert branch.trajectory_time_s[-1] == 6.0
            assert branch.trajectory_validation["within_position_limits"] is True
            assert branch.trajectory_validation["within_velocity_limits"] is True
            assert branch.trajectory_validation["within_acceleration_limits"] is True


def _geometry_points() -> np.ndarray:
    return np.asarray(
        [
            [-0.025, -0.0010, 1.0],
            [-0.015, 0.0007, 1.0],
            [-0.005, -0.0004, 1.0],
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


def _visibility(supported: set[int]) -> tuple[RenderedVisibility, ...]:
    return tuple(
        RenderedVisibility(
            name=name,
            visible_fraction=0.95 if index in supported else 0.0,
            robot_front_fraction=0.0 if index in supported else 1.0,
            valid_depth_fraction=1.0,
            mask_edge_distance_px=20.0 if index in supported else 0.0,
        )
        for index, name in enumerate(POINT_NAMES)
    )


def _gate_frame(
    supported: set[int],
    noise: experiment.PredeclaredObservationNoise,
    *,
    target_name: str = "R1",
    world_from_cv_camera: np.ndarray | None = None,
    evidence_sha256: str = "a" * 64,
) -> experiment.RenderedGeometryGateFrame:
    points = _geometry_points()
    observed = points + noise.error_camera_m
    intervals = np.full((len(POINT_NAMES), 2), np.nan)
    midpoint = np.full(len(POINT_NAMES), np.nan)
    axial_sigma = np.full(len(POINT_NAMES), np.nan)
    depth_valid = np.zeros(len(POINT_NAMES), dtype=np.bool_)
    for index in supported:
        intervals[index] = observed[index, 2]
        midpoint[index] = observed[index, 2]
        axial_sigma[index] = noise.axial_noise_std_m[index]
        depth_valid[index] = True
    return experiment.RenderedGeometryGateFrame(
        point_names=POINT_NAMES,
        reference_anchor_camera_m=points,
        observed_anchor_camera_m=observed,
        observed_projected_pixels=np.zeros((len(POINT_NAMES), 2)),
        depth_interval_m=intervals,
        depth_mid_m=midpoint,
        axial_sigma_effective_m=axial_sigma,
        depth_valid=depth_valid,
        visibility=_visibility(supported),
        target_name=target_name,
        world_from_cv_camera=(
            np.eye(4) if world_from_cv_camera is None else world_from_cv_camera
        ),
        evidence_sha256=evidence_sha256,
        raw_render_sha256="b" * 64,
        predeclared_noise_sha256=noise.contract_sha256,
        conversion_release_id=(
            f"official_meshes-{experiment.EXPECTED_CONVERSION_MANIFEST_SHA256[:20]}"
        ),
        conversion_manifest_sha256=experiment.EXPECTED_CONVERSION_MANIFEST_SHA256,
        asset_release_id=(
            f"official_assets-{experiment.EXPECTED_ASSET_MANIFEST_SHA256[:20]}"
        ),
        asset_manifest_sha256=experiment.EXPECTED_ASSET_MANIFEST_SHA256,
        runtime_manifest_sha256=experiment.EXPECTED_RUNTIME_MANIFEST_SHA256,
        scene_contract_sha256=experiment.EXPECTED_SCENE_CONTRACT_SHA256,
        depth_footprint=experiment.DEPTH_FOOTPRINT,
        evidence_source=(
            "oracle_landmark_noise_simulation_exact_labcoat_segmentation_and_metric_depth"
        ),
    )


def test_rendered_gate_uses_geometry_and_depth_not_a_visible_count_rule() -> None:
    noise = experiment.make_predeclared_observation_noise(51020260715)
    clustered_six = experiment.evaluate_rendered_geometry_gate(
        _gate_frame(set(range(6)), noise)
    )
    broad_four = experiment.evaluate_rendered_geometry_gate(
        _gate_frame(set(range(6, 10)), noise)
    )

    assert clustered_six.visible_point_count_report_only == 6
    assert broad_four.visible_point_count_report_only == 4
    assert not clustered_six.accepted
    assert broad_four.accepted
    assert clustered_six.geometry_result.effective_support > (
        broad_four.geometry_result.effective_support
    )
    assert broad_four.geometry_result.decision_surface == experiment.DECISION_SURFACE
    assert broad_four.accepted_estimate_world_m is not None
    assert broad_four.accepted_error_p95_m is not None
    assert broad_four.accepted_error_p95_m <= 0.007
    assert broad_four.predeclared_noise_sha256 == noise.contract_sha256

    bad_depth = _gate_frame(set(range(6, 10)), noise)
    changed_interval = bad_depth.depth_interval_m.copy()
    changed_midpoint = bad_depth.depth_mid_m.copy()
    changed_interval[6:] += 0.020
    changed_midpoint[6:] += 0.020
    depth_rejected = experiment.evaluate_rendered_geometry_gate(
        replace(
            bad_depth,
            depth_interval_m=changed_interval,
            depth_mid_m=changed_midpoint,
        )
    )
    assert not depth_rejected.accepted
    assert "depth_observation_mismatch" in depth_rejected.geometry_result.failure_reasons


def test_rendered_gate_keeps_radius_six_visibility_strictly_report_only() -> None:
    noise = experiment.make_predeclared_observation_noise(51020260715)
    exact_depth_support = _gate_frame(set(range(6, 10)), noise)

    visible_report = experiment.evaluate_rendered_geometry_gate(exact_depth_support)
    hidden_report = experiment.evaluate_rendered_geometry_gate(
        replace(exact_depth_support, visibility=_visibility(set()))
    )

    assert visible_report.visible_point_count_report_only == 4
    assert hidden_report.visible_point_count_report_only == 0
    assert visible_report.accepted
    assert hidden_report.accepted
    assert visible_report.supported_point_names == hidden_report.supported_point_names
    assert (
        visible_report.geometry_result.failure_reasons
        == hidden_report.geometry_result.failure_reasons
    )
    assert (
        visible_report.geometry_result.target_error_p95_m
        == hidden_report.geometry_result.target_error_p95_m
    )
    np.testing.assert_array_equal(
        visible_report.geometry_result.transform,
        hidden_report.geometry_result.transform,
    )


def test_predeclared_noise_is_seeded_hashed_and_immutable() -> None:
    first = experiment.make_predeclared_observation_noise(51020260715)
    second = experiment.make_predeclared_observation_noise(51020260715)
    changed = experiment.make_predeclared_observation_noise(51020260716)

    np.testing.assert_array_equal(first.error_camera_m, second.error_camera_m)
    assert first.contract_sha256 == second.contract_sha256
    assert first.contract_sha256 != changed.contract_sha256
    assert not first.error_camera_m.flags.writeable
    assert not first.lateral_noise_std_m.flags.writeable
    assert not first.axial_noise_std_m.flags.writeable
    assert np.max(np.abs(first.error_camera_m[:, :2])) <= 2.5e-3
    assert np.max(np.abs(first.error_camera_m[:, 2])) <= 5.0e-3


_SYNTHETIC_GATE_TEMPLATES: dict[bool, experiment.GeometryGateDecision] | None = None


def _synthetic_gate_templates() -> dict[bool, experiment.GeometryGateDecision]:
    global _SYNTHETIC_GATE_TEMPLATES
    if _SYNTHETIC_GATE_TEMPLATES is None:
        noise = experiment.make_predeclared_observation_noise(1)
        _SYNTHETIC_GATE_TEMPLATES = {
            accepted: experiment.evaluate_rendered_geometry_gate(
                _gate_frame(
                    set(range(6, 10)) if accepted else set(range(6)),
                    noise,
                )
            )
            for accepted in (False, True)
        }
    return _SYNTHETIC_GATE_TEMPLATES


def _sparse_gate_provider(
    event: experiment.SparseEventSpec,
) -> experiment.GeometryGateDecision:
    # Test-only geometry: broad support away from the contact/hold interval;
    # clustered support begins at contact and ends on the first depart frame.
    rejected = (
        (event.phase == "base_precontact" and event.trajectory_index == 60)
        or (
            event.phase == "contact_hold"
            and event.scenario_id
            not in {"R1__w0__occ0250ms", "L1__w0__occ0100ms"}
        )
        or (
            event.scenario_id == "R1__w0__occ0100ms"
            and event.phase == "base_precontact"
        )
        or (
            event.scenario_id == "R1__w0__occ0500ms"
            and event.phase == "base_precontact"
            and event.trajectory_index >= 59
        )
        or (
            event.scenario_id == "L1__w0__occ0100ms"
            and event.phase == "contact_hold"
            and event.hold_frame_index == 1
        )
    )
    accepted = not rejected
    template = _synthetic_gate_templates()[accepted]
    estimate = None
    error_p95 = None
    if accepted:
        target_index = POINT_NAMES.index(event.target_name)
        estimate = (
            event.target_ground_truth_world_m
            + event.noise.error_camera_m[target_index]
        )
        error_p95 = template.accepted_error_p95_m
    supported_names = tuple(POINT_NAMES[index] for index in (
        range(6, 10) if accepted else range(6)
    ))
    return experiment.GeometryGateDecision(
        accepted=accepted,
        accepted_estimate_world_m=estimate,
        accepted_error_p95_m=error_p95,
        visible_point_count_report_only=4 if accepted else 6,
        supported_point_names=supported_names,
        predeclared_noise_sha256=event.noise.contract_sha256,
        evidence_sha256=hashlib.sha256(
            event.render_reuse_key_sha256.encode("ascii")
        ).hexdigest(),
        raw_render_sha256=hashlib.sha256(
            ("raw:" + event.render_reuse_key_sha256).encode("ascii")
        ).hexdigest(),
        conversion_manifest_sha256=experiment.EXPECTED_CONVERSION_MANIFEST_SHA256,
        asset_manifest_sha256=experiment.EXPECTED_ASSET_MANIFEST_SHA256,
        runtime_manifest_sha256=experiment.EXPECTED_RUNTIME_MANIFEST_SHA256,
        scene_contract_sha256=experiment.EXPECTED_SCENE_CONTRACT_SHA256,
        geometry_result=template.geometry_result,
    )


@pytest.mark.skipif(
    os.environ.get("WAM_RUN_REAL_MESH_TEST") != "1",
    reason="set WAM_RUN_REAL_MESH_TEST=1 for the single-frame official release test",
)
def test_v2_real_evidence_uses_observed_uv_exact_labcoat_depth_and_no_truth() -> None:
    _assert_v2_real_evidence_contract()


def test_in_memory_contract_keeps_full_twenty_hz_eighty_stratum_matrix(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    anchors = experiment.load_official_anchor_contract()

    result = experiment.build_in_memory_sparse_experiment(
        official_motion_plans,
        anchors,
        urdf_raw=OFFICIAL_URDF.read_bytes(),
        gate_provider=_sparse_gate_provider,
        allow_test_only=True,
    )

    assert result.input_contract["schema"] == (
        "wam_oracle_landmark_noise_simulation_v1"
    )
    assert result.input_contract["status"] == "oracle_landmark_noise_simulation"
    assert result.input_contract["scope"] == "oracle_landmark_noise_simulation"
    assert result.input_contract["production_activation"] is False
    assert result.input_contract["model_validation"] is False
    assert result.input_contract["landmark_source"] == (
        "oracle_anchor_plus_predeclared_noise"
    )
    assert result.input_contract["ground_truth_role"] == (
        "method_upper_bound_and_policy_scoring_only"
    )
    assert result.input_contract["provider_contract"] == (
        "test_only_synthetic_decision"
    )
    assert result.input_contract["test_only_synthetic_provider"] is True
    assert result.input_contract["metric_eligibility_contract"] == (
        "trajectory_index_59_geometry_accepted",
        "at_least_one_contact_hold_geometry_rejected",
        "contact_hold_exit_geometry_rejected",
        "first_post_rejection_acceptance_occurs_in_depart",
    )
    assert "gate_provider_mode" not in result.input_contract
    assert result.input_contract["required_strata_count"] == 80
    assert result.input_contract["event_schedule"] == (
        "official_trajectory_indices_0_through_60_patient_identity",
        "contact_hold_2_5_10_20_frames_patient_rigid_translation",
        "official_trajectory_indices_61_through_120_shifted_by_hold",
    )
    assert result.input_contract["patient_translation_speed_m_s"] == 0.012
    assert result.input_contract["patient_motion_bound_m_s"] == 0.013
    assert len(result.strata) == 80
    assert len(result.required_strata) == 80
    assert len(result.observations) == 10420
    assert len(result.ground_truth_frames) == 10420
    assert all(row.execution_status == "COMPLETE" for row in result.required_strata)
    assert {row.declared_frame_count for row in result.required_strata} == {
        123,
        126,
        131,
        141,
    }
    assert len({row.observation_stream_sha256 for row in result.required_strata}) == 80
    assert tuple(result.policy_outputs) == experiment.POLICY_NAMES
    assert all(len(rows) == 10420 for rows in result.policy_outputs.values())
    assert len(
        {
            event.render_reuse_key_sha256
            for stratum in result.strata
            for event in stratum.events
        }
    ) == 6420

    r1_w0 = {
        row.occlusion_duration_s: row
        for row in result.strata
        if row.target_name == "R1" and row.wrist_index == 0
    }
    assert len(
        {
            r1_w0[duration].events[0].render_reuse_key_sha256
            for duration in experiment.OCCLUSION_DURATIONS_S
        }
    ) == 1
    first_hold_keys = {
        next(
            event.render_reuse_key_sha256
            for event in r1_w0[duration].events
            if event.phase == "contact_hold" and event.hold_frame_index == 1
        )
        for duration in experiment.OCCLUSION_DURATIONS_S
    }
    assert len(first_hold_keys) == 1
    assert len(
        {
            next(
                event.render_reuse_key_sha256
                for event in r1_w0[duration].events
                if event.phase == "base_depart"
            )
            for duration in experiment.OCCLUSION_DURATIONS_S
        }
    ) == 4

    for stratum in result.strata:
        assert len(stratum.events) == 121 + experiment.OCCLUSION_HOLD_FRAMES[
            stratum.occlusion_duration_s
        ]
        np.testing.assert_allclose(
            np.diff([event.timestamp_s for event in stratum.events]),
            0.05,
            atol=1.0e-12,
            rtol=0.0,
        )
        assert stratum.events[0].trajectory_index == 0
        assert stratum.events[60].trajectory_index == 60
        assert stratum.events[60].phase == "base_precontact"
        assert stratum.events[-1].trajectory_index == 120
        assert stratum.events[-1].phase == "base_depart"
        if stratum.scenario_id == "R1__w0__occ0100ms":
            assert stratum.status == "COMPLETE"
            assert stratum.failure_reason is None
            assert not stratum.geometry_eligible
            assert stratum.geometry_failure_reason == (
                "no_immediate_precontact_geometry_baseline"
            )
        elif stratum.scenario_id == "R1__w0__occ0250ms":
            assert not stratum.geometry_eligible
            assert stratum.geometry_failure_reason == "no_hold_geometry_rejection"
        elif stratum.scenario_id == "R1__w0__occ0500ms":
            assert not stratum.geometry_eligible
            assert stratum.geometry_failure_reason == (
                "no_immediate_precontact_geometry_baseline"
            )
        elif stratum.scenario_id == "L1__w0__occ0100ms":
            # It rejects at hold frame 1 but accepts again at hold frame 2.
            # A later accepted depart frame is not a new reacquisition episode.
            assert not stratum.geometry_eligible
            assert stratum.geometry_failure_reason == (
                "no_depart_geometry_reacquisition"
            )
        else:
            assert stratum.geometry_eligible
            assert stratum.geometry_failure_reason is None
        for event in stratum.events:
            anchor = anchors[POINT_NAMES.index(event.target_name)]
            displacement = event.target_ground_truth_world_m - anchor.position_m
            if event.phase == "base_precontact":
                expected_displacement = 0.0
            elif event.phase == "contact_hold":
                assert event.hold_frame_index is not None
                expected_displacement = 0.012 * event.hold_frame_index * 0.05
            else:
                expected_displacement = 0.012 * stratum.occlusion_duration_s
            assert math.isclose(float(np.linalg.norm(displacement)), expected_displacement, abs_tol=1.0e-12)
            assert event.patient_motion_bound_m_s == 0.013
            assert not event.bias_applied
            assert not event.delay_applied

    rejected_contact = next(
        row
        for row in result.observations
        if row.scenario_id == "R1__w0__occ1000ms" and row.frame_index == 60
    )
    assert not rejected_contact.geometry_accepted
    lost_bounded = next(
        row
        for row in result.policy_outputs["bounded_freeze_then_lost"]
        if row.scenario_id == rejected_contact.scenario_id
        and row.state == "LOST"
        and row.frame_index >= rejected_contact.frame_index
    )
    assert lost_bounded.timestamp_s > rejected_contact.timestamp_s
    denominators = result.metrics["bounded_freeze_then_lost"]["denominators"]
    assert denominators["required_strata_count"] == 80
    assert denominators["completed_strata_count"] == 80
    assert denominators["failed_strata_count"] == 0
    assert denominators["eligible_paired_frames"] == 9917
    assert denominators["metric_eligible_strata_count"] == 76
    assert denominators["metric_ineligible_strata_count"] == 4
    assert result.metrics["bounded_freeze_then_lost"]["pooled_frame"][
        "coverage"
    ]["denominator"] == 9917
    assert len(result.metrics["bounded_freeze_then_lost"]["per_stratum"]) == 80
    ineligible_rows = [
        row
        for row in result.metrics["bounded_freeze_then_lost"]["per_stratum"]
        if row["execution_status"] == "COMPLETE" and not row["metric_eligible"]
    ]
    assert len(ineligible_rows) == 4
    assert all(not row["metrics_available"] for row in ineligible_rows)
    assert result.metrics["bounded_freeze_then_lost"]["time_to_lost_s"]["p50"] > 0.0

    motion_digest = experiment.revalidate_official_motion_plans(
        OFFICIAL_URDF.read_bytes(), anchors, official_motion_plans
    )
    _, repeated_contract_digest = experiment.build_experiment_input_contract(
        motion_plan_sha256=motion_digest,
        allow_test_only=True,
    )
    assert repeated_contract_digest == result.input_contract_sha256


def test_v2_motion_contract_uses_the_validated_twenty_hz_trajectory(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    assert experiment.TRAJECTORY_FRAMES_PER_SEGMENT == 31
    assert experiment.SCORING_TRAJECTORY_INDICES == tuple(range(121))
    assert experiment.OCCLUSION_HOLD_FRAMES == {
        0.10: 2,
        0.25: 5,
        0.50: 10,
        1.00: 20,
    }
    digest = experiment.revalidate_official_motion_plans(
        OFFICIAL_URDF.read_bytes(),
        experiment.load_official_anchor_contract(),
        official_motion_plans,
    )
    assert len(digest) == 64
    contract, contract_digest = experiment.build_experiment_input_contract(
        motion_plan_sha256=digest
    )
    test_only_contract, test_only_contract_digest = (
        experiment.build_experiment_input_contract(
            motion_plan_sha256=digest,
            allow_test_only=True,
        )
    )
    assert contract["motion_plan_sha256"] == digest
    assert contract["provider_contract"] == "scene_owned_oracle_evidence"
    assert contract["test_only_synthetic_provider"] is False
    assert test_only_contract["provider_contract"] == (
        "test_only_synthetic_decision"
    )
    assert test_only_contract["test_only_synthetic_provider"] is True
    assert test_only_contract_digest != contract_digest
    assert contract["scene_contract_sha256"] == (
        "ec571eb51deb21738145c63bbf5db9a6a89974bca8ca53a02131b23bcddc499c"
    )
    assert contract["robot_world_from_root"] == ROBOT_WORLD_FROM_ROOT.tolist()
    assert len(contract_digest) == 64

    for plan in official_motion_plans:
        for branch in plan.branches:
            assert branch.trajectory_joint_positions.shape == (121, 7)
            assert branch.trajectory_time_s.shape == (121,)
            assert branch.trajectory_root_poses.shape == (121, 4, 4)
            np.testing.assert_allclose(
                branch.trajectory_joint_positions[[0, 30, 60, 90, 120]],
                np.vstack(
                    (
                        experiment.HOME_JOINT_POSITIONS_RAD,
                        branch.approach_joint_positions_rad,
                        branch.contact_joint_positions_rad,
                        branch.approach_joint_positions_rad,
                        experiment.HOME_JOINT_POSITIONS_RAD,
                    )
                ),
                atol=1.0e-15,
                rtol=0.0,
            )
            np.testing.assert_allclose(np.diff(branch.trajectory_time_s), 0.05)


def test_oracle_builder_rejects_unbound_synthetic_gate_without_production_claim(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    with pytest.raises(
        experiment.OfflineOfficialMeshExperimentError,
        match="scene-owned OfflineFrameEvidence",
    ):
        experiment.build_in_memory_sparse_experiment(
            official_motion_plans,
            experiment.load_official_anchor_contract(),
            urdf_raw=OFFICIAL_URDF.read_bytes(),
            gate_provider=lambda _request: _synthetic_gate_templates()[False],
        )


def test_default_oracle_provider_request_cannot_read_scoring_ground_truth(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    class RequestInspected(RuntimeError):
        pass

    def inspect_request(request: object) -> object:
        assert request.__class__.__name__ == "OracleRenderRequest"
        assert not hasattr(request, "target_ground_truth_world_m")
        assert not hasattr(request, "noise")
        assert set(request.__dataclass_fields__) == {
            "frame_index",
            "timestamp_s",
            "left_joint_positions_rad",
            "world_from_patient",
            "render_reuse_key_sha256",
        }
        assert not hasattr(request, "target_name")
        assert not hasattr(request, "scenario_id")
        assert not hasattr(request, "wrist_index")
        raise RequestInspected

    with pytest.raises(RequestInspected):
        experiment.build_in_memory_sparse_experiment(
            official_motion_plans,
            experiment.load_official_anchor_contract(),
            urdf_raw=OFFICIAL_URDF.read_bytes(),
            gate_provider=inspect_request,
        )


def test_reused_render_key_rejects_a_second_raw_render_identity(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    seen: dict[str, int] = {}

    def mismatched_reuse_provider(
        event: experiment.SparseEventSpec,
    ) -> experiment.GeometryGateDecision:
        decision = _sparse_gate_provider(event)
        count = seen.get(event.render_reuse_key_sha256, 0)
        seen[event.render_reuse_key_sha256] = count + 1
        if count == 0:
            return decision
        return replace(
            decision,
            raw_render_sha256=hashlib.sha256(
                f"forged:{event.render_reuse_key_sha256}:{count}".encode("ascii")
            ).hexdigest(),
        )

    with pytest.raises(
        experiment.OfflineOfficialMeshExperimentError,
        match="reuse key.*raw render",
    ):
        experiment.build_in_memory_sparse_experiment(
            official_motion_plans,
            experiment.load_official_anchor_contract(),
            urdf_raw=OFFICIAL_URDF.read_bytes(),
            gate_provider=mismatched_reuse_provider,
            allow_test_only=True,
        )


def test_v2_revalidation_rejects_untrusted_motion_plan_payloads(
    official_motion_plans: tuple[experiment.TargetMotionPlan, ...],
) -> None:
    anchors = experiment.load_official_anchor_contract()
    first = official_motion_plans[0]
    forged_branch = replace(
        first.branches[0],
        contact_joint_positions_rad=np.full(7, 999.0),
    )
    forged_plan = replace(
        first,
        branches=(forged_branch, first.branches[1]),
    )
    with pytest.raises(
        experiment.OfflineOfficialMeshExperimentError,
        match="joint|motion|trajectory|limit",
    ):
        experiment.revalidate_official_motion_plans(
            OFFICIAL_URDF.read_bytes(),
            anchors,
            (forged_plan, *official_motion_plans[1:]),
        )


def test_v2_sweep_contract_declares_three_but_executes_only_zero() -> None:
    value = experiment.observation_sweep_contract()

    assert value["declared_sweep_count"] == 3
    assert value["executed_sweep_count"] == 1
    assert value["executed_sweep_indices"] == (0,)
    assert len(value["sweeps"]) == 3
    assert all(set(row) == {
        "sweep_index",
        "noise_seed_root",
        "bias_seed_root",
        "delay_seed_root",
        "noise_applied",
        "bias_applied",
        "delay_applied",
    } for row in value["sweeps"])
    assert value["sweeps"][0]["noise_applied"] is True
    assert value["sweeps"][0]["bias_applied"] is False
    assert value["sweeps"][0]["delay_applied"] is False
    assert all(row["noise_applied"] is False for row in value["sweeps"][1:])


def _assert_v2_real_evidence_contract() -> None:
    release = (
        WORKSPACE_ROOT
        / ".wam-build/offline-official-openarm-meshes/releases"
        / "official_meshes-b5e826af2539bceafea1"
    )
    scene = OfflineMeshScene.from_verified_conversion(release)
    try:
        converted_anchor_positions = scene.anchor_local_position_m
        scene.set_frame(experiment.HOME_JOINT_POSITIONS_RAD, np.eye(4))
        reference = scene.project_current_anchors().anchor_camera_position_m
        evidence = scene.capture_frame_evidence(0.0)
        noise = experiment.make_predeclared_observation_noise(51020260715)

        decision = experiment.rendered_gate_decision_from_evidence(
            evidence,
            reference_anchor_camera_m=reference,
            target_name="R1",
            noise=noise,
        )
        frame = experiment.rendered_gate_frame_from_evidence(
            evidence,
            reference_anchor_camera_m=reference,
            target_name="R1",
            noise=noise,
        )
    finally:
        scene.close()

    assert experiment.ANCHOR_QUANTIZATION_TOLERANCE_M == 1.0e-9
    manifest_anchor_positions = np.stack(
        [row.position_m for row in experiment.load_official_anchor_contract()]
    )
    quantization_error = np.max(
        np.abs(converted_anchor_positions - manifest_anchor_positions)
    )
    assert 1.0e-10 < quantization_error < experiment.ANCHOR_QUANTIZATION_TOLERANCE_M
    experiment.validate_anchor_quantization(
        manifest_anchor_positions, converted_anchor_positions
    )
    outside = converted_anchor_positions.copy()
    outside[0, 0] = (
        manifest_anchor_positions[0, 0]
        + experiment.ANCHOR_QUANTIZATION_TOLERANCE_M
        + 1.0e-12
    )
    with pytest.raises(
        experiment.OfflineOfficialMeshExperimentError,
        match="quantization tolerance",
    ):
        experiment.validate_anchor_quantization(manifest_anchor_positions, outside)
    assert frame.depth_footprint == "observed_uv_bilinear_2x2_exact_labcoat_4_of_4"
    assert frame.observed_anchor_camera_m.shape == (10, 3)
    assert frame.observed_projected_pixels.shape == (10, 2)
    assert frame.depth_interval_m.shape == (10, 2)
    assert frame.depth_mid_m.shape == (10,)
    assert frame.axial_sigma_effective_m.shape == (10,)
    assert frame.depth_valid.dtype == np.bool_
    assert np.all(frame.depth_interval_m[frame.depth_valid, 0] <= frame.depth_mid_m[frame.depth_valid])
    assert np.all(frame.depth_mid_m[frame.depth_valid] <= frame.depth_interval_m[frame.depth_valid, 1])
    expected_sigma = noise.axial_noise_std_m[frame.depth_valid] + 0.5 * np.ptp(
        frame.depth_interval_m[frame.depth_valid], axis=1
    )
    np.testing.assert_allclose(
        frame.axial_sigma_effective_m[frame.depth_valid], expected_sigma
    )
    assert len(frame.evidence_sha256) == 64
    assert decision.evidence_sha256 == frame.evidence_sha256
    assert decision.predeclared_noise_sha256 == noise.contract_sha256
    assert not hasattr(decision, "target_ground_truth_world_m")

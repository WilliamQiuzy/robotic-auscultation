"""Verify portable release-file hashes using only the Python standard library.

Verification is read-only. --generate creates a new manifest without replacing
an existing file. Only the explicit core list and scene sources named in an
expert snapshot are included; raw traces, control logs and dataset arrays are
excluded. Matching hashes establish file integrity, not simulation acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


SCHEMA = "auscultation_release_v1"
MODULE = "sim/auscultation_training/"
DEFAULT_MANIFEST = MODULE + "RELEASE_MANIFEST.json"
DEFAULT_SNAPSHOT = MODULE + "results/low_elbow_release_v1/physics_all10/input_snapshot/manifest.json"
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
CODE_NAMES = (
    "__init__.py", "audit_episode.py", "audit_tool_fit.py", "collision.py",
    "controller.py", "dataset.py", "dataset_v2.py", "env.py", "evaluate_bc.py", "evaluate_bc_v2.py",
    "export_collision_geometry.py", "inspect_geometry.py", "inspect_runtime.py",
    "kinematics.py", "phase_manager.py", "plan_contact_pose.py", "plan_paths.py",
    "policy.py", "probe_contact.py", "refine_normals.py", "render_expert.py",
    "revalidate_plans.py", "run_checked_evaluation.py", "run_expert.py", "run_sequence.py", "scene.py",
    "smoke_runtime.py", "student.py", "student_v2.py", "targets.py", "tool_frames.py",
    "train_bc.py", "train_bc_v2.py", "verify_release.py",
    "audit_low_elbow.py", "compare_experts.py", "low_elbow_controller.py",
    "plan_low_elbow.py", "pose_ik.py", "posture.py", "run_low_elbow.py",
    "evaluate_bc_v3.py", "policy_v3.py", "student_v3.py", "train_bc_v3.py",
)
POINTS = tuple(side + str(level) for level in range(1, 6) for side in ("R", "L"))
CORE_PATHS = frozenset(
    [MODULE + name for name in CODE_NAMES]
    + [MODULE + name for name in (
        "README.md", "RELEASE_NOTES.md", "EXPERT_REVIEW.md", "tests/test_release.py", "openarm_v10_follower_stetho.urdf",
        "tests/test_low_elbow_audit.py", "tests/test_low_elbow_planner.py",
        "tests/test_low_elbow_runtime.py", "tests/test_low_elbow_sequence.py",
        "tests/test_pose_ik.py", "tests/test_compare_experts_review.py",
        "tests/test_contact_ready_plan.py", "tests/test_validate_video_review.py",
        "results/low_elbow_final_v5/validate_video.py",
        "results/low_elbow_recorded_review/review_recorded_sequence.py",
        "results/low_elbow_final_v8_L5_margin1/plans/search_report.json",
        "results/low_elbow_final_v8_L5_margin1/plans/posture_limits.json",
        "results/low_elbow_final_v8_L5_margin1/plans/tool_fit_L5_joint2_margin_refined_2.0mm.json",
        "tests/test_render_sequence.py", "tests/test_replay_timing.py",
        "runtime_inspection.json", "results/contact_targets.json",
        "results/collision_geometry.json", "results/collision_geometry.npz",
        "results/collision_regeneration_verification.json",
        "results/bc_dataset_v1/manifest.json", "results/bc_baseline_v1/policy.pt",
        "results/bc_baseline_v1/run_manifest.json",
        "results/bc_dataset_v2/manifest.json",
        "results/bc_dataset_v2/waypoint_annotations_0.json",
        "results/bc_dataset_v2/waypoint_annotations_1.json",
        "results/bc_baseline_v2/policy.pt", "results/bc_baseline_v2/run_manifest.json",
        "results/bc_baseline_v2/source_snapshot/dataset_v2.py",
        "results/bc_baseline_v2/source_snapshot/train_bc_v2.py",
        "results/bc_dataset_v2_precision/manifest.json",
        "results/bc_dataset_v2_precision/waypoint_annotations_0.json",
        "results/bc_dataset_v2_precision/waypoint_annotations_1.json",
        "results/bc_baseline_v2_precision/policy.pt",
        "results/bc_baseline_v2_precision/run_manifest.json",
        "results/video_all10_v2/expert_sequence_all10_v2_recorded.mp4",
        "results/video_all10_v2/video_manifest.json",
        "results/video_all10_v2/video_validation.json",
        "results/student_r1_v1/outcome.json",
        "results/student_r1_v1/independent_audit.json",
        "results/student_r1_v1/evaluation_summary.json",
        "results/student_r1_v2/outcome.json",
        "results/student_r1_v2/independent_audit.json",
        "results/student_r1_v2/evaluation_summary.json",
        "results/student_r1_v2_precision/outcome.json",
        "results/student_r1_v2_precision/independent_audit.json",
        "results/student_r1_v2_precision/evaluation_summary.json",
    )]
    + [MODULE + "results/planned_paths_" + point + ".json" for point in POINTS]
    + [MODULE + "results/tool_fit_" + point + "_2.0mm.json" for point in POINTS]
    + [MODULE + "results/" + run + "/" + name
       for run in ("sequence_all10_v2", "sequence_all10_speed085")
       for name in ("outcome.json", "independent_audit.json", "input_snapshot/manifest.json")]
    + [MODULE + "results/low_elbow_release_v1/plans/" + prefix + point + suffix
       for point in POINTS
       for prefix, suffix in (("planned_paths_", ".json"), ("tool_fit_", "_2.0mm.json"))
       if prefix == "planned_paths_" or point not in ("R3", "L5")]
    + [MODULE + "results/low_elbow_release_v1/" + name for name in (
        "plans/contact_targets.json", "plans/collision_geometry.json", "plans/collision_geometry.npz",
        "plans/posture_limits.json", "assembly_manifest.json", "geometry_review.json",
        "comparison.json", "recorded_review.json", "generation_dependencies.json",
        "independent_collection_mapping_review.json",
        "plans/tool_fit_R3_low_elbow_refined_2.0mm.json",
        "plans/tool_fit_L5_low_elbow_refined_2.0mm.json",
        "physics_all10/outcome.json", "physics_all10/independent_audit.json",
        "physics_all10/posture_audit.json", "physics_all10/verification_summary.json",
        "physics_all10/runner_summary.json", "physics_all10/world_from_left_base.json",
        "physics_all10/input_snapshot/manifest.json",
        "video_all10/low_elbow_expert_physics_all10_recorded.mp4",
        "video_all10/video_manifest.json", "video_all10/video_validation.json",
    )]
    + ["sim/auscultation_assets_20260922/model_predictions.json"]
)
# Frozen generation dependency allowlist; no external manifest can expand it.
CORE_PATHS = CORE_PATHS | frozenset((
    'sim/auscultation_training/__init__.py',
    'sim/auscultation_training/audit_tool_fit.py',
    'sim/auscultation_training/collision.py',
    'sim/auscultation_training/controller.py',
    'sim/auscultation_training/kinematics.py',
    'sim/auscultation_training/openarm_v10_follower_stetho.urdf',
    'sim/auscultation_training/plan_contact_pose.py',
    'sim/auscultation_training/plan_low_elbow.py',
    'sim/auscultation_training/pose_ik.py',
    'sim/auscultation_training/posture.py',
    'sim/auscultation_training/results/collision_geometry.json',
    'sim/auscultation_training/results/collision_geometry.npz',
    'sim/auscultation_training/results/collision_regeneration_verification.json',
    'sim/auscultation_training/results/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_L1.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_L2.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_L3.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_R1.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_R2.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_R3.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_R4.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/planned_paths_R5.json',
    'sim/auscultation_training/results/low_elbow_contact_v11/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_contact_v7/controller_preflight_review.json',
    'sim/auscultation_training/results/low_elbow_contact_v7/generation_dependencies.partial.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/geometry_contact_review.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/geometry_review.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/planned_paths_R3.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/planned_paths_R3.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v11_R3/plans/tool_fit_R3_low_elbow_refined_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_final_v5/L2_route_diversity.json',
    'sim/auscultation_training/results/low_elbow_final_v5/L3_route_diversity.json',
    'sim/auscultation_training/results/low_elbow_final_v5/R2_route_diversity.json',
    'sim/auscultation_training/results/low_elbow_final_v5/all10_final_plan_validation.json',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/check_L5_command_corridor.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/contact_gap_geometry.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_L2.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_L3.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_L4_v8_pre_first.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_L5_diagnostic_gap0.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_R2.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_R2_v7_old_contact.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/progressive_R3_v11_refined_fit.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/refine_L5_contact_fit.py',
    'sim/auscultation_training/results/low_elbow_final_v5/exploration/refine_R3_contact_fit.py',
    'sim/auscultation_training/results/low_elbow_final_v5/geometry_review.json',
    'sim/auscultation_training/results/low_elbow_final_v5/geometry_review_check.py',
    'sim/auscultation_training/results/low_elbow_final_v5/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v5/geometry_review_update.py',
    'sim/auscultation_training/results/low_elbow_final_v5/plans/planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_final_v5/plans/planned_paths_L5.json',
    'sim/auscultation_training/results/low_elbow_final_v5/plans/planned_paths_R3.json',
    'sim/auscultation_training/results/low_elbow_final_v5/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/README.md',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/preparation_verification.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/release_additions.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/reproduce_progressive.py',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/search_dependencies.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/06b91e903888b316a8bfa8df1c3dd86d11850ec8b209f668fb89c398b297ffa5_planned_paths_R4.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/0b1beb8d42559313937e71ac6ba4645e20a561cf667f4ea2d02b07f45cdd3ae2_planned_paths_L5.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/1afdb7cc6fc6b4c795d0b49a38a7481635c01a556bbd6793ff67d6d36b742cdb_min_elbow_margin_L2.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/24823e8c3daac3a408b9e413731bb4dc59a8bbb1c938395325998ad837e07b55_L1_exact_min_elbow.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/6a8fd1eca3e24f5a4f68d50286b435a2f31bff9d1a14a6f43421f817301dc88e_planned_paths_L5.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/77c18250887443d0a262f10e185b44ce1a7b3031a0dddba9989f403bd36c61eb_low_R2.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/9ab93d03de475592513bfa6f9988ebce0614efaa08696c643f23a89f273e4199_planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/a36346e67f533c01e7f21efbf681dcf6364c667bfa0d0d980b51670c114522a9_planned_paths_R1.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/aa74e33c6ebde5d6c990bbc820cac1dab1a7ac0cbd9a855431b181b60109e391_planned_paths_R3.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/bd74fa178a261b8429c724c9c44dca7ed89bce3d9dc6be2bc77a3e75d428e606_planned_paths_R4.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/beca3b2bc0e51ff6eb9f0b95e17ca032b6e6f1a748d2cfc29b7d119812e324bb_low_L3.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/c1783a3f37116a17345646bd415daac4073017accad0ee352a39e46af5197dc5_planned_paths_R3.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/c371a27823f9318af4f09be51cf3288df55947b4ef777d6c77e9ad9df03cfed7_planned_paths_L3.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/dcd19684e48dc602c49012c670ca5c1b6dd7cc00b80291b4e3148fe6e936e136_planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/dd11713c5b5cdfa154076a75f4f44b09e9d9c6f85b317b45fd9f79e7d7e3e010_planned_paths_R5.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/ddf741283bed5d151b9567edb3ee3e4d3224cfb2b05392c078112a2e71ccdef0_planned_paths_L2.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/dee7bd82b2f4071a3ee927294aa8d857a1ba0d39a96f19301ba0c08fe5896da5_planned_paths_L1.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/e5f80ea6ca6ac32c94e9229f493e42fa6d53446b217d4d9b4ad651814c3b6a1e_planned_paths_R1.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/e606bde6d2e8a627ca2bc0be879ed27b5ee2385f9376d8ef10683b0c7b951257_planned_paths_R5.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/seed_sources/e9d0aaf6d562da39390360e70388ccd33793a848b66cbf79bf0e6f70770da301_planned_paths_R2.json',
    'sim/auscultation_training/results/low_elbow_final_v5/reproduction/test_reproduce_progressive.py',
    'sim/auscultation_training/results/low_elbow_final_v5/search_hints.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/planned_paths_L2.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L2/plans/tool_fit_L2_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/planned_paths_L3.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v6_L3/plans/tool_fit_L3_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_final_v6_R2/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v6_R2/plans/planned_paths_R2.json',
    'sim/auscultation_training/results/low_elbow_final_v6_R2/plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_contact_gap0_profile.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_contact_geometry_review.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_geometry_review.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_joint_margin_review.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_loaded_corridor_review.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/actual_normal_corridor_check.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/command_corridor_input_plan.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/diagnostic_metrics.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/planned_paths_L5.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/planned_paths_L5.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_refined_plans/tool_fit_L5_low_elbow_refined_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_final_v7_L5/diagnostic_gap0_runtime_normal_posture_review.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/contact_pose_revision.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/geometry_review.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/planned_paths_R2.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/planned_paths_R2.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v7_R2/plans/tool_fit_R2_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/contact_corridor_review.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/geometry_review.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/geometry_review_samples.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/planned_paths_L4.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/search_report.json',
    'sim/auscultation_training/results/low_elbow_final_v8_L4/plans/tool_fit_L4_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_margin_v2/plans/planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_margin_v2/plans/planned_paths_L5.json',
    'sim/auscultation_training/results/low_elbow_release_v1/REVIEW.md',
    'sim/auscultation_training/results/low_elbow_release_v1/assemble_collection.py',
    'sim/auscultation_training/results/low_elbow_release_v1/assembly_manifest.json',
    'sim/auscultation_training/results/low_elbow_release_v1/geometry_review.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/collision_geometry.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/collision_geometry.npz',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/contact_targets.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L1.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L1.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L2.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L2.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L3.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L3.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L4.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L4.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L5.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_L5.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R1.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R1.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R2.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R2.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R3.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R3.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R4.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R4.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R5.before_revalidation.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/planned_paths_R5.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/posture_limits.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_L1_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_L2_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_L3_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_L4_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_L5_low_elbow_refined_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_R1_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_R2_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_R3_low_elbow_refined_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_R4_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/plans/tool_fit_R5_2.0mm.json',
    'sim/auscultation_training/results/low_elbow_release_v1/software_tests.txt',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_L1.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_L2.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_L3.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_L4.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_L5.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_R1.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_R2.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_R3.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_R4.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/planned_paths_R5.source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/posture_limits_L5_source.json',
    'sim/auscultation_training/results/low_elbow_release_v1/source_plans/posture_limits_before_contact0.json',
    'sim/auscultation_training/results/planned_paths_L2.json',
    'sim/auscultation_training/results/planned_paths_L3.json',
    'sim/auscultation_training/results/planned_paths_L4.json',
    'sim/auscultation_training/results/planned_paths_L5.json',
    'sim/auscultation_training/results/planned_paths_R2.json',
    'sim/auscultation_training/results/planned_paths_R3.json',
    'sim/auscultation_training/results/tool_fit_L1_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_L2_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_L3_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_L4_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_L5_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_L5_low_elbow_refined_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_R1_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_R2_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_R3_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_R3_low_elbow_refined_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_R4_2.0mm.json',
    'sim/auscultation_training/results/tool_fit_R5_2.0mm.json',
    'sim/auscultation_training/revalidate_plans.py',
    'sim/auscultation_training/runtime_inspection.json',
    'sim/auscultation_training/scene.py',
    'sim/auscultation_training/targets.py',
    'sim/auscultation_training/tests/test_contact_ready_plan.py',
    'sim/auscultation_training/tool_frames.py',
))
SCENE_PARENTS = frozenset((
    "sim/auscultation_assets_20260922/contact_static.usda",
    "sim/auscultation_assets_20260922/raised_seat.usda",
    "sim/auscultation_assets_20260922/asset_manifest.json",
    "sim/real2sim_20260922/auscultation_real2sim.usdc",
))
ASSET_PREFIX = "sim/real2sim_20260922/assets/"
ASSET_SUFFIXES = frozenset((".obj", ".npz", ".json", ".py"))
SELECTION_KIND = "explicit_core_and_snapshot_scene_v1"
SOURCE_SNAPSHOTS = frozenset(name for name in CORE_PATHS if name.endswith("/input_snapshot/manifest.json"))


def safe_name(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(character) < 32 for character in value)):
        raise ValueError("invalid portable relative path")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in (".", "..") or part.startswith(".") for part in path.parts)):
        raise ValueError("unsafe relative path: " + value)
    return value


def permitted(name):
    return (name in CORE_PATHS or name in SCENE_PARENTS
            or (name.startswith(ASSET_PREFIX) and PurePosixPath(name).suffix in ASSET_SUFFIXES))


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_artifact(root, name, *, collect=False, limit=MAX_ARTIFACT_BYTES):
    """Read one approved file through no-follow descriptors, checking stability.

    Keep every parent descriptor open, including the resolved root's parents.
    A renamed directory cannot redirect the read, and final pathname checks
    reject a file or directory replaced while the descriptor was being read.
    """
    name = safe_name(name)
    if not permitted(name):
        raise ValueError("not an approved core artifact: " + name)
    if (not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd
            or os.stat not in os.supports_follow_symlinks):
        raise ValueError("platform lacks safe no-follow artifact reads")
    root = Path(root).resolve(strict=True)
    parts = root.parts[1:] + PurePosixPath(name).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors, links = [], []
    try:
        descriptors.append(os.open(root.anchor, directory_flags))
        for part in parts[:-1]:
            parent = descriptors[-1]
            child = os.open(part, directory_flags, dir_fd=parent)
            descriptors.append(child)
            links.append((parent, part, child))
        parent = descriptors[-1]
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("artifact is not regular or exceeds byte limit: " + name)
        checksum, size, chunks = hashlib.sha256(), 0, []
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError("artifact grew beyond byte limit: " + name)
            checksum.update(chunk)
            if collect:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if (_identity(before) != _identity(after) or _identity(after) != _identity(named)
                or not stat.S_ISREG(named.st_mode) or size != after.st_size):
            raise ValueError("artifact changed while hashing: " + name)
        for parent, part, child in links:
            named = os.stat(part, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(child)
            if (not stat.S_ISDIR(named.st_mode)
                    or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)):
                raise ValueError("artifact parent changed while hashing: " + name)
        return {"path": name, "sha256": checksum.hexdigest(), "bytes": size}, b"".join(chunks)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def fingerprint(root, name):
    return _read_artifact(root, name)[0]


def read_json(path):
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds 4 MiB")
    with path.open("rb") as handle:
        data = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds 4 MiB")
    return json.loads(data)


def scene_paths_from_snapshot(snapshot):
    """Map only explicit scene entries from a recorded source_paths() snapshot."""
    files = snapshot.get("files") if isinstance(snapshot, dict) else None
    if not isinstance(files, dict):
        raise ValueError("source snapshot has no files mapping")
    selected = set()
    for key, item in files.items():
        if not key.startswith("source/scene/"):
            continue
        relative = key.removeprefix("source/scene/")
        if relative.startswith("assets/") or relative == "auscultation_real2sim.usdc":
            name = "sim/real2sim_20260922/" + relative
        else:
            name = "sim/auscultation_assets_20260922/" + relative
        safe_name(name)
        if not permitted(name) or not isinstance(item, dict):
            raise ValueError("unapproved source snapshot entry: " + key)
        source = item.get("source_path")
        if not isinstance(source, str) or not source.endswith("/" + name):
            raise ValueError("source snapshot path disagrees with entry: " + key)
        selected.add(name)
    if not SCENE_PARENTS.issubset(selected) or not any(name.startswith(ASSET_PREFIX) for name in selected):
        raise ValueError("source snapshot is missing scene parents or assets")
    return selected


def _source_snapshot(root, snapshot_name):
    if safe_name(snapshot_name) not in SOURCE_SNAPSHOTS:
        raise ValueError("unapproved source snapshot: " + snapshot_name)
    return _read_artifact(root, snapshot_name, collect=True, limit=MAX_MANIFEST_BYTES)


def scene_paths(root, snapshot_name):
    _, data = _source_snapshot(root, snapshot_name)
    return scene_paths_from_snapshot(json.loads(data))


def generate(root, snapshot_name):
    snapshot_entry, data = _source_snapshot(root, snapshot_name)
    names = CORE_PATHS | scene_paths_from_snapshot(json.loads(data))
    entries = [snapshot_entry if name == snapshot_name else fingerprint(root, name) for name in sorted(names)]
    if fingerprint(root, snapshot_name) != snapshot_entry:
        raise ValueError("source snapshot changed during generation")
    return {"schema": SCHEMA, "algorithm": "sha256", "paths_relative_to": "WAM project root",
            "scope": "Core file integrity only; no physical, learned-policy, or acoustic certification.",
            "selection": {"kind": SELECTION_KIND, "source_snapshot": snapshot_name,
                          "source_snapshot_sha256": snapshot_entry["sha256"]},
            "excluded": ["raw physics traces", "control logs", "replay arrays", "BC dataset arrays"],
            "files": entries}


def verify(root, manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("algorithm") != "sha256":
        raise ValueError("unsupported release manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2048:
        raise ValueError("manifest must name 1..2048 core files")
    errors, by_name, checked = [], {}, 0
    for entry in entries:
        try:
            if not isinstance(entry, dict):
                raise ValueError("invalid file entry")
            name = safe_name(entry.get("path"))
            if name in by_name:
                raise ValueError("duplicate artifact: " + name)
            if not permitted(name):
                raise ValueError("not an approved core artifact: " + name)
            checksum, size = entry.get("sha256"), entry.get("bytes")
            if (not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
                    or type(size) is not int or not 0 <= size <= MAX_ARTIFACT_BYTES):
                raise ValueError("invalid hash or size: " + name)
            by_name[name] = entry
        except (OSError, ValueError) as error:
            errors.append(str(error))
    # The snapshot must be bound and verified before any of its contents can
    # select additional paths. The manifest never gets to reduce CORE_PATHS.
    snapshot_name = None
    if not errors:
        try:
            selection = manifest.get("selection")
            if not isinstance(selection, dict) or selection.get("kind") != SELECTION_KIND:
                raise ValueError("missing or unsupported artifact selection")
            snapshot_name = safe_name(selection.get("source_snapshot"))
            if snapshot_name not in SOURCE_SNAPSHOTS or snapshot_name not in by_name:
                raise ValueError("missing or unapproved source snapshot entry")
            snapshot_entry = by_name[snapshot_name]
            if selection.get("source_snapshot_sha256") != snapshot_entry["sha256"]:
                raise ValueError("source snapshot selection hash disagrees with file entry")
            actual, data = _source_snapshot(root, snapshot_name)
            if actual["sha256"] != snapshot_entry["sha256"] or actual["bytes"] != snapshot_entry["bytes"]:
                raise ValueError("hash or size mismatch: " + snapshot_name)
            checked += 1
            expected = CORE_PATHS | scene_paths_from_snapshot(json.loads(data))
            for name in sorted(expected - by_name.keys()):
                errors.append("missing required artifact entry: " + name)
            for name in sorted(by_name.keys() - expected):
                errors.append("unexpected artifact entry: " + name)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    if not errors:
        for name, entry in by_name.items():
            if name == snapshot_name:
                continue
            try:
                actual = fingerprint(root, name)
                if actual["sha256"] != entry["sha256"] or actual["bytes"] != entry["bytes"]:
                    raise ValueError("hash or size mismatch: " + name)
                checked += 1
            except (OSError, ValueError) as error:
                errors.append(str(error))
    return {"ok": not errors, "checked": checked, "listed": len(entries),
            "errors": errors[:8], "additional_errors": max(0, len(errors) - 8)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--generate", action="store_true", help="create a new manifest; never overwrite")
    parser.add_argument("--source-snapshot", default=DEFAULT_SNAPSHOT,
                        help="project-relative expert snapshot used only during generation")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project root must be a directory")
        path = Path(args.manifest)
        path = path if path.is_absolute() else root / path
        if args.generate:
            manifest = generate(root, args.source_snapshot)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            result = {"ok": True, "generated": len(manifest["files"]), "manifest": str(path)}
        else:
            result = verify(root, read_json(path))
    except (OSError, ValueError, TypeError) as error:
        result = {"ok": False, "errors": [str(error)]}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

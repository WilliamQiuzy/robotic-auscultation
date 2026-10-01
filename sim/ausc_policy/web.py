"""Serve a local experiment blog and explicitly allowed evidence/media only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    '/media/m2-observed-seat-pilot.png': 'results/ausc_policy/observed_scene_pilot_v1/observed_scene_pilot.png',
    '/evidence/m2-observed-seat-pilot': 'results/ausc_policy/observed_scene_pilot_v1/summary.json',
    '/evidence/m2-observed-source-review': 'results/ausc_policy/observed_scene_pilot_v1/source_semantic_review.json',
    '/media/retainer-l4l5-near.mp4': 'results/ausc_policy/retainer_l4l5_near_v4/actual_fresh_expert.mp4',
    '/media/retainer-l4l5-near-poster.png': 'results/ausc_policy/retainer_l4l5_near_v4/contact_frame.png',
    '/media/retainer-l4l5-force.svg': 'results/ausc_policy/retainer_l4l5_near_v4/independent_force.svg',
    '/evidence/retainer-l4l5-independent': 'results/ausc_policy/retainer_l4l5_near_v4/independent_summary.json',
    '/evidence/retainer-l4l5-native': 'results/ausc_policy/retainer_l4l5_near_v4/fresh_expert_result.json',
    '/evidence/retainer-l4l5-owner': 'results/ausc_policy/retainer_l4l5_near_v4/result.json',

    '/media/m2-codec-comparison.svg': 'results/ausc_policy/compact_codec_benchmark_v1/codec_comparison.svg',
    '/evidence/m2-storage-reachability': 'reports/ausc_policy/m2_storage_reachability_20260930.md',
    '/evidence/m2-codec-delta': 'results/ausc_policy/compact_codec_benchmark_v1/compact_codec_benchmark_20260930_u32_xor20_v3__summary.json',
    '/evidence/m2-codec-independent': 'results/ausc_policy/compact_codec_benchmark_v1/compact_codec_benchmark_20260930_u32_v2__summary.json',
    '/evidence/tenpoint-schedule': 'results/ausc_policy/compact_codec_benchmark_v1/tenpoint_schedule_prescreen_20260930_v4.json',
    '/media/retainer-r3-single-preload.mp4': 'results/ausc_policy/retainer_r3_single_preload_v3/output/review/actual_fresh_expert.mp4',
    '/media/retainer-r3-single-preload-poster.png': 'results/ausc_policy/retainer_r3_single_preload_v3/output/review/contact_frame.png',
    '/media/retainer-preload-comparison.svg': 'results/ausc_policy/retainer_preload_comparison_v3/comparison.svg',
    '/evidence/retainer-r3-single-preload-force': 'results/ausc_policy/retainer_r3_single_preload_analysis_v3/summary.json',
    '/evidence/retainer-r3-single-preload-capture': 'results/ausc_policy/retainer_r3_raw_audit_v3.json',
    '/evidence/retainer-preload-comparison': 'results/ausc_policy/retainer_preload_comparison_v3/comparison.json',
    '/media/retainer-r3-contact.mp4': 'results/ausc_policy/retainer_r3_contact_v2/output/review/actual_fresh_expert.mp4',
    '/media/retainer-r3-contact-poster.png': 'results/ausc_policy/retainer_r3_contact_v2/output/review/contact_frame.png',
    '/media/retainer-r3-force.svg': 'results/ausc_policy/retainer_r3_contact_analysis_v2/contact.svg',
    '/evidence/retainer-r3-force': 'results/ausc_policy/retainer_r3_contact_analysis_v2/summary.json',
    '/evidence/retainer-preload': 'results/ausc_policy/retainer_r3_contact_analysis_v2/preload_diagnosis.json',
    '/evidence/retainer-tenpoint': 'results/ausc_policy/retainer_tenpoint_plan_v2/output/review/COMPLETE.json',
    '/media/retainer-recess-oblique.png': 'results/ausc_policy/retainer_geometry_review_v2/output/review/after_oblique.png',
    '/evidence/retainer-contact': 'reports/ausc_policy/retainer_contact_revision_20260930.md',
    '/evidence/retainer-geometry': 'results/ausc_policy/retainer_geometry_review_v2/output/review/COMPLETE.json',
    '/media/velocity-r3-v1.mp4': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_gravity_velocity_v1/output/review/actual_fresh_expert.mp4',
    '/media/velocity-r3-v1-poster.png': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_gravity_velocity_v1/output/review/poster.png',
    '/media/velocity-r3-comparison.svg': 'results/ausc_policy/velocity_comparison_v1/comparison.svg',
    '/evidence/velocity-r3-native': 'reports/ausc_policy/velocity_feedforward_native.md',
    '/evidence/head-contact-envelope': 'reports/ausc_policy/head_contact_envelope_audit.md',
    '/evidence/m2-readiness': 'reports/ausc_policy/m2_readiness_20260930.md',
    '/media/gravity-r3-v2.mp4': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_gravity_v2/output/review/actual_fresh_expert.mp4',
    '/media/gravity-r3-v2-poster.png': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_gravity_v2/output/review/video_check_70s.png',
    '/media/contact-clearance-audit.svg': 'results/ausc_policy/contact_geometry_profile_v1/clearance_diagnosis.svg',
    '/evidence/actual-carriage-clearance': 'results/ausc_policy/actual_carriage_clearance_v1/COMPLETE.json',
    '/evidence/r4-contact-profile': 'results/ausc_policy/contact_geometry_profile_v1/R4_profile_summary.json',
    '/media/m2-native-rgbd.png': 'results/ausc_policy/m2_native_sensors_v3/output/review/rgbd_pair.png',
    '/evidence/m2-native-sensors': 'results/ausc_policy/m2_native_sensors_v3/output/review/sensor_probe_result.json',
    '/evidence/approved-camera-restoration': 'results/ausc_policy/m2_native_sensors_v3/output/review/approved_camera_restoration.json',
    '/media/fresh-r3-v3.mp4': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_v3/output/review/actual_fresh_expert.mp4',
    '/media/fresh-r3-v3-poster.png': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_v3/output/review/poster.png',
    '/media/fresh-r3-tracking.svg': 'results/ausc_policy/fresh_r3_dynamic_diagnosis_v1/tracking_and_force.svg',
    '/evidence/fresh-r3-dynamics': 'results/ausc_policy/fresh_r3_dynamic_diagnosis_v1/REPORT.json',
    '/media/fresh-r3-poster.png': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_v1/output/review/poster.png',
    '/media/fresh-r3-failure.mp4': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_v1/output/review/actual_fresh_expert.mp4',
    '/media/fresh-r3-contact.svg': 'results/ausc_policy/geometry_5mm_fresh_expert_r3_v1/output/review/contact_diagnosis.svg',
    '/media/m2-patient-anchors.png': 'results/ausc_policy/m2_patient_preview_v1/output/review/patient_anchor_review.png',
    '/evidence/5mm-m2': 'reports/ausc_policy/geometry_5mm_m2_start.md',
    '/media/5mm-contact.mp4': 'results/ausc_policy/geometry_5mm_contact_preload_v1/output/review/actual_contact_probe.mp4',
    '/media/5mm-contact-poster.png': 'results/ausc_policy/geometry_5mm_contact_preload_v1/output/review/contact_frame.png',
    '/media/5mm-force.svg': 'results/ausc_policy/geometry_5mm_summary_v1/force_comparison.svg',
    '/media/5mm-contact-before.mp4': 'results/ausc_policy/geometry_5mm_contact_v2/output/review/actual_contact_probe.mp4',
    '/media/tool-2mm-oblique.png': 'results/ausc_policy/tool_protrusion_review_v1/review/after_oblique.png',
    '/media/tool-2mm-side.png': 'results/ausc_policy/tool_protrusion_review_v1/review/after_side.png',
    '/evidence/tool-protrusion-2mm': 'reports/ausc_policy/tool_protrusion_review.md',
    '/evidence/tool-protrusion': 'reports/ausc_policy/tool_protrusion_5mm_review.md',
    '/media/tool-after-oblique.png': 'results/ausc_policy/tool_protrusion_review_5mm_v1/review/after_oblique.png',
    '/media/tool-before-oblique.png': 'results/ausc_policy/tool_protrusion_review_v1/review/before_oblique.png',
    '/media/tool-after-side.png': 'results/ausc_policy/tool_protrusion_review_5mm_v1/review/after_side.png',
    '/media/tool-before-side.png': 'results/ausc_policy/tool_protrusion_review_v1/review/before_side.png',
    '/media/tool-after-face.png': 'results/ausc_policy/tool_protrusion_review_5mm_v1/review/after_face.png',
    '/media/tool-after-scene.png': 'results/ausc_policy/tool_protrusion_review_5mm_v1/review/after_scene.png',
    '/evidence/redesign-plan': 'docs/superpowers/plans/2026-09-29-ausc-diverse-data-policy-redesign.md',
    '/evidence/pi05-v11-r4': 'results/pi05_v11_delivery/ISAAC_R4_FAILED.md',
    '/evidence/pi05-v12-fit': 'results/diagnostics/pi05_v12_train_fit_terminal_evidence_v1/REPORT.md',
    '/evidence/molmo-interface': 'results/diagnostics/molmoact2_wam_one_update_gpu_evidence_v1/REPORT.md',
    '/evidence/molmo-native-fit': 'results/ausc_policy/m0_diagnostics/REPORT.md',
    '/evidence/molmo-seed-sampling': 'results/ausc_policy/m0_seed512/REPORT.md',
    '/evidence/act-fit-diagnosis': 'reports/ausc_policy/m0_act_fit_diagnosis.md',
    '/evidence/act-action-comparison': 'results/ausc_policy/m0_act_representation/REPORT.md',
    '/evidence/act-fk-probe': 'results/ausc_policy/m0_act_fk_probe/REPORT.md',
    '/evidence/act-d0': 'results/ausc_policy/m0_act_d0/REPORT.md',
    '/media/act-d0.mp4': 'results/ausc_policy/m0_act_d0/review.mp4',
    '/media/act-d0-trajectory.svg': 'results/ausc_policy/m0_act_d0/trajectory.svg',
    '/evidence/executor-probe': 'results/ausc_policy/m0_executor_probe/REPORT.md',
    '/evidence/act-input-audit': 'results/ausc_policy/m0_act_input_audit/REPORT.md',
    '/evidence/motion-preview': 'results/ausc_policy/m0_motion_preview_probe/REPORT.md',
    '/evidence/act-path-probe': 'results/ausc_policy/m0_act_path_probe/REPORT.md',
    '/evidence/act-executor-results': 'reports/ausc_policy/m0_executor_closed_loop.md',
    '/evidence/asset-preflight': 'reports/ausc_policy/m2_asset_preflight.md',
    '/evidence/act-feedback': 'reports/ausc_policy/m0_act_feedback_probe.md',
    '/evidence/act-robot-feedback': 'reports/ausc_policy/m0_robot_feedback_probe.md',
    '/media/act-feedback-response.svg': 'results/ausc_policy/m0_feedback_comparison/response.svg',
    '/evidence/recovery-pilot': 'reports/ausc_policy/m0_recovery_pilot.md',
    '/evidence/velocity-definition': 'reports/ausc_policy/m0_velocity_definition.md',
    '/evidence/recovery-sensor-audit': 'reports/ausc_policy/m0_recovery_sensor_audit.md',
    '/media/velocity-definition.svg': 'results/ausc_policy/m0_velocity_probe/comparison.svg',
    '/media/recovery-stationarity.svg': 'results/ausc_policy/m0_recovery_pilot_report/recovery.svg',
    '/evidence/moving-recovery': 'reports/ausc_policy/m0_moving_recovery.md',
    '/evidence/near-state-labels': 'reports/ausc_policy/m0_near_state_labels.md',
    '/media/near-state-labels.svg': 'results/ausc_policy/m0_velocity_probe/near_state_labels.svg',
    '/evidence/capture-region': 'reports/ausc_policy/m0_capture_region.md',
    '/media/capture-geometry.svg': 'results/ausc_policy/m0_capture_geometry_report/geometry.svg',
    '/evidence/causal-history': 'reports/ausc_policy/m0_causal_history.md',
    '/media/causal-history.svg': 'results/ausc_policy/m0_history_probe_matched/history.svg',
    '/evidence/history-fit': 'reports/ausc_policy/m0_history_fit_protocol.md',
    '/media/history-fit.svg': 'results/ausc_policy/m0_history_fit_report/paired_fit.svg',
    '/evidence/history-rollouts': 'reports/ausc_policy/m0_history_closed_loop.md',
    '/evidence/history-recovery-edges': 'reports/ausc_policy/m0_history_recovery_edges.md',
    '/media/history-recovery-edges.mp4': 'results/ausc_policy/m0_history_free20_recovery_edges/m0-history-free20-recovery-edges-20260929/output/review-history20.mp4',
    '/media/history-recovery-edges-poster.jpg': 'results/ausc_policy/m0_history_free20_edges_media/poster.jpg',
    '/media/history-recovery-contact.svg': 'results/ausc_policy/m0_history_free20_edges_report_v2/contact.svg',
    '/media/history-recovery-coverage.svg': 'results/ausc_policy/m0_history_edges_coverage_final/coverage.svg',
    '/media/history-recovery-120.mp4': 'results/ausc_policy/m0_history_free120_recovery_edges/m0-history-free120-recovery-edges-20260929/output/review-history120.mp4',
    '/media/history-recovery-220.mp4': 'results/ausc_policy/m0_history_free220_recovery_edges/m0-history-free220-recovery-edges-20260929/output/review-history220.mp4',
    '/media/history-rollouts.svg': 'results/ausc_policy/m0_history_closed_loop_report_v2/rollouts.svg',
    '/media/history-control.mp4': 'results/ausc_policy/m0_history_control_d0/m0-history-control-d0-grid12-20260929/output/review.mp4',
    '/media/history-control-poster.jpg': 'results/ausc_policy/m0_history_control_media/review-poster.jpg',
    '/media/history-causal.mp4': 'results/ausc_policy/m0_history_causal_d0/m0-history-causal-d0-grid12-20260929/output/review.mp4',
    '/media/capture-normal-plus.mp4': 'results/ausc_policy/m0_capture_normal_plus/m0-capture-normal-plus-20260929/output/review-entry-normal-plus.mp4',
    '/media/capture-normal-plus-poster.jpg': 'results/ausc_policy/m0_capture_normal_plus_media/review-entry-normal-plus-poster.jpg',
    '/media/capture-normal-plus.svg': 'results/ausc_policy/m0_capture_normal_plus_report/contact.svg',
    '/media/recovery-moving.mp4': 'results/ausc_policy/m0_moving_recovery/m0-moving-recovery-act609-20260929/output/review-act609.mp4',
    '/media/recovery-moving-poster.jpg': 'results/ausc_policy/m0_moving_recovery_media/review-moving-poster.jpg',
    '/media/recovery-perturbed.svg': 'results/ausc_policy/m0_recovery_perturbed_report/recovery.svg',
    '/media/recovery-plus.mp4': 'results/ausc_policy/m0_recovery_perturbed/m0-recovery-perturbed-position-history-20260929/output/review-joint3-plus.mp4',
    '/media/recovery-minus.mp4': 'results/ausc_policy/m0_recovery_perturbed/m0-recovery-perturbed-position-history-20260929/output/review-joint3-minus.mp4',
    '/media/recovery-plus-poster.jpg': 'results/ausc_policy/m0_recovery_perturbed_media/review-plus-poster.jpg',
    '/media/recovery-nominal.svg': 'results/ausc_policy/m0_recovery_nominal_report/recovery.svg',
    '/media/recovery-nominal.mp4': 'results/ausc_policy/m0_recovery_nominal/m0-recovery-nominal-position-history-20260929/output/review-nominal.mp4',
    '/media/recovery-nominal-poster.jpg': 'results/ausc_policy/m0_recovery_nominal_media/review-nominal-poster.jpg',
    '/media/act-nearest.mp4': 'results/ausc_policy/m0_act_d0_nearest/review.mp4',
    '/media/act-nearest-comparison.svg': 'results/ausc_policy/m0_act_nearest_comparison/comparison.svg',
    '/media/act-preview.mp4': 'results/ausc_policy/m0_act_d0_preview/review.mp4',
    '/media/act-executor-comparison.svg': 'results/ausc_policy/m0_act_executor_comparison/comparison.svg',
    '/media/executor-probe.svg': 'results/ausc_policy/m0_executor_probe/comparison.svg',
    '/media/pi05-r4-failure.mp4': 'results/diagnostics/pi05_v11_R4_terminal_evidence_v1/verified_files/diagnostics/pi05_v11/cases_v1/R4_selected3500_v11_002_lifecycle/review.mp4',
    '/media/m0-fit.svg': 'results/ausc_policy/m0_diagnostics/fit_vs_baselines.svg',
    '/media/act-action-comparison.svg': 'results/ausc_policy/m0_act_representation/act_action_comparison.svg',
}
for name in ('index.html', 'styles.css', 'app.js', 'entries.json'):
    SOURCES['/' + name] = 'docs/experiment_blog/' + name
SOURCES['/'] = SOURCES['/index.html']


def byte_range(header, length):
    if header is None:
        return 0, length - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()):
        raise ValueError('Unsupported range')
    first, last = match.groups()
    if not first:
        count = int(last)
        if count <= 0:
            raise ValueError('Empty suffix')
        start, end = max(0, length-count), length-1
    else:
        start, end = int(first), min(int(last), length-1) if last else length-1
    if start > end or start >= length:
        raise ValueError('Unsatisfiable range')
    return start, end, True


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.serve(False)

    def do_GET(self):
        self.serve(True)

    def serve(self, body):
        source = SOURCES.get(urlsplit(self.path).path)
        if source is None:
            self.send_error(404)
            return
        path = ROOT / source
        if not path.is_file():
            self.send_error(404, 'Evidence is not available yet')
            return
        length = path.stat().st_size
        try:
            start, end, partial = byte_range(self.headers.get('Range'), length)
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{length}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        self.send_response(206 if partial else 200)
        content_type = 'text/plain; charset=utf-8' if path.suffix == '.md' else (
            mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{length}')
        self.end_headers()
        if not body:
            return
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining:
                    block = stream.read(min(remaining, 128*1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A browser may cancel a video range when seeking.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8774)
    args = parser.parse_args()
    print(f'Experiment blog: http://127.0.0.1:{args.port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

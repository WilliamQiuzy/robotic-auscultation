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

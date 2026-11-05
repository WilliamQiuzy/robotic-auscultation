"""Portable transport checks with synthetic packets and no real model/native run."""
import base64
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

RGB = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=')
CALIBRATION = 'ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054'


def observation(boundary='HOME'):
    """Explicit synthetic protocol fixture, not a captured simulator packet."""
    grid = [[None] * 32 for _ in range(18)]
    counts = [[0] * 32 for _ in range(18)]
    grid[0][0] = 1.25
    counts[0][0] = 1
    hall = dict(packet=dict(source='simulated_adc', raw=32768,
                    calibration_sha256=CALIBRATION, sample_time_s=0., received_time_s=0.),
                decoded=dict(calibration_sha256=CALIBRATION, hall_voltage_v=2.5,
                             spring_force_estimate_n=0.),
                observed_at_s=0., actual_adc_measurement=False,
                force_label_used=False, bench_verified=False, calibration_sha256=CALIBRATION)
    sensors = dict(schema='ausc_astra_rgbd_hall_adc_v1',
        single_camera_path='/World/RobotCam', physics_row_end_exclusive=0,
        observed_at_s=0., hall=hall, raw_rgbd=dict(exact_arrays_transferred=False),
        depth=dict(units='m', semantics='metric_optical_z', annotator='distance_to_image_plane',
                   grid_shape=[18, 32], valid_count=1, invalid_count=360*640-1,
                   median_grid_m=grid, valid_count_grid=counts))
    return dict(schema='ausc_astra_native_observation_v1', simulation_only=True,
        observation_id='synthetic-episode:0', episode_id='synthetic-episode',
        physics_row=0, time_s=0., boundary=boundary, phase='APPROACH',
        q_rad=[0.] * 7, dq_rad_s=[0.] * 7, hall_force_N=0., compression_m=0.,
        hall_voltage_v=2.5, hall_voltage_source='simulated_adc',
        hall_source='calibration_decoded_simulated_adc_spring_load',
        target_id='L1', target_source='given_contact_pose', reference_sha256='a' * 64,
        reference_pose_base=[[float(r == c) for c in range(4)] for r in range(4)],
        rgb_sha256=hashlib.sha256(RGB).hexdigest(), local_executor='GivenContactExpert',
        local_motor_assistance=True, sensor_contract=sensors['schema'], sensor_bundle=sensors)


def decision(obs, skill=None):
    return dict(observation_id=obs['observation_id'],
                skill=skill or {'HOME': 'approach_target', 'PARKED': 'continue_local_control',
                                'RELEASED': 'retract'}[obs['boundary']],
                target_id=obs['target_id'], reason='Synthetic transport test decision.')



class PortableBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = importlib.import_module('demo.astra_supervisor')
        cls.service = importlib.import_module('demo.astra_service')

    def proposal(self, obs):
        return dict(model='gpt-6-astra', backend='codex_exec', decision=decision(obs),
            executed=False, physical_success=None,
            service_source_sha256=hashlib.sha256(Path(self.service.__file__).read_bytes()).hexdigest(),
            protocol_source_sha256=hashlib.sha256(Path(self.protocol.__file__).read_bytes()).hexdigest())

    @contextmanager
    def server(self, infer):
        token = 'synthetic-test-' + 'x' * 32
        server = self.service.make_server(0, token, infer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/decision', token
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_end_to_end_http_binds_sensor_packet_image_and_source_hashes(self):
        received = []
        def infer(obs, rgb):
            received.append((obs, rgb))
            return self.proposal(obs)
        obs = observation()
        with self.server(infer) as (endpoint, token):
            result = self.protocol.HTTPClient(endpoint, token)(obs, RGB)
        self.assertEqual(received, [(obs, RGB)])
        self.assertEqual(result['decision']['observation_id'], obs['observation_id'])
        self.assertFalse(result['executed'])
        self.assertIsNone(result['physical_success'])

    def test_missing_auth_and_oversize_request_never_call_inference(self):
        calls = []
        with self.server(lambda *args: calls.append(args)) as (endpoint, token):
            for headers, expected in [({}, 401),
                ({'Authorization': 'Bearer ' + token,
                  'Content-Length': str(self.protocol.MAX_REQUEST_BYTES + 1)}, 400)]:
                with self.subTest(status=expected), self.assertRaises(HTTPError) as error:
                    urlopen(Request(endpoint, data=b'{}', headers=headers, method='POST'), timeout=2)
                self.assertEqual(error.exception.code, expected)
        self.assertEqual(calls, [])

    def test_source_drift_is_rejected_at_client(self):
        def infer(obs, rgb):
            return dict(self.proposal(obs), service_source_sha256='0' * 64)
        with self.server(infer) as (endpoint, token), self.assertRaisesRegex(ValueError, 'source'):
            self.protocol.HTTPClient(endpoint, token)(observation(), RGB)

    def test_boundaries_reject_stale_ids_targets_and_motor_overrides(self):
        for boundary in ('HOME', 'PARKED', 'RELEASED'):
            obs = observation(boundary)
            self.assertEqual(self.protocol.validate_decision(decision(obs), obs), decision(obs))
            for change in [dict(observation_id='stale'), dict(target_id='R5'),
                           dict(force_N=4), dict(skill='invented')]:
                with self.subTest(boundary=boundary, change=change), self.assertRaises(ValueError):
                    self.protocol.validate_decision(dict(decision(obs), **change), obs)

    def test_sensor_provenance_and_depth_counts_cannot_be_relabeled(self):
        for change in ('hardware', 'calibration', 'depth_count', 'raw_arrays'):
            obs = observation()
            if change == 'hardware': obs['sensor_bundle']['hall']['actual_adc_measurement'] = True
            if change == 'calibration': obs['sensor_bundle']['hall']['calibration_sha256'] = '0' * 64
            if change == 'depth_count': obs['sensor_bundle']['depth']['valid_count'] += 1
            if change == 'raw_arrays': obs['sensor_bundle']['raw_rgbd']['exact_arrays_transferred'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.protocol.validate_observation(obs)

    def test_image_digest_mismatch_is_rejected_before_transport(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.protocol.request_payload(observation(), b'different-image')

    def test_client_requires_explicit_loopback_token_and_bounded_timeout(self):
        for endpoint in ('https://example.invalid/decision', 'http://0.0.0.0:8123/decision',
                         'http://127.0.0.1:8123/decision?token=unsafe'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self.protocol.HTTPClient(endpoint, 'synthetic-token')
        with self.assertRaises(ValueError):
            self.protocol.HTTPClient('http://127.0.0.1:8123/decision', '')
        with self.assertRaises(ValueError):
            self.protocol.HTTPClient('http://127.0.0.1:8123/decision', 'synthetic-token', 181)

    def test_cli_pins_exact_model_and_cleans_image_schema_and_response(self):
        seen = []
        def run(argv, **kwargs):
            seen.append((argv, kwargs))
            self.assertEqual(Path(argv[argv.index('-i') + 1]).read_bytes(), RGB)
            Path(argv[argv.index('-o') + 1]).write_text(json.dumps(decision(observation())))
            return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed"}\n', stderr='')
        with patch.object(self.service.subprocess, 'run', side_effect=run):
            result = self.service.CodexClient('codex')(observation(), RGB)
        argv, kwargs = seen[0]
        self.assertEqual(argv[argv.index('--model') + 1], 'gpt-6-astra')
        self.assertEqual(argv[argv.index('--sandbox') + 1], 'read-only')
        for flag in ('--ignore-user-config', '--ignore-rules', '--ephemeral', '--output-schema'):
            self.assertIn(flag, argv)
        self.assertFalse(Path(kwargs['cwd']).exists())
        self.assertEqual(kwargs['timeout'], 120.)
        self.assertFalse(result['model_fallback'])
        self.assertFalse(result['executed'])
        self.assertIsNone(result['physical_success'])

    def test_cli_timeout_is_one_attempt_without_fallback(self):
        with patch.object(self.service.subprocess, 'run',
                          side_effect=subprocess.TimeoutExpired('codex', 120)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.service.CodexClient('codex')(observation(), RGB)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(Path(run.call_args.kwargs['cwd']).exists())

    def test_failed_cli_turn_cannot_become_a_proposal(self):
        def run(argv, **kwargs):
            Path(argv[argv.index('-o') + 1]).write_text(json.dumps(decision(observation())))
            return SimpleNamespace(returncode=0, stdout='{"type":"turn.failed"}\n', stderr='')
        with patch.object(self.service.subprocess, 'run', side_effect=run), self.assertRaisesRegex(ValueError, 'failed'):
            self.service.CodexClient('codex')(observation(), RGB)


if __name__ == '__main__':
    unittest.main()

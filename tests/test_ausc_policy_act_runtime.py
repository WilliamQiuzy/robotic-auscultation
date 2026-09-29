"""CPU protocol/adapter tests with injected engines; no native model is loaded."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from sim.ausc_policy import act_baseline as a
try:
    from sim.ausc_policy import act_runtime as runtime
except ImportError:
    runtime = None


class FakeEngine:
    def __init__(self, prediction=None):
        self.prediction = np.zeros((1, 10, 7), np.float32) if prediction is None else prediction
        self.batches = []

    def predict(self, batch):
        self.batches.append(batch)
        if set(batch) != {'observation.state', 'observation.environment_state'}:
            raise AssertionError('Labels/images reached inference')
        return self.prediction.copy()


class ACTRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, 'CPU ACT runtime not implemented')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root/'output'; self.output.mkdir()
        self.checkpoint = self.output/'checkpoint_2000'; self.checkpoint.mkdir()
        config = a.read_json('configs/ausc_policy/m0_act_relative_train.json')
        self.config = self.root/'source.json'; a.save_json(self.config, config)
        a.save_json(self.output/'configuration.json', config)
        a.save_json(self.output/'COMPLETE.json', dict(completed=True, mode='train', steps=2000,
            checkpoint_step=2000, action_representation=a.BLOCK_RELATIVE))
        a.save_json(self.output/'native.json', dict(upstream_commit=a.UPSTREAM_COMMIT, source_sha256={}))
        policy_config = dict(type='act', n_obs_steps=1, chunk_size=10, n_action_steps=5,
            input_features={'observation.state': dict(type='STATE',shape=[7]),
                            'observation.environment_state': dict(type='ENV',shape=[12])},
            output_features={'action':dict(type='ACTION',shape=[7])},
            normalization_mapping={key:'IDENTITY' for key in ('STATE','ENV','ACTION')},
            temporal_ensemble_coeff=None, pretrained_backbone_weights=None, use_amp=False,
            pre_norm=False, feedforward_activation='relu', latent_dim=32, **config['model'])
        a.save_json(self.checkpoint/'config.json', policy_config)
        a.save_json(self.root/'result.json', dict(completed=True, config=config))
        self.weights = self.checkpoint/'model.safetensors'; self.weights.write_bytes(b'fake-test-weights')
        self.checkpoint_sha = a.sha256(self.weights)
        stats = {key: dict(low=[lo]*size, high=[hi]*size, actual_clamp=False)
                 for key, size, lo, hi in [('state',19,-2.,2.), ('action',7,-.1,.3)]}
        a.save_json(self.output/'normalization.json', stats)
        source = dict(normalization_denominator='each eligible TRAIN L2 origin once; actions each origin/horizon once',
            train_windows=8685, normalization=dict(path='/source/absolute.json', sha256='a'*64))
        a.save_action_contract(self.output, config, source)
        self.artifacts = runtime.load_artifacts(self.config, self.checkpoint, self.checkpoint_sha)
        self.state = np.arange(19, dtype=np.float32)/20
        self.state[17:] = [-1., -1.]

    def request(self, **changes):
        request = runtime.build_request(self.state, self.artifacts.receipt, request_id='request-1',
            task_id='run-1', episode_id='episode-1', frame_id=12, target_revision=3,
            sim_stamp_s=.6, target_id='R1', phase='FREE')
        request.update(changes)
        return request

    def test_label_free_scaling_and_single_original_anchor_restore(self):
        engine = FakeEngine(); service = runtime.InferenceService(self.artifacts, engine)
        response = service.handle(self.request())
        # z=0 -> midpoint relative action +0.1; add actual q exactly once.
        expected = np.broadcast_to(self.state[:7]+.1, (10, 7))
        np.testing.assert_allclose(response['raw'], expected, atol=1e-7)
        batch = engine.batches[0]
        np.testing.assert_array_equal(batch['observation.state'][0], self.state[:7]/2)
        self.assertEqual(batch['observation.environment_state'].shape, (1, 12))
        self.assertEqual(response['checkpoint_sha256'], self.checkpoint_sha)
        self.assertEqual(response['receipt'], self.artifacts.receipt)
        self.assertEqual(response['frame_id'], 12)
        self.assertEqual(response['episode_id'], 'episode-1')
        self.assertGreaterEqual(response['latency_ms'], 0)

    def test_fail_closed_bad_state_task_receipt_and_model_output(self):
        service = runtime.InferenceService(self.artifacts, FakeEngine())
        for changes in [dict(state=[0]*18), dict(state=[float('nan')]*19), dict(action=[[0]*7]*10),
                        dict(target_id='L1'), dict(phase='APPROACH'), dict(frame_id=-1),
                        dict(target_revision=True), dict(receipt_sha256='b'*64)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                service.handle(self.request(**changes))
        for prediction in [np.zeros((1, 9, 7)), np.full((1, 10, 7), np.nan)]:
            with self.assertRaises(ValueError):
                runtime.InferenceService(self.artifacts, FakeEngine(prediction)).handle(self.request())

    def test_wrong_source_checkpoint_or_absolute_normalizer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            runtime.load_artifacts(self.config, self.checkpoint, '0'*64)
        config = a.read_json(self.config); config['seed'] = 62; a.save_json(self.config, config)
        with self.assertRaises(ValueError):
            runtime.load_artifacts(self.config, self.checkpoint, self.checkpoint_sha)
        a.save_json(self.config, a.read_json(self.output/'configuration.json'))
        contract = a.read_json(self.output/'action_contract.json')
        contract['normalization']['path'] = '/source/absolute.json'
        a.save_json(self.output/'action_contract.json', contract)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            runtime.load_artifacts(self.config, self.checkpoint, self.checkpoint_sha)

    def test_native_constructor_requires_owner_before_model_import(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(a, 'make_native') as native:
            with self.assertRaisesRegex(ValueError, 'owner'):
                runtime.NativeEngine(self.artifacts)
            native.assert_not_called()

    def test_mocked_native_strict_cpu_load_eval_and_no_gradient(self):
        parameter = SimpleNamespace(device=SimpleNamespace(type='cpu'), requires_grad=True)
        policy = Mock(config=SimpleNamespace(device='cpu'), training=True)
        policy.parameters.return_value = [parameter]; policy.buffers.return_value = []
        policy.requires_grad_.side_effect = lambda flag: setattr(parameter, 'requires_grad', flag)
        policy.eval.side_effect = lambda: setattr(policy, 'training', False)
        tensor = Mock(device=SimpleNamespace(type='cpu'))
        tensor.detach.return_value.numpy.return_value = np.zeros((1,10,7), np.float32)
        policy.predict_action_chunk.return_value = tensor
        torch = Mock(); torch.inference_mode.side_effect = nullcontext
        loader = Mock(return_value=(set(), set()))
        package = ModuleType('safetensors'); package.__path__ = []
        child = ModuleType('safetensors.torch'); child.load_model = loader
        with patch.dict('os.environ', {'AUSC_OWNED_RUN':'1'}, clear=True), patch.object(runtime.sys, 'platform', 'linux'), \
             patch.object(a, 'make_native', return_value=(torch,policy,{})) as native, \
             patch.dict(sys.modules, {'safetensors':package, 'safetensors.torch':child}):
            engine = runtime.NativeEngine(self.artifacts)
            self.assertEqual(native.call_args.args[0]['device'], 'cpu')
            self.assertEqual(native.call_args.args[0]['runtime_scope'], 'inference_only')
            self.assertEqual(self.artifacts.config['device'], 'cuda')
            loader.assert_called_once_with(policy, str(self.weights.resolve()), strict=True, device='cpu')
            engine.predict({'observation.state':np.zeros((1,7)), 'observation.environment_state':np.zeros((1,12))})
            torch.inference_mode.assert_called_once()
            parameter.device.type = 'cuda'
            with self.assertRaisesRegex(ValueError, 'CPU'):
                engine.predict({'observation.state':np.zeros((1,7)), 'observation.environment_state':np.zeros((1,12))})

    def test_owner_cli_verify_reads_small_arrays_and_exits_without_server(self):
        inputs, reference = self.root/'inputs.npz', self.root/'reference.npz'
        np.savez(inputs, state19=self.state[None]); np.savez(reference, raw=np.zeros((1,10,7)))
        config = dict(experiment='act_runtime', mode='verify_reload', base=str(self.root), run_root=str(self.root),
            checkpoint_step=2000, inputs_npz=str(inputs), reference_npz=str(reference), atol_rad=1e-6,
            checkpoint_sha256=self.checkpoint_sha, time_limit_seconds=600)
        cp = self.root/'runtime.json'; a.save_json(cp, config); destination = self.root/'verification'
        with patch.object(runtime, 'verify_reload', return_value=dict(passed=True, native_model_loaded=False)) as verify, \
             patch.object(runtime, 'make_server') as server, \
             patch.object(sys, 'argv', ['act_runtime','--config',str(cp),'--output',str(destination)]):
            self.assertEqual(runtime.main(), 0)
            server.assert_not_called()
            np.testing.assert_array_equal(verify.call_args.args[1], self.state[None])
            self.assertEqual(verify.call_args.kwargs['atol'], 1e-6)
            result = a.read_json(destination/'COMPLETE.json')
            self.assertTrue(result['completed'])
            self.assertEqual(result['mode'], 'verify_reload')
            self.assertFalse(result['native_model_loaded'])

    def test_localhost_client_preserves_bindings_and_server_is_single_threaded(self):
        service = runtime.InferenceService(self.artifacts, FakeEngine())
        server = runtime.make_server(service, port=0)
        self.addCleanup(server.server_close)
        self.assertEqual(server.server_address[0], '127.0.0.1')
        import socketserver
        self.assertNotIsInstance(server, socketserver.ThreadingMixIn)
        thread = threading.Thread(target=server.handle_request); thread.start()
        client = runtime.ACTClient(server.server_address[1], self.artifacts.receipt, timeout_s=2.)
        response = client.predict(self.state, request_id='request-1', task_id='run-1', episode_id='episode-1',
            frame_id=12, target_revision=3, sim_stamp_s=.6, target_id='R1', phase='FREE')
        thread.join(2); self.assertFalse(thread.is_alive())
        self.assertEqual(response['target_revision'], 3)

    def test_client_rejects_response_binding_and_deadline(self):
        service = runtime.InferenceService(self.artifacts, FakeEngine())
        correct = service.handle(self.request())
        for key, value in [('request_id','wrong'), ('task_id','wrong'), ('episode_id','wrong'),
                           ('frame_id',13), ('target_revision',4), ('sim_stamp_s',.65),
                           ('checkpoint_sha256','b'*64), ('input_sha256','b'*64)]:
            changed = copy.deepcopy(correct); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_response(changed, self.request(), self.artifacts.receipt)
        first, second = socket.socketpair()
        with first, second, self.assertRaises(TimeoutError):
            runtime.read_message(first, time.monotonic()+.01)

    def test_reload_function_explicitly_reports_injected_engine(self):
        result = runtime.verify_reload(self.artifacts, self.state[None], engine_factory=lambda _: FakeEngine())
        self.assertTrue(result['passed'])
        self.assertFalse(result['native_model_loaded'])
        self.assertEqual(result['max_reload_difference_rad'], 0.)
        with self.assertRaisesRegex(ValueError, 'reference') as failure:
            runtime.verify_reload(self.artifacts, self.state[None], engine_factory=lambda _: FakeEngine(),
                                  expected_raw=np.zeros((1, 10, 7)))
        self.assertFalse(failure.exception.evidence['passed'])
        self.assertGreater(failure.exception.evidence['max_reference_difference_rad'], .09)

    def test_reload_npz_refuses_large_or_wrong_shape_arrays(self):
        path = self.root/'large.npz'
        np.savez(path, state19=np.zeros((33,19), np.float32))
        with self.assertRaisesRegex(ValueError, '32'):
            runtime.read_small_npz(path, 'state19', (19,))
        np.savez(path, state19=np.zeros((1,19)), unused=np.zeros(300000))
        with self.assertRaisesRegex(ValueError, 'small'):
            runtime.read_small_npz(path, 'state19', (19,))

    def test_saved_checkpoint_stub_reproduces_physical_predictions(self):
        # This exercises saved small predictions; it is deliberately not a native reload test.
        root = Path('results/ausc_policy')
        prediction_path = root/'m0_act_relative_train16/checkpoint_2000/predictions.npz'
        input_path = root/'m0_act_fit_analysis/selected_inputs.npz'
        if not prediction_path.is_file() or not input_path.is_file():
            self.skipTest('Local small diagnostic artifacts unavailable; no model download')
        stats = a.read_json(root/'m0_act_relative_train16/normalization.json')
        a.save_json(self.output/'normalization.json', stats)
        contract = a.read_json(self.output/'action_contract.json')
        contract['normalization']['sha256'] = a.sha256(self.output/'normalization.json')
        a.save_json(self.output/'action_contract.json', contract)
        self.artifacts = runtime.load_artifacts(self.config, self.checkpoint, self.checkpoint_sha)
        with np.load(prediction_path,allow_pickle=False) as saved, np.load(input_path,allow_pickle=False) as inputs:
            service = runtime.InferenceService(self.artifacts,
                FakeEngine(a.scale(saved['model_action_physical'][:1], stats['action'])))
            response = service.handle(self.request(state=inputs['state19'][0].tolist()))
            np.testing.assert_allclose(response['raw'], saved['raw'][0], atol=2e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()

"""Source28 publication adapter contracts; all generated payloads are synthetic."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / 'repro/source28'
ADAPTER = CAPSULE / 'verify_capsule.py'


def hashes(body):
    return dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                git_blob=hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest())


class Source28Capsule(unittest.TestCase):
    def verifier(self):
        self.assertTrue(ADAPTER.is_file(), 'Independent source28 capsule verifier required')
        spec = importlib.util.spec_from_file_location('source28_public_verifier', ADAPTER)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        return module

    def test_normal_python_import_preserves_bytecode_flag_and_closed_inventory(self):
        env = dict(os.environ)
        env.pop('PYTHONDONTWRITEBYTECODE', None)
        env.pop('PYTHONPYCACHEPREFIX', None)
        result = subprocess.run([sys.executable, '-c', '''
import sys
from demo_tests.test_source28_capsule import Source28Capsule, CAPSULE
assert not sys.dont_write_bytecode, 'Regression must start with normal non-B Python'
for previous in (False, True):
    sys.dont_write_bytecode = previous
    verifier = Source28Capsule().verifier()
    assert sys.dont_write_bytecode is previous, 'Adapter changed caller bytecode policy'
    assert verifier.verify(CAPSULE) == (244, 2)
'''], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_policy_ci_runs_source28_adapter_and_isolated_original_fixtures(self):
        workflow = (ROOT / '.github/workflows/tests.yml').read_text()
        marker = '      - name: Verify source28 partial capsule and selected CPU fixtures\n'
        self.assertIn(marker, workflow)
        start = workflow.index(marker)
        self.assertGreater(start, workflow.index('      - name: Verify source26'))
        self.assertLess(start, workflow.index('\n  perception:'))
        step = workflow[start:workflow.index('\n  perception:')]
        self.assertIn('python -B -m unittest demo_tests.test_source28_capsule -v\n', step)
        self.assertIn('python -I -B repro/source28/verify_capsule.py --fixtures\n', step)

    def fixture(self, root, verifier):
        body = b'value = 28\n'
        source = root / 'sim/example.py'
        source.parent.mkdir()
        source.write_bytes(body)
        selection = dict(schema='wam_source28_public_selection_v1',
                         scope='PARTIAL_INSPECTION_CPU_CAPSULE',
                         original_runtime=verifier.ORIGINAL_RUNTIME,
                         files=[dict(file='sim/example.py', **hashes(body))],
                         external_inspection_artifacts=[],
                         public_adapters=[])
        (root / 'CODE_COMMIT').write_text(verifier.ORIGINAL_RUNTIME['code_commit'] + '\n')
        (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
        return source, selection

    def test_authenticates_bytes_sha_and_git_blob(self):
        verifier = self.verifier()
        for fault in ('bytes', 'sha256', 'git_blob'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, selection = self.fixture(root, verifier)
                self.assertEqual(verifier.verify(root), (1, 0))
                selection['files'][0][fault] = 1 if fault == 'bytes' else '0' * (40 if fault == 'git_blob' else 64)
                (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
                with self.assertRaisesRegex(ValueError, 'bytes|sha256|git_blob'):
                    verifier.verify(root)

    def test_rejects_unsafe_paths_duplicate_symlink_and_unknown_payload(self):
        verifier = self.verifier()
        for fault in ('traversal', 'absolute', 'duplicate', 'symlink', 'unknown'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, selection = self.fixture(root, verifier)
                if fault == 'traversal':
                    selection['files'][0]['file'] = '../outside.py'
                elif fault == 'absolute':
                    selection['files'][0]['file'] = str(source)
                elif fault == 'duplicate':
                    selection['files'].append(selection['files'][0].copy())
                elif fault == 'symlink':
                    source.rename(root / 'real.py')
                    source.symlink_to(root / 'real.py')
                else:
                    (root / 'unlisted.npz').write_bytes(b'unlisted')
                (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
                with self.assertRaises(ValueError):
                    verifier.verify(root)

    def test_pins_full_runtime_identity_without_claiming_full_inventory(self):
        verifier = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, selection = self.fixture(root, verifier)
            selection['original_runtime'] = dict(selection['original_runtime'], file_count=1)
            (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                verifier.verify(root)
        with self.assertRaisesRegex(ValueError, 'outside capsule'):
            verifier.require_local_module(ROOT / 'sim/ausc_policy/run.py', CAPSULE)

    def test_external_operator_must_stay_separate_from_git_members(self):
        verifier = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, selection = self.fixture(root, verifier)
            selection['files'][0]['file'] = 'ops/wam-act6-heldout-20261107.py'
            (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'External|ops'):
                verifier.verify(root)

    def test_external_operator_bytes_and_original_linux_path_are_pinned(self):
        verifier = self.verifier()
        published = json.loads((CAPSULE / 'SOURCE_SELECTION.json').read_text())
        entry = published['external_inspection_artifacts'][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, selection = self.fixture(root, verifier)
            target = root / entry['file']
            target.parent.mkdir()
            body = (CAPSULE / entry['file']).read_bytes()
            target.write_bytes(body)
            selection['external_inspection_artifacts'] = [entry.copy()]
            (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
            self.assertEqual(verifier.verify(root), (1, 1))
            target.write_bytes(body + b'\n')
            with self.assertRaisesRegex(ValueError, 'bytes|sha256'):
                verifier.verify(root)
            target.write_bytes(body)
            selection['external_inspection_artifacts'][0]['original_linux_runtime_path'] = '/tmp/relocated.py'
            (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'External|runtime path'):
                verifier.verify(root)

    def test_real_capsule_is_bound_and_has_v4_entries(self):
        verifier = self.verifier()
        selected, external = verifier.verify(CAPSULE)
        self.assertGreaterEqual(selected, 237)
        self.assertEqual(external, 2)
        selection = json.loads((CAPSULE / 'SOURCE_SELECTION.json').read_text())
        names = {row['file'] for row in selection['files']}
        for stem in ('m2_act_terminal_data', 'm2_act_frozen26_data', 'm2_act_warm_start'):
            self.assertIn(f'sim/ausc_policy/{stem}.py', names)
        for name in ('test_ausc_act_terminal_v4.py', 'test_ausc_act_training_v4.py'):
            self.assertIn(f'tests/{name}', selection['selected_cpu_tests'])


if __name__ == '__main__':
    unittest.main()

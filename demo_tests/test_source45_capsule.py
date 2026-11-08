"""Partial Source45 publication contracts; payloads in these tests are invented."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / 'repro/source45'
ADAPTER = CAPSULE / 'verify_capsule.py'


class Source45Capsule(unittest.TestCase):
    def verifier(self):
        self.assertTrue(ADAPTER.is_file(), 'Independent Source45 verifier required')
        spec = importlib.util.spec_from_file_location('source45_public_verifier', ADAPTER)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        self.assertIs(sys.dont_write_bytecode, previous)
        return module

    def fixture(self, root, verifier):
        body = b'value = 45\n'
        source = root / 'sim/example.py'
        source.parent.mkdir()
        source.write_bytes(body)
        row = dict(file='sim/example.py', bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                   git_blob=hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest())
        selection = dict(schema='wam_source45_public_selection_v1',
                         scope='PARTIAL_INSPECTION_CPU_CAPSULE',
                         original_runtime=verifier.ORIGINAL_RUNTIME, files=[row], public_adapters=[])
        (root / 'CODE_COMMIT').write_text(verifier.ORIGINAL_RUNTIME['code_commit'] + '\n')
        (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
        return source, selection

    def test_authenticates_bytes_sha_and_original_git_blob(self):
        verifier = self.verifier()
        for fault in ('bytes', 'sha256', 'git_blob'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, selection = self.fixture(root, verifier)
                self.assertEqual(verifier.verify(root), 1)
                selection['files'][0][fault] = 1 if fault == 'bytes' else '0' * (40 if fault == 'git_blob' else 64)
                (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
                with self.assertRaisesRegex(ValueError, 'bytes|sha256|git_blob'):
                    verifier.verify(root)

    def test_rejects_unsafe_duplicate_symlink_and_unlisted_payloads(self):
        verifier = self.verifier()
        for fault in ('traversal', 'absolute', 'duplicate', 'symlink', 'unlisted', 'ops'):
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
                elif fault == 'ops':
                    selection['files'][0]['file'] = 'ops/unreviewed.py'
                else:
                    (root / 'unlisted.npz').write_bytes(b'not a source file')
                (root / 'SOURCE_SELECTION.json').write_text(json.dumps(selection))
                with self.assertRaises(ValueError):
                    verifier.verify(root)

    def test_full_original_context_is_distinct_from_public_selection(self):
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

    def test_fixture_profile_requires_bounded_original_test_nodes(self):
        verifier = self.verifier()
        selected = {'tests/test_ausc_example.py'}
        valid = ['tests/test_ausc_example.py::test_contract']
        self.assertEqual(verifier.fixture_nodes(valid, selected), valid)
        for invalid in ([], valid * 2, ['tests/test_ausc_example.py'], ['tests/other.py::test_contract'],
                        ['../tests/test_ausc_example.py::test_contract'], ['tests/test_ausc_example.py::-k']):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                verifier.fixture_nodes(invalid, selected)

    def test_real_selection_has_current_entries_and_explicit_optional_tensors(self):
        verifier = self.verifier()
        count = verifier.verify(CAPSULE)
        selection = json.loads((CAPSULE / 'SOURCE_SELECTION.json').read_text())
        self.assertEqual(count, len(selection['files']))
        self.assertLess(count, verifier.ORIGINAL_RUNTIME['file_count'])
        self.assertEqual(len(selection['entry_points']), 10)
        names = {row['file'] for row in selection['files']}
        self.assertTrue(set(selection['entry_points']) <= names)
        self.assertIn('sim/ausc_policy/controller.py', names)
        self.assertIn('sim/ausc_policy/scene_store.py', names)
        verifier.fixture_nodes(selection['selected_cpu_tests'], names)
        verifier.fixture_nodes(selection['optional_torch_cpu_tests'], names)
        self.assertIn('test_native_transform_storage_and_signed_compression_are_not_analytic_equality',
                      ' '.join(selection['selected_cpu_tests']))
        tensor_candidate = ('tests/test_ausc_act_training_v9.py::'
                            'test_v9_all40_saved_task_ranking_states_predictions_and_exact_corruption')
        self.assertIn(tensor_candidate, selection['optional_torch_cpu_tests'])
        self.assertNotIn(tensor_candidate, selection['selected_cpu_tests'])


if __name__ == '__main__':
    unittest.main()

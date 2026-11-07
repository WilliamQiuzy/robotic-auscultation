"""Public capsule adapter contract; synthetic files only."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'repro/source26/verify_capsule.py'


class CapsuleVerification(unittest.TestCase):
    def adapter(self):
        self.assertTrue(ADAPTER.is_file(), 'Independent source26 capsule verifier required')
        spec = importlib.util.spec_from_file_location('public_capsule_verifier', ADAPTER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root):
        path = root / 'sim/example.py'
        path.parent.mkdir()
        path.write_bytes(b'value = 26\n')
        entry = dict(file='sim/example.py', bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scope='project_source',
                     dependency_provenance=['synthetic test entry'])
        manifest = dict(scope='PARTIAL_PUBLIC_CAPSULE', code_commit='05e972ca184383a525aff293fc9f3590e546da9c',
                        original_runtime_manifest_sha256='315f021e76f4e606f4a327b053abc22bacf8a1b8c649d35b4bd8ba9e644c743d',
                        files=[entry], public_adapters=['verify_capsule.py'])
        (root / 'CODE_COMMIT').write_text(manifest['code_commit'] + '\n')
        (root / 'SOURCE_SELECTION.json').write_text(json.dumps(manifest))
        return path, manifest

    def test_verifies_selected_bytes_then_rejects_tampering(self):
        verifier = self.adapter()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, _ = self.fixture(root)
            self.assertEqual(verifier.verify(root), 1)
            path.write_bytes(b'value = 19\n')
            with self.assertRaisesRegex(ValueError, 'bytes|sha256'):
                verifier.verify(root)

    def test_rejects_traversal_duplicate_and_symlink(self):
        verifier = self.adapter()
        for fault in ['traversal', 'duplicate', 'symlink']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path, manifest = self.fixture(root)
                if fault == 'traversal':
                    manifest['files'][0]['file'] = '../outside.py'
                elif fault == 'duplicate':
                    manifest['files'].append(manifest['files'][0].copy())
                else:
                    path.rename(root / 'real.py')
                    path.symlink_to(root / 'real.py')
                (root / 'SOURCE_SELECTION.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    verifier.verify(root)

    def test_rejects_old_root_fallback_and_unlisted_source(self):
        verifier = self.adapter()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / 'sim/extra.py').write_text('unexpected = True\n')
            with self.assertRaisesRegex(ValueError, 'Unlisted'):
                verifier.verify(root)
        with self.assertRaisesRegex(ValueError, 'outside capsule'):
            verifier.require_local_module(ROOT / 'sim/ausc_policy/run.py', ROOT / 'repro/source26')

    def test_real_capsule_matches_selection(self):
        verifier = self.adapter()
        self.assertGreaterEqual(verifier.verify(ADAPTER.parent), 228)


if __name__ == '__main__':
    unittest.main()

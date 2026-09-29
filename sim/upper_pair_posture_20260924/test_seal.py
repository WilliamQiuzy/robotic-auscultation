"""Temporary-directory tests for immutable core inventories and mirror receipts."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class SealTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()/"package";self.root.mkdir()
        (self.root/"nested").mkdir()
        (self.root/"nested/data.bin").write_bytes(b"abc\x00def")
        (self.root/"报告 with space.md").write_text("fixed report\n")
        self.manifest=self.root/"package_manifest.json"
        try:self.seal=importlib.import_module("sim.upper_pair_posture_20260924.seal")
        except ImportError:self.seal=None
        self.assertIsNotNone(self.seal,"portable package sealing implementation is missing")

    def create(self):
        return self.seal.create_manifest(self.manifest)

    def test_relative_inventory_verifies_after_copy_to_another_root(self):
        made=self.create();document=json.loads(self.manifest.read_text())
        self.assertEqual(set(document['files']),{'nested/data.bin','报告 with space.md'})
        self.assertEqual(made['file_count'],2)
        self.assertEqual(made['total_bytes'],20)
        mirror=self.root.parent/'mirror';shutil.copytree(self.root,mirror)
        checked=self.seal.verify_manifest(mirror/self.manifest.name)
        self.assertEqual(checked['manifest_sha256'],made['manifest_sha256'])
        self.assertEqual(checked['file_count'],made['file_count'])

    def test_explicit_runtime_junk_is_excluded_but_other_hidden_files_are_core(self):
        (self.root/'__pycache__').mkdir();(self.root/'__pycache__/cache.pyc').write_bytes(b'cache')
        for name in ('.DS_Store','._data.bin','orphan.pyc'):(self.root/name).write_bytes(b'ignored')
        (self.root/'.important').write_bytes(b'core')
        self.create();document=json.loads(self.manifest.read_text())
        self.assertEqual(len(document['files']),3)
        self.assertIn('.important',document['files'])
        (self.root/'__pycache__/new.pyc').write_bytes(b'new')
        self.seal.verify_manifest(self.manifest)

    def test_existing_manifest_is_never_overwritten(self):
        self.create();original=self.manifest.read_bytes()
        with self.assertRaises(ValueError):self.create()
        self.assertEqual(self.manifest.read_bytes(),original)

    def test_missing_extra_and_same_size_changed_core_files_are_rejected(self):
        self.create();target=self.root/'nested/data.bin'
        original=target.read_bytes();target.write_bytes(b'xxxxxxx')
        with self.assertRaisesRegex(ValueError,'changed'):self.seal.verify_manifest(self.manifest)
        target.write_bytes(original);(self.root/'extra.txt').write_text('extra')
        with self.assertRaisesRegex(ValueError,'extra'):self.seal.verify_manifest(self.manifest)
        (self.root/'extra.txt').unlink();target.unlink()
        with self.assertRaisesRegex(ValueError,'missing'):self.seal.verify_manifest(self.manifest)

    def test_symlinks_are_rejected_even_when_name_would_be_ignored(self):
        for name in ('link.bin','.DS_Store'):
            link=self.root/name;link.symlink_to(self.root/'nested/data.bin')
            with self.assertRaisesRegex(ValueError,'symlink'):self.create()
            link.unlink()
        self.assertFalse(self.manifest.exists())

    def test_unsafe_manifest_paths_and_changed_exclusion_policy_are_rejected(self):
        self.create();document=json.loads(self.manifest.read_text())
        for unsafe in ('../outside','/absolute','nested/../escape','nested\\data.bin'):
            changed=dict(document);changed['files']={unsafe:{'sha256':'a'*64,'bytes':0}}
            self.manifest.write_text(json.dumps(changed))
            with self.assertRaises(ValueError,msg=unsafe):self.seal.verify_manifest(self.manifest)
        changed=dict(document);changed['exclusions']={}
        self.manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'exclusion'):self.seal.verify_manifest(self.manifest)

    def test_mid_capture_changes_do_not_publish_manifest(self):
        original=self.seal.digest_file;changed=False
        def mutate_after_hash(path):
            nonlocal changed
            result=original(path)
            if not changed and Path(path).name=='data.bin':
                changed=True;Path(path).write_bytes(b'different bytes')
            return result
        with patch.object(self.seal,'digest_file',side_effect=mutate_after_hash):
            with self.assertRaisesRegex(ValueError,'changed'):self.create()
        self.assertFalse(self.manifest.exists())

    def test_publication_race_removes_new_manifest(self):
        original=self.seal._write_exclusive
        def mutate_after_publish(path,raw):
            original(path,raw)
            (self.root/'nested/data.bin').write_bytes(b'raced publication')
        with patch.object(self.seal,'_write_exclusive',side_effect=mutate_after_publish):
            with self.assertRaisesRegex(ValueError,'changed'):self.create()
        self.assertFalse(self.manifest.exists())

    def test_interrupted_write_does_not_leave_partial_seal(self):
        with patch.object(self.seal.os,'fsync',side_effect=OSError('interrupted write')):
            with self.assertRaises(OSError):self.create()
        self.assertFalse(self.manifest.exists())

    def test_new_mirror_receipts_are_excluded_and_bound_to_manifest(self):
        made=self.create()
        receipt=self.root/'mirror_receipts/mac.json'
        result=self.seal.verify_manifest(self.manifest,receipt=receipt)
        document=json.loads(receipt.read_text())
        self.assertEqual(document['manifest_sha256'],made['manifest_sha256'])
        self.assertEqual(document['status'],'verified')
        self.assertIn('platform',document);self.assertIn('hostname',document)
        checked=self.seal.verify_manifest(self.manifest)
        self.assertEqual(checked['file_count'],2)
        self.assertEqual(result['receipt_path'],'mirror_receipts/mac.json')
        with self.assertRaises(ValueError):self.seal.verify_manifest(self.manifest,receipt=receipt)

    def test_receipt_path_cannot_escape_reserved_directory(self):
        self.create()
        for path in ('outside.json','mirror_receipts/../outside.json','mirror_receipts/nested/../../outside.json'):
            with self.assertRaises(ValueError,msg=path):self.seal.verify_manifest(self.manifest,receipt=path)
        self.assertFalse((self.root/'outside.json').exists())

    def test_arbitrary_files_cannot_hide_inside_receipt_exclusion(self):
        self.create();directory=self.root/'mirror_receipts';directory.mkdir()
        (directory/'hidden_core.mp4').write_bytes(b'not a receipt')
        with self.assertRaisesRegex(ValueError,'receipt'):self.seal.verify_manifest(self.manifest)

    def test_modified_receipt_binding_is_rejected(self):
        self.create();receipt=self.root/'mirror_receipts/mac.json'
        self.seal.verify_manifest(self.manifest,receipt=receipt)
        record=json.loads(receipt.read_text());record['manifest_sha256']='0'*64
        receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'binding'):self.seal.verify_manifest(self.manifest)

    def test_cli_create_verify_and_failure_are_machine_readable(self):
        cli=[sys.executable,str(Path(self.seal.__file__).resolve())]
        made=subprocess.run(cli+['--create',str(self.manifest)],capture_output=True,text=True)
        self.assertEqual(made.returncode,0,made.stdout+made.stderr)
        created=json.loads(made.stdout);self.assertEqual(created['status'],'created')
        checked=subprocess.run(cli+['--verify',str(self.manifest),'--receipt','mirror_receipts/cli.json'],capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
        verified=json.loads(checked.stdout)
        self.assertEqual(verified['manifest_sha256'],created['manifest_sha256'])
        (self.root/'new_core.txt').write_text('extra')
        failed=subprocess.run(cli+['--verify',str(self.manifest)],capture_output=True,text=True)
        self.assertEqual(failed.returncode,1)
        self.assertEqual(json.loads(failed.stdout)['status'],'failed')

    def test_duplicate_json_paths_are_rejected(self):
        self.create();raw=self.manifest.read_text()
        raw=raw.replace('"files": {','"files": {}, "files": {',1)
        self.manifest.write_text(raw)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.seal.verify_manifest(self.manifest)


if __name__=='__main__':unittest.main()

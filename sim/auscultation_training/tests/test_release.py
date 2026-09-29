import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'verify_release.py'
spec = importlib.util.spec_from_file_location('release_under_test', SOURCE)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
ASSET = v.ASSET_PREFIX + 'components/fixed_bridge.obj'

class ReleaseRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wam-release-regression-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in v.CORE_PATHS | v.SCENE_PARENTS | {ASSET}:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture\n')
        files = {}
        for name in v.SCENE_PARENTS | {ASSET}:
            prefix = 'sim/real2sim_20260922/' if name.startswith('sim/real2sim_20260922/') else 'sim/auscultation_assets_20260922/'
            files['source/scene/' + name.removeprefix(prefix)] = {'source_path': '/recorded/project/' + name}
        (self.root / v.DEFAULT_SNAPSHOT).write_text(json.dumps({'files': files}))
        self.manifest = v.generate(self.root, v.DEFAULT_SNAPSHOT)

    def reject(self, manifest):
        result = v.verify(self.root, manifest)
        self.assertFalse(result['ok'], result)
        return result

    def test_complete_and_stdlib_only(self):
        result = v.verify(self.root, self.manifest)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['checked'], len(self.manifest['files']))
        # A larger pytest run may already have imported torch or numpy.
        # Test the verifier's dependency boundary in a fresh isolated process.
        script = ("import runpy,sys; runpy.run_path(sys.argv[1], run_name='release_import_only'); "
                  "assert {'torch','numpy','isaacsim'}.isdisjoint(sys.modules)")
        isolated = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', script, str(SOURCE)],
                                  capture_output=True, text=True)
        self.assertEqual(isolated.returncode, 0, isolated.stdout + isolated.stderr)

    def test_removed_core_and_scene_entries(self):
        for name in (v.MODULE+'results/bc_baseline_v1/policy.pt',
                     v.MODULE+'results/bc_baseline_v2/policy.pt',
                     v.MODULE+'results/bc_dataset_v2/waypoint_annotations_0.json',
                     v.MODULE+'results/planned_paths_R1.json',
                     sorted(v.SCENE_PARENTS)[0], ASSET):
            with self.subTest(name=name):
                m=copy.deepcopy(self.manifest)
                m['files']=[item for item in m['files'] if item['path'] != name]
                self.assertIn('missing required artifact entry: '+name, self.reject(m)['errors'])

    def test_readme_only(self):
        m=copy.deepcopy(self.manifest)
        m['files']=[item for item in m['files'] if item['path']==v.MODULE+'README.md']
        self.reject(m)

    def test_unexpected_allowed_scene_entry(self):
        name=v.ASSET_PREFIX+'unexpected.obj'
        (self.root/name).write_bytes(b'extra')
        self.manifest['files'].append(v.fingerprint(self.root,name))
        self.assertIn('unexpected artifact entry: '+name,self.reject(self.manifest)['errors'])

    def test_selection_required_and_hash_bound(self):
        for mutation in ('missing','wrong_pin','unapproved_snapshot'):
            with self.subTest(mutation=mutation):
                m=copy.deepcopy(self.manifest)
                if mutation=='missing':
                    del m['selection']
                elif mutation=='wrong_pin':
                    m['selection']['source_snapshot_sha256']='0'*64
                else:
                    m['selection']['source_snapshot']=v.MODULE+'README.md'
                self.reject(m)

    def test_snapshot_hash_precedes_selection_parsing(self):
        (self.root/v.DEFAULT_SNAPSHOT).write_text('{"files":{"malicious":"not parsed"}}')
        with patch.object(v,'scene_paths_from_snapshot',side_effect=AssertionError('parsed unverified snapshot')):
            self.reject(self.manifest)

    def test_snapshot_must_have_scene_closure(self):
        (self.root/v.DEFAULT_SNAPSHOT).write_text('{"files":{}}')
        entry=v.fingerprint(self.root,v.DEFAULT_SNAPSHOT)
        self.manifest['files']=[entry if x['path']==v.DEFAULT_SNAPSHOT else x for x in self.manifest['files']]
        self.manifest['selection']['source_snapshot_sha256']=entry['sha256']
        self.reject(self.manifest)

    def test_invalid_and_duplicate_entries(self):
        for name in ('../README.md','/tmp/outside.json','sim/.env', 'sim\\bad.json',v.MODULE+'results/sequence_all10_v2/physics_trace.npz'):
            with self.subTest(name=name):
                m=copy.deepcopy(self.manifest)
                m['files'].append({'path':name,'sha256':'0'*64,'bytes':0})
                self.reject(m)
        m=copy.deepcopy(self.manifest)
        m['files'].append(m['files'][0])
        self.reject(m)

    def test_changed_or_missing_file(self):
        path=self.root/(v.MODULE+'README.md')
        path.write_bytes(b'changed')
        self.reject(self.manifest)
        path.unlink()
        self.reject(self.manifest)

    def test_symlink_rejects(self):
        path=self.root/(v.MODULE+'README.md')
        path.unlink()
        path.symlink_to(self.root/(v.MODULE+'controller.py'))
        self.reject(self.manifest)

    def test_size_cap(self):
        path=self.root/(v.MODULE+'README.md')
        with path.open('wb') as handle:
            handle.truncate(v.MAX_ARTIFACT_BYTES+1)
        self.reject(self.manifest)

    def test_final_open_symlink_race(self):
        name=v.MODULE+'README.md'
        path=self.root/name
        backup=path.with_suffix('.saved')
        outside=self.root.parent/(self.root.name+'-outside')
        outside.write_bytes(b'outside-content')
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        real_open=os.open
        calls=[]
        def swapped(target,flags,*args,**kwargs):
            if target=='README.md':
                calls.append(flags)
                path.rename(backup)
                path.symlink_to(outside)
                try:
                    return real_open(target,flags,*args,**kwargs)
                finally:
                    path.unlink()
                    backup.rename(path)
            return real_open(target,flags,*args,**kwargs)
        with patch.object(os,'open',swapped), patch.object(os,'supports_dir_fd',os.supports_dir_fd|{swapped}):
            with self.assertRaises(OSError):
                v.fingerprint(self.root,name)
        self.assertEqual(len(calls),1)
        self.assertTrue(calls[0] & os.O_NOFOLLOW)
        self.assertEqual(path.read_bytes(),b'fixture\n')

    def test_parent_open_symlink_race(self):
        name=v.MODULE+'README.md'
        path=self.root/'sim'/'auscultation_training'
        backup=path.with_name('saved-training')
        with tempfile.TemporaryDirectory(prefix='outside-release-') as external:
            (Path(external)/'README.md').write_bytes(b'outside-content')
            real_open=os.open
            calls=[]
            def swapped(target,flags,*args,**kwargs):
                if target=='auscultation_training':
                    calls.append(flags)
                    path.rename(backup)
                    path.symlink_to(external,target_is_directory=True)
                    try:
                        return real_open(target,flags,*args,**kwargs)
                    finally:
                        path.unlink()
                        backup.rename(path)
                return real_open(target,flags,*args,**kwargs)
            with patch.object(os,'open',swapped),patch.object(os,'supports_dir_fd',os.supports_dir_fd|{swapped}):
                with self.assertRaises(OSError):
                    v.fingerprint(self.root,name)
            self.assertEqual(len(calls),1)
            self.assertTrue(calls[0] & os.O_NOFOLLOW)
            self.assertTrue(calls[0] & os.O_DIRECTORY)

    def test_open_file_pathname_replacement(self):
        name=v.MODULE+'README.md'
        path=self.root/name
        backup=path.with_suffix('.saved')
        real_read=os.read
        swapped=False
        def read(descriptor,size):
            nonlocal swapped
            data=real_read(descriptor,size)
            if not swapped:
                swapped=True
                path.rename(backup)
                path.write_bytes(b'fixture\n')
            return data
        with patch.object(os,'read',read):
            with self.assertRaisesRegex(ValueError,'changed while hashing'):
                v.fingerprint(self.root,name)

    def test_in_place_rewrite_restoring_mtime(self):
        name=v.MODULE+'README.md'
        path=self.root/name
        before=path.stat()
        real_read=os.read
        changed=False
        def read(descriptor,size):
            nonlocal changed
            data=real_read(descriptor,size)
            if not changed:
                changed=True
                path.write_bytes(b'newdata\n')
                os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            return data
        with patch.object(os,'read',read):
            with self.assertRaisesRegex(ValueError,'changed while hashing'):
                v.fingerprint(self.root,name)

    def test_cli_generation_verification_no_overwrite(self):
        command=[sys.executable,'-B',str(SOURCE),'release.json','--root',str(self.root)]
        generated=subprocess.run(command+['--generate'],capture_output=True,text=True)
        self.assertEqual(generated.returncode,0,generated.stdout+generated.stderr)
        checked=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
        self.assertEqual(subprocess.run(command+['--generate'],capture_output=True).returncode,1)
        manifest=json.loads((self.root/'release.json').read_text())
        manifest['files']=[x for x in manifest['files'] if x['path']!=ASSET]
        (self.root/'release.json').write_text(json.dumps(manifest))
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,1)

if __name__=='__main__':
    unittest.main(verbosity=2)

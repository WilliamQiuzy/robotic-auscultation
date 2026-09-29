import json
from pathlib import Path
import tempfile
import unittest

from sim.auscultation_skill_v1 import provenance as p


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        objects = self.root/'input_snapshot/objects'
        objects.mkdir(parents=True)
        files = {}
        for name in p.REQUIRED_SOURCES:
            data = name.encode()
            import hashlib
            digest = hashlib.sha256(data).hexdigest()
            (objects/digest).write_bytes(data)
            files[p.PREFIX+name] = dict(sha256=digest, source_path='/unused/'+name)
        manifest = self.root/'input_snapshot/manifest.json'
        manifest.write_text(json.dumps(dict(files=files)))
        self.outcome = dict(executed_controller_class=p.CLASS, candidate_method=p.CLASS,
            candidate_policy='revised_shortcut', input_snapshot='input_snapshot/manifest.json',
            input_snapshot_manifest_sha256=p.sha256_file(manifest), planner_calls=[])
        self.save()

    def tearDown(self): self.tmp.cleanup()
    def save(self): (self.root/'outcome.json').write_text(json.dumps(self.outcome))
    def bind(self):
        p.bind_manifest(self.root)
        self.outcome = json.loads((self.root/'outcome.json').read_text())

    def test_actual_identity_bound_and_parent_impersonation_rejected(self):
        self.bind()
        self.assertEqual(p.verify_candidate_sources(self.root,self.outcome)['status'],'passed')
        self.outcome['candidate_method']='sim.curobo_expert_v1.controller.ExpertController'
        self.assertEqual(p.verify_candidate_sources(self.root,self.outcome)['status'],'failed')

    def test_source_tampering_detected(self):
        self.bind()
        manifest=json.loads((self.root/'candidate_sources_manifest.json').read_text())
        (self.root/manifest['files']['executor.py']['path']).write_text('changed')
        self.assertEqual(p.verify_candidate_sources(self.root,self.outcome)['status'],'failed')

    def test_student_without_checkpoint_rejected(self):
        self.outcome.update(executed_controller_class=p.STUDENT_CLASS,candidate_method=p.STUDENT_CLASS)
        self.save(); self.bind()
        self.assertEqual(p.verify_candidate_sources(self.root,self.outcome)['status'],'failed')

    def test_bind_refuses_to_supply_controller_identity(self):
        del self.outcome['executed_controller_class']; self.save()
        with self.assertRaises(ValueError): p.bind_manifest(self.root)


if __name__=='__main__': unittest.main()

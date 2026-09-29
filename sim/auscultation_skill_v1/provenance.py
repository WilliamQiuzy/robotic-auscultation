"""Bind the executed skill implementation without relabeling it as its parent."""
import json
from pathlib import Path

from sim.expert_quality_20260924.quality import sha256_file, verify_hashes

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CLASS = 'sim.auscultation_skill_v1.controller.ExpertController'
STUDENT_CLASS = 'sim.auscultation_skill_v1.student.StudentController'
CLASSES = (CLASS, STUDENT_CLASS)
SCHEMA = 'wam_contact_skill_sources_v1'
PREFIX = 'source/contact_skill/'
REQUIRED_SOURCES = frozenset(('controller.py', 'curve.py', 'executor.py',
    'config.py', 'contract.py', 'contract.json', 'provenance.py', 'run.py', 'runtime.py', 'audits.py', 'trace_binding.py',
    'parent/curobo_expert_v1/controller.py', 'parent/curobo_expert_v1/withdrawal.py',
    'parent/curobo_expert_v1/timing.py', 'parent/curobo_expert_v1/static_world.py',
    'parent/curobo_expert_v1/fast_collision.py',
    'parent/expert_quality_20260924/candidate_controller.py',
    'parent/expert_quality_20260924/candidate_planner.py'))


def implementation_sources(extra=None):
    files = {p.name: p for p in ROOT.glob('*.py')}
    for name in ('contract.json', 'conditions.json'):
        if (ROOT/name).is_file(): files[name] = ROOT/name
    for package in ('curobo_expert_v1', 'expert_quality_20260924',
                    'upper_pair_posture_20260924'):
        for path in (ROOT.parent/package).glob('*.py'):
            files['parent/'+package+'/'+path.name] = path
    if extra:
        for key, path in extra.items():
            if key in files or Path(key).is_absolute() or '..' in Path(key).parts:
                raise ValueError('invalid extra source key: '+key)
            files[key] = Path(path).resolve()
    if not REQUIRED_SOURCES.issubset(files):
        raise ValueError('missing skill implementation: '+str(sorted(REQUIRED_SOURCES-set(files))))
    return {'contact_skill/'+key: path for key, path in files.items()}


def bind_manifest(output):
    output = Path(output).resolve()
    outcome_path = output/'outcome.json'
    outcome = json.loads(outcome_path.read_text())
    identity = outcome.get('executed_controller_class')
    if identity not in CLASSES or outcome.get('candidate_method') != identity:
        raise ValueError('controller must report its actual skill identity before binding')
    if outcome.get('candidate_policy') != 'revised_shortcut':
        raise ValueError('only the checked revised shortcut planner is supported')
    snapshot = json.loads((output/'input_snapshot/manifest.json').read_text())
    files = {key[len(PREFIX):]: {'path': 'input_snapshot/objects/'+item['sha256'],
             'sha256': item['sha256']} for key, item in snapshot['files'].items()
             if key.startswith(PREFIX)}
    if not REQUIRED_SOURCES.issubset(files):
        raise ValueError('incomplete captured skill implementation')
    generated = {str(p.relative_to(output)): sha256_file(p)
        for p in sorted((output/'planner_requests').rglob('*')) if p.is_file()}
    manifest = dict(schema=SCHEMA, derived_controller=identity,
        candidate_policy='revised_shortcut', files=files, generated_files=generated)
    destination = output/'candidate_sources_manifest.json'
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    outcome.update(candidate_sources_manifest=destination.name,
                   candidate_sources_manifest_sha256=sha256_file(destination))
    outcome_path.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
    return manifest


def verify_candidate_sources(source, outcome):
    source = Path(source).resolve(); hashes = {}; reasons = []
    try:
        identity = outcome.get('executed_controller_class')
        if identity not in CLASSES or outcome.get('candidate_method') != identity:
            raise ValueError('executed controller identity mismatch')
        if outcome.get('candidate_policy') != 'revised_shortcut':
            raise ValueError('unexpected planner policy')
        name = outcome['candidate_sources_manifest']
        path = (source/name).resolve()
        if source not in path.parents: raise ValueError('unsafe manifest path')
        expected = outcome['candidate_sources_manifest_sha256']
        if sha256_file(path) != expected: raise ValueError('candidate manifest hash mismatch')
        hashes[name] = expected
        manifest = json.loads(path.read_text())
        if (manifest.get('schema') != SCHEMA or manifest.get('derived_controller') != identity
                or manifest.get('candidate_policy') != outcome['candidate_policy']):
            raise ValueError('manifest identity mismatch')
        files = manifest['files']
        if not isinstance(files, dict) or not REQUIRED_SOURCES.issubset(files):
            raise ValueError('missing required implementation')
        if identity == STUDENT_CLASS and not {'student.py', 'policy.py', 'observations.py',
                                             'learning/checkpoint', 'learning/manifest'}.issubset(files):
            raise ValueError('student requires captured network and checkpoint')
        snapshot_path = (source/outcome['input_snapshot']).resolve()
        if source not in snapshot_path.parents: raise ValueError('unsafe snapshot path')
        snapshot_sha = outcome['input_snapshot_manifest_sha256']
        if sha256_file(snapshot_path) != snapshot_sha: raise ValueError('snapshot hash mismatch')
        hashes[outcome['input_snapshot']] = snapshot_sha
        snapshot = json.loads(snapshot_path.read_text())
        captured = {k[len(PREFIX):]: v for k,v in snapshot['files'].items() if k.startswith(PREFIX)}
        if set(files) != set(captured): raise ValueError('captured source set differs')
        for key, item in files.items():
            digest = captured[key]['sha256']
            if (item['sha256'] != digest or (source/item['path']).resolve() !=
                    (snapshot_path.parent/'objects'/digest).resolve()):
                raise ValueError('source differs from snapshot: '+key)
            hashes[item['path']] = digest
        generated = manifest['generated_files']
        actual = {str(p.relative_to(source)) for p in (source/'planner_requests').rglob('*') if p.is_file()}
        if not isinstance(generated, dict) or set(generated) != actual:
            raise ValueError('planner evidence file set differs')
        # This release uses no external planner subprocess or cuRobo fallback.
        if outcome.get('planner_calls') or generated:
            raise ValueError('unexpected external planner calls')
        hashes.update(generated)
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append('contact_skill_sources:'+str(error))
    reasons.extend(verify_hashes(source, hashes))
    return dict(status='failed' if reasons else 'passed', artifact_sha256=hashes, reasons=reasons)

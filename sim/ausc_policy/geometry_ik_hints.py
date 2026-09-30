"""Hash-bound current-geometry seeds; no IK, collision or motion authorization.

Optional geometry_plan config: ``ik_hints: {L3: {file: ABSOLUTE_PATH,
sha256: HEX_SHA256}}``. Each ``current_geometry_ik_hints_v1`` document binds
geometry_id, geometry_source_sha256, geometry_artifacts, contact_surface,
the full target record, world_from_left_base, loaded_compression_m and
patient_provenance (explicit null when absent). ``candidate_file`` is absolute
or relative to the hint document; ``candidate_file_sha256`` must also match
the freshly written fit JSON byte for byte. Candidate indices are never remapped.

``hints`` contains unique candidate_index/loaded_q and optional pre_q records.
collision_checked, path_checked, physics_executed and training_eligible must
all be false. An optional note is descriptive only. The existing planner must
solve each seed again and apply all its original posture/collision/path checks.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .geometry_epoch import require_current
from .retainer_contact import CONTACT_SURFACE


_FALSE_FLAGS = ('collision_checked', 'path_checked', 'physics_executed', 'training_eligible')
_ARTIFACT_KEYS = ('collision_geometry_sha256', 'collision_arrays_sha256')


def _check(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    _check(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
           'IK hints require a lowercase SHA256')
    return value


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _check(key not in result, 'Duplicate IK hint JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite IK hint JSON: '+value)
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    _check(isinstance(result, dict), 'IK hint/fit JSON object required')
    return result


def _equal(left, right):
    # Preserve bool/number and integer/float distinctions in binding metadata.
    options = dict(sort_keys=True, separators=(',', ':'), allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


class BoundIKHints:
    """Bind fresh fit indices once, then recheck files before use and completion."""
    def __init__(self, config, geometry_artifacts, patient_provenance):
        self.entries = deepcopy(config.get('ik_hints', {}))
        _check(isinstance(self.entries, dict), 'ik_hints must be an explicit target/file/SHA mapping')
        self._files, self._bound = {}, {}
        if not self.entries:
            return
        require_current(config)
        _check(config.get('fit_refinement_diagnostic') is None,
               'IK hints cannot be consumed by refinement-only diagnostics')
        targets = config['targets']
        self._targets = {target['name']: deepcopy(target) for target in targets}
        _check(len(self._targets) == len(targets) and set(self.entries) <= set(self._targets),
               'Duplicate targets or unused IK hint target binding')
        _check(set(geometry_artifacts) == set(_ARTIFACT_KEYS) and
               all(_digest(geometry_artifacts[key]) == config[key] for key in _ARTIFACT_KEYS),
               'IK hint collision artifacts differ from current config')
        self._context = dict(geometry_id=config['geometry_id'],
            geometry_source_sha256=config['geometry_source_sha256'], geometry_artifacts=deepcopy(geometry_artifacts),
            contact_surface=CONTACT_SURFACE, world_from_left_base=deepcopy(config['world_from_left_base']),
            loaded_compression_m=config['loaded_compression_m'], patient_provenance=deepcopy(patient_provenance))
        for name, binding in self.entries.items():
            _check(isinstance(name, str) and re.fullmatch('[LR][1-5]', name) is not None,
                   'Invalid IK hint target name')
            _check(isinstance(binding, dict) and set(binding) == {'file', 'sha256'} and
                   isinstance(binding['file'], str) and Path(binding['file']).is_absolute(),
                   'IK hint config requires an absolute file path and SHA256')
            _digest(binding['sha256'])

    def _pin(self, path, expected):
        path = Path(path).absolute()
        _digest(expected)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError('IK hint bound file unavailable: '+str(path)) from error
        _check(hashlib.sha256(data).hexdigest() == expected, 'IK hint bound file SHA changed: '+str(path))
        _check(path not in self._files or self._files[path] == expected, 'Conflicting IK hint file SHA')
        self._files[path] = expected
        return data

    def _recheck(self):
        for path, digest in list(self._files.items()):
            self._pin(path, digest)

    def bind(self, target, fresh_fit_path):
        name = target['name']
        if name not in self.entries:
            return None
        _check(name not in self._bound, 'IK hint target already bound')
        _check(_equal(target, self._targets[name]), 'IK hint planning target changed')
        binding = self.entries[name]
        source_hint = Path(binding['file'])
        hint_bytes = self._pin(source_hint, binding['sha256'])
        document = _json(hint_bytes)
        expected = dict(self._context, target=target)
        keys = set(expected) | set(_FALSE_FLAGS) | {'schema', 'candidate_file', 'candidate_file_sha256', 'hints'}
        _check(set(document) in (keys, keys | {'note'}) and
               document.get('schema') == 'current_geometry_ik_hints_v1', 'Unsupported or incomplete IK hint schema')
        _check('note' not in document or isinstance(document['note'], str), 'Invalid IK hint note')
        for key, value in expected.items():
            _check(_equal(document[key], value), 'IK hint current binding mismatch: '+key)
        _check(all(document[key] is False for key in _FALSE_FLAGS), 'IK hints must remain unqualified seeds')
        filename = document['candidate_file']
        _check(isinstance(filename, str) and filename.strip(), 'IK hint candidate file required')
        source_fit = Path(filename)
        if not source_fit.is_absolute():
            source_fit = source_hint.parent/source_fit
        source_fit = source_fit.absolute()
        fit_digest = _digest(document['candidate_file_sha256'])
        self._pin(source_fit, fit_digest)
        fresh_fit = Path(fresh_fit_path).absolute()
        fresh_bytes = self._pin(fresh_fit, fit_digest)
        report = _json(fresh_bytes)
        for key in ('geometry_artifacts', 'contact_surface', 'loaded_compression_m'):
            _check(_equal(report.get(key), expected[key]), 'IK hint fresh fit binding mismatch: '+key)
        candidates, hints = report.get('candidates'), document['hints']
        _check(isinstance(candidates, list) and isinstance(hints, list) and hints, 'IK hints require candidates and seeds')
        indices = set()
        for hint in hints:
            _check(isinstance(hint, dict) and set(hint) in
                   ({'candidate_index', 'loaded_q'}, {'candidate_index', 'loaded_q', 'pre_q'}), 'Invalid IK seed schema')
            index = hint['candidate_index']
            _check(type(index) is int and 0 <= index < len(candidates) and index not in indices,
                   'Invalid/duplicate IK hint candidate index')
            indices.add(index)
            candidate = candidates[index]
            _check(isinstance(candidate, dict) and candidate.get('passed') is True and
                   candidate.get('contact_surface') == CONTACT_SURFACE and
                   _equal(candidate.get('raw_anchor_world_m'), target['position_world_m']),
                   'IK hint candidate does not bind the current target fit')
            for key in ('loaded_q', 'pre_q'):
                if key not in hint:
                    continue
                value = hint[key]
                _check(isinstance(value, list) and len(value) == 7 and
                       all(type(v) in (int, float) for v in value) and np.isfinite(value).all(),
                       'IK hint '+key+' requires seven finite numeric seed values')
        copy_path = fresh_fit.parent/(name+'_ik_hints.json')
        with copy_path.open('xb') as stream:
            stream.write(hint_bytes)
        self._pin(copy_path, binding['sha256'])
        receipt = dict(schema='current_geometry_ik_hint_binding_v1', **deepcopy(expected),
            source_hint=dict(file=str(source_hint), sha256=binding['sha256']),
            source_fit=dict(file=str(source_fit), sha256=fit_digest),
            fresh_fit=dict(file=str(fresh_fit), sha256=fit_digest),
            hint_copy=dict(file=copy_path.name, sha256=binding['sha256']), candidate_indices=sorted(indices),
            seeds_only=True, **{key: False for key in _FALSE_FLAGS},
            scope='Seed provenance only; the planner must re-solve and apply all original posture, collision and path checks')
        receipt_path = fresh_fit.parent/(name+'_ik_hint_receipt.json')
        receipt_bytes = (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()
        with receipt_path.open('xb') as stream:
            stream.write(receipt_bytes)
        receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
        self._pin(receipt_path, receipt_digest)
        reference = dict(file=receipt_path.name, sha256=receipt_digest)
        self._bound[name] = dict(hints=deepcopy(hints), reference=reference)
        return deepcopy(reference)

    def planner_hints(self, name):
        _check(name in self._bound, 'IK hint target has not been bound to fresh fits')
        self._recheck()
        return deepcopy(self._bound[name]['hints'])

    def finalize(self):
        if not self.entries:
            return None
        _check(set(self._bound) == set(self.entries), 'Unused IK hints were not bound to fresh fits')
        self._recheck()
        return dict(schema='current_geometry_ik_hint_final_verification_v1', verified=True,
                    bindings={name: deepcopy(value['reference']) for name, value in self._bound.items()},
                    **{key: False for key in _FALSE_FLAGS}, scope='All bound hint/fit bytes rechecked after planning')

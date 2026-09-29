"""Require independently repeated exports before binding collision geometry."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .scene import ROOT, PROJECT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a, b = ROOT/'results/export_a', ROOT/'results/export_b'
    metadata = [json.loads((p/'collision_geometry.json').read_text()) for p in (a, b)]
    assert metadata[0] == metadata[1], 'Repeated collider metadata differs'
    with np.load(a/'collision_geometry.npz', allow_pickle=False) as x, np.load(b/'collision_geometry.npz', allow_pickle=False) as y:
        assert set(x.files) == set(y.files)
        count = len(x.files)
        for name in x.files:
            assert x[name].dtype == y[name].dtype and np.array_equal(x[name], y[name]), name
    manifests = [json.loads((p/'collision_source_manifest.json').read_text()) for p in (a, b)]
    assert manifests[0]['source_files'] == manifests[1]['source_files'], 'Export source set changed'
    sources = {}
    for absolute, expected in manifests[0]['source_files'].items():
        p = Path(absolute).resolve()
        assert p.is_relative_to(PROJECT.resolve()) and sha(p) == expected, absolute
        sources[str(p.relative_to(PROJECT.resolve()))] = expected
    for name in ['collision_geometry.json', 'collision_geometry.npz']:
        for directory, manifest in zip((a, b), manifests):
            assert sha(directory/name) == manifest['outputs'][name]
        target = ROOT/'results'/name
        assert not target.exists(), 'Refuse replacing bound geometry'
        shutil.copy2(a/name, target)
    report = {'equal_exported_arrays': True, 'equal_metadata': True, 'array_count': count,
              'collider_count': len(metadata[0]['colliders']),
              'unsupported': metadata[0]['unsupported'],
              'existing_outputs_sha256': {n: sha(ROOT/'results'/n) for n in ['collision_geometry.json', 'collision_geometry.npz']},
              'source_files_relative': sources,
              'independent_export_manifests': {str(p.relative_to(PROJECT)): sha(p) for p in [a/'collision_source_manifest.json', b/'collision_source_manifest.json']},
              'scope': 'Two exports of the approved physical stage match every collider array and metadata; not continuous collision proof.'}
    (ROOT/'results/collision_regeneration_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GEOMETRY_BOUND', count, 'arrays;', report['collider_count'], 'colliders', flush=True)


if __name__ == '__main__':
    main()

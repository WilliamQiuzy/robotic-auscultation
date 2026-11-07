"""Read-only CPU bridge to the complete, immutable original source26 archive.

No caller-supplied Python or archive root is executed. The original package
initializer and its original absolute path checks run unchanged in -I -B.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile

import numpy as np

from . import m2_act_auxiliary_data as old, m2_act_recovery_scope as scope

rt = old.rt
need = rt.need
SOURCE26_ROOT = '/opt/wam-runtime/code/wam-act-recovery-20261106-v26'
SOURCE26_COMMIT = '05e972ca184383a525aff293fc9f3590e546da9c'
SOURCE26_MANIFEST = dict(path=SOURCE26_ROOT+'/SOURCE_MANIFEST.json', bytes=287532,
    sha256='315f021e76f4e606f4a327b053abc22bacf8a1b8c649d35b4bd8ba9e644c743d')
SOURCE26_FILES = 1660
PARENT_COMPLETE = dict(path='/opt/wam-runtime/fast-wam/datasets/act-conditional-full-chunks-20261107-v1/COMPLETE.json',
    bytes=1202, sha256='d84afc9f705f25de34bc2aeb253ee79b43a52981ad6fdc57e3f19265da47b075')


def array_proof(array):
    value = np.asarray(array)
    return dict(shape=list(value.shape), dtype=value.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def _regular(path):
    path = Path(path)
    for part in (path, *path.parents):
        need(not part.is_symlink(), 'Symlink in frozen source26 archive')
    need(stat.S_ISREG(path.lstat().st_mode), 'Regular frozen source26 file required')


def _verify_inventory(root, manifest_ref, *, expected_count):
    """Internal inventory checker; production entry point pins every argument."""
    root = Path(root)
    need(root.is_absolute() and root.resolve() == root, 'Canonical frozen source26 root required')
    _regular(root/'SOURCE_MANIFEST.json'); _regular(root/'CODE_COMMIT')
    need((root/'CODE_COMMIT').read_text().strip() == SOURCE26_COMMIT, 'Frozen source26 commit differs')
    rows = rt.read_bound(manifest_ref)
    need(isinstance(rows, list) and len(rows) == expected_count, 'Complete frozen source26 inventory required')
    files = {}
    for row in rows:
        need(isinstance(row, dict) and set(row) == {'file', 'bytes', 'sha256'}, 'Exact frozen source26 manifest row required')
        name = row['file']; relative = PurePosixPath(name) if isinstance(name, str) else None
        need(relative is not None and name == relative.as_posix() and not relative.is_absolute() and
            relative.parts and all(p not in ('.', '..') for p in relative.parts) and
            name not in files and name not in ('CODE_COMMIT', 'SOURCE_MANIFEST.json'),
            'Unique safe relative frozen source26 file required')
        path = root/name; _regular(path)
        ref = dict(path=str(path), bytes=row['bytes'], sha256=row['sha256'])
        rt.bound_path(ref); files[name] = ref
    actual = set()
    for path in root.rglob('*'):
        mode = path.lstat().st_mode
        need(not stat.S_ISLNK(mode), 'Symlink in frozen source26 inventory')
        if stat.S_ISDIR(mode): continue
        _regular(path); actual.add(path.relative_to(root).as_posix())
    need(actual == set(files) | {'SOURCE_MANIFEST.json', 'CODE_COMMIT'}, 'Unknown or omitted frozen source26 file')
    return files


def require_archive(manifest_ref=None):
    manifest_ref = SOURCE26_MANIFEST if manifest_ref is None else manifest_ref
    need(manifest_ref == SOURCE26_MANIFEST, 'Pinned original source26 manifest required; arbitrary archives rejected')
    return _verify_inventory(Path(SOURCE26_ROOT), manifest_ref, expected_count=SOURCE26_FILES)


def generator_reference():
    return require_archive()['sim/ausc_policy/m2_student_oracle_proposals.py']


# This fixed bridge imports only original26 project modules. No project module
# from the future V4 numerical freeze is added to the child's import path.
_PROGRAM = '''import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
sys.path.insert(0,str(root))
from sim.ausc_policy import m2_act_conditional_data as original
from sim.ausc_policy import m2_student_oracle_proposals as oracle
for module in (original,oracle):
    assert Path(module.__file__).resolve()==root/('sim/ausc_policy/'+module.__name__.split('.')[-1]+'.py')
payload=json.load(sys.stdin)
if payload['operation']=='parent':
    value=original.load_preparation(payload['complete'],payload['allowed'])
    original.validate_lineage(value,payload['allowed'])
    arrays={k:value['samples'][k] for k in original.old.ARRAY_KEYS}
    meta={k:v for k,v in value.items() if k!='samples'}
    meta['samples']={k:v for k,v in value['samples'].items() if k not in arrays}
    meta['array_proofs']={k:dict(shape=list(v.shape),dtype=v.dtype.str,sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in arrays.items()}
    directory=Path(payload['output'])
    with (directory/'ARRAYS.npz').open('xb') as stream: np.savez_compressed(stream,**arrays)
    metadata=original.rt.write_new(directory/'METADATA.json',meta)
    print(json.dumps(dict(metadata=metadata,arrays=original.rt.reference(directory/'ARRAYS.npz')),allow_nan=False))
elif payload['operation']=='sources':
    results=[]
    for source in payload['sources']:
        binding=oracle.authenticate_source(source)
        results.append(dict(metadata=binding['metadata'],pose_id=original.old.installed_pose_id(binding['metadata'])))
    print(json.dumps(results,allow_nan=False))
elif payload['operation']=='queries':
    results=[]
    for group in payload['groups']:
        binding=oracle.authenticate_source(group['source'])
        rows=[]
        for row in group['rows']:
            q,snapshot,commands=oracle.query_at(binding,row)
            record=dict(query=q,snapshot=snapshot,commands=commands.tolist())
            proposal_ref=group.get('proposals',{}).get(str(row))
            if proposal_ref is not None:
                proposal=original.rt.read_bound(proposal_ref)
                assert proposal.get('query')==q, 'Proposal differs from authenticated actual query'
                if proposal.get('kind')=='recovery_proposal':
                    preview=proposal['preview']
                    assert preview.get('input_state')==snapshot and preview.get('input_sha256')==original.rt.digest(snapshot), 'Retained q/v/a differs'
                    from sim.ausc_policy import m2_given_contact_recovery as recovery
                    replay=recovery.preview_cells(original.rt.handoff.restore_trajectory(snapshot),commands,preview['cells'],lambda q:None)
                    assert all(preview.get(k)==v for k,v in replay.items()), 'Original full computed command prefix/seam differs'
                record['proposal_authenticated_sha256']=proposal_ref['sha256']
            rows.append(record)
        results.append(rows)
    print(json.dumps(results,allow_nan=False))
else: raise ValueError('Unknown fixed source26 bridge operation')
'''


def _run(payload):
    require_archive()  # Every file is authenticated before ANY archive execution.
    need(sys.platform == 'linux', 'Original source26 validation requires Linux CPU environment')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-I', '-B', '-c', _PROGRAM, SOURCE26_ROOT],
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, check=False,
        timeout=1800, env=environment, cwd=SOURCE26_ROOT)
    need(result.returncode == 0, 'Original source26 validation failed: '+result.stderr[-8000:])
    require_archive()  # Reject changes during validation as well.
    return json.loads(result.stdout)


def load_parent(complete_ref, allowed_train_ids):
    need(complete_ref==PARENT_COMPLETE, 'Pinned complete ACT5 parent required')
    with tempfile.TemporaryDirectory(prefix='wam-source26-parent-') as directory:
        directory = str(Path(directory).resolve())
        refs = _run(dict(operation='parent', complete=complete_ref, allowed=sorted(allowed_train_ids), output=directory))
        need(set(refs) == {'metadata', 'arrays'} and refs['metadata']['path'] == str(Path(directory)/'METADATA.json') and
            refs['arrays']['path'] == str(Path(directory)/'ARRAYS.npz'), 'Exact source26 transport artifacts required')
        meta = rt.read_bound(refs['metadata'])
        with np.load(rt.bound_path(refs['arrays']), allow_pickle=False) as archive:
            need(set(archive.files) == set(old.ARRAY_KEYS), 'Exact original26 parent transport arrays required')
            arrays = {k:archive[k].copy() for k in old.ARRAY_KEYS}
        need(meta.pop('array_proofs') == {k:array_proof(v) for k,v in arrays.items()}, 'Original26 parent transport bit proof differs')
        meta['samples'].update(arrays)
        meta['archive_validation'] = dict(manifest=SOURCE26_MANIFEST, code_commit=SOURCE26_COMMIT,
            transport_metadata_sha256=refs['metadata']['sha256'], array_proofs={k:array_proof(v) for k,v in arrays.items()})
        return meta


def authenticate_queries(groups):
    """Authenticate owner/child/model/raw/H2 and retained q/v/a + three commands."""
    _gate_sources([g['source'] for g in groups])
    return _run(dict(operation='queries', groups=groups))


def authenticate_sources(sources):
    """Authenticate a failed source even when all its semantic slots are missing."""
    _gate_sources(sources)
    return _run(dict(operation='sources', sources=sources))


def _gate_sources(sources):
    for source in sources:
        need(isinstance(source,dict) and set(source)=={'config','owner','child','native'}, 'Original source26 source refs required')
        config = rt.read_bound(source['config'])
        meta = config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
        old._train(meta.get('subject_id'),meta.get('split'),scope.TRAIN16_IDS)

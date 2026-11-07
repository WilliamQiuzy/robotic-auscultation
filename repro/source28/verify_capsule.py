#!/usr/bin/env python3
"""Verify the partial source28 selection; optionally run isolated CPU fixtures.

New public tooling, not an original frozen Git blob or experiment owner.
External operator copies are inspected and hashed only, never executed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

ORIGINAL_RUNTIME = dict(
    code_commit='d932993b13030da7873d04a4a9471df99bb836ec', file_count=1673,
    source_manifest_bytes=289698,
    source_manifest_sha256='f05a78822bcc6febbaa7f4a7a8de2cabd9aa33daed3d648b90053cb336edb89c',
    archive_bytes=6122793,
    archive_sha256='ac7da2332d17484fd3775679264e9cd1f56cbe10381f2234b76c7ef298397092')
EXTERNAL_OPERATORS = {
    'ops/wam-act6-official-round-20261107.py': (
        46964, 'ba7cbe84bffe7c4fcc016653de9b20dcfddb99296746d95b3146bbb1a87684f8',
        '/opt/wam-runtime/fast-wam/diagnostics/run_act6_official_round_20261107.py'),
    'ops/wam-act6-heldout-20261107.py': (
        52214, '22f197d16f7a5345e454319306ab4bf8bfaa1203a358e24bd2728433d587f346',
        '/opt/wam-runtime/fast-wam/diagnostics/run_act6_heldout_20261107.py')}
PUBLIC_ADAPTERS = {'verify_capsule.py', 'PLAN.md', 'CPU_FIXTURE_CHECK.json'}


def require_local_module(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Project module outside capsule: {resolved}')
    return resolved


def verify(root):
    root = Path(root).resolve()
    selection = json.loads((root / 'SOURCE_SELECTION.json').read_text())
    if (selection.get('schema') != 'wam_source28_public_selection_v1'
            or selection.get('scope') != 'PARTIAL_INSPECTION_CPU_CAPSULE'
            or selection.get('original_runtime') != ORIGINAL_RUNTIME
            or (root / 'CODE_COMMIT').read_text().strip() != ORIGINAL_RUNTIME['code_commit']):
        raise ValueError('Partial capsule provenance differs')
    names = set()
    for external, rows in ((False, selection['files']),
                           (True, selection['external_inspection_artifacts'])):
        for entry in rows:
            name = entry['file']
            relative = PurePosixPath(name)
            if (name != relative.as_posix() or relative.is_absolute() or not relative.parts
                    or '..' in relative.parts or name in names
                    or name in {'CODE_COMMIT', 'SOURCE_SELECTION.json'}):
                raise ValueError('Unique safe relative capsule paths required')
            if external:
                if (name not in EXTERNAL_OPERATORS or entry.get('git_blob') is not None
                        or entry.get('scope') != 'EXTERNAL_INSPECTION_ONLY_NOT_FULL_MANIFEST_MEMBER'
                        or (entry['bytes'], entry['sha256'], entry.get('original_linux_runtime_path'))
                        != EXTERNAL_OPERATORS[name]):
                    raise ValueError('External inspection artifact identity differs')
            elif relative.parts[0] == 'ops':
                raise ValueError('External ops cannot be original full-manifest Git members')
            elif relative.suffix not in {'.py', '.json', '.urdf'}:
                raise ValueError('Only original selected source/config/URDF text is allowed')
            path = root / relative
            require_local_module(path, root)
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('Symlink in capsule path')
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('Regular selected file required')
            body = path.read_bytes()
            if len(body) != entry['bytes'] or hashlib.sha256(body).hexdigest() != entry['sha256']:
                raise ValueError(f'Selected bytes/sha256 differ: {name}')
            if not external:
                blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
                if blob != entry['git_blob']:
                    raise ValueError(f'Selected git_blob differs: {name}')
            names.add(name)
    adapters = set(selection['public_adapters'])
    if not adapters <= PUBLIC_ADAPTERS:
        raise ValueError('Unknown public adapter')
    expected = names | adapters | {'SOURCE_SELECTION.json', 'CODE_COMMIT'}
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in capsule inventory')
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Regular capsule file required')
        actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(f'Unlisted or missing capsule files: {sorted(actual ^ expected)}')
    return len(selection['files']), len(selection['external_inspection_artifacts'])


class ImportBoundary:
    """Reject loaded project/test-helper modules supplied by another checkout."""
    def __init__(self, root):
        self.root = root

    def pytest_sessionfinish(self, session, exitstatus):
        checked = []
        try:
            for name, module in list(sys.modules.items()):
                if name.split('.')[0] not in {'sim', 'auscultation', 'scripts'} and not name.startswith('test_ausc_'):
                    continue
                paths = [module.__file__] if getattr(module, '__file__', None) else []
                paths += list(getattr(module, '__path__', []))
                for path in paths:
                    require_local_module(path, self.root)
                checked.append(name)
        except ValueError as error:
            print(f'\nIMPORT BOUNDARY FAILED: {error}', flush=True)
            session.exitstatus = 1
            return
        print(f'\nImport boundary: {len(checked)} loaded project/test modules inside source28.', flush=True)


def run_fixtures(root):
    import pytest
    selection = json.loads((root / 'SOURCE_SELECTION.json').read_text())
    selected = {entry['file'] for entry in selection['files']}
    tests = selection['selected_cpu_tests']
    if not tests or len(tests) != len(set(tests)) or any(name not in selected for name in tests):
        raise ValueError('Explicit unique original selected CPU tests required')
    # Python -I excludes the caller's working directory, user site and PYTHONPATH.
    # Only these two capsule directories become project/test-helper import roots.
    sys.path[:0] = [str(root), str(root / 'tests')]
    return pytest.main(['-c', '/dev/null', '--rootdir', str(root), '--confcutdir', str(root),
                        '-p', 'no:cacheprovider', '-q', '-rs',
                        *[str(root / name) for name in tests]], plugins=[ImportBoundary(root)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', action='store_true', help='Run only the selected original CPU test suites')
    parser.add_argument('--_isolated-fixtures', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selected, external = verify(root)
    print(f'Verified {selected} original selected files and {external} external inspection artifacts; '
          'PARTIAL_INSPECTION_CPU_CAPSULE, no runtime/physics qualification.', flush=True)
    if args._isolated_fixtures:
        if not sys.flags.isolated or sys.flags.optimize:
            raise ValueError('Fixtures require unoptimized isolated Python (-I -B)')
        return run_fixtures(root)
    if args.fixtures:
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CUDA_VISIBLE_DEVICES='',
                   PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                               '--_isolated-fixtures'], cwd=root, env=env, check=False).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

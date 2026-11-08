#!/usr/bin/env python3
"""Verify a partial Source45 selection; optionally run bounded synthetic CPU tests.

Public adapter derived from the Source28 capsule verifier, not an original
Source45 Git blob, historical archive reader, native owner or model trainer.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

ORIGINAL_RUNTIME = dict(
    code_commit='29024f00a8656865fe79242605580278561f463e', file_count=1724,
    source_manifest_bytes=298378,
    source_manifest_sha256='40fe7c7361e7cb4ab3d7acaafdc4cf97eced57ab084a2afd49a136d9d8336545',
    archive_bytes=6458473,
    archive_sha256='b34c6c21c271c48bedd33bc490be3a892b051e6aaca664ff491c1fce17186a95')
PUBLIC_ADAPTERS = {'verify_capsule.py', 'PLAN.md', 'CPU_FIXTURE_CHECK.json'}


def require_local_module(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Project module outside capsule: {resolved}')
    return resolved


def verify(root):
    root = Path(root).resolve()
    selection = json.loads((root / 'SOURCE_SELECTION.json').read_text())
    if (selection.get('schema') != 'wam_source45_public_selection_v1'
            or selection.get('scope') != 'PARTIAL_INSPECTION_CPU_CAPSULE'
            or selection.get('original_runtime') != ORIGINAL_RUNTIME
            or (root / 'CODE_COMMIT').read_text().strip() != ORIGINAL_RUNTIME['code_commit']):
        raise ValueError('Partial capsule provenance differs')
    names = set()
    for entry in selection['files']:
        name = entry['file']
        relative = PurePosixPath(name)
        if (name != relative.as_posix() or relative.is_absolute() or not relative.parts
                or '..' in relative.parts or name in names
                or name in {'CODE_COMMIT', 'SOURCE_SELECTION.json'}):
            raise ValueError('Unique safe relative capsule paths required')
        if relative.parts[0] == 'ops' or relative.suffix not in {'.py', '.json', '.urdf'}:
            raise ValueError('Only selected original source/config/URDF text; no ops artifacts')
        path = root / relative
        require_local_module(path, root)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlink in capsule path')
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Regular selected file required')
        body = path.read_bytes()
        if len(body) != entry['bytes'] or hashlib.sha256(body).hexdigest() != entry['sha256']:
            raise ValueError(f'Selected bytes/sha256 differ: {name}')
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
    return len(names)


def fixture_nodes(nodes, selected):
    """Permit explicit original test functions, never a whole legacy suite."""
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError('Explicit unique bounded original CPU test nodes required')
    for node in nodes:
        parts = node.split('::')
        if (len(parts) != 2 or parts[0] not in selected
                or not parts[0].startswith('tests/test_ausc_')
                or not re.fullmatch(r'test_[A-Za-z0-9_]+', parts[1])):
            raise ValueError(f'Bounded original test function required: {node}')
    return nodes


class ImportBoundary:
    """Reject loaded project/test-helper modules supplied by another checkout."""
    def __init__(self, root):
        self.root = root

    def pytest_sessionfinish(self, session, exitstatus):
        checked = []
        try:
            for name, module in list(sys.modules.items()):
                if name.split('.')[0] not in {'sim', 'auscultation', 'ausc', 'scripts'} and not name.startswith('test_ausc_'):
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
        print(f'\nImport boundary: {len(checked)} loaded project/test modules inside source45.', flush=True)


def run_fixtures(root, optional_torch=False):
    import pytest
    selection = json.loads((root / 'SOURCE_SELECTION.json').read_text())
    selected = {entry['file'] for entry in selection['files']}
    tests = fixture_nodes(selection['selected_cpu_tests'], selected)
    if optional_torch:
        tests = tests + fixture_nodes(selection['optional_torch_cpu_tests'], selected)
    # -I excludes caller cwd, user site and PYTHONPATH. These are the only
    # added project/test-helper roots; no fallback to root/source26/source28.
    sys.path[:0] = [str(root), str(root / 'tests')]
    for name in selection['entry_points']:
        module = importlib.import_module(name.removesuffix('.py').replace('/', '.'))
        if require_local_module(module.__file__, root) != root / name:
            raise ValueError(f'Entry module origin differs: {name}')
    print(f'Entry imports: {len(selection["entry_points"])} exact Source45 capsule paths.', flush=True)
    return pytest.main(['-c', '/dev/null', '--rootdir', str(root), '--confcutdir', str(root),
                        '-p', 'no:cacheprovider', '-q', '-rs',
                        *[str(root / name) for name in tests]], plugins=[ImportBoundary(root)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', action='store_true', help='Run only bounded original synthetic CPU tests')
    parser.add_argument('--optional-torch', action='store_true', help='Also run tiny synthetic CPU tensor admission tests')
    parser.add_argument('--_isolated-fixtures', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.optional_torch and not (args.fixtures or args._isolated_fixtures):
        parser.error('--optional-torch requires --fixtures')
    root = Path(__file__).resolve().parent
    selected = verify(root)
    print(f'Verified {selected} original selected files; PARTIAL_INSPECTION_CPU_CAPSULE. '
          'No training, model, native episode or physical qualification.', flush=True)
    if args._isolated_fixtures:
        if not sys.flags.isolated or sys.flags.optimize or not sys.dont_write_bytecode:
            raise ValueError('Fixtures require unoptimized isolated Python (-I -B)')
        return run_fixtures(root, args.optional_torch)
    if args.fixtures:
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CUDA_VISIBLE_DEVICES='',
                   PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--_isolated-fixtures']
        if args.optional_torch:
            command.append('--optional-torch')
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

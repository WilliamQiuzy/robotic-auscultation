#!/usr/bin/env python3
"""Public adapter: verify this partial selection and run nine isolated CPU fixtures.

This file is new publication tooling, not an original source26 Git blob. It does
not launch the experiment owner, change admission, or qualify native physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CODE_COMMIT = '05e972ca184383a525aff293fc9f3590e546da9c'
RUNTIME_MANIFEST_SHA256 = '315f021e76f4e606f4a327b053abc22bacf8a1b8c649d35b4bd8ba9e644c743d'


def require_local_module(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Project module outside capsule: {resolved}')
    return resolved


def verify(root):
    root = Path(root).resolve()
    selection = json.loads((root / 'SOURCE_SELECTION.json').read_text())
    if (selection.get('scope') != 'PARTIAL_PUBLIC_CAPSULE'
            or selection.get('code_commit') != CODE_COMMIT
            or selection.get('original_runtime_manifest_sha256') != RUNTIME_MANIFEST_SHA256
            or (root / 'CODE_COMMIT').read_text().strip() != CODE_COMMIT):
        raise ValueError('Partial capsule provenance differs')
    names = set()
    for entry in selection['files']:
        relative = Path(entry['file'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in names:
            raise ValueError('Unique relative capsule paths required')
        path = root / relative
        require_local_module(path, root)
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError('Symlink in selected source path')
        payload = path.read_bytes()
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError(f'Selected bytes/sha256 differ: {relative}')
        names.add(relative.as_posix())
    adapters = set(selection.get('public_adapters', []))
    for path in root.rglob('*.py'):
        if '__pycache__' not in path.parts:
            relative = path.relative_to(root).as_posix()
            if relative not in names and relative not in adapters:
                raise ValueError(f'Unlisted Python source: {relative}')
    return len(names)


class ImportBoundary:
    """Check actual loaded project modules after the selected fixture run."""
    def __init__(self, root):
        self.root = root

    def pytest_sessionfinish(self, session, exitstatus):
        checked = []
        try:
            for name, module in list(sys.modules.items()):
                if name.split('.')[0] not in {'sim', 'auscultation', 'scripts'} and not name.startswith('test_ausc_'):
                    continue
                paths = ([module.__file__] if getattr(module, '__file__', None) else [])
                paths += list(getattr(module, '__path__', []))
                for path in paths:
                    require_local_module(path, self.root)
                checked.append(name)
        except ValueError as error:
            print(f'\nIMPORT BOUNDARY FAILED: {error}', flush=True)
            session.exitstatus = 1
            return
        print(f'\nImport boundary: {len(checked)} loaded project/test modules resolve inside this capsule.', flush=True)


def run_fixtures(root):
    import pytest
    selection = json.loads((root / 'SOURCE_SELECTION.json').read_text())
    # -I removes the caller's working directory, user site and PYTHONPATH. Only
    # this capsule and its exact helper directory are added as project roots.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'tests'))
    tests = [str(root / name) for name in selection['selected_cpu_tests']]
    return pytest.main(['-c', '/dev/null', '--rootdir', str(root), '-p', 'no:cacheprovider',
                        '-q', '-rs', *tests], plugins=[ImportBoundary(root)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', action='store_true', help='Run only the nine selected original CPU test files')
    parser.add_argument('--_isolated-fixtures', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    count = verify(root)
    print(f'Verified {count} exact selected files; PARTIAL_PUBLIC_CAPSULE, no runtime/physics qualification.', flush=True)
    if args._isolated_fixtures:
        if not sys.flags.isolated:
            raise ValueError('Fixture subprocess requires isolated Python (-I)')
        return run_fixtures(root)
    if args.fixtures:
        return subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                               '--_isolated-fixtures'], cwd=root, check=False).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

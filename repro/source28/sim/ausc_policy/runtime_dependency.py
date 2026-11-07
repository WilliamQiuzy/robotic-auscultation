"""Versioned, closed import composition for ACT5 source26 + native runtime27.

Only stdlib is imported here. The selected model loader and numerical modules
are unmodified. A deployment lock is generated after freezing native sources;
the full native manifest includes the lock, whose native manifest digest uses
canonical rows excluding only the lock row (see deployment documentation).
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys

SCHEMA = 'm2_act5_runtime_dependency_lock_v1'
LOCK_NAME = 'RUNTIME_DEPENDENCY_LOCK.json'
SOURCE26_ROOT = '/opt/wam-runtime/code/wam-act-recovery-20261106-v26'
SOURCE26_COMMIT = '05e972ca184383a525aff293fc9f3590e546da9c'
SOURCE26_MANIFEST_SHA256 = '315f021e76f4e606f4a327b053abc22bacf8a1b8c649d35b4bd8ba9e644c743d'
SOURCE26_FILE_COUNT = 1660
NATIVE_ROOT = '/opt/wam-runtime/code/wam-act-recovery-20261106-v27'
MANIFEST_NAME = 'SOURCE_MANIFEST.json'
_INSTALLED = None


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value):
    _need(isinstance(value, (str, Path)), 'Real absolute dependency path required')
    path = Path(value)
    _need(path.is_absolute() and '..' not in path.parts and str(path) == str(value),
          'Canonical absolute dependency path required')
    _need(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink dependency path forbidden')
    return path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _need(key not in result, 'Duplicate dependency JSON key: '+key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite dependency JSON value: '+value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _regular(path):
    info = _path(path).lstat()
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Regular unlinked dependency file required: '+str(path))
    return info


def _ref(path):
    path = _path(path)
    _regular(path)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=_sha(data))


def _read(ref):
    _need(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
          type(ref['bytes']) is int and ref['bytes'] >= 0 and isinstance(ref['sha256'], str) and
          re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None, 'Exact dependency file reference required')
    path = _path(ref['path'])
    _regular(path)
    data = path.read_bytes()
    _need(len(data) == ref['bytes'] and _sha(data) == ref['sha256'], 'Dependency file changed: '+str(path))
    return data


def _relative(value):
    _need(isinstance(value, str) and value and not Path(value).is_absolute() and
          '..' not in Path(value).parts and Path(value).as_posix() == value,
          'Safe relative dependency inventory path required')
    return value


def _manifest(root, *, require_lock=False):
    root = _path(root)
    _need(root.is_dir(), 'Dependency root missing')
    rows = _json((root/MANIFEST_NAME).read_bytes())
    _need(isinstance(rows, list) and rows, 'Nonempty closed dependency manifest required')
    files = {}
    for row in rows:
        _need(isinstance(row, dict) and set(row) == {'file', 'bytes', 'sha256'}, 'Exact dependency manifest row required')
        name = _relative(row['file'])
        _need(name not in files and name not in (MANIFEST_NAME, 'CODE_COMMIT'), 'Unique dependency manifest row required')
        ref = dict(path=str(root/name), bytes=row['bytes'], sha256=row['sha256'])
        _read(ref)
        files[name] = ref
    actual = set()
    for path in root.rglob('*'):
        info = path.lstat()
        _need(not stat.S_ISLNK(info.st_mode), 'Symlink in dependency inventory')
        if stat.S_ISDIR(info.st_mode):
            continue
        _regular(path)
        actual.add(path.relative_to(root).as_posix())
    _need(actual == set(files) | {MANIFEST_NAME, 'CODE_COMMIT'}, 'Unknown or omitted dependency inventory file')
    _regular(root/MANIFEST_NAME)
    _regular(root/'CODE_COMMIT')
    if require_lock:
        _need(LOCK_NAME in files, 'Deployment dependency lock must be covered by full native manifest')
    return files


def _native_digest(files):
    rows = [dict(file=name, bytes=ref['bytes'], sha256=ref['sha256'])
            for name, ref in sorted(files.items()) if name != LOCK_NAME]
    return _sha(_canonical(rows))


def _source26():
    root = _path(SOURCE26_ROOT)
    _need(_ref(root/MANIFEST_NAME)['sha256'] == SOURCE26_MANIFEST_SHA256, 'Exact frozen source26 full manifest required')
    _need((root/'CODE_COMMIT').read_text().strip() == SOURCE26_COMMIT, 'Exact frozen source26 commit required')
    files = _manifest(root)
    _need(len(files) == SOURCE26_FILE_COUNT, 'Exact frozen source26 file count required')
    return files


def _trained_modules(complete_ref, source_files):
    complete = _json(_read(complete_ref))
    _need(isinstance(complete, dict) and complete.get('schema') == 'm2_act_auxiliary_act_pilot_v3' and
          complete.get('complete') is True, 'Pinned completed ACT5 version-3 metadata required')
    code = complete.get('inputs', {}).get('code', {})
    _need(isinstance(code, dict) and set(code) == {'code_commit', 'files'} and
          code['code_commit'] == SOURCE26_COMMIT and isinstance(code['files'], dict) and len(code['files']) == 19,
          'Exact trained ACT5 source26 code binding required')
    modules = {}
    for name, ref in code['files'].items():
        _need(isinstance(name, str) and re.fullmatch('[A-Za-z_][A-Za-z_0-9]*[.]py', name) is not None and
              name != '__init__.py', 'Finite trained numerical module filename required')
        _need(ref == source_files.get('sim/ausc_policy/'+name), 'Trained source26 module reference differs: '+name)
        modules['sim.ausc_policy.'+name[:-3]] = dict(ref)
    return modules


def build_lock(trained_complete, *, native_root=NATIVE_ROOT):
    """Return a lock from pinned COMPLETE metadata; no numerical module imports.

    Write the returned JSON to LOCK_NAME and regenerate the full native manifest
    including that file. The final external operator pins that full manifest SHA.
    """
    root = _path(native_root)
    _need(str(root) == NATIVE_ROOT, 'Exact native runtime27 root required')
    source_files = _source26()
    native_files = _manifest(root)
    commit = (root/'CODE_COMMIT').read_text().strip()
    _need(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Exact native commit required')
    modules = _trained_modules(trained_complete, source_files)
    return dict(schema=SCHEMA, trained_complete=dict(trained_complete), modules=modules,
        model_source=dict(root=SOURCE26_ROOT, code_commit=SOURCE26_COMMIT,
                          manifest=_ref(Path(SOURCE26_ROOT)/MANIFEST_NAME)),
        native_runtime=dict(root=str(root), code_commit=commit,
            manifest=dict(path=str(root/MANIFEST_NAME), canonical_sha256=_native_digest(native_files),
                          canonical_excluded_files=[LOCK_NAME])))


def _validated(root):
    root = _path(root)
    _need(str(root) == NATIVE_ROOT, 'Exact native runtime27 root required')
    lock = _json((root/LOCK_NAME).read_bytes())
    _need(isinstance(lock, dict) and set(lock) == {'schema', 'trained_complete', 'modules', 'model_source', 'native_runtime'} and
          lock['schema'] == SCHEMA, 'Exact versioned runtime dependency lock required')
    source_files = _source26()
    _need(lock['model_source'] == dict(root=SOURCE26_ROOT, code_commit=SOURCE26_COMMIT,
          manifest=_ref(Path(SOURCE26_ROOT)/MANIFEST_NAME)), 'Frozen numerical source root/manifest binding changed')
    modules = _trained_modules(lock['trained_complete'], source_files)
    _need(lock['modules'] == modules, 'Exact COMPLETE-derived numerical module map required')
    native_files = _manifest(root, require_lock=True)
    commit = (root/'CODE_COMMIT').read_text().strip()
    _need(re.fullmatch('[0-9a-f]{40}', commit) is not None and lock['native_runtime'] == dict(
        root=str(root), code_commit=commit, manifest=dict(path=str(root/MANIFEST_NAME),
        canonical_sha256=_native_digest(native_files), canonical_excluded_files=[LOCK_NAME])),
        'Native runtime root/commit/canonical manifest binding changed')
    return lock, modules, native_files


def _project(name):
    return name in ('sim', 'scripts', 'auscultation') or name.startswith(('sim.', 'scripts.', 'auscultation.'))


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, ref):
        super().__init__(name, ref['path'])
        self.reference = dict(ref)

    def get_code(self, fullname):
        # Compile authenticated source directly; never consume an alternative pyc.
        return self.source_to_code(_read(self.reference), self.path)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, root, modules, native_files):
        self.root = root
        self.modules = dict(modules)
        self.files = dict(native_files)

    def expected(self, fullname):
        if fullname in self.modules:
            return self.modules[fullname], False
        relative = fullname.replace('.', '/')
        file = self.files.get(relative+'.py')
        package = self.files.get(relative+'/__init__.py')
        _need(not (file and package), 'Ambiguous project module in dependency manifest: '+fullname)
        if file or package:
            return file or package, bool(package)
        prefix = relative+'/'
        if any(name.startswith(prefix) for name in self.files):
            return None, True
        raise ImportError('Unknown project module in closed runtime composition: '+fullname)

    def find_spec(self, fullname, path=None, target=None):
        if not _project(fullname):
            return None
        ref, package = self.expected(fullname)
        if ref is None:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(self.root/fullname.replace('.', '/'))]
            return spec
        _read(ref)
        return importlib.util.spec_from_file_location(fullname, ref['path'],
            loader=_SourceLoader(fullname, ref),
            submodule_search_locations=[str(Path(ref['path']).parent)] if package else None)

    def check_loaded(self):
        for name, module in tuple(sys.modules.items()):
            if not _project(name):
                continue
            try:
                ref, package = self.expected(name)
            except ImportError as error:
                raise ValueError('Preloaded unknown project module: '+name) from error
            path = getattr(module, '__file__', None)
            spec = getattr(module, '__spec__', None)
            if ref is not None:
                _need(path == ref['path'] and spec is not None and spec.origin == ref['path'],
                      'Preloaded foreign project module: '+name)
                _read(ref)
            else:
                # SDKs may contribute to the two unexecuted root namespaces.
                # Their search roots are narrowed only after all imports pass.
                locations = list(getattr(module, '__path__', []))
                expected = [str(self.root/name.replace('.', '/'))]
                _need(path is None and spec is not None and (name in ('sim', 'scripts') or locations == expected),
                      'Preloaded foreign project namespace: '+name)
            if package and ref is not None:
                _need(list(getattr(module, '__path__', [])) == [str(Path(ref['path']).parent)],
                      'Preloaded foreign project package path: '+name)


def install(*, native_root=None):
    """Authenticate both closed roots and install once before numerical imports."""
    global _INSTALLED
    root = _path(native_root if native_root is not None else Path(__file__).parents[2])
    _need(Path(__file__) == root/'sim/ausc_policy/runtime_dependency.py', 'Bootstrap must execute from actual native root')
    lock, modules, files = _validated(root)
    finder = _Finder(root, modules, files)
    finder.check_loaded()
    if _INSTALLED is not None:
        _need(_INSTALLED[0] == lock and _INSTALLED[1] in sys.meta_path, 'Installed runtime dependency composition changed')
        return verify()
    for name in ('sim', 'scripts'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__file__', None) is None:
            locations = [str(root/name)]
            module.__path__ = locations
            module.__spec__.submodule_search_locations = locations
    sys.meta_path.insert(0, finder)
    _INSTALLED = (lock, finder)
    return verify()


def verify():
    """Reauthenticate full inventories and truthful origins without importing numerics."""
    _need(_INSTALLED is not None, 'Explicit runtime dependency composition is not installed')
    lock, finder = _INSTALLED
    current, modules, _ = _validated(finder.root)
    _need(lock == current and finder in sys.meta_path, 'Installed runtime dependency composition changed')
    finder.check_loaded()
    loaded = {name: dict(path=module.__file__, origin=module.__spec__.origin)
              for name, module in tuple(sys.modules.items()) if _project(name) and getattr(module, '__file__', None)}
    return dict(schema=SCHEMA, model_source=lock['model_source'], native_runtime=lock['native_runtime'],
                dependency_lock=_ref(finder.root/LOCK_NAME),
                native_full_manifest=_ref(finder.root/MANIFEST_NAME), modules=modules, loaded_modules=loaded)

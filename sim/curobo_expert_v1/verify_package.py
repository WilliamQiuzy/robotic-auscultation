"""Seal, verify, or cross-check host receipts for the complete core package.

Run --seal once on the completed package, copy that manifest to the other host,
and verify there without --seal. After retrieving both host receipts, use
--verify-pair. Pair verification is read-only and also checks current files.
Receipts record locally observed host identity; they are unsigned records.
"""
import argparse
import datetime
import hashlib
import json
import platform
import re
import socket
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'package_manifest.json', 'package_verification_mac.json',
            'package_verification_linux.json'}
LOCAL_ONLY = 'results/graph_path_gate_probe_v1'
MANIFEST_SCHEMA = 'wam.expert_comparison.package.v1'
RECEIPT_SCHEMA = 'wam.expert_comparison.package_receipt.v1'
HOST_SYSTEMS = {'mac': 'Darwin', 'linux': 'Linux'}
REQUIRED_FILES = frozenset({
    'README.md', 'BACKEND.md', 'RESULTS_ZH.md', 'comparison.html',
    'results/comparison_summary.json', 'configs/README.md',
    'configs/bridge_v2/bridge_manifest.json', 'verify_package.py',
})
SCOPE = (
    'Complete core package inventory, excluding root manifest/host receipts, '
    'results/graph_path_gate_probe_v1 (local-only; upload denied), '
    'disposable_render.usdc clones, Python/pytest caches and Apple metadata. '
    'Pre-existing source dependencies are bound by episode provenance. '
    'Model weights and disposable render clones remain on Linux.'
)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def excluded(relative):
    """The same exclusion policy applies to sealing and verification."""
    path = PurePosixPath(relative)
    return (relative in EXCLUDED or relative == LOCAL_ONLY
            or relative.startswith(LOCAL_ONLY + '/')
            or path.name in {'.DS_Store', 'disposable_render.usdc'}
            or path.suffix == '.pyc'
            or any(part in {'__pycache__', '.pytest_cache', '.AppleDouble',
                            '.__AppleDouble'} or part.startswith('._')
                   for part in path.parts))


def core_files(root, problems):
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if excluded(relative):
            continue
        if path.is_symlink():
            problems.append({'path': relative, 'reason': 'symlink_not_supported'})
        elif path.is_file():
            files[relative] = path
    return files


def read_json(path, problems):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError) as exc:
        problems.append({'path': path.name, 'reason': 'unreadable_json',
                         'detail': str(exc)})
        return None


def validate_manifest(manifest, actual, problems):
    """Validate structure before using any untrusted entry as a path or count."""
    if not isinstance(manifest, dict) or manifest.get('schema') != MANIFEST_SCHEMA:
        problems.append({'path': 'package_manifest.json', 'reason': 'invalid_schema'})
        return {}
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        problems.append({'path': 'package_manifest.json', 'reason': 'invalid_files_list'})
        return {}
    files = {}
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append({'path': 'package_manifest.json', 'reason': 'invalid_entry'})
            continue
        relative = entry.get('path')
        if (not isinstance(relative, str) or not relative or '\\' in relative
                or PurePosixPath(relative).is_absolute()
                or any(part in {'', '.', '..'} for part in relative.split('/'))):
            problems.append({'path': str(relative), 'reason': 'invalid_relative_path'})
            continue
        if excluded(relative):
            problems.append({'path': relative, 'reason': 'excluded_file_in_manifest'})
            continue
        if relative in files:
            problems.append({'path': relative, 'reason': 'duplicate_entry'})
            continue
        if (type(entry.get('bytes')) is not int or entry['bytes'] < 0
                or not isinstance(entry.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', entry['sha256']) is None):
            problems.append({'path': relative, 'reason': 'invalid_size_or_digest'})
            continue
        files[relative] = entry
    for relative in sorted(REQUIRED_FILES - files.keys()):
        problems.append({'path': relative, 'reason': 'required_delivery_file_not_listed'})
    for relative in sorted(files.keys() - actual.keys()):
        problems.append({'path': relative, 'reason': 'missing'})
    for relative in sorted(actual.keys() - files.keys()):
        problems.append({'path': relative, 'reason': 'unlisted_core_file'})
    for relative in sorted(files.keys() & actual.keys()):
        path, entry = actual[relative], files[relative]
        try:
            if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                problems.append({'path': relative, 'reason': 'content_mismatch'})
        except OSError as exc:
            problems.append({'path': relative, 'reason': 'unreadable_file',
                             'detail': str(exc)})
    return files


def verify_pair(root, report, problems):
    """Check both saved receipts without refreshing either one."""
    for label, system in HOST_SYSTEMS.items():
        path = root / ('package_verification_' + label + '.json')
        receipt = read_json(path, problems)
        if not isinstance(receipt, dict):
            problems.append({'path': path.name, 'reason': 'invalid_receipt'})
            continue
        expected = {'schema': RECEIPT_SCHEMA, 'host_label': label,
                    'platform_system': system, 'passed': True, 'problems': [],
                    'verification_mode': 'host',
                    'manifest_sha256': report['manifest_sha256'],
                    'files_checked': report['files_checked'],
                    'bytes_checked': report['bytes_checked']}
        for field, value in expected.items():
            observed = receipt.get(field)
            if type(observed) is not type(value) or observed != value:
                problems.append({'path': path.name, 'reason': 'receipt_field_mismatch',
                                 'field': field})
        if not isinstance(receipt.get('hostname'), str) or not receipt['hostname'].strip():
            problems.append({'path': path.name, 'reason': 'missing_hostname'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--seal', action='store_true')
    modes.add_argument('--verify-pair', action='store_true',
                       help='Read-only verification of current files and both host receipts')
    parser.add_argument('--host-label', choices=tuple(HOST_SYSTEMS), required=True)
    args = parser.parse_args()
    system = platform.system()
    if system != HOST_SYSTEMS[args.host_label]:
        parser.error('--host-label ' + args.host_label + ' requires '
                     + HOST_SYSTEMS[args.host_label] + ', observed ' + system)
    root = ROOT.resolve()
    manifest_path = root / 'package_manifest.json'
    problems = []
    if args.seal:
        # Invalidate both receipts even when a new seal later fails validation.
        for label in HOST_SYSTEMS:
            (root / ('package_verification_' + label + '.json')).unlink(missing_ok=True)
    actual = core_files(root, problems)
    if args.seal:
        for relative in sorted(REQUIRED_FILES - actual.keys()):
            problems.append({'path': relative, 'reason': 'required_delivery_file_missing'})
        if not problems:
            entries = [{'path': relative, 'bytes': path.stat().st_size,
                        'sha256': sha256(path)} for relative, path in actual.items()]
            manifest_path.write_text(json.dumps({
                'schema': MANIFEST_SCHEMA, 'scope': SCOPE, 'files': entries},
                ensure_ascii=False, indent=2) + '\n')
    manifest = read_json(manifest_path, problems)
    files = validate_manifest(manifest, actual, problems)
    receipt = {
        'schema': RECEIPT_SCHEMA,
        'host_label': args.host_label, 'platform_system': system,
        'hostname': socket.gethostname(), 'root': str(root),
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'manifest_sha256': sha256(manifest_path) if manifest_path.is_file() else None,
        'files_checked': len(files),
        'bytes_checked': sum(x['bytes'] for x in files.values()),
        'verification_mode': 'pair' if args.verify_pair else 'host',
        'problems': problems, 'passed': not problems}
    if args.verify_pair:
        verify_pair(root, receipt, problems)
        receipt['passed'] = not problems
    else:
        (root / ('package_verification_' + args.host_label + '.json')).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

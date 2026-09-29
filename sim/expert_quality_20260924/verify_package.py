"""Seal or verify the comparative evidence package, excluding generated receipts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'package_manifest.json', 'package_verification_mac.json',
            'package_verification_linux.json'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--host-label', choices=('mac', 'linux'), required=True)
    args = parser.parse_args()
    manifest_path = ROOT / 'package_manifest.json'
    if args.seal:
        entries = []
        for path in sorted(ROOT.rglob('*')):
            if (not path.is_file() or path.name in EXCLUDED
                    or '__pycache__' in path.parts or path.suffix == '.pyc'
                    or path.name == '.DS_Store' or '.pytest_cache' in path.parts):
                continue
            entries.append({'path': str(path.relative_to(ROOT)),
                            'bytes': path.stat().st_size, 'sha256': sha256(path)})
        manifest_path.write_text(json.dumps({
            'schema': 'wam.expert_comparison.package.v1',
            'scope': 'Listed package files only; pre-existing source dependencies '
                     'are bound by episode provenance. Model weights and disposable '
                     'render clones remain on Linux.',
            'files': entries}, ensure_ascii=False, indent=2) + '\n')
    manifest = json.loads(manifest_path.read_text())
    problems = []
    for entry in manifest['files']:
        path = (ROOT / entry['path']).resolve()
        if ROOT not in path.parents:
            raise ValueError('manifest path escapes package')
        if not path.is_file():
            problems.append({'path': entry['path'], 'reason': 'missing'})
        elif (path.stat().st_size != entry['bytes']
              or sha256(path) != entry['sha256']):
            problems.append({'path': entry['path'], 'reason': 'content_mismatch'})
    receipt = {
        'host_label': args.host_label, 'root': str(ROOT),
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'manifest_sha256': sha256(manifest_path),
        'files_checked': len(manifest['files']),
        'bytes_checked': sum(x['bytes'] for x in manifest['files']),
        'problems': problems, 'passed': not problems}
    (ROOT / ('package_verification_' + args.host_label + '.json')).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

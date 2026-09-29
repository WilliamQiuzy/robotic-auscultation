"""Verify a frozen camera package and its source scene dependencies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
project = root.parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


manifest = json.loads((root/'MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert sha(root/name) == expected, name
for name, expected in manifest['source_dependencies'].items():
    assert sha(project/name) == expected, name
result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'all_match': True,
          'package_file_count': len(manifest['files']), 'dependency_count': len(manifest['source_dependencies']),
          'manifest_sha256': sha(root/'MANIFEST.json'), 'verified_root': str(root)}
(root/'SYNC_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))

"""Linux owns acquisition; Mac receives explicitly selected review artifacts.

These guards do not delete files, promote training data, or copy anything.
Call checked_mac_exports before each selected transfer, then verify the hashes
at the destination. Recursive raw mirroring is outside this contract.
"""
import hashlib
import math
from pathlib import Path, PurePosixPath

EPISODE_LIMIT_BYTES = 100_000_000
EXPORT_LIMITS = {'summary': 2_000_000, 'chart': 10_000_000, 'video': 100_000_000}
EXPORT_SUFFIXES = {'summary': {'.json', '.md', '.txt'},
                   'chart': {'.png', '.svg', '.pdf'}, 'video': {'.mp4', '.webm'}}
RAW_PARTS = {'raw', 'rgbd', 'frames', 'hall', 'raw_capture', 'multimodal',
             'raw_multimodal_capture', 'training_capture'}


def _directory(root):
    supplied = Path(root).absolute()
    if any(p.is_symlink() for p in (supplied, *supplied.parents)):
        raise ValueError('Root directory cannot use symlink aliases')
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError('A real directory root is required')
    return resolved


def checked_mac_exports(root, entries):
    root = _directory(root)
    if any(p.lower() in RAW_PARTS or p.lower().startswith('raw_') for p in root.parts):
        raise ValueError('Raw directories cannot be Mac export roots')
    result = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'path', 'kind', 'sha256'}:
            raise ValueError('Export needs explicit path, kind and source SHA256')
        name, kind, expected = entry['path'], entry['kind'], entry['sha256']
        if not isinstance(name, str) or '\\' in name or not name:
            raise ValueError('Export must use a relative POSIX path')
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or str(path) != name or
                any(p.lower() in RAW_PARTS or p.lower().startswith('raw_') for p in path.parts) or
                kind not in EXPORT_LIMITS or path.suffix.lower() not in EXPORT_SUFFIXES[kind] or
                name in seen):
            raise ValueError('Raw, unclassified, duplicate or escaping Mac export')
        selected = root / name
        if any(p.is_symlink() for p in (selected, *selected.parents) if p != root):
            raise ValueError('Mac exports cannot use symlink aliases')
        source = selected.resolve(strict=True)
        if root not in source.parents or not source.is_file():
            raise ValueError('Mac export escapes the source root')
        size = source.stat().st_size
        if not 0 < size <= EXPORT_LIMITS[kind]:
            raise ValueError('Mac evidence exceeds its bounded export budget')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('Mac export source hash changed')
        result.append(dict(path=name, kind=kind, sha256=digest, bytes=size))
        seen.add(name)
    return result


def assign_capture_tiers(episode_ids, *, seed, percent=2):
    ids = list(episode_ids)
    if (not isinstance(seed, str) or not seed or
            any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids) or
            isinstance(percent, bool) or not isinstance(percent, (int, float)) or
            not math.isfinite(percent) or not 1 <= percent <= 2):
        raise ValueError('Unique episodes, a frozen seed and 1–2 percent are required')
    # Floor avoids turning one small pilot sample into a misleading >2% tier.
    n = int(len(ids) * percent / 100)
    ranked = sorted(ids, key=lambda x: (hashlib.sha256((seed+'\0'+x).encode()).hexdigest(), x))
    tier_b = set(ranked[:n])
    return {x: 'tier_b' if x in tier_b else 'training' for x in sorted(ids)}


def episode_size(root, *, limit_bytes=EPISODE_LIMIT_BYTES):
    if type(limit_bytes) is not int or limit_bytes <= 0:
        raise ValueError('Positive integer byte limit required')
    root = _directory(root)
    count = total = 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('An episode cannot hide payloads behind symlinks')
        if path.is_file():
            count += 1
            total += path.stat().st_size
    return dict(total_bytes=total, files=count, limit_bytes=limit_bytes,
                within_budget=total <= limit_bytes, training_eligible=False)

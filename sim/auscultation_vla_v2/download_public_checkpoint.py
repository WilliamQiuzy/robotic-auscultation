"""Download a pre-inventoried, public checkpoint; fail loudly on integrity errors."""
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import google_crc32c


def main():
    root = Path('/home/william/wam-vla-v2')
    inventory = json.loads((root / 'public_checkpoint_inventory.json').read_text())
    # The inventory is public GCS metadata, never user experiment data.
    entries = inventory.get('items', inventory.get('objects', []))
    if not entries:
        raise ValueError('Missing public object inventory')
    destination = root / 'public-assets' / 'pi05_base'
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise RuntimeError('Insufficient free disk for public checkpoint')
    started = time.time()

    def fetch(entry):
        name = entry['name']
        relative = name.removeprefix('checkpoints/pi05_base/')
        if relative == name or '..' in Path(relative).parts or relative.startswith('/'):
            raise ValueError('Unexpected public object path')
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        expected = entry.get('md5Hash')
        expected_crc = entry['crc32c']
        def digest(local):
            md5, sha, crc = hashlib.md5(), hashlib.sha256(), google_crc32c.Checksum()
            with local.open('rb') as stream:
                while block := stream.read(8 * 2**20):
                    md5.update(block); sha.update(block); crc.update(block)
            return base64.b64encode(md5.digest()).decode(), sha.hexdigest(), base64.b64encode(crc.digest()).decode()
        if path.is_file() and path.stat().st_size == int(entry['size']):
            md5, sha, crc = digest(path)
            if (expected is None or md5 == expected) and crc == expected_crc:
                return dict(name=name, bytes=path.stat().st_size, sha256=sha, md5_verified=expected is not None, crc32c_verified=True)
        url = 'https://storage.googleapis.com/openpi-assets/' + urllib.parse.quote(name, safe='/')
        url += '?generation=' + str(entry['generation'])
        partial = path.with_name(path.name + '.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output, length=8 * 2**20)
                if partial.stat().st_size != int(entry['size']):
                    raise ValueError('Object length mismatch: ' + name)
                md5, sha, crc = digest(partial)
                if (expected is not None and md5 != expected) or crc != expected_crc:
                    raise ValueError('Object checksum mismatch: ' + name)
                partial.chmod(0o600)
                partial.replace(path)
                print(json.dumps(dict(downloaded=name, bytes=path.stat().st_size)), flush=True)
                return dict(name=name, bytes=path.stat().st_size, sha256=sha, md5_verified=expected is not None, crc32c_verified=True)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch, entries))
    receipt = dict(schema='verified_public_checkpoint_v1', source='gs://openpi-assets/checkpoints/pi05_base',
                   files=results, elapsed_s=time.time()-started, exit_status=0)
    (root / 'public_checkpoint_verified.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(verified_files=len(results), elapsed_s=receipt['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()

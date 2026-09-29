"""Read-only tree receipt used before deleting verified duplicate raw media."""
import hashlib
import json
from pathlib import Path
import sys


def receipt(root):
    root=Path(root);tree=hashlib.sha256();count=size=0
    if not root.is_dir():raise ValueError('Missing archive root')
    for file in sorted(root.rglob('*')):
        if file.is_symlink():raise ValueError('Symlink in archive')
        if not file.is_file():continue
        h=hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
        tree.update((file.relative_to(root).as_posix()+'\0'+h.hexdigest()+'\n').encode())
        count+=1;size+=file.stat().st_size
    return dict(file_count=count,total_bytes=size,tree_sha256=tree.hexdigest())


if __name__=='__main__':print(json.dumps(receipt(sys.argv[1])))

"""Pure CPU manifest for one approved Isaac 6.0 MDL library, not qualification.

No Kit, USD, GPU, renderer or generic dependency parser is imported here.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import stat
import tempfile

SCHEMA='m2_isaacsim60_sdk_mdl_manifest_v1'
APPROVED_ROOT=Path('/opt/wam-runtime/envs/isaacsim60/lib/python3.12/site-packages/isaacsim/kit/mdl')
ENTRY_RELATIVE='core/Base/OmniPBR.mdl'
APPROVED_ENTRY_BYTES=13989
APPROVED_ENTRY_SHA256='d384fc800f120e3cad1b7ca8837f9c554f9c34343d194813fca27caa493001f2'
APPROVED_FILE_COUNT=244
APPROVED_TOTAL_BYTES=5432503
APPROVED_MANIFEST_SHA256='223126cf1c9891611e8422533b596acf6ed7d0b703962f2186899765242f5936'


def need(ok,message):
    if not ok:raise ValueError(message)


def real_path(value,*,exists=True):
    need(isinstance(value,(str,Path)),'Absolute SDK/manifest path required')
    path=Path(value)
    need(path.is_absolute() and '..' not in path.parts,'Absolute path without traversal required')
    need(not any(p.is_symlink() for p in (path,*path.parents)),'SDK/manifest symlink aliases are forbidden')
    if exists:need(path.exists(),'Missing SDK/manifest path: '+str(path))
    return path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def _entry():return dict(path=ENTRY_RELATIVE,bytes=APPROVED_ENTRY_BYTES,sha256=APPROVED_ENTRY_SHA256)


def _inventory(root):
    root=real_path(root)
    need(root==APPROVED_ROOT and root.is_dir(),'Unapproved SDK MDL root: '+str(root))
    rows=[]
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(),'SDK MDL inventory contains a symlink: '+str(path))
        if path.is_dir():continue
        need(stat.S_ISREG(path.stat().st_mode),'SDK MDL inventory requires regular files')
        rows.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    need(len(rows)==APPROVED_FILE_COUNT and sum(row['bytes'] for row in rows)==APPROVED_TOTAL_BYTES,
         'SDK MDL full inventory count/bytes differ from approved profile')
    need([row for row in rows if row['path']==ENTRY_RELATIVE]==[_entry()],
         'Approved SDK OmniPBR entry hash/bytes changed')
    return rows


def prepare_manifest(output,*,sdk_root=None):
    """Atomically write a new immutable manifest outside the SDK; never edit it."""
    root=APPROVED_ROOT if sdk_root is None else Path(sdk_root)
    rows=_inventory(root);output=real_path(output,exists=False)
    need(output!=root and root not in output.parents,'Manifest must be outside the SDK input tree')
    output.parent.mkdir(parents=True,exist_ok=True);real_path(output.parent)
    document=dict(schema=SCHEMA,sdk_root=str(root),entry=_entry(),files=rows,
        file_count=len(rows),total_bytes=sum(row['bytes'] for row in rows))
    payload=json.dumps(document,sort_keys=True,allow_nan=False).encode()+b'\n'
    need(hashlib.sha256(payload).hexdigest()==APPROVED_MANIFEST_SHA256,
         'Complete SDK MDL library differs from approved canonical manifest')
    lock_path=output.parent/'.sdk-mdl-manifest.lock';real_path(lock_path,exists=False)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        need(not output.exists(),'SDK MDL manifest already exists')
        fd,name=tempfile.mkstemp(prefix='.sdk-mdl-',dir=str(output.parent))
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            need(_inventory(root)==rows,'SDK MDL library changed during manifest preparation')
            os.chmod(name,0o444);os.replace(name,output)
        finally:Path(name).unlink(missing_ok=True)
    ref=dict(file=str(output),sha256=hashlib.sha256(payload).hexdigest())
    validate_manifest(ref)
    return ref


def validate_manifest(ref):
    """Verify exact manifest bytes, complete paths and every current SDK byte."""
    need(isinstance(ref,dict) and set(ref) in ({'file','sha256'},{'file','sha256','bytes'}) and
         isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
         'Hash-bound SDK MDL manifest reference required')
    path=real_path(ref['file']);need(path.is_file(),'SDK manifest must be a regular file')
    payload=path.read_bytes()
    need(hashlib.sha256(payload).hexdigest()==ref['sha256']==APPROVED_MANIFEST_SHA256 and
         ('bytes' not in ref or type(ref['bytes']) is int and ref['bytes']==len(payload)),
         'SDK MDL manifest hash/bytes changed')
    document=json.loads(payload)
    need(isinstance(document,dict) and set(document)=={'schema','sdk_root','entry','files','file_count','total_bytes'} and
         document['schema']==SCHEMA and document['sdk_root']==str(APPROVED_ROOT) and document['entry']==_entry() and
         type(document['file_count']) is int and document['file_count']==APPROVED_FILE_COUNT and
         type(document['total_bytes']) is int and document['total_bytes']==APPROVED_TOTAL_BYTES,
         'Unsupported or unapproved SDK MDL manifest profile')
    root=real_path(document['sdk_root']);need(root not in path.parents,'Manifest cannot be inside SDK input tree')
    rows=document['files'];need(isinstance(rows,list) and len(rows)==APPROVED_FILE_COUNT,'Incomplete SDK MDL manifest inventory')
    names=[]
    for row in rows:
        need(isinstance(row,dict) and set(row)=={'path','sha256','bytes'} and
             isinstance(row['path'],str) and row['path'] and '\\' not in row['path'],
             'Invalid SDK MDL manifest file row')
        relative=PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts and str(relative)==row['path'] and
             type(row['bytes']) is int and row['bytes']>=0 and isinstance(row['sha256'],str) and
             re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,'Invalid SDK MDL file path/hash/bytes')
        names.append(row['path'])
    need(names==sorted(set(names)),'Duplicate or noncanonical SDK MDL file inventory')
    need(rows==_inventory(root),'SDK MDL file paths, hashes or bytes changed')
    files=[dict(file=str(root/row['path']),sha256=row['sha256'],bytes=row['bytes']) for row in rows]
    entry=next(row for row in files if row['file']==str(root/ENTRY_RELATIVE))
    return dict(manifest=dict(file=str(path),sha256=ref['sha256'],bytes=len(payload)),
        files=files,entry=entry,document=document)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sdk-root',type=Path,default=APPROVED_ROOT)
    args=parser.parse_args()
    print(json.dumps(prepare_manifest(args.output,sdk_root=args.sdk_root),sort_keys=True))


if __name__=='__main__':main()

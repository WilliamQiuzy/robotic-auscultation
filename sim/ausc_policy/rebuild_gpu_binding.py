"""Explicit replacement-Pod resource identity for rebuilt diagnostic ACT only.

The issuer reads an existing lock and GPU-0 UUID without flocking, creating or
replacing the lock. Acquisition uses the same original batch lock and owner
checks; this resource migration grants no formal or physical qualification.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

SCHEMA='m2_rebuild_expanded_gpu_resource_binding_v1'
EPOCH='observed-reconstruction-20261105'
TRAIN_IDS=frozenset(('p001120','p000758','p001031','p000833'))
SELECTION_IDS=frozenset(('p000923',))


def need(ok,message):
    if not ok:raise ValueError(message)


def validate(binding,*,base=None):
    """Metadata-only validation also works during later CPU checkpoint loading."""
    need(isinstance(binding,dict) and set(binding)=={'schema','base','path','device','inode','gpu_uuid'} and
        binding['schema']==SCHEMA,'Explicit rebuilt diagnostic resource binding required')
    declared=binding['base'];path=binding['path']
    need(isinstance(declared,str) and Path(declared).is_absolute() and str(Path(declared))==declared and
        os.path.normpath(declared)==declared and path==str(Path(declared)/'locks/isaac_gpu.lock') and
        (base is None or declared==str(base)),'Exact canonical base/locks/isaac_gpu.lock path required')
    need(type(binding['device']) is int and binding['device']>=0 and type(binding['inode']) is int and binding['inode']>0,
        'Actual POSIX device/inode required')
    need(isinstance(binding['gpu_uuid'],str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',binding['gpu_uuid']),
        'Exact physical GPU UUID required')
    return binding


def gpu_uuid():
    """Read GPU identity only; never initialize CUDA or create a compute process."""
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','--id=0'],
        text=True,timeout=15)
    rows=[row.strip() for row in raw.splitlines() if row.strip()]
    need(len(rows)==1 and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',rows[0]),
        'One actual GPU-0 UUID required')
    return rows[0]


def _existing(path):
    path=Path(path)
    need(path.is_absolute() and str(path)==os.path.normpath(str(path)), 'Canonical existing lock path required')
    for directory in reversed(path.parents):
        info=os.lstat(directory)
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            'GPU lock path ancestors cannot be links or non-directories')
    info=os.lstat(path)
    need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'GPU lock must be a regular single-link file, without aliases')
    need(str(path.resolve(strict=True))==str(path),'GPU lock path is aliased')
    return info


def _identity(info):return (info.st_dev,info.st_ino)


def _check_fd(path,fd,expected):
    current=_existing(path);opened=os.fstat(fd)
    need(stat.S_ISREG(opened.st_mode) and opened.st_nlink==1 and
        _identity(opened)==_identity(current)==expected,'Existing GPU lock FD/path identity changed')
    return opened


def _open_existing(path,*,writable=False):
    return os.open(path,(os.O_RDWR if writable else os.O_RDONLY)|os.O_NOFOLLOW|os.O_CLOEXEC)


def issue(base):
    """Read-only issuance is allowed while the original capture holds the lock."""
    need(sys.platform=='linux','Replacement Pod resource issuance requires Linux')
    base=str(base);path=Path(base)/'locks/isaac_gpu.lock'
    need(Path(base).is_absolute() and str(Path(base))==base and os.path.normpath(base)==base,
        'Explicit canonical existing base required')
    before=_existing(path);expected=_identity(before);fd=_open_existing(path)
    try:
        _check_fd(path,fd,expected);uuid=gpu_uuid();_check_fd(path,fd,expected)
        return validate(dict(schema=SCHEMA,base=base,path=str(path),device=expected[0],inode=expected[1],gpu_uuid=uuid))
    finally:os.close(fd)


def validate_dataset(value,*,schema,task_contract):
    """No numeric payload access; the original loader still verifies all arrays."""
    need(value.get('schema')==schema and value.get('complete') is True and value.get('rebuild_epoch')==EPOCH and
        value.get('original_cases')==0 and value.get('diagnostic_only') is True and value.get('formal_acceptance') is False and
        value.get('training_eligible') is False and value.get('test_payloads_opened') is False and
        value.get('task_contract')==task_contract,'Resource migration is restricted to the new rebuilt diagnostic dataset')
    splits=value.get('split_subject_ids',{})
    need(set(splits)=={'train','validation'} and isinstance(splits['train'],list) and isinstance(splits['validation'],list) and
        len(splits['train'])==len(TRAIN_IDS) and set(splits['train'])==TRAIN_IDS and
        len(splits['validation'])==len(SELECTION_IDS) and set(splits['validation'])==SELECTION_IDS,
        'Resource migration requires actual rebuilt TRAIN4/selection-VAL1; frozen/TEST identities excluded')
    return value


@contextmanager
def owned_gpu(binding):
    from . import process_guard as guard
    binding=validate(binding)
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0',
        'Original owned GPU process required')
    pids=sorted(guard.gpu_pids());need(not pids,'Conflicting GPU process before CUDA initialization')
    path=Path(binding['path']);expected=(binding['device'],binding['inode'])
    need(_identity(_existing(path))==expected,'Existing GPU lock identity differs from issued binding')
    before_uuid=gpu_uuid();need(before_uuid==binding['gpu_uuid'],'Actual GPU UUID differs from issued binding')
    fd=_open_existing(path,writable=True);locked=False
    try:
        _check_fd(path,fd,expected)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
        _check_fd(path,fd,expected)
        after_uuid=gpu_uuid();need(after_uuid==binding['gpu_uuid'],'Actual GPU UUID changed after lock acquisition')
        after_pids=sorted(guard.gpu_pids());need(not after_pids,'Conflicting GPU process after lock acquisition')
        _check_fd(path,fd,expected)
        yield dict(schema=SCHEMA,binding=deepcopy(binding),path=str(path),device=expected[0],inode=expected[1],
            gpu_uuid=after_uuid,pid=os.getpid(),checks=dict(gpu_pids_before=pids,gpu_pids_after_lock=after_pids,
                gpu_uuid_before=before_uuid,gpu_uuid_after_lock=after_uuid,fd_path_identity_preserved=True))
    finally:
        try:
            if locked:fcntl.flock(fd,fcntl.LOCK_UN)
        finally:os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--issue',action='store_true',required=True)
    parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=issue(args.base)
    with args.output.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(value,sort_keys=True),flush=True)


if __name__=='__main__':main()

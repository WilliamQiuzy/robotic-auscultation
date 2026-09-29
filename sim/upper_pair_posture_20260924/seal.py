"""Portable exact-file package seals; mirror receipts are the sole mutable output.

The manifest lives directly in the package root. It records every regular core
file using relative POSIX paths, byte sizes, and streamed SHA256. Runtime cache
and Apple metadata exclusions are explicit. The reserved mirror_receipts tree
may contain only JSON verification receipts bound to this exact manifest; it
cannot hide arbitrary report, code, or media files. This is integrity checking,
not a signature, physics acceptance, or a claim that excluded caches are sealed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import socket
import stat


SCHEMA="wam_upper_pair_package_seal_v1"
RECEIPT_SCHEMA="wam_upper_pair_package_verification_receipt_v1"
CHUNK_BYTES=1024*1024
EXCLUSIONS={"directory_names":["__pycache__"], "file_suffixes":[".pyc"],
            "file_names":[".DS_Store"], "entry_prefixes":["._"],
            "reserved_root_directory":"mirror_receipts",
            "reserved_directory_scope":"Only new JSON verification receipts bound to this manifest; never core artifacts.",
            "manifest_scope":"Only the manifest itself at the package root is excluded."}


def require(condition,message):
    if not condition:raise ValueError(message)


def _safe_relative(value):
    require(isinstance(value,str) and bool(value) and "\\" not in value and "\x00" not in value,
            "unsafe relative path")
    parts=value.split("/")
    require(not PurePosixPath(value).is_absolute() and all(part not in ("",".","..") for part in parts),
            "unsafe relative path: "+value)
    return parts


def _stamp(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _stream(path,keep=False):
    path=Path(path);before=path.lstat()
    require(not stat.S_ISLNK(before.st_mode),"symlink rejected: "+str(path))
    require(stat.S_ISREG(before.st_mode),"nonregular file rejected: "+str(path))
    descriptor=os.open(str(path),os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    digest=hashlib.sha256();size=0;blocks=[] if keep else None
    with os.fdopen(descriptor,"rb") as stream:
        require(_stamp(os.fstat(stream.fileno()))==_stamp(before),"file changed before hashing: "+str(path))
        for block in iter(lambda:stream.read(CHUNK_BYTES),b""):
            digest.update(block);size+=len(block)
            if keep:blocks.append(block)
        require(_stamp(os.fstat(stream.fileno()))==_stamp(before),"file changed while hashing: "+str(path))
    require(_stamp(path.lstat())==_stamp(before) and size==before.st_size,"file changed after hashing: "+str(path))
    record={"sha256":digest.hexdigest(),"bytes":size}
    return (record,b"".join(blocks)) if keep else record


def digest_file(path):
    """Read in 1 MiB blocks and reject path/inode/size/time changes during read."""
    return _stream(path)


def _json(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,"duplicate JSON key: "+key)
            result[key]=value
        return result
    def nonfinite(value):raise ValueError("nonfinite JSON: "+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=nonfinite)


def _context(manifest):
    path=Path(manifest).expanduser().absolute()
    require(".." not in path.parts,"unsafe manifest path")
    _safe_relative(path.name)
    require(path.name not in (".DS_Store","__pycache__","mirror_receipts")
            and not path.name.startswith("._") and not path.name.endswith(".pyc"),"excluded manifest name")
    require(not path.parent.is_symlink(),"symlink package root rejected")
    require(path.parent.is_dir(),"package directory does not exist")
    root=path.parent.resolve();path=root/path.name
    require(not path.is_symlink(),"symlink manifest rejected")
    return root,path


def _inventory(root,manifest_name):
    files={};receipts=[]
    def walk(directory,parts=(),ignored=False,receipt_zone=False):
        with os.scandir(directory) as stream:entries=sorted(stream,key=lambda e:e.name)
        for entry in entries:
            relative="/".join(parts+(entry.name,));_safe_relative(relative)
            info=entry.stat(follow_symlinks=False)
            require(not stat.S_ISLNK(info.st_mode),"symlink rejected: "+relative)
            require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),"nonregular entry rejected: "+relative)
            if not parts and entry.name==manifest_name:
                require(stat.S_ISREG(info.st_mode),"manifest is not regular")
                continue
            apple=entry.name.startswith("._")
            if stat.S_ISDIR(info.st_mode):
                reserved=not parts and entry.name=="mirror_receipts"
                walk(Path(entry.path),parts+(entry.name,),ignored or apple or entry.name=="__pycache__",
                     receipt_zone or reserved)
            else:
                require(not (not parts and entry.name=="mirror_receipts"),"receipt directory is a file")
                if ignored or apple or entry.name==".DS_Store" or entry.name.endswith(".pyc"):continue
                if receipt_zone:receipts.append(relative)
                else:files[relative]=digest_file(Path(entry.path))
    walk(root)
    return files,receipts


def _inventory_matches(actual,expected):
    missing=sorted(set(expected)-set(actual));extra=sorted(set(actual)-set(expected))
    changed=sorted(name for name in set(actual)&set(expected) if actual[name]!=expected[name])
    require(not missing and not extra and not changed,
            "core inventory changed: "+json.dumps({"missing":missing,"extra":extra,"changed":changed},ensure_ascii=False))


def _receipts(root,paths,manifest_digest,file_count,total_bytes):
    if manifest_digest is None:
        require(not paths,"receipt directory must be empty before creating the first seal")
        return
    allowed={"schema","status","manifest_sha256","file_count","total_bytes","verified_utc","hostname","platform"}
    for relative in paths:
        require(relative.endswith(".json"),"only JSON verification receipts allowed: "+relative)
        _,raw=_stream(root/relative,keep=True);record=_json(raw)
        require(isinstance(record,dict) and set(record)==allowed
                and record.get("schema")==RECEIPT_SCHEMA and record.get("status")=="verified",
                "invalid mirror receipt: "+relative)
        require(record.get("manifest_sha256")==manifest_digest and record.get("file_count")==file_count
                and record.get("total_bytes")==total_bytes,"mirror receipt binding changed: "+relative)
        require(isinstance(record.get("hostname"),str) and isinstance(record.get("platform"),dict)
                and isinstance(record.get("verified_utc"),str),"missing mirror receipt host/platform: "+relative)


def _encoded(document):
    return (json.dumps(document,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")


def _write_exclusive(path,raw):
    descriptor=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o644)
    identity=os.fstat(descriptor)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
    except BaseException:
        # An interrupted publication must not leave a partial seal or receipt.
        # Inode identity prevents deleting a replacement from another writer.
        try:
            current=Path(path).lstat()
            if (current.st_dev,current.st_ino)==(identity.st_dev,identity.st_ino):Path(path).unlink()
        except FileNotFoundError:pass
        raise


def _summary(document,digest,status):
    return {"status":status,"file_count":document["file_count"],"total_bytes":document["total_bytes"],
            "manifest_sha256":digest,"manifest_path":document["manifest_path"]}


def create_manifest(manifest):
    root,path=_context(manifest)
    require(not path.exists(),"manifest already exists; seals are immutable")
    files,receipts=_inventory(root,path.name)
    _receipts(root,receipts,None,len(files),sum(x["bytes"] for x in files.values()))
    document={"schema":SCHEMA,"hash_algorithm":"sha256","manifest_path":path.name,
              "created_utc":datetime.now(timezone.utc).isoformat(),"exclusions":EXCLUSIONS,
              "files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files.values()),
              "scope":"Exact regular core-file inventory relative to manifest parent. Excluded caches/Apple metadata are unsealed; mirror_receipts contains only new verification receipts. No physics acceptance or cryptographic signature is inferred."}
    # Hash every file again before publication; a changed or extra source means
    # no manifest is written. A final post-write check removes our new seal if
    # a source raced the publication interval.
    repeated,receipts=_inventory(root,path.name);_inventory_matches(repeated,files)
    _receipts(root,receipts,None,document["file_count"],document["total_bytes"])
    raw=_encoded(document);digest=hashlib.sha256(raw).hexdigest()
    _write_exclusive(path,raw)
    try:
        repeated,receipts=_inventory(root,path.name);_inventory_matches(repeated,files)
        _receipts(root,receipts,None,document["file_count"],document["total_bytes"])
        require(digest_file(path)["sha256"]==digest,"manifest changed during publication")
    except Exception:
        # Never remove a replacement written by somebody else.
        if path.is_file() and not path.is_symlink() and digest_file(path)["sha256"]==digest:path.unlink()
        raise
    return _summary(document,digest,"created")


def _validate_document(document,path):
    require(isinstance(document,dict) and document.get("schema")==SCHEMA
            and document.get("hash_algorithm")=="sha256","unsupported seal schema or hash algorithm")
    require(document.get("manifest_path")==path.name,"manifest filename changed")
    require(document.get("exclusions")==EXCLUSIONS,"manifest exclusion policy changed")
    files=document.get("files");require(isinstance(files,dict),"invalid file inventory")
    for relative,record in files.items():
        parts=_safe_relative(relative)
        require(relative!=path.name and parts[0]!="mirror_receipts" and "__pycache__" not in parts
                and not any(part.startswith("._") for part in parts) and parts[-1]!=".DS_Store"
                and not relative.endswith(".pyc"),"excluded or unsafe inventory path: "+relative)
        require(isinstance(record,dict) and set(record)=={"sha256","bytes"}
                and type(record.get("bytes")) is int and record["bytes"]>=0,"invalid size record: "+relative)
        digest=record.get("sha256")
        require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),
                "invalid SHA256: "+relative)
    require(type(document.get("file_count")) is int and document["file_count"]==len(files)
            and type(document.get("total_bytes")) is int
            and document["total_bytes"]==sum(record["bytes"] for record in files.values()),"inventory totals disagree")
    return files


def _receipt_target(root,receipt):
    given=Path(receipt)
    if given.is_absolute():
        try:relative=str(given.relative_to(root))
        except ValueError:raise ValueError("receipt must be under mirror_receipts")
    else:relative=given.as_posix()
    parts=_safe_relative(relative)
    require(len(parts)>=2 and parts[0]=="mirror_receipts" and parts[-1].endswith(".json")
            and not any(part.startswith("._") or part=="__pycache__" for part in parts),
            "receipt must be a new JSON file under mirror_receipts")
    current=root
    for part in parts:
        current=current/part
        require(not current.is_symlink(),"symlink receipt path rejected")
    require(not current.exists(),"receipt already exists; verification receipts are immutable")
    return current,relative


def verify_manifest(manifest,receipt=None):
    root,path=_context(manifest)
    record,raw=_stream(path,keep=True);document=_json(raw);expected=_validate_document(document,path)
    digest=record["sha256"]
    target,relative=_receipt_target(root,receipt) if receipt is not None else (None,None)
    for _ in range(2):
        actual,receipts=_inventory(root,path.name);_inventory_matches(actual,expected)
        _receipts(root,receipts,digest,document["file_count"],document["total_bytes"])
    require(digest_file(path)==record,"manifest changed during verification")
    result=_summary(document,digest,"verified")
    if target is not None:
        value={"schema":RECEIPT_SCHEMA,"status":"verified","manifest_sha256":digest,
               "file_count":document["file_count"],"total_bytes":document["total_bytes"],
               "verified_utc":datetime.now(timezone.utc).isoformat(),"hostname":socket.gethostname(),
               "platform":{"system":platform.system(),"release":platform.release(),
                           "machine":platform.machine(),"python":platform.python_version()}}
        target.parent.mkdir(parents=True,exist_ok=True)
        # Recheck parent components immediately before the exclusive write.
        _receipt_target(root,relative)
        _write_exclusive(target,_encoded(value));result["receipt_path"]=relative
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--create",type=Path,metavar="MANIFEST",help="new manifest directly in the finished package root")
    modes.add_argument("--verify",type=Path,metavar="MANIFEST",help="verify the exact core inventory on any mirror host")
    parser.add_argument("--receipt",type=Path,help="new JSON file under package/mirror_receipts only")
    args=parser.parse_args(argv)
    try:
        if args.create:
            result=create_manifest(args.create)
            if args.receipt is not None:result["receipt_path"]=verify_manifest(args.create,receipt=args.receipt)["receipt_path"]
        else:result=verify_manifest(args.verify,receipt=args.receipt)
    except (OSError,ValueError,TypeError,KeyError) as error:
        print(json.dumps({"status":"failed","error":str(error)},ensure_ascii=False));return 1
    print(json.dumps(result,ensure_ascii=False,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())

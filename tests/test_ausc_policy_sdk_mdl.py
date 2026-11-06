"""CPU-only SDK MDL library integrity; synthetic content is never rendered."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec('sim.ausc_policy.sdk_mdl') is not None,'Missing pure-CPU SDK MDL manifest helper'
    return importlib.import_module('sim.ausc_policy.sdk_mdl')


def sdk_fixture(tmp_path,monkeypatch):
    m=module();root=tmp_path/'sdk/kit/mdl'
    contents={m.ENTRY_RELATIVE:b'mdl 1.7; import OmniPBR_ClearCoat::*; export material OmniPBR() = material();',
        'core/Base/OmniPBR_ClearCoat.mdl':b'mdl 1.7; // synthetic clear coat',
        'core/mdl/nvidia/core_definitions.mdl':b'mdl 1.7; // synthetic core definitions',
        'rtx/helper.mdl':b'mdl 1.7; // synthetic helper','core/.packman.lock':b''}
    for name,data in contents.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr(m,'APPROVED_ROOT',root)
    monkeypatch.setattr(m,'APPROVED_ENTRY_BYTES',len(contents[m.ENTRY_RELATIVE]))
    monkeypatch.setattr(m,'APPROVED_ENTRY_SHA256',hashlib.sha256(contents[m.ENTRY_RELATIVE]).hexdigest())
    monkeypatch.setattr(m,'APPROVED_FILE_COUNT',len(contents))
    monkeypatch.setattr(m,'APPROVED_TOTAL_BYTES',sum(map(len,contents.values())))
    rows=[dict(path=name,bytes=len(contents[name]),sha256=hashlib.sha256(contents[name]).hexdigest()) for name in sorted(contents)]
    document=dict(schema=m.SCHEMA,sdk_root=str(root),entry=dict(path=m.ENTRY_RELATIVE,
        bytes=m.APPROVED_ENTRY_BYTES,sha256=m.APPROVED_ENTRY_SHA256),files=rows,
        file_count=len(rows),total_bytes=m.APPROVED_TOTAL_BYTES)
    payload=json.dumps(document,sort_keys=True,allow_nan=False).encode()+b'\n'
    monkeypatch.setattr(m,'APPROVED_MANIFEST_SHA256',hashlib.sha256(payload).hexdigest(),raising=False)
    return m,root,contents


def test_production_profile_is_exact_and_not_generic_sdk_search():
    m=module()
    assert str(m.APPROVED_ROOT)=='/opt/wam-runtime/envs/isaacsim60/lib/python3.12/site-packages/isaacsim/kit/mdl'
    assert m.ENTRY_RELATIVE=='core/Base/OmniPBR.mdl'
    assert m.APPROVED_ENTRY_BYTES==13989
    assert m.APPROVED_ENTRY_SHA256=='d384fc800f120e3cad1b7ca8837f9c554f9c34343d194813fca27caa493001f2'
    assert m.APPROVED_FILE_COUNT==244 and m.APPROVED_TOTAL_BYTES==5432503
    assert m.APPROVED_MANIFEST_SHA256=='223126cf1c9891611e8422533b596acf6ed7d0b703962f2186899765242f5936'


def test_atomic_manifest_covers_whole_library_without_modifying_it(tmp_path,monkeypatch):
    m,root,contents=sdk_fixture(tmp_path,monkeypatch);before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    ref=m.prepare_manifest(tmp_path/'mdl-manifest.json',sdk_root=root)
    assert set(ref)=={'file','sha256'}
    bound=m.validate_manifest(ref);document=json.loads(Path(ref['file']).read_bytes())
    assert document['file_count']==len(contents) and document['total_bytes']==sum(map(len,contents.values()))
    assert [row['path'] for row in document['files']]==sorted(contents)
    assert bound['entry']['file']==str(root/m.ENTRY_RELATIVE)
    assert len(bound['files'])==len(contents)
    assert {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before
    assert Path(ref['file']).stat().st_mode & 0o222==0
    with pytest.raises(ValueError):m.prepare_manifest(tmp_path/'mdl-manifest.json',sdk_root=root)
    with pytest.raises(ValueError):m.prepare_manifest(root/'manifest.json',sdk_root=root)


def test_initial_freeze_rejects_changed_nonentry_file_even_with_approved_count_and_bytes(tmp_path,monkeypatch):
    m,root,_=sdk_fixture(tmp_path,monkeypatch);path=root/'rtx/helper.mdl'
    data=path.read_bytes();path.write_bytes(b'X'+data[1:])
    with pytest.raises(ValueError):m.prepare_manifest(tmp_path/'manifest.json')


@pytest.mark.parametrize('kind',['mutation','missing','extra','file_symlink','directory_symlink','manifest_mutation'])
def test_manifest_rejects_mutation_missing_extra_or_symlink_inventory(tmp_path,monkeypatch,kind):
    m,root,contents=sdk_fixture(tmp_path,monkeypatch);ref=m.prepare_manifest(tmp_path/'manifest.json')
    path=root/'rtx/helper.mdl'
    if kind=='mutation':path.write_bytes(b'changed but unbound')
    elif kind=='missing':path.unlink()
    elif kind=='extra':(root/'unknown.mdl').write_bytes(b'unknown')
    elif kind=='file_symlink':path.unlink();path.symlink_to(root/m.ENTRY_RELATIVE)
    elif kind=='directory_symlink':(root/'aliased').symlink_to(root/'core',target_is_directory=True)
    else:
        target=Path(ref['file']);target.chmod(0o644);target.write_text('{}')
    with pytest.raises(ValueError):m.validate_manifest(ref)


@pytest.mark.parametrize('kind',['wrong_entry','incomplete','duplicate','traversal','absolute','root_alias'])
def test_even_rehashed_forged_manifest_is_rejected(tmp_path,monkeypatch,kind):
    m,root,_=sdk_fixture(tmp_path,monkeypatch);ref=m.prepare_manifest(tmp_path/'manifest.json')
    path=Path(ref['file']);document=json.loads(path.read_bytes())
    if kind=='wrong_entry':document['entry']['path']='rtx/helper.mdl'
    elif kind=='incomplete':document['files'].pop()
    elif kind=='duplicate':document['files'][1]=document['files'][0]
    elif kind=='traversal':document['files'][0]['path']='../outside.mdl'
    elif kind=='absolute':document['files'][0]['path']=str(root/m.ENTRY_RELATIVE)
    else:document['sdk_root']=str(tmp_path/'other-sdk')
    path.chmod(0o644);path.write_text(json.dumps(document));ref['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):m.validate_manifest(ref)

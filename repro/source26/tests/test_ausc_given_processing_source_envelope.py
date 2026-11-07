"""Closed future source cap; all fixtures here are synthetic CPU metadata."""
from copy import deepcopy
import io
from pathlib import Path
import tarfile

import pytest

from sim.ausc_policy import m2_given_contact_dataset as d
from test_ausc_given_contact_dataset import save


def source_scope(tmp_path,monkeypatch,*,final=True):
    live=tmp_path/'final-live';live.mkdir();archive_root=tmp_path/'final-archive';archive_root.mkdir()
    monkeypatch.setattr(d,'FINAL_PROCESSING_ROOT',str(live),raising=False)
    source=live/'sim/ausc_policy/m2_given_contact_dataset.py';source.parent.mkdir(parents=True);source.write_text('frozen gate\n')
    factory=source.with_name('m2_given_contact_act.py');factory.write_text('old explicit factory\n')
    (live/'CODE_COMMIT').write_text('a'*40+'\n')
    save(live/'SOURCE_MANIFEST.json',[dict(file=p.relative_to(live).as_posix(),sha256=d.sha256(p)) for p in (source,factory)])
    archive=archive_root/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(live.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(live).as_posix())
    descriptor=tmp_path/'metadata/final_processing_freeze.json'
    files={p.relative_to(live).as_posix():dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha256(p)) for p in live.rglob('*') if p.is_file()}
    scope=dict(scope_id='final_processing_sources_v1',live_root=str(live),archive_root=str(archive_root),
        archive_file='source.tar.gz',descriptor_path=str(descriptor),limit_bytes=67108864,charged_bytes=67108864,
        allowed_files=sorted(files),required_unchanged_gate_sources={source.relative_to(live).as_posix():d.sha256(source)})
    historical=tmp_path/'historical-gate';__import__('shutil').copytree(live,historical)
    oldfiles={p.relative_to(historical).as_posix():dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha256(p))
        for p in historical.rglob('*') if p.is_file()}
    dep=tmp_path/'cpu-fixture-dependency';dep.write_text('CPU metadata only')
    depref=dict(path=str(dep),bytes=dep.stat().st_size,sha256=d.sha256(dep))
    value=dict(schema='m2_given_collection_execution_freeze_v1',root=str(historical),files=oldfiles,
        reviewed_commit='a'*40,code_commit=oldfiles['CODE_COMMIT'],source_manifest=oldfiles['SOURCE_MANIFEST.json'],
        environment=dict(python=dict(depref,invoked_path=str(dep)),ruckig=depref,ruckig_version='0.15.3'))
    oldref=save(tmp_path/'historical-freeze.json',value)
    scope['original_execution_freeze']=dict(reference=oldref,root=str(historical),files=oldfiles,environment=value['environment'])
    if final:
        save(descriptor,dict(schema='m2_given_final_processing_freeze_v1',scope_id=scope['scope_id'],reviewed_commit='a'*40,
            live_root=str(live),code_commit=files['CODE_COMMIT'],source_manifest=files['SOURCE_MANIFEST.json'],files=files,
            archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=d.sha256(archive)),
            preparer_sha256=d.sha256(source),official_environment=dict(path='/official/python',sha256='a'*64),
            required_unchanged_gate_sources=deepcopy(scope['required_unchanged_gate_sources']),
            original_execution_freeze=deepcopy(scope['original_execution_freeze'])))
    return scope


def test_source_cap_precharges_both_copies_and_allows_only_pending_reconciliation(tmp_path,monkeypatch):
    scope=source_scope(tmp_path,monkeypatch,final=False)
    result=d.processing_source_accounting(scope,tmp_path,snapshot_phase='reconcile')
    assert result['charged_bytes']==67108864 and result['pending'] is True and result['observed_bytes']>0
    with pytest.raises(ValueError,match='descriptor'):
        d.processing_source_accounting(scope,tmp_path,snapshot_phase='pre_export')


def test_exact_source_descriptor_closes_live_and_archive_inventory(tmp_path,monkeypatch):
    scope=source_scope(tmp_path,monkeypatch)
    report=d.processing_source_accounting(scope,tmp_path,snapshot_phase='post_export')
    assert report['pending'] is False and report['observed_bytes']==report['live_bytes']+report['archive_bytes']
    assert report['charged_bytes']==67108864 and report['descriptor']['path']==scope['descriptor_path']


def test_final_source_reopens_original_history_at_its_own_root(tmp_path,monkeypatch):
    scope=source_scope(tmp_path,monkeypatch)
    original=Path(scope['original_execution_freeze']['root'])/'sim/ausc_policy/m2_given_contact_dataset.py'
    original.write_text('changed after original gate ran')
    with pytest.raises(ValueError,match='Original executed tree'):
        d.processing_source_accounting(scope,tmp_path,snapshot_phase='pre_export')


def test_future_reviewed_factory_may_differ_while_original_gate_stays_frozen(tmp_path,monkeypatch):
    scope=source_scope(tmp_path,monkeypatch);live=Path(scope['live_root']);descriptor=Path(scope['descriptor_path'])
    factory=live/'sim/ausc_policy/m2_given_contact_act.py';factory.write_text('future reviewed explicit route\n')
    sources=[p for p in live.rglob('*') if p.is_file() and p.name not in ('SOURCE_MANIFEST.json','CODE_COMMIT')]
    save(live/'SOURCE_MANIFEST.json',[dict(file=p.relative_to(live).as_posix(),sha256=d.sha256(p)) for p in sources])
    value=d.shared._json(descriptor)
    value['files']={p.relative_to(live).as_posix():dict(path=str(p),bytes=p.stat().st_size,sha256=d.sha256(p))
        for p in live.rglob('*') if p.is_file()}
    value['source_manifest']=value['files']['SOURCE_MANIFEST.json']
    archive=Path(value['archive']['path'])
    with tarfile.open(archive,'w:gz') as tar:
        for p in live.rglob('*'):
            if p.is_file():tar.add(p,arcname=p.relative_to(live).as_posix())
    value['archive'].update(bytes=archive.stat().st_size,sha256=d.sha256(archive));save(descriptor,value)
    assert d.processing_source_accounting(scope,tmp_path,snapshot_phase='pre_export')['pending'] is False


@pytest.mark.parametrize('fault',['cap','charge','root','unknown','symlink','hardlink','overrun','changed_live',
    'omitted_live','archive_extra','archive_duplicate','archive_traversal','archive_link','replaced_gate','overlap'])
def test_source_cap_rejects_unclosed_or_retroactively_changed_gate(tmp_path,monkeypatch,fault):
    scope=source_scope(tmp_path,monkeypatch);live=Path(scope['live_root']);descriptor=Path(scope['descriptor_path'])
    value=d.shared._json(descriptor);source=live/'sim/ausc_policy/m2_given_contact_dataset.py';excluded=[]
    if fault=='cap':scope['limit_bytes']+=1
    elif fault=='charge':scope['charged_bytes']-=1
    elif fault=='root':scope['live_root']=str(tmp_path/'another-root')
    elif fault=='unknown':(live/'unexpected.py').write_text('x')
    elif fault=='symlink':source.unlink();source.symlink_to(tmp_path/'elsewhere')
    elif fault=='hardlink':__import__('os').link(source,tmp_path/'linked.py')
    elif fault=='overrun':
        with source.open('r+b') as out:out.truncate(67108865)
    elif fault=='changed_live':source.write_text('changed')
    elif fault=='omitted_live':del value['files'][source.relative_to(live).as_posix()];save(descriptor,value)
    elif fault=='replaced_gate':
        value['required_unchanged_gate_sources']={};save(descriptor,value)
    elif fault=='overlap':excluded=[str(source)]
    else:
        archive=Path(value['archive']['path'])
        with tarfile.open(archive,'w:gz') as tar:
            for p in sorted(live.rglob('*')):
                if p.is_file():tar.add(p,arcname=p.relative_to(live).as_posix())
            name='unexpected.py' if fault=='archive_extra' else '../outside' if fault=='archive_traversal' else source.relative_to(live).as_posix()
            info=tarfile.TarInfo(name)
            if fault=='archive_link':info.type=tarfile.SYMTYPE;info.linkname='/outside'
            else:info.size=1
            tar.addfile(info,None if fault=='archive_link' else io.BytesIO(b'x'))
        value['archive'].update(bytes=archive.stat().st_size,sha256=d.sha256(archive));save(descriptor,value)
    with pytest.raises(ValueError):
        d.processing_source_accounting(scope,tmp_path,snapshot_phase='training_preflight',excluded_paths=excluded)

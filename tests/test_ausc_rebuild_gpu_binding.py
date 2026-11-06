"""POSIX lock migration tests use local files and CPU-only GPU-query doubles."""
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import importlib
import importlib.util
import os
from pathlib import Path

import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt
from sim.ausc_policy import m2_given_contact_expanded_pilot_train as trainer

UUID='GPU-12345678-1234-1234-1234-123456789abc'


def module():
    name='sim.ausc_policy.rebuild_gpu_binding'
    assert importlib.util.find_spec(name) is not None,'Explicit replacement resource binding missing'
    return importlib.import_module(name)


def setup(tmp_path,monkeypatch):
    m=module();base=tmp_path/'base';(base/'locks').mkdir(parents=True)
    lock=base/'locks/isaac_gpu.lock';lock.write_text('original lock bytes\n')
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setattr(m,'gpu_uuid',lambda:UUID)
    from sim.ausc_policy import process_guard
    monkeypatch.setattr(process_guard,'gpu_pids',lambda:set())
    monkeypatch.setenv('AUSC_OWNED_RUN','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    return m,base,lock


def test_read_only_issuer_works_while_capture_holds_original_lock(tmp_path,monkeypatch):
    m,base,lock=setup(tmp_path,monkeypatch);before=lock.stat();raw=lock.read_bytes()
    with lock.open('r+') as capture:
        fcntl.flock(capture.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=m.issue(base)
        assert binding==dict(schema=m.SCHEMA,base=str(base),path=str(lock),device=before.st_dev,inode=before.st_ino,gpu_uuid=UUID)
        with pytest.raises(BlockingIOError):
            with rt.owned_gpu(binding):pytest.fail('Active capture lock bypassed')
    assert lock.read_bytes()==raw and lock.stat().st_ino==before.st_ino
    with rt.owned_gpu(binding) as ownership:
        assert ownership['path']==str(lock) and ownership['binding']==binding and ownership['gpu_uuid']==UUID
        assert ownership['checks']['gpu_pids_before']==[]
    with lock.open('r+') as again:fcntl.flock(again.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('fault',['inode','device','uuid','leaf_symlink','parent_symlink','hardlink','missing','not_regular','wrong_path'])
def test_resource_changes_fail_closed_without_creating_or_replacing_lock(tmp_path,monkeypatch,fault):
    m,base,lock=setup(tmp_path,monkeypatch);binding=m.issue(base)
    if fault=='inode':binding['inode']+=1
    elif fault=='device':binding['device']+=1
    elif fault=='uuid':binding['gpu_uuid']='GPU-aaaaaaaa-1234-1234-1234-123456789abc'
    elif fault=='leaf_symlink':
        target=base/'saved-lock';lock.rename(target);lock.symlink_to(target)
    elif fault=='parent_symlink':
        target=base/'saved-locks';lock.parent.rename(target);(base/'locks').symlink_to(target,target_is_directory=True)
    elif fault=='hardlink':os.link(lock,base/'alias')
    elif fault=='missing':lock.unlink()
    elif fault=='not_regular':lock.unlink();lock.mkdir()
    elif fault=='wrong_path':binding['path']=str(base/'locks/another.lock')
    with pytest.raises((ValueError,OSError)):
        with rt.owned_gpu(binding):pytest.fail('Changed resource admitted')
    if fault=='missing':assert not lock.exists()
    if fault=='wrong_path':assert not Path(binding['path']).exists()


def test_identity_replacement_after_flock_is_rejected_and_releases_original_fd(tmp_path,monkeypatch):
    m,base,lock=setup(tmp_path,monkeypatch);binding=m.issue(base);real=m.fcntl.flock;old=base/'old-lock'
    def flock(fd,operation):
        real(fd,operation)
        if operation==(fcntl.LOCK_EX|fcntl.LOCK_NB):
            lock.rename(old);lock.write_text('replacement')
    monkeypatch.setattr(m.fcntl,'flock',flock)
    with pytest.raises(ValueError,match='identity'):
        with rt.owned_gpu(binding):pytest.fail('Lock replacement after flock admitted')
    with old.open('r+') as descriptor:real(descriptor.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('fault',['unowned','cuda','busy','uuid_after_lock'])
def test_original_owner_visibility_idle_and_exact_uuid_checks_remain_mandatory(tmp_path,monkeypatch,fault):
    m,base,lock=setup(tmp_path,monkeypatch);binding=m.issue(base)
    from sim.ausc_policy import process_guard
    if fault=='unowned':monkeypatch.delenv('AUSC_OWNED_RUN')
    elif fault=='cuda':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    elif fault=='busy':monkeypatch.setattr(process_guard,'gpu_pids',lambda:{123})
    else:
        values=iter([UUID,'GPU-aaaaaaaa-1234-1234-1234-123456789abc'])
        monkeypatch.setattr(m,'gpu_uuid',lambda:next(values))
    with pytest.raises(ValueError):
        with rt.owned_gpu(binding):pytest.fail('Original ownership/resource gate bypassed')


def rebuilt_header():
    return dict(schema=trainer.PREPARATION_SCHEMA,complete=True,diagnostic_only=True,formal_acceptance=False,
        training_eligible=False,test_payloads_opened=False,rebuild_epoch='observed-reconstruction-20261105',original_cases=0,
        split_subject_ids=dict(train=['p001120','p000758','p001031','p000833'],validation=['p000923']),
        task_contract=rt.task_contract())


def config(tmp_path,base,binding,header):
    return dict(trainer.SCHEDULE,base=str(base),dataset=rt.write_new(tmp_path/'data.json',header),replacement_pod_gpu_binding=binding)


def test_opt_in_config_binds_only_exact_rebuilt_diagnostic_train4_selection1(tmp_path,monkeypatch):
    m,base,lock=setup(tmp_path,monkeypatch);binding=m.issue(base);value=config(tmp_path,base,binding,rebuilt_header())
    assert trainer.validate_config(value)==value
    assert trainer.training_protocol(value)['schedule']['replacement_pod_gpu_binding']==binding
    legacy=deepcopy(value);legacy.pop('replacement_pod_gpu_binding')
    assert trainer.validate_config(legacy)==legacy
    # Metadata validation never needs the original lock or GPU on CPU reload.
    lock.unlink();monkeypatch.setattr(m,'gpu_uuid',lambda:pytest.fail('CPU validation queried GPU'))
    assert trainer.validate_config(value)==value


@pytest.mark.parametrize('fault',['old_epoch','formal','test','old_cases','selection_leak','train_scope','wrong_base','schema'])
def test_replacement_binding_refuses_old_or_formal_or_scope_changed_datasets(tmp_path,monkeypatch,fault):
    m,base,lock=setup(tmp_path,monkeypatch);binding=m.issue(base);header=rebuilt_header()
    if fault=='old_epoch':header.pop('rebuild_epoch')
    elif fault=='formal':header['formal_acceptance']=True
    elif fault=='test':header['test_payloads_opened']=True
    elif fault=='old_cases':header['original_cases']=8
    elif fault=='selection_leak':header['split_subject_ids']['validation'].append('p001030')
    elif fault=='train_scope':header['split_subject_ids']['train'].pop()
    elif fault=='wrong_base':binding['base']=str(tmp_path)
    elif fault=='schema':binding['schema']='formal_gpu_permission'
    with pytest.raises(ValueError):trainer.validate_config(config(tmp_path,base,binding,header))


@pytest.mark.parametrize('opt_in',[False,True])
def test_expanded_fit_passes_binding_only_when_explicit_and_stops_before_cuda(tmp_path,monkeypatch,opt_in):
    m,base,lock=setup(tmp_path,monkeypatch);binding=m.issue(base)
    value=config(tmp_path,base,binding,rebuilt_header())
    if not opt_in:value.pop('replacement_pod_gpu_binding')
    config_ref=rt.write_new(tmp_path/'config.json',value);calls=[]
    @contextmanager
    def lease(*args):
        calls.append(args);raise RuntimeError('lease reached; CUDA intentionally not initialized')
        yield
    monkeypatch.setattr(rt,'owned_gpu',lease)
    monkeypatch.setattr(trainer,'load_dataset',lambda ref:{'scope':{}})
    monkeypatch.setattr(rt,'make_model',lambda *a,**k:pytest.fail('Model initialized in resource test'))
    with pytest.raises(RuntimeError,match='lease reached'):trainer.fit(config_ref,tmp_path/'training')
    assert calls==[(binding,)] if opt_in else calls==[()]


def test_gpu_uuid_query_is_fixed_to_physical_gpu0_and_rejects_ambiguous_output(monkeypatch):
    m=module();calls=[]
    def query(command,**kwargs):calls.append((command,kwargs));return UUID+'\n'
    monkeypatch.setattr(m.subprocess,'check_output',query)
    assert m.gpu_uuid()==UUID
    assert calls[0][0]==['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','--id=0']
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:UUID+'\n'+UUID+'\n')
    with pytest.raises(ValueError,match='One actual'):m.gpu_uuid()

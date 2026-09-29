from pathlib import Path
import pytest
from sim.auscultation_pi05_v11 import execution as X


def stage():
    return dict(stage='training',command=['python','train'],exit_code=0,passed=True,process_group_drained=True,errors=[],signals=[],deferred_signals=[],
        process_identity=dict(pid=1234,pgid=1234,sid=1234,start_ticks='5678'),started_utc='start',ended_utc='end')


@pytest.mark.parametrize('key,value',[('exit_code',True),('exit_code',1),('passed',False),('process_group_drained',False),('signals',['SIGINT']),('deferred_signals',['SIGTERM']),('errors',['failed'])])
def test_non_successful_owner_rejected(key,value):
    r=stage();r[key]=value
    with pytest.raises(ValueError):X.check_stage(r,'training',['python','train'])


def test_exact_owned_stage_and_no_foreign_command():
    X.check_stage(stage(),'training',['python','train'])
    with pytest.raises(ValueError):X.check_stage(stage(),'training',['python','replay'])


def test_late_dangling_cancellation_still_rejects(tmp_path):
    (tmp_path/'training.late_interrupt.json').symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError):X.no_cancel(tmp_path)


import json
from types import SimpleNamespace

def fixture(root,terminal_pid=123):
    run=root/'runs/new-run';run.mkdir(parents=True)
    folder=root/'diagnostics/pi05_v11_training_lifecycles/one';folder.mkdir(parents=True)
    launcher=root/'diagnostics/pi05_v11_training_cloud_launcher_v1/launch.py';launcher.parent.mkdir(parents=True);launcher.write_text('# synthetic')
    guard=root/'guard.py';guard.write_text('# synthetic guard')
    def write(path,value):
        path.write_text(json.dumps(value));return dict(path=str(path),sha256=X.file_sha(path))
    binding=dict(manifest_binding=dict(path=str(root/'manifest.json'),sha256='a'*64),
                 normalization_file_binding=dict(path=str(root/'normalization.json'),sha256='b'*64))
    context=dict(config=dict(source_sha256={'synthetic':'source'},dataset_binding=binding),
                 dataset=SimpleNamespace(data_hash='synthetic-data'),terminal=dict(pid=terminal_pid))
    terminal=run/'training_run.json';tb=write(terminal,context['terminal'])
    context['artifact_sha256']={str(terminal):tb['sha256']}
    prep=dict(schema='wam_pi05_v11_training_preparation_v1',run=str(run),training_source_sha256=context['config']['source_sha256'],
              dataset_binding=binding,launcher_binding=dict(path=str(launcher),sha256=X.file_sha(launcher)))
    pb=write(root/'preparation.json',prep)
    command=[str(root/'vendor/openpi/.venv/bin/python'),'-u','-B','-m','sim.auscultation_pi05_v11.train',
             '--manifest',binding['manifest_binding']['path'],'--normalization',binding['normalization_file_binding']['path'],'--output',str(run)]
    completion=[str(root/'vendor/openpi/.venv/bin/python'),'-u','-B',str(launcher),'--preparation',pb['path'],
                '--preparation-sha256',pb['sha256'],'--evidence',str(folder),'--cpu-completion']
    stages=[]
    for i,(name,cmd) in enumerate((('cpu_preflight',command+['--verify-only']),('training',command),('cpu_completion',completion))):
        pid=(122,123,124)[i]
        stage=dict(stage=name,command=cmd,exit_code=0,passed=True,process_group_drained=True,errors=[],signals=[],deferred_signals=[],
                   process_identity=dict(pid=pid,pgid=pid,sid=pid,start_ticks=str(100+i)),started_utc='synthetic-start',ended_utc='synthetic-end')
        write(folder/(name+'.process.json'),stage);stages.append(stage)
    proof=dict(schema='wam_pi05_v11_full_context_completion_v1',completed=True,run=str(run),source_sha256=context['config']['source_sha256'],
               data_hash=context['dataset'].data_hash,terminal=context['terminal'],artifact_sha256=context['artifact_sha256'])
    cb=write(folder/'context_verified.json',proof)
    life=dict(schema=X.LIFECYCLE_SCHEMA,run=str(run),completed=True,training_complete=True,prepared_only=False,gpu_released=True,
              errors=[],guard_sha256=X.file_sha(guard),preparation=pb,training_command=command,completion_command=completion,
              stages=stages,completion=cb)
    lb=write(folder/'lifecycle.json',life);write(run/'TRAINING_OWNER.json',dict(schema=X.SCHEMA,run=str(run),lifecycle=lb))
    return run,folder,guard,context


@pytest.mark.parametrize('terminal_pid,marker,accepted',[(123,None,True),(999,None,False),(123,'late_interrupt.json',False),(123,'training.late_interrupt.json',False),(123,'training.cancelled.json',False)])
def test_full_owned_completion_links_terminal_and_rejects_cancellation(tmp_path,monkeypatch,terminal_pid,marker,accepted):
    root=tmp_path.resolve();run,folder,guard,context=fixture(root,terminal_pid)
    if marker:(folder/marker).symlink_to(root/'absent')
    monkeypatch.setattr(X,'BASE',root);monkeypatch.setattr(X,'GUARD',guard);monkeypatch.setattr(X,'GUARD_SHA',X.file_sha(guard))
    if accepted:assert X.verify_training_owner(run,context)['run']==str(run)
    else:
        with pytest.raises(ValueError):X.verify_training_owner(run,context)


@pytest.mark.parametrize('ticks',[None,0,'0','-1','abc','١٢'])
def test_process_start_clock_is_positive_ascii_decimal(ticks):
    r=stage();r['process_identity']['start_ticks']=ticks
    with pytest.raises(ValueError):X.check_stage(r,'training',['python','train'])

"""Read actual guarded training ownership before allowing checkpoint selection."""
from pathlib import Path
import json
from sim.auscultation_pi05_v7.coverage_pilot_schedule import file_sha

SCHEMA='wam_pi05_v11_training_owner_pointer_v1'
LIFECYCLE_SCHEMA='wam_pi05_v11_guarded_training_lifecycle_v1'
GUARD_SHA='48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2'
BASE=Path('/home/william/wam-vla-v2')
GUARD=BASE/'diagnostics/coverage_pilot_cloud_launcher/process_guard.py'


def need(value,message):
    if not value:raise ValueError(message)


def pin(path,artifacts,expected=None):
    p=Path(path);need(p.is_absolute() and p==p.resolve() and p.is_file() and not p.is_symlink(),'Canonical regular process evidence required')
    h=file_sha(p);need(expected is None or expected==h,'Execution evidence SHA differs')
    need(str(p) not in artifacts or artifacts[str(p)]==h,'Consumed execution evidence changed')
    artifacts[str(p)]=h;return json.loads(p.read_text())


def check_stage(record,name,command):
    need(record.get('stage')==name and record.get('command')==command,'Actual guarded stage command differs')
    need(type(record.get('exit_code')) is int and record['exit_code']==0 and record.get('passed') is True and record.get('process_group_drained') is True,'Actual owned process must exit0 and drain')
    need(record.get('errors')==[] and record.get('signals')==[] and record.get('deferred_signals')==[],'Stage failed or cancelled')
    identity=record.get('process_identity',{})
    need(type(identity.get('pid')) is int and identity['pid']>0 and identity.get('pgid')==identity['pid'] and identity.get('sid')==identity['pid'],'Actual owned session identity required')
    ticks=identity.get('start_ticks')
    need(isinstance(ticks,str) and ticks.isascii() and ticks.isdecimal() and int(ticks)>0 and bool(record.get('started_utc')) and bool(record.get('ended_utc')),'Actual process timing required')


def no_cancel(folder):
    need(not any(p.name=='late_interrupt.json' or p.name.endswith(('.late_interrupt.json','.cancelled.json')) for p in Path(folder).iterdir()),'Training owner cancellation evidence present')


def verify_training_owner(run,context):
    run=Path(run);artifacts={};pointer=pin(run/'TRAINING_OWNER.json',artifacts)
    need(pointer.get('schema')==SCHEMA and pointer.get('run')==str(run),'Original V11 training owner pointer required')
    b=pointer['lifecycle'];path=Path(b['path']);life=pin(path,artifacts,b['sha256']);folder=path.parent
    need(folder.parent==BASE/'diagnostics/pi05_v11_training_lifecycles' and path.name=='lifecycle.json','Original bounded owner evidence directory required')
    no_cancel(folder)
    need(life.get('schema')==LIFECYCLE_SCHEMA and life.get('run')==str(run) and life.get('completed') is True and life.get('training_complete') is True
         and life.get('prepared_only') is False and life.get('gpu_released') is True and life.get('errors')==[],'Complete actual V11 training lifecycle required')
    need(file_sha(GUARD)==GUARD_SHA and life.get('guard_sha256')==GUARD_SHA,'Original owned resource guard changed')
    prep=life['preparation'];prepared=pin(prep['path'],artifacts,prep['sha256'])
    need(prepared.get('schema')=='wam_pi05_v11_training_preparation_v1' and prepared.get('run')==str(run),'Actual V11 launch preparation differs')
    need(prepared['training_source_sha256']==context['config']['source_sha256'] and prepared['dataset_binding']==context['config']['dataset_binding'],'Actual launcher source/data differs')
    launcher=prepared['launcher_binding'];lp=Path(launcher['path'])
    need(lp==BASE/'diagnostics/pi05_v11_training_cloud_launcher_v1/launch.py' and not lp.is_symlink() and file_sha(lp)==launcher['sha256'],'Actual reviewed V11 launcher differs')
    artifacts[str(lp)]=launcher['sha256']
    norm=context['config']['dataset_binding']['normalization_file_binding'];manifest=context['config']['dataset_binding']['manifest_binding']
    command=[str(BASE/'vendor/openpi/.venv/bin/python'),'-u','-B','-m','sim.auscultation_pi05_v11.train','--manifest',manifest['path'],'--normalization',norm['path'],'--output',str(run)]
    if life.get('resume') is not None:
        resume=Path(life['resume']);need(resume.parent in (run/'model',run/'recovery'),'Same-run recovery path required');command+=['--resume',str(resume)]
    need(life['training_command']==command,'Actual training command differs from selected run')
    completion_command=[str(BASE/'vendor/openpi/.venv/bin/python'),'-u','-B',str(lp),'--preparation',prep['path'],'--preparation-sha256',prep['sha256'],'--evidence',str(folder),'--cpu-completion']
    need(life['completion_command']==completion_command,'Actual CPU verifier entry differs')
    need([s['stage'] for s in life['stages']]==['cpu_preflight','training','cpu_completion'],'All three owned stages required')
    for stage in life['stages']:
        name=stage['stage'];actual=pin(folder/(name+'.process.json'),artifacts)
        need(actual==stage,'Lifecycle stage differs from original process receipt')
        expected=command+['--verify-only'] if name=='cpu_preflight' else command if name=='training' else life['completion_command']
        check_stage(actual,name,expected)
        if name=='training':
            need(type(context['terminal'].get('pid')) is int and context['terminal']['pid']==actual['process_identity']['pid'],'Terminal must belong to the actual owned training process')
    cb=life['completion'];proof=pin(cb['path'],artifacts,cb['sha256'])
    need(Path(cb['path'])==folder/'context_verified.json' and proof.get('schema')=='wam_pi05_v11_full_context_completion_v1' and proof.get('completed') is True and proof.get('run')==str(run),'Actual CPU context reconstruction required')
    need(proof.get('source_sha256')==context['config']['source_sha256'] and proof.get('data_hash')==context['dataset'].data_hash and proof.get('terminal')==context['terminal'],'Completed CPU context differs')
    for p,h in proof['artifact_sha256'].items():
        need(Path(p).is_absolute() and file_sha(p)==h and (p not in artifacts or artifacts[p]==h),'CPU-consumed training evidence changed');artifacts[p]=h
    need(proof['artifact_sha256']==context['artifact_sha256'],'CPU context artifact closure differs')
    for p,h in artifacts.items():need(file_sha(p)==h,'Owner evidence changed during consumption')
    no_cancel(folder)
    return dict(schema='wam_pi05_v11_verified_training_owner_v1',artifact_sha256=artifacts,lifecycle_binding=b,run=str(run))

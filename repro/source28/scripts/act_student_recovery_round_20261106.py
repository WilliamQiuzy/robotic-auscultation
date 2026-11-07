"""One finite student-state recovery -> ACT -> development continuation.

Only the original sim.ausc_policy.run launches GPU work. No retries, teacher
substitution, held-out execution, automation or native guard changes occur here.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

from sim.ausc_policy import m2_given_contact_runtime as rt
from sim.ausc_policy import m2_given_contact_recovery_dataset as recovery
from sim.ausc_policy import m2_given_contact_masked_dataset as masked
from sim.ausc_policy import m2_rebuild_capture_dataset as rebuild
from sim.ausc_policy import m2_given_contact_expanded_pilot_train as trainer
from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as deployment
from sim.ausc_policy import m2_given_contact_pilot_native_audit as audit
from sim.ausc_policy import isaac_batch, process_guard, rebuild_gpu_binding

BASE=Path('/opt/wam-runtime/fast-wam')
CODE=Path(__file__).resolve().parents[1]
CAPTURE_RUN='act-student-recovery-20261106-v1'
CONTROL=BASE/'diagnostics/act-student-recovery-development-20261106-v1'
EXPORT=BASE/'datasets/act-student-recovery-masked-20261106-v1'
TRAIN_RUN='act-student-recovery-train-20261106-v1'
SMOKE_RUN='act-student-recovery-smoke-20261106-v1'
COHORT=BASE/'diagnostics/act-recovery-cohort-20261106-v1/COHORT_FREEZE.json'
NATIVE='/opt/wam-runtime/envs/isaacsim60/bin/python'
OFFICIAL='/opt/wam-runtime/vendor/lerobot-eb7a54c/.venv/bin/python'


def normal_owner(value):
    stage=value.get('stage',{})
    return (value.get('completed') is True and value.get('gpu_released') is True and
        stage.get('passed') is True and type(stage.get('exit_code')) is int and stage.get('exit_code')==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')))


def environment():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in ('AUSC_OWNED_RUN','PYTHONPATH','HF_TOKEN'):env.pop(key,None)
    return env


def snapshot():
    return dict(code_commit=rt.reference(CODE/'CODE_COMMIT'),source_manifest=rt.reference(CODE/'SOURCE_MANIFEST.json'),
        continuation=rt.reference(__file__),guard=rt.reference(process_guard.__file__),native=deployment.sources())


def unchanged(source):
    for k in ('code_commit','source_manifest','continuation','guard'):
        rt.need(rt.reference(source[k]['path'])==source[k],'Frozen continuation source changed: '+k)
    rt.need(deployment.sources()==source['native'],'Frozen numerical/native source inventory changed')


def wait_capture():
    path=BASE/'runs/ausc_policy'/CAPTURE_RUN/'result.json';deadline=time.monotonic()+21600
    while not path.is_file():
        rt.need(time.monotonic()<deadline,'Finite capture-terminal wait expired')
        time.sleep(5)
    ref=rt.reference(path);value=rt.read_bound(ref)
    rt.need(normal_owner(value) and value.get('run_id')==CAPTURE_RUN,'Capture owner did not finish normally')
    return ref,value


def build_recovery_manifests(owner_ref,owner):
    source_root=Path(owner['resource_guard']['path']).resolve().parents[2]
    source=dict(code_commit=owner['code_commit'],code_commit_file=rt.reference(source_root/'CODE_COMMIT'),
        source_manifest=rt.reference(source_root/'SOURCE_MANIFEST.json'))
    rt.need(rt.read_bound(rt.reference(BASE/'diagnostics'/CAPTURE_RUN/'FROZEN_PLAN.json'))['fixed_denominator']==4,
        'Exact predeclared four-case student-state recovery batch required')
    rows=[]
    for job in owner['config']['jobs']:
        cfg_ref=rt.reference(job['config']);rt.need(cfg_ref['sha256']==job['sha256'],'Original capture config changed')
        config=rt.read_bound(cfg_ref);meta=config['m2_given_contact_control']['metadata']
        root=BASE/'runs/ausc_policy'/CAPTURE_RUN/'output'/job['id'];capture=root/'compact_multimodal'
        report=rt.read_bound(rt.reference(root/'COMPLETE.json'));header=rt.shared._json(capture/'manifest.json')
        freeze=rebuild.source_freeze(report,header,source,CONTROL/'capture_source_freezes',job['id'])
        inventory={}
        for path in sorted(capture.rglob('*')):
            rt.need(not path.is_symlink(),'Symlink in recovery capture')
            if path.is_file():
                ref=rt.reference(path);inventory[path.relative_to(capture).as_posix()]={k:ref[k] for k in ('bytes','sha256')}
        pose=next(p for p in rebuild.POSE_IDS if p in job['id'])
        rows.append(dict(episode_id=meta['episode_id'],subject_id=meta['subject_id'],split=meta['split'],
            target_id=meta['target_id'],target=meta['target_id'],reset_id=meta['episode_id']+'-reset',
            run_id=CAPTURE_RUN,job_id=job['id'],config=cfg_ref,owner_terminal=owner_ref,
            child_result=rt.reference(root.parent/(job['id']+'.result.json')),run_result=rt.reference(root/'COMPLETE.json'),
            capture_root=str(capture),source_freeze=freeze,artifacts=inventory,pose_id=pose,cohort='train'))
    rt.need(len(rows)==4,'No recovery attempt may disappear from its denominator')
    return [rt.write_new(CONTROL/f'recovery-manifest-{i//4}.json',dict(schema=recovery.MANIFEST_SCHEMA,
        episodes=rows[i:i+4])) for i in range(0,len(rows),4)]


def recovery_gate(preparation):
    episodes=[e for e in preparation['episodes'] if e.get('origin')=='qualified_actual_recovery' and
        rt.read_bound(e['config']).get('m2_given_contact_recovery',{}).get('schema')=='m2_given_contact_student_prefix_recovery_v3']
    rt.need(len(episodes)>=2 and {e['target_id'] for e in episodes}=={'R3','L3'},
        'New actual student-state expert recovery at both sites is required; old random DART alone cannot pass')
    return dict(recovery_episodes=len(episodes),targets=sorted({e['target_id'] for e in episodes}),
        subjects=sorted({e['subject_id'] for e in episodes}),formal_population_qualified=False)


def verify_existing_prefix(preparation,old_ref=None):
    old_ref=rt.reference(BASE/'datasets/act-recovery-masked-20261106-v1/COMPLETE.json') if old_ref is None else old_ref
    old=rt.read_bound(old_ref);checks={}
    for split in ('train','validation'):
        count=old['counts'][split]
        rt.need(preparation['counts'][split]>=count,'Previously qualified data disappeared')
        with np.load(rt.bound_path(old['arrays'][split]),allow_pickle=False) as before,\
             np.load(rt.bound_path(preparation['arrays'][split]),allow_pickle=False) as after:
            for key in masked.ARRAY_KEYS:
                rt.need(np.array_equal(before[key],after[key][:count]),
                    'Previously qualified ordered data changed: '+split+'/'+key)
        checks[split]=dict(origins=count,arrays_compared=len(masked.ARRAY_KEYS),bit_identical=True)
    return dict(source=old_ref,checks=checks)


def run_owner(config_ref,run_id,python,source,*,control=CONTROL):
    unchanged(source)
    rt.need(not (BASE/'runs/ausc_policy'/run_id).exists(),'Owner already exists; no duplicate or automatic retry')
    rt.need(not process_guard.gpu_pids(),'Another GPU owner is active')
    time.sleep(2);rt.need(not process_guard.gpu_pids(),'GPU became occupied before original owner launch')
    command=[python,'-u','-B','-m','sim.ausc_policy.run','--config',config_ref['path'],'--run-id',run_id]
    rt.write_new(control/(run_id+'-COMMAND.json'),dict(command=command,config=config_ref,started_utc=process_guard.now()))
    with (control/(run_id+'.log')).open('xb') as log:
        result=subprocess.run(command,cwd=CODE,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=False)
    path=BASE/'runs/ausc_policy'/run_id/'result.json';ref=rt.reference(path);owner=rt.read_bound(ref)
    rt.need(result.returncode==0 and normal_owner(owner) and owner['config_sha256']==config_ref['sha256'],
        'Original owner failed, did not drain or changed its configuration: '+run_id)
    unchanged(source)
    return ref,owner


def prepare_smokes(model,complete,dataset):
    cases=[];jobs=[];inventory=deployment.sources()
    for subject,split in (('p001120','train'),('p000923','validation')):
        for target in ('R3','L3'):
            matches=[e for e in dataset['episodes'] if e.get('origin')=='qualified_actual_nominal' and
                e['subject_id']==subject and e['split']==split and e['target_id']==target and
                e.get('pose_id')==rebuild.POSE_IDS[0]]
            rt.need(len(matches)==1,'Exact qualified nominal development source case required')
            row=matches[0];original=rt.read_bound(row['config']);job=subject+'-'+target.lower()
            config=deployment.derive_config(original,source_config=row['config'],model=model,run_id=SMOKE_RUN,
                source_inventory=inventory,scope=complete['scope'])
            deployment.preflight(config,output=BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/job)
            ref=rt.write_new(CONTROL/'smokes'/(job+'.json'),config)
            jobs.append(dict(id=job,config=ref['path'],sha256=ref['sha256']))
            cases.append(dict(id=job,config=ref,source_episode_id=row['episode_id'],subject_id=subject,
                split=split,target_id=target,pose_id=rebuild.POSE_IDS[0]))
    batch=dict(experiment='isaac_batch',base=str(BASE),parallelism=2,time_limit_seconds=20000,jobs=jobs)
    isaac_batch.validate_jobs(batch)
    return rt.write_new(CONTROL/'smokes/batch.json',batch),cases


def main():
    rt.need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Linux CPU continuation required')
    rt.need(not CONTROL.exists(),'New finite continuation required; no resume')
    source=snapshot();CONTROL.mkdir()
    rt.write_new(CONTROL/'STARTED.json',dict(source=source,pid=os.getpid(),started_utc=process_guard.now(),
        capture_run=CAPTURE_RUN,training_run=TRAIN_RUN,smoke_run=SMOKE_RUN,
        evaluation_cohort=rt.reference(COHORT),heldout_execution=False,automatic_retries=False))
    phase='wait_capture'
    try:
        owner_ref,owner=wait_capture();unchanged(source)
        phase='qualify_and_export';new=build_recovery_manifests(owner_ref,owner)
        prior=BASE/'diagnostics/act-recovery-development-20261106-v1'
        manifests=[rt.reference(prior/f'recovery-manifest-{i}.json') for i in (0,1)]+new
        request=rt.write_new(CONTROL/'masked-request.json',dict(schema=masked.REQUEST_SCHEMA,
            source_dataset=rt.reference(BASE/'datasets/rebuilt-act-export-20261105-v2/dataset/COMPLETE.json'),
            recovery_manifests=manifests))
        prepared_ref=masked.export(request,EXPORT,cpu_workers=3);dataset=rt.read_bound(prepared_ref)
        rt.write_new(CONTROL/'EXISTING_DATA_PREFIX.json',verify_existing_prefix(dataset))
        gate=recovery_gate(dataset);rt.write_new(CONTROL/'DATA_QUALIFIED.json',dict(dataset=prepared_ref,**gate))
        phase='train'
        config=dict(trainer.SCHEDULE,base=str(BASE),dataset=prepared_ref,seed=20261106,
            replacement_pod_gpu_binding=rebuild_gpu_binding.issue(str(BASE)),evaluation_cohort=rt.reference(COHORT))
        trainer.validate_config(config);train_ref=rt.write_new(CONTROL/'train-config.json',config)
        training_owner,_=run_owner(train_ref,TRAIN_RUN,OFFICIAL,source)
        trained_ref=rt.reference(BASE/'runs/ausc_policy'/TRAIN_RUN/'output/COMPLETE.json')
        model,complete=deployment.model_binding(trained_ref)
        rt.write_new(CONTROL/'TRAINED.json',dict(training_owner=training_owner,complete=trained_ref,model=model))
        phase='native_development';batch,cases=prepare_smokes(model,complete,dataset)
        smoke_owner,_=run_owner(batch,SMOKE_RUN,NATIVE,source)
        phase='independent_physical_qualification';outcomes=[]
        for case in cases:
            path=BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/case['id']
            ref=audit.qualify(path,case['config']['path']);value=rt.read_bound(ref)
            outcomes.append(dict(case,qualification=ref,passed=value['passed'],error=value.get('error')))
        result=dict(training=trained_ref,model=model,smoke_owner=smoke_owner,outcomes=outcomes,
            physical_successes=sum(r['passed'] for r in outcomes),attempted_cases=4,
            heldout_executed=False,model_qualified=False,formal_acceptance=False,physics_gates_changed=False)
        rt.write_new(CONTROL/'DEVELOPMENT_RESULTS.json',result)
        if all(r['passed'] for r in outcomes):
            rt.write_new(CONTROL/'DEVELOPMENT_COMPLETE.json',dict(result,next='Fresh32-case expert/ACT evaluation is still required'))
        else:
            rt.write_new(CONTROL/'NEEDS_REPAIR.json',dict(result,next='Diagnose only development failures; held-out cohort is still unexposed'))
    except BaseException as error:
        (CONTROL/'traceback.txt').write_text(traceback.format_exc())
        rt.write_new(CONTROL/'FAILED.json',dict(phase=phase,error=repr(error),heldout_executed=False,
            native_source=source,automatic_retry=False))
        raise


if __name__=='__main__':main()

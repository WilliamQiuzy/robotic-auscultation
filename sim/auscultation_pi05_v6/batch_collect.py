"""Mac orchestration for the explicitly authorized remaining eight V6 points.

Default invocation only prints the proposed order. --execute is an explicit
operation switch, not permission: launch only after the root task confirms.
One held Isaac lock, no retries, and unchanged gate failures stop the sequence.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

POINTS=('R1','R2','R3','L2','L3','L4','R5','L5')
REPO=Path('/Users/williamqiu/Desktop/Harvard/WAM')
MIRROR=REPO/'results/pi05_v6_pilot'
BASE='/home/robot/wam-sim-media'
PROJECT=BASE+'/hall-modality-v1/project'
SOURCE=BASE+'/skill_v4/project'
RESULTS=BASE+'/pi05-v6-results'
QUALIFIED=BASE+'/pi05-v6-qualified'
AUDITS=BASE+'/pi05-v6-audits'
PROTOCOL=BASE+'/pi05-v5-audits/expert_samephysics_analysis_v1/pi05_frozen_parent_protocol.json'
RECIPE='aff4caf176365bf244b545b5dce658eed6cc75a8e9218e39d74f773dedb51dfa'
PYTHON='/home/robot/miniconda3/envs/env_isaacsim/bin/python'
ENV=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join((PROJECT,SOURCE,BASE+'/pi05-v4-deps','/home/robot/Desktop/stethoscope/sim/auscultation_training/python_deps')))


def remote_json(code):
    return json.loads(subprocess.check_output(['ssh','linbox','python3 -B -'],input=code,text=True))


def run_remote(args):
    command=shlex.join(['env',*[key+'='+value for key,value in ENV.items()],PYTHON,'-u','-B',*args])
    return subprocess.call(['ssh','linbox','cd '+shlex.quote(PROJECT)+' && '+command])


def emit(event,**fields):
    data=dict(event=event,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    MIRROR.mkdir(parents=True,exist_ok=True)
    with (MIRROR/'batch_events.jsonl').open('a') as stream:stream.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)


def preflight(point):
    return remote_json(f'''
from pathlib import Path
import json,hashlib,shutil
root=Path({RESULTS!r});source=Path({PROJECT!r})/'sim/auscultation_pi05_v6'
reference=root/'R4_expert_observed_v1'
snapshot=json.loads((reference/'input_snapshot/manifest.json').read_text())
for key,item in snapshot['files'].items():
 if key.startswith('source/'):
  path=Path(item['source_path'])
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Frozen acquisition source differs: '+key)
audit=json.loads((reference/'observed_supervisor_audit.json').read_text())
if audit.get('passed') is not True or hashlib.sha256((source/'audit_observed_supervisor.py').read_bytes()).hexdigest()!=audit['audit_source_sha256']:raise ValueError('Frozen independent auditor differs')
qualified=Path({QUALIFIED!r})/'R4_observed_v1'
receipt=json.loads((qualified/'derivation_receipt.json').read_text())
if hashlib.sha256((source/'qualify_expert.py').read_bytes()).hexdigest()!=receipt['helper_source_sha256']:raise ValueError('Frozen qualifier differs')
if (root/{point+'_expert_observed_v1'!r}).exists():raise ValueError('New output required; no automatic retry/resume')
free=shutil.disk_usage(root).free
if free<3*2**30:raise RuntimeError('Less than 3 GiB free before collection')
print(json.dumps(dict(free_bytes=free,source_entries=sum(k.startswith('source/') for k in snapshot['files']),sources_unchanged=True)))
''')


def completed_status(episode):
    return remote_json(f'''
from pathlib import Path
import json
p=Path({RESULTS!r})/{episode!r}
c=json.loads((p/'collection_result.json').read_text());o=json.loads((p/'outcome.json').read_text());t=json.loads((p/'teardown.json').read_text())
print(json.dumps(dict(passed=c.get('passed') is True and c.get('smoothness_target_met') is True and o.get('success') is True,
 completed=True,teardown_complete=t.get('close_returned') is True and t.get('errors')==[],
 sensor_frames=o.get('sensor_frames'),failure_reason=o.get('failure_reason'),centered=c.get('measured_dynamics',{{}}).get('centered'))))
''')


def local_module(module,*args):
    subprocess.run([sys.executable,'-u','-B','-m','sim.auscultation_pi05_v6.'+module,*args],cwd=REPO,check=True)


def mirror_qualified(episode,qid):
    destination=MIRROR/episode/'qualified';destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-az','linbox:'+QUALIFIED+'/'+qid+'/',str(destination)+'/'],check=True)


def archive(episode):
    local_module('archive_trial','--episode',episode,'--mirror',str(MIRROR/episode))


def run_episode(point):
    episode=point+'_expert_observed_v1';qid=point+'_observed_v1'
    status=preflight(point);emit('preflight_passed',episode=episode,**status)
    command=shlex.join(['bash',PROJECT+'/sim/auscultation_pi05_v6/collect_demo.sh',
        '--output',RESULTS+'/'+episode,'--point',point,'--sensors','--speed-scale','.7',
        '--position-iterations','32','--armature','.01'])
    emit('collection_started',episode=episode)
    code=subprocess.call(['ssh','linbox',command+' > '+shlex.quote(RESULTS+'/'+episode+'.log')+' 2>&1'])
    status=completed_status(episode);emit('collection_completed',episode=episode,exit_code=code,**status)
    if code or not status['passed'] or not status['teardown_complete']:
        if status['teardown_complete']:archive(episode)
        raise RuntimeError('Collection gate failed; no retry, qualification or next point: '+episode)
    if run_remote(['-m','sim.auscultation_pi05_v6.audit_observed_supervisor',RESULTS+'/'+episode]):
        archive(episode)
        raise RuntimeError('Independent supervisor replay failed; no next point: '+episode)
    emit('supervisor_replay_passed',episode=episode)
    args=['-m','sim.auscultation_pi05_v6.qualify_expert','--episode',RESULTS+'/'+episode,
          '--output',QUALIFIED+'/'+qid,'--working-copy',AUDITS+'/'+qid+'_working','--episode-id',qid,
          '--protocol',PROTOCOL,'--camera-source',SOURCE+'/sim/camera_framing_20260923/recommended_camera.json',
          '--expected-recipe-sha256',RECIPE]
    if run_remote(args):
        try:
            mirror_qualified(episode,qid)
        except Exception as error:
            emit('failed_qualification_mirror_unavailable',episode=episode,reason=str(error))
        archive(episode)
        raise RuntimeError('Qualification failed; original reports/raw retained; no next point: '+episode)
    mirror_qualified(episode,qid);emit('qualification_passed',episode=episode)
    archive(episode)
    local_module('mirror_qualification','--episode',episode)
    emit('archive_and_qualified_copy_verified',episode=episode)
    for module in ('cleanup_raw_duplicates','cleanup_scene_duplicates','cleanup_qualified_arrays','cleanup_evidence_duplicates'):
        local_module(module,'--episode',episode)
    emit('episode_lifecycle_complete',episode=episode)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True,points=POINTS,requires_root_launch_confirmation=True,
            failure_behavior='stop at first failed gate; no retries or threshold/source edits'),indent=2));return 0
    # The prerequisites are completed pilots, not inferred success from source provenance.
    for point in ('R4','L1'):
        qid=point+'_observed_v1';path=MIRROR/(point+'_expert_observed_v1')/'qualification_copy_receipt.json'
        receipt=json.loads(path.read_text())
        if receipt.get('verified_identical') is not True or receipt.get('episode_id')!=qid:
            raise ValueError('Both pilot qualifications must be verified before batch')
    emit('remaining_batch_started',points=POINTS,orchestration_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        for point in POINTS:run_episode(point)
    except Exception as error:
        emit('remaining_batch_stopped',reason=type(error).__name__+': '+str(error));raise
    emit('remaining_batch_completed',points=POINTS)
    return 0


if __name__=='__main__':raise SystemExit(main())

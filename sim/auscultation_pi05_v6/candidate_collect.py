"""One explicitly authorized R5 redundancy candidate; default is CPU dry-run.

This operational module is outside the frozen acquisition/training source set.
The original failed R5 episode is never overwritten or relabeled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from . import batch_collect as batch
from .operational_identity import CANDIDATE_EPISODE as EPISODE, qualification_id

QID=qualification_id(EPISODE)
CANDIDATE_ROOT='/home/robot/wam-sim-media/r5-redundancy-diagnostic-20260927a/candidate_30deg_v4_contained'
CANDIDATE_SOURCE=CANDIDATE_ROOT+'/source_project'
PLAN_SHA='09a57d7055fa64dcb125ff2305b8e16b2273b99d7c7607b6e0a3be118c57b960'
CANDIDATE_RECEIPT_SHA='2570075b150f0b14205fec3fe40bfed19ba1aa01d48be5d1f57e0540d58e275c'
PREPARED=batch.MIRROR/'r5_redundancy30_preflight.json'


def verify_manifest(original,candidate):
    old=original['files'];new=candidate['files']
    source={k:v['sha256'] for k,v in old.items() if k.startswith('source/')}
    current={k:v['sha256'] for k,v in new.items() if k.startswith('source/')}
    if len(source)!=143 or source!=current:raise ValueError('Original 143 acquisition source bytes differ')
    if candidate.get('points')!=['R5'] or set(old)!=set(new):raise ValueError('Candidate snapshot input set differs')
    if new['R5/plan']['sha256']!=PLAN_SHA:raise ValueError('Candidate plan differs from approved bytes')
    for key in old:
        if key not in {'R5/plan','R5/audit'} and old[key]['sha256']!=new[key]['sha256']:
            raise ValueError('Candidate changed non-plan input: '+key)
    return {k:v['sha256'] for k,v in new.items()}


def collection_command():
    return ['bash',batch.PROJECT+'/sim/auscultation_pi05_v6/collect_demo.sh',
        '--output',batch.RESULTS+'/'+EPISODE,'--point','R5','--sensors','--speed-scale','.7',
        '--position-iterations','32','--armature','.01','--source-project',CANDIDATE_SOURCE]


def cpu_remote(code):
    command=shlex.join(['env',*[key+'='+value for key,value in batch.ENV.items()],batch.PYTHON,'-B','-'])
    result=subprocess.run(['ssh','linbox','cd '+shlex.quote(batch.PROJECT)+' && '+command],input=code,text=True,capture_output=True)
    if result.returncode:raise RuntimeError('CPU preflight failed: '+result.stderr.strip())
    return json.loads(result.stdout)


def preflight():
    script=f'''
from pathlib import Path
import hashlib,importlib,json,shutil,tempfile,subprocess
from sim.auscultation_pi05_v6.collect_demo import collector_source_paths
from sim.auscultation_expert_layout_v1.controller import validate_plan,create_snapshot,verify_snapshot
from sim.auscultation_expert_layout_v1.run_sequence import source_paths
from sim.auscultation_expert_layout_v1.revalidate_plans import validate_geometry_binding
from sim.auscultation_expert_layout_v1 import scene
project=Path({batch.PROJECT!r});source=Path({CANDIDATE_SOURCE!r});runtime=source/'sim/auscultation_expert_layout_v1'
reference=Path({batch.RESULTS!r})/'R5_expert_observed_v1'
original=json.loads((reference/'input_snapshot/manifest.json').read_text())
expected_scene=Path({batch.SOURCE!r})/'sim/auscultation_expert_layout_v1/scene.py'
if Path(scene.__file__).resolve()!=expected_scene.resolve() or scene.ROOT.resolve()!=expected_scene.parent.resolve():raise ValueError('Actual scene import is not the original frozen source')
receipt=source.parent/'candidate_receipt.json'
if hashlib.sha256(receipt.read_bytes()).hexdigest()!={CANDIDATE_RECEIPT_SHA!r}:raise ValueError('Candidate receipt differs')
geometry=json.loads(receipt.read_text())
for name,digest in {{'runtime_curve_preflight.json':'e52d13461f3afc7ec89b75d2885ef817819f39347d8755db19c75c6d837ecdec','dense_route_revalidation.json':'3024ced3aa6c1c52164c7e3292c5ee1431149efc63d43a3ab0ccef62456ef35d'}}.items():
 if hashlib.sha256((source.parent/name).read_bytes()).hexdigest()!=digest:raise ValueError('Candidate geometry evidence differs')
plans=source/'sim/upper_pair_posture_20260924/results/candidate_b_plans'
bundle=validate_plan(plans/'planned_paths_R5.json',plans,runtime)
binding=validate_geometry_binding(runtime)
if bundle.plan.get('revalidation',{{}}).get('collision_regeneration_sha256')!=binding:raise ValueError('Candidate collision binding differs')
sources=source_paths(runtime);sources['collision_regeneration']=runtime/'results/collision_regeneration_verification.json'
for key,item in original['files'].items():
 if key.startswith(('source/vla/','source/repair/')):sources[key.removeprefix('source/')]=Path(item['source_path'])
sources.update(collector_source_paths())
for package in ('auscultation_skill_v1','curobo_expert_v1','expert_quality_20260924'):
 for directory in importlib.import_module('sim.'+package).__path__:
  for path in Path(directory).glob('*.py'):sources['runtime/'+package+'/'+path.name]=path
with tempfile.TemporaryDirectory(prefix='r5_candidate_snapshot_') as temporary:
 snapshot=create_snapshot([bundle],Path(temporary)/'input_snapshot',sources)
 candidate=verify_snapshot(snapshot,verify_sources=True)
freeze=project/'sim/auscultation_pi05_v6'
r4=Path({batch.RESULTS!r})/'R4_expert_observed_v1'
audit=json.loads((r4/'observed_supervisor_audit.json').read_text())
qualified=json.loads((Path({batch.QUALIFIED!r})/'R4_observed_v1/derivation_receipt.json').read_text())
if hashlib.sha256((freeze/'audit_observed_supervisor.py').read_bytes()).hexdigest()!=audit['audit_source_sha256']:raise ValueError('Independent auditor changed')
if hashlib.sha256((freeze/'qualify_expert.py').read_bytes()).hexdigest()!=qualified['helper_source_sha256']:raise ValueError('Qualifier changed')
for target in (Path({batch.RESULTS!r})/{EPISODE!r},Path({batch.QUALIFIED!r})/{QID!r},Path({batch.AUDITS!r})/{QID+'_working'!r}):
 if target.exists():raise ValueError('New candidate output required: '+str(target))
free=shutil.disk_usage(reference).free
if free<3*2**30:raise ValueError('Less than 3 GiB before candidate launch')
if subprocess.run(['flock','-n','/home/robot/wam-sim-media/skill_v4/isaac.lock','true']).returncode:raise ValueError('Shared Isaac lock is busy')
print(json.dumps(dict(original=original,candidate=candidate,scene_module_file=str(Path(scene.__file__).resolve()),scene_root=str(scene.ROOT.resolve()),
 candidate_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),free_bytes=free,source_project=str(source),
 independent_auditor_sha256=audit['audit_source_sha256'],qualifier_sha256=qualified['helper_source_sha256'],no_simulation_started=True)))
'''
    result=cpu_remote(script)
    if (batch.MIRROR/EPISODE).exists():raise ValueError('New Mac candidate mirror required')
    external=json.loads(subprocess.check_output(['ssh','qwen-agent-l40s','python3 -B -'],input=f'''from pathlib import Path
import json
paths=[Path('/home/william/wam-vla-v2/isaac-trials-v5')/{EPISODE!r},Path('/home/william/wam-vla-v2/qualified-v6')/{QID!r}]
print(json.dumps(dict(new_outputs=all(not p.exists() for p in paths))))
''',text=True))
    if external['new_outputs'] is not True:raise ValueError('New cloud candidate outputs required')
    manifest=verify_manifest(result.pop('original'),result['candidate'])
    result['manifest_fingerprint']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    result['manifest_hashes']=manifest
    result['expected_runtime_recipe_sha256']=batch.RECIPE
    result['collection_argv']=collection_command()
    result['acquisition_sources_unchanged']=143
    return result


def prepare():
    record=preflight();record['scope']='CPU dry-run only; physics acceptance still required'
    PREPARED.parent.mkdir(parents=True,exist_ok=True)
    PREPARED.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('candidate','manifest_hashes')},indent=2),flush=True)
    return record


def run_collection():
    log=batch.RESULTS+'/'+EPISODE+'.log'
    return subprocess.call(['ssh','linbox',shlex.join(collection_command())+' > '+shlex.quote(log)+' 2>&1'])


def verify_completed_inputs(prepared):
    result=batch.remote_json(f'''
from pathlib import Path
import json,hashlib
p=Path({batch.RESULTS!r})/{EPISODE!r}
manifest=json.loads((p/'input_snapshot/manifest.json').read_text())
for key,item in manifest['files'].items():
 if hashlib.sha256((p/'input_snapshot/objects'/item['sha256']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Saved snapshot object differs')
contract=json.loads((p/'dynamics_contract.json').read_text())
outcome=json.loads((p/'outcome.json').read_text())
print(json.dumps(dict(manifest=manifest,recipe=contract.get('recipe_sha256'),complete=contract.get('complete'),selected_scope=contract.get('selected_scope'),source_project=outcome.get('source_project'))))
''')
    actual={k:v['sha256'] for k,v in result['manifest']['files'].items()}
    if actual!=prepared['manifest_hashes']:raise ValueError('Executed snapshot differs from prepared source/plan bytes')
    if result['recipe']!=batch.RECIPE or result['complete'] is not True or result['selected_scope']!='simulator_recipe':raise ValueError('Actual candidate runtime recipe differs')
    if result['source_project']!=CANDIDATE_SOURCE:raise ValueError('Executed candidate source root differs')


def execute(prepared):
    fresh=preflight()
    if fresh['manifest_fingerprint']!=prepared['manifest_fingerprint']:raise ValueError('Prepared candidate inputs changed')
    batch.emit('candidate_collection_started',episode=EPISODE,argv=collection_command(),preflight_fingerprint=fresh['manifest_fingerprint'])
    code=run_collection();status=batch.completed_status(EPISODE)
    batch.emit('candidate_collection_completed',episode=EPISODE,exit_code=code,**status)
    if code or not status['passed'] or not status['teardown_complete']:
        if status['teardown_complete']:batch.archive(EPISODE)
        raise RuntimeError('Collection gate failed; preserve candidate and stop')
    try:verify_completed_inputs(fresh)
    except Exception:
        batch.archive(EPISODE);raise
    if batch.run_remote(['-m','sim.auscultation_pi05_v6.audit_observed_supervisor',batch.RESULTS+'/'+EPISODE]):
        batch.archive(EPISODE);raise RuntimeError('Independent supervisor replay failed')
    batch.emit('candidate_supervisor_replay_passed',episode=EPISODE)
    args=['-m','sim.auscultation_pi05_v6.qualify_expert','--episode',batch.RESULTS+'/'+EPISODE,
        '--output',batch.QUALIFIED+'/'+QID,'--working-copy',batch.AUDITS+'/'+QID+'_working','--episode-id',QID,
        '--protocol',batch.PROTOCOL,'--camera-source',batch.SOURCE+'/sim/camera_framing_20260923/recommended_camera.json','--expected-recipe-sha256',batch.RECIPE]
    if batch.run_remote(args):
        try:batch.mirror_qualified(EPISODE,QID)
        except Exception as error:batch.emit('candidate_failed_qualification_mirror_unavailable',episode=EPISODE,reason=str(error))
        batch.archive(EPISODE);raise RuntimeError('Candidate qualification failed')
    batch.mirror_qualified(EPISODE,QID);batch.emit('candidate_qualification_passed',episode=EPISODE)
    batch.archive(EPISODE)
    batch.local_module('mirror_qualification','--episode',EPISODE)
    batch.emit('candidate_archive_and_qualified_copy_verified',episode=EPISODE)
    for module in ('cleanup_raw_duplicates','cleanup_scene_duplicates','cleanup_qualified_arrays','cleanup_evidence_duplicates'):
        batch.local_module(module,'--episode',EPISODE)
    batch.emit('candidate_lifecycle_complete',episode=EPISODE)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:prepare();return 0
    prepared=json.loads(PREPARED.read_text())
    try:execute(prepared)
    except Exception as error:
        batch.emit('candidate_lifecycle_stopped',episode=EPISODE,reason=type(error).__name__+': '+str(error));raise
    return 0


if __name__=='__main__':raise SystemExit(main())

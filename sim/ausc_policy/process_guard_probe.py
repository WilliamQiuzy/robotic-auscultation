"""Linux CPU lifecycle and explicitly counterfactual v10 GPU-snapshot replay.

No GPU work, no other project's process changes, no overwrite of old evidence.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('guard',type=Path)
    parser.add_argument('v10',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Actual /proc probe requires Linux')
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('probe_guard',args.guard)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    source=guard.sha(args.guard)
    record=dict(guard_sha256=source,new_gpu_work_started=False,original_v10_modified=False)
    original_lookup=guard.process_identity
    resource=args.v10/'lifecycle/geometry_qualification.resources.jsonl'
    owner=json.loads((args.v10/'result.json').read_text())
    leader=owner['stage']['process_identity']['pid'];identity=dict(pid=leader,members={})
    replay=[]
    for line in resource.open():
        row=json.loads(line)
        members={p['pid']:dict(pid=p['pid'],start_ticks=p['start_ticks'],pgid=leader,sid=leader,state='S')
                 for p in row['owned_process_memory']}
        identity['members'].update({str(pid):p['start_ticks'] for pid,p in members.items()})
        def recorded_lookup(pid):
            if pid not in members:raise FileNotFoundError(str(pid))
            return members[pid]
        guard.process_identity=recorded_lookup
        enriched=dict(row,gpu_ownership=guard.classify_gpu_ownership(identity,row['gpu_pids'],row['elapsed_s']))
        replay.append(dict(time=row['sampled_utc'],old=guard.guard_violation(row,row['elapsed_s'],9000.),
                           revised=guard.guard_violation(enriched,row['elapsed_s'],9000.),
                           ownership=enriched['gpu_ownership']))
    guard.process_identity=original_lookup
    assert sum(r['old'] is not None for r in replay)==1
    assert all(r['revised'] is None for r in replay)
    record['v10_replay']=dict(samples=len(replay),old_rejections=1,new_rejections=0,
        final=replay[-1],resource_sha256=guard.sha(resource),
        scope='Recorded process snapshots; counterfactual classification only, not a new GPU run')
    # Real child identities, disappearance and expiry; no NVIDIA PID spoofing.
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1.5)'],start_new_session=True)
    try:
        ident=guard.process_identity(child.pid);guard.remember_group(ident)
        live=guard.classify_gpu_ownership(ident,{child.pid},100.)
        child.wait(timeout=10)
        exited=guard.classify_gpu_ownership(ident,{child.pid},101.)
        expired=guard.classify_gpu_ownership(ident,{child.pid},116.)
        assert live['verified_owned_pids']==[child.pid]
        assert exited['exiting_owned_pids']==[child.pid]
        assert expired['rejected_pids']==[child.pid]
        record['real_proc_probe']=dict(passed=True,pid=child.pid,start_ticks=ident['start_ticks'],
            live=live,after_exit=exited,expired=expired,
            scope='Real CPU process identities; candidate GPU PID and monotonic times injected for classification')
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=10)
    if guard.gpu_pids():
        record['cpu_run_stage']=dict(passed=False,skipped='GPU occupied; do not contend or terminate other project')
    else:
        evidence=args.output/'lifecycle';evidence.mkdir()
        guard.POLL_S=.5;guard.STAGE_LIMIT_S=20.
        stage=guard.run_stage('cpu_lifecycle',[sys.executable,'-c','import time; time.sleep(1.5)'],
            SimpleNamespace(output=args.output,evidence=evidence),dict(runtime_project=str(args.output)))
        record['cpu_run_stage']=stage
        assert stage['passed'] and stage['exit_code']==0 and stage['process_group_drained'] and not stage['signals']
    record['real_isaac_teardown_verified']=False
    (args.output/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(v10_replay='passed',real_proc='passed',cpu_stage=record['cpu_run_stage']['passed'],
                         real_isaac_teardown_verified=False,output=str(args.output))),flush=True)


if __name__=='__main__':main()

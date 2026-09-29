"""Finite CPU geometry check at three actual failed ACT rollout states.

No physics playback, state reset, model, stopping-success or recovery-success
claim. The original failure and its sensor/command history remain immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

from .isaac_case import prepare,write
from .moving_brake import plan_moving_brake


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def acquired_query(outcome,physics,frames,index,calibration):
    from sim.auscultation_vla_v2.hall import HallPacket
    action=outcome['l2_actions'][index];row_index=action['physics_row_start']-1
    row=physics[row_index]
    if (not action['completed'] or row['time_s']!=action['time_s']
            or action['target']!='R4' or action['phase']!='APPROACH'):
        raise ValueError('Probe must bind an actual completed APPROACH action origin')
    available=[f for f in frames if f['time_s']<=row['time_s']]
    if not available or row['time_s']-available[-1]['time_s']>.05+1e-8:
        raise ValueError('Missing actual fresh Hall frame')
    frame=available[-1]
    hall=calibration.decode(HallPacket(**frame['hall']),now_s=row['time_s'])
    from sim.auscultation_pi05_v6.supervision import contact_load_estimate
    force=sum(row[k] for k in ('diaphragm_N','lip_N','contact_rim_N','carriage_other_N','frame_N','forbidden_N'))
    return dict(q=row['q'],dq=row['dq'],time_s=row['time_s'],physics_row=row_index,
        target='R4',phase=action['phase'],hall_load_N=float(contact_load_estimate(hall)),
        physical_contact_N=float(force),compression_m=row['compression_m'],
        command_state=action['posture_hint_preview']['source_command_state'],
        recent_measurements=[dict(q=r['q'],q_command=r['q_command'],time_s=r['time_s'],physics_row=j)
            for j,r in enumerate(physics[row_index-12:row_index+1],start=row_index-12)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=read(a.config)
    if (sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1'
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or config['experiment']!='moving_brake_probe'
            or config['action_indices']!=[609,900,1250] or config['point']!='R4'):
        raise ValueError('Declared CPU-only three-origin geometry probe required')
    source=Path(config['source_run']);bindings={}
    for relative,expected in config['source_files'].items():
        path=source/relative
        if not path.resolve().is_relative_to(source.resolve()) or sha(path)!=expected:
            raise ValueError('Actual failed source changed: '+relative)
        bindings[path]=expected
    owner=read(source/'result.json');outcome=read(source/'output/case/outcome.json')
    if not (owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
            and owner['stage']['exit_code']==0 and not outcome['success']
            and not outcome['expert_demonstration'] and outcome['expert_l2_fallbacks']==0):
        raise ValueError('Completed original failed learned rollout required')
    a.output.mkdir(parents=True,exist_ok=False)
    bundle,_,_=prepare(config,a.output/'geometry_source')
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureChecker,PostureLimits
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.curobo_expert_v1.static_world import load_static_world,install_fcl_static_world
    from sim.auscultation_vla_v2.hall import default_calibration
    kin=OpenArmKinematics();base=np.asarray(bundle.plan['world_from_left_base']);folder=a.output/'geometry_source'
    checker=CertifiedCollisionChecker(kin,base,directory=folder/'geometry')
    install_fcl_static_world(checker,load_static_world(folder/'static_world'),geometry_dir=folder/'geometry')
    limits=PostureLimits.from_dict(bundle.row['posture_limits'],kin)
    checker=PostureChecker(checker,base,limits)
    physics=[json.loads(x) for x in (source/'output/case/expert_trace.jsonl').open()]
    frames=[json.loads(x) for x in (source/'output/case/sensors/frames.jsonl').open()]
    cases=[]
    for index in config['action_indices']:
        query=acquired_query(outcome,physics,frames,index,default_calibration())
        report=plan_moving_brake(query,lambda q:checker.check(q,clearance=.003,
            allow_patient_contact=False,compression=0.),lower=limits.joint_lower_rad,upper=limits.joint_upper_rad)
        report.update(source_action_index=index,source_files={str(k):v for k,v in bindings.items()})
        write(a.output/f'action-{index}.json',report)
        short={k:report.get(k) for k in ('kind','reason','native_stop_steps','minimum_sampled_clearance_m','command_derivative_peaks')}
        cases.append(dict(action_index=index,source_time_s=query['time_s'],**short))
        print(json.dumps(cases[-1]),flush=True)
    if any(sha(path)!=digest for path,digest in bindings.items()): raise ValueError('Source changed during probe')
    write(a.output/'COMPLETE.json',dict(completed=True,cases=cases,m0_passed=False,
        physics_executed=False,training_dataset_eligible=False,model_loaded=False,
        scope='Actual recorded moving ACT states, native braking plus sampled geometry only; no physical stopping or recovery success'))


if __name__=='__main__': main()

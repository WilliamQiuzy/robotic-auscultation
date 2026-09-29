"""Audit acquired history/input provenance and plot actual paired-case geometry.

This is saved-frame reconstruction, not a raw RGB-D tracker replay or a new
model inference. No simulator or hardware is constructed.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import act_baseline as a
from .history_runtime import history_from_frames,validate_history
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_pi05_v6.task_observation import encode_task_state
from sim.auscultation_vla_v2.hall import HallCalibration,HallPacket


def lines(path):
    import json
    return [json.loads(line) for line in Path(path).open()]


def audit(root):
    owner=a.read_json(root/'result.json');case=root/'output/case';result=a.read_json(case/'result.json');outcome=a.read_json(case/'outcome.json')
    a.need(owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
        and owner['stage']['exit_code']==0 and result['execution_completed'] and owner['config']['causal_history_input']
        and outcome['expert_demonstration'] is False and outcome['training_dataset_eligible'] is False,
        'Actual completed owned history-case acquisition required')
    snapshot=case/'input_snapshot/manifest.json';a.need(a.sha256(snapshot)==outcome['input_snapshot_manifest_sha256'],'Acquisition snapshot changed')
    files=a.read_json(snapshot)['files'];code=Path(__file__).resolve().parents[2]
    for name in ('auscultation_expert_layout_v1/kinematics.py','auscultation_expert_layout_v1/tool_frames.py',
                 'auscultation_pi05_v6/task_observation.py','auscultation_vla_v2/hall.py','ausc_policy/act_baseline.py'):
        a.need(a.sha256(code/'sim'/name)==files['source/code/'+name]['sha256'],'Acquisition decoding source changed')
    calibration=code/'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
    a.need(a.sha256(calibration)==owner['config']['core_assets']['sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json'],'Hall calibration changed')
    kin=OpenArmKinematics();cal=HallCalibration.from_path(calibration)
    a.need(a.sha256(kin.urdf_path)==files['source/urdf']['sha256'],'FK URDF differs')
    frames=lines(case/'sensors/frames.jsonl');bodies={r['frame_index']:r for r in lines(case/'sensors/body_poses.jsonl')}
    physics=lines(case/'expert_trace.jsonl');task=outcome['task_frame_registration']['R4']['task_pose_in_body_reference']
    a.need([f['index'] for f in frames]==list(range(len(frames))),'Frame identities differ')
    geometry=[];raw_by={};volts=[]
    for frame in frames:
        body=bodies[frame['index']];row=physics[frame['physics_row']]
        a.need(frame['time_s']==body['time_s']==row['time_s'] and frame['eligible'] is False
            and np.array_equal(frame['q'],row['q']),'Actual measured q/time differs from physics ledger')
        raw=encode_task_state(frame['state'],measured_q=frame['q'],hall_packet=HallPacket(**frame['hall']),
            observation_time_s=frame['time_s'],now_s=frame['time_s'],body_time_s=body['time_s'],
            body_reference_to_current_base=body['body_reference_to_current_base'],task_pose_in_body_reference=task,
            kinematics=kin,hall_calibration=cal)
        raw_by[frame['index']]=raw
        hall=cal.decode(HallPacket(**frame['hall']),now_s=frame['time_s']);volts.append(hall.hall_voltage_v)
        target=np.asarray(body['body_reference_to_current_base'])@np.asarray(task);tcp=kin.tcp(frame['q'],hall.displacement_estimate_m)
        error=tcp[:3,3]-target[:3,3];normal=-target[:3,2];gap=float(error@normal)
        geometry.append(dict(time_s=frame['time_s'],gap_mm=gap*1000,tangent_mm=float(np.linalg.norm(error-gap*normal))*1000,
            axis_deg=float(np.degrees(np.arccos(np.clip(tcp[:3,2]@target[:3,2],-1,1)))),
            full_rotation_deg=float(np.degrees(Rotation.from_matrix(tcp[:3,:3]@target[:3,:3].T).magnitude()))))
    accepted=[q for q in outcome['l2_queries'] if q['accepted']];errors=[]
    for query in accepted:
        frame=frames[query['observation_frame']]
        expected=history_from_frames(frames[:frame['index']+1],frame)
        a.need(expected==query['causal_history'],'Served history differs from saved acquired frames')
        rebuilt=a.small_state(raw_by[frame['index']],query['target'],query['phase'])
        error=float(abs(rebuilt-np.asarray(query['state19'],np.float32)).max());errors.append(error)
        validate_history(expected,dict(frame_id=frame['index'],sim_stamp_s=query['time_s'],target_id=query['target'],state=query['state19']))
        a.need(error==0 and query['time_s']==frame['time_s'] and query['revision']==frame['target_estimate']['revision'],
            'Served current input differs from same-frame measured reconstruction')
    actions=outcome['l2_actions'];completed=[r for r in actions if r.get('completed')]
    for row in completed:
        q=outcome['l2_queries'][row['query_index']]
        a.need(abs(row['label_time_s']-q['time_s']-(row['prefix_index']+1)*.05)<1e-8,'Action label clock differs')
    summary=dict(run_id=root.name,variant=owner['config']['variant'],source_code=owner['code_commit'],case=result,
        accepted_queries=len(accepted),attempted_queries=len(outcome['l2_queries']),proposed_actions=len(actions),
        completed_actions=len(completed),same_frame_input_max_error=max(errors,default=None),
        historical_samples_bound=len(accepted)*5,physics_samples=len(physics),sensor_frames=len(frames),
        final_observed_geometry=geometry[-1],final_physics_time_s=physics[-1]['time_s'],
        simulated_hall_voltage_range_v=[min(volts),max(volts)],
        contact_component_peak_N={k:max(row[k] for row in physics) for k in ('diaphragm_N','lip_N','contact_rim_N','forbidden_N')},
        best_physical_audit_hold_s=max(row['audit_hold_s'] for row in physics),
        measured_dynamics=a.read_json(case/'measured_dynamics.json')['centered'],
        raw_pixels_replayed=False,native_outputs_recomputed=False,training_dataset_eligible=False,m0_passed=False,
        source_hashes={name:a.sha256(root/name) for name in ('result.json','output/case/result.json','output/case/outcome.json',
            'output/case/expert_trace.jsonl','output/case/sensors/frames.jsonl','output/case/sensors/body_poses.jsonl')})
    return summary,geometry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();summary,geometry=audit(args.run);args.output.mkdir(parents=True,exist_ok=False)
    a.save_json(args.output/'summary.json',summary);a.save_json(args.output/'geometry.json',geometry)
    print({k:summary[k] for k in ('run_id','accepted_queries','historical_samples_bound','same_frame_input_max_error','final_observed_geometry')})


if __name__=='__main__':main()

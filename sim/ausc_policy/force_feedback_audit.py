"""Independent streaming audit of native expert trace; raw stays on Linux."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from sim.ausc_policy.contact_geometry_audit import group
from sim.ausc_policy.retainer_contact import ALL_FORCE_GROUPS
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    started=time.monotonic()
    owner=json.loads((args.run/'result.json').read_text())
    case=args.run/'output/review'
    native=json.loads((case/'fresh_expert_result.json').read_text())
    trace=case/'fresh_expert/fresh_expert_trace.jsonl'
    config=owner['config']
    expected=[json.loads(Path(x['file']).read_text())['name'] for x in config['fresh_expert']['plans']]
    bounds=config['posture_limits']
    lower=np.asarray(bounds['joint_lower_rad']);upper=np.asarray(bounds['joint_upper_rad'])
    kin=OpenArmKinematics()
    native_points={p['target']:p for p in native['points']}
    points={};series=[];digest=hashlib.sha256();count=0;max_error=clock_error=0.
    recorded_violations=nonfront_bad=0
    for line in trace.open('rb'):
        digest.update(line);row=json.loads(line);t=row['target']
        if t not in points:
            points[t]=dict(target=t,first_row=count,last_row=count,count=0,hold_rows=0,
                run=0,longest=0,force_max=0.,forbidden_max=0.,hall_max=0.,compression_max_m=0.,
                j2_software_command_margin_deg=math.inf,j2_software_observed_margin_deg=math.inf,
                j2_mechanical_command_margin_deg=math.inf,j2_mechanical_observed_margin_deg=math.inf,
                hold=defaultdict(list),balance=defaultdict(list),last_release=None)
        p=points[t];p['last_row']=count;p['count']+=1
        values=np.asarray([row[k+'_N'] for k in ALL_FORCE_GROUPS],float)
        assert np.isfinite(values).all() and (values>=0).all()
        raw=np.zeros(7)
        for contact in row['contact_points']:
            label=group(contact['colliders'],revised=True)
            if label is not None:raw[ALL_FORCE_GROUPS.index(label)]+=np.linalg.norm(contact['impulse_Ns'])*240
        max_error=max(max_error,float(np.max(np.abs(values-raw))))
        force=float(raw[:2].sum());total=float(raw.sum())
        p['run']=p['run']+1 if 3.5<=force<=4.5 else 0;p['longest']=max(p['longest'],p['run'])
        p['force_max']=max(p['force_max'],force)
        p['forbidden_max']=max(p['forbidden_max'],float(raw[2:].max()))
        p['total_max_N']=max(p.get('total_max_N',0.),total)
        p['hall_max']=max(p['hall_max'],row['hall_N'])
        p['compression_max_m']=max(p['compression_max_m'],row['compression_m'])
        for field,label in [('q_command','command'),('q','observed')]:
            q=np.asarray(row[field]);j2=q[1]
            for kind,lo,hi in [('software',lower[1],upper[1]),('mechanical',kin.lower_limits[1],kin.upper_limits[1])]:
                key='j2_'+kind+'_'+label+'_margin_deg'
                p[key]=min(p[key],math.degrees(min(j2-lo,hi-j2)))
        if row.get('physical_violations') or row.get('posture_violations'):recorded_violations+=1
        if 'force_balance' in row:
            b=row['force_balance']
            assert b['actor_input'] is False and abs(b['normal_force_N']-force)<1e-8
            combined=np.asarray(b['normal_force_vector_N'])+np.asarray(b['friction_force_vector_N'])
            p['combined_vector_peak_N']=max(p.get('combined_vector_peak_N',0.),float(np.linalg.norm(combined)))
            p['friction_vector_peak_N']=max(p.get('friction_vector_peak_N',0.),float(np.linalg.norm(b['friction_force_vector_N'])))
        front=row.get('retainer_front_audit',{})
        if values[1]>0 and (front.get('passed') is not True or front.get('force_matches_ledger') is not True
                or front.get('off_front_force_N')!=0 or not np.isclose(front.get('force_N'),values[1],atol=1e-8,rtol=1e-8)):
            nonfront_bad+=1
        if row.get('subphase') in ('withdraw_settle','normal_release_confirm','home_settle'):
            p['last_release']=dict(row=count,total_N=total,hall_N=row['hall_N'],compression_m=row['compression_m'])
        if row['phase']=='HOLD':
            p['hold_rows']+=1
            for k,v in [('Hall_N',row['hall_N']),('normal_N',force),('bias_N',row['hall_N']-force)]:p['hold'][k].append(v)
            if 'force_balance' in row:
                b=row['force_balance'];assert b['actor_input'] is False
                assert abs(b['normal_force_N']-force)<1e-8
                for k in ['normal_axial_N','friction_axial_N','combined_axial_N','axial_model_minus_report_N',
                          'spring_preload_N','damping_N','gravity_N']:
                    p['balance'][k].append(b[k])
        if t=='R2' and row['phase'] in ('CONTACT','HOLD','RETRACT') and count%12==0:
            b=row.get('force_balance',{})
            series.append(dict(time_s=row['time_s'],phase=row['phase'],Hall_N=row['hall_N'],normal_N=force,
                compression_mm=row['compression_m']*1000,normal_axial_N=b.get('normal_axial_N'),
                friction_axial_N=b.get('friction_axial_N'),combined_axial_N=b.get('combined_axial_N')))
        clock_error=max(clock_error,abs(row['time_s']-(count+1)/240))
        count+=1;last=row
    assert digest.hexdigest()==native['trace_sha256']
    assert count==native['physics_steps']
    assert max_error<1e-8 and clock_error<1e-7
    for t,p in points.items():
        p['continuous_contact_s']=p.pop('longest')/240;p.pop('run')
        for key in ['hold','balance']:
            p[key]={k:dict(n=len(v),mean=float(np.mean(v)),min=float(np.min(v)),max=float(np.max(v))) for k,v in p[key].items()}
        a=native_points[t]['force_audit'];p['native_force_pass']=a['passed']
        if config['fresh_expert'].get('force_feedback'):
            assert p['balance']['combined_axial_N']['n']==p['hold_rows']>0
        assert abs(a['continuous_contact_hold_s']-p['continuous_contact_s'])<1e-9
        p['physics_pass']=bool(p['continuous_contact_s']>=3.-1e-9 and p['forbidden_max']==0
            and p['total_max_N']<=6 and p['hall_max']<5.5 and p['compression_max_m']<=.006
            and p.get('combined_vector_peak_N',0.)<=6.)
        release=p['last_release'];p['release_pass']=bool(release and release['total_N']<.2 and release['hall_N']<.2 and release['compression_m']<=.00015)
    home=bool(last.get('subphase')=='home_settle' and max(abs(x) for x in last['q'])<.03
              and max(abs(x) for x in last['q_command'])<1e-9 and sum(last[k+'_N'] for k in ALL_FORCE_GROUPS)<.2)
    diagnostics=json.loads((case/'fresh_expert_controller_diagnostics.json').read_text())
    returns=[s for s in diagnostics['free_motion_timing'] if s.get('subphase')=='return_home']
    returned=bool(returns and all(s.get('motion_complete') for s in returns))
    for t,p in points.items():
        assert p['first_row']==native_points[t]['start_physics_row']
        assert p['last_row']+1==native_points[t]['end_physics_row']
    passed=bool(owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
        and native['controller_motion_complete'] and native['physical_validation'] and native['success']
        and list(points)==expected and native['resets_between_points']==0 and home and returned
        and not recorded_violations and not nonfront_bad
        and all(p['physics_pass'] and p['release_pass'] and p['native_force_pass'] for p in points.values()))
    out=dict(schema='force_feedback_independent_audit_v1',run_id=args.run.name,passed=passed,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checked_utc=datetime.now(timezone.utc).isoformat(),code_commit=owner['code_commit'],config_sha256=owner['config_sha256'],
        expected_points=expected,point_order=list(points),points=list(points.values()),physics_rows=count,
        simulation_s=count/240,wall_s=(datetime.fromisoformat(owner['ended_utc'])-datetime.fromisoformat(owner['started_utc'])).total_seconds(),
        trace_sha256=digest.hexdigest(),raw_impulse_max_error_N=max_error,clock_error_s=clock_error,
        nonfront_bad_rows=nonfront_bad,recorded_violations=recorded_violations,final_home_verified=home,
        return_home_segments=returns,
        final_q_max_rad=max(abs(x) for x in last['q']),resets_between_points=native['resets_between_points'],
        video=native.get('video'),runtime_cost=native.get('runtime_cost'),audit_wall_s=time.monotonic()-started,
        R2_20hz_series=series,learned_policy=False,training_eligible=False,
        scope='Streaming original impulses and recorded front-location audit; no new physics or patient generalization')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('points','R2_20hz_series','video')},indent=2))


if __name__=='__main__':main()

"""Read-only contact footprint/angle audit; no acoustic or training qualification."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sim.ausc_policy.retainer_contact import RETAINER_PATHS, is_retainer_patient_pair


def unit(value):
    vector=np.asarray(value,float)
    if vector.shape!=(3,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1.,atol=1e-6,rtol=0):
        raise ValueError('Finite unit vector required')
    return vector/np.linalg.norm(vector)


def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(unit(a)@unit(b),-1.,1.))))


def measure(row, target_normal, command_normal):
    outward=-unit(row['axis_world'])
    theta=angle(outward,target_normal)
    forces=np.zeros(32); candidates=set(); vector=np.zeros(3)
    for point in row['contact_points']:
        a,b=point['colliders']
        if not is_retainer_patient_pair(a,b):continue
        moving=a if a in RETAINER_PATHS else b
        index=int(moving.rsplit('_',1)[1])
        impulse=np.asarray(point['impulse_Ns'],float)
        if impulse.shape!=(3,) or not np.isfinite(impulse).all():raise ValueError('Invalid impulse')
        force=impulse*240
        forces[index]+=np.linalg.norm(force)
        vector+=force if a==moving else -force
        candidates.add(index)
    if not np.isfinite(row['retainer_N']) or not np.isclose(forces.sum(),row['retainer_N'],atol=1e-8,rtol=1e-8):
        raise ValueError('Raw sector force differs from retainer ledger')
    active=np.flatnonzero(forces>.05)
    arc=(32-int(np.max(np.diff(np.r_[active,active[0]+32]))))*11.25 if len(active) else 0.
    return dict(target_normal_angle_deg=theta,
        command_normal_angle_deg=angle(outward,command_normal),
        normal_resultant_angle_deg=angle(outward,vector/np.linalg.norm(vector)) if np.linalg.norm(vector)>1e-10 else None,
        tangent_plane_opposite_edge_height_mm=46*math.sin(math.radians(theta)),
        reported_candidate_sectors=len(candidates),sectors_gt_zero=int(np.sum(forces>0)),
        sectors_gt_0_01N=int(np.sum(forces>.01)),sectors_gt_0_05N=int(len(active)),
        loaded_center_arc_deg=float(arc),
        largest_sector_force_fraction=float(forces.max()/forces.sum()) if forces.sum()>0 else None,
        sector_normal_force_N=forces.tolist())


def stats(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Finite nonempty measurements required')
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),
                p95=float(np.percentile(a,95)),max=float(a.max()))


def audit(run, output):
    run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError('Do not overwrite a completed diagnostic')
    config=json.loads((run/'run.json').read_text())['config']
    native=json.loads((run/'output/review/fresh_expert_result.json').read_text())
    plans={};bindings=[]
    for item in config['fresh_expert']['plans']:
        data=Path(item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Plan SHA mismatch')
        plan=json.loads(data);plans[plan['name']]=plan;bindings.append(item)
    buckets=defaultdict(list);timeline=[];digest=hashlib.sha256();count=0
    with (run/'output/review/fresh_expert/fresh_expert_trace.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line);row=json.loads(line);count+=1
            if row['phase']!='HOLD':continue
            p=plans[row['target']]
            result=measure(row,p['normal_world'],p['command_normal_world'])
            buckets[row['target']].append(result)
            if count%12==0:timeline.append(dict(time_s=row['time_s'],target=row['target'],**result))
    if digest.hexdigest()!=native['trace_sha256'] or count!=native['physics_steps']:
        raise ValueError('Trace provenance mismatch')
    if list(buckets)!=list(plans):raise ValueError('Incomplete point order')
    summary=[]
    for target,rows in buckets.items():
        if len(rows)!=720:raise ValueError('Expected complete three-second 240 Hz HOLD')
        forces=np.asarray([r['sector_normal_force_N'] for r in rows])
        result=dict(target=target,hold_rows=len(rows),hold_s=len(rows)/240,
            metrics={key:stats([r[key] for r in rows]) for key in rows[0] if key!='sector_normal_force_N'},
            sector_mean_force_N=forces.mean(axis=0).tolist(),
            sector_loaded_fraction_gt_0_05N=(forces>.05).mean(axis=0).tolist(),
            planned_normal_deviation_deg=angle(plans[target]['normal_world'],plans[target]['command_normal_world']))
        summary.append(result)
    report=dict(schema='retainer_flatness_diagnostic_v1',run_id=run.name,
        checked_utc=datetime.now(timezone.utc).isoformat(),trace_sha256=digest.hexdigest(),physics_rows=count,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plans=bindings,points=summary,
        nominal_ring_diameter_mm=46.,sector_count=32,
        axis_source='native measured wrist get_world_pose rotation column 2; prismatic carriage axis',
        target_normal_source='frozen plan normal_world, distinct from fitted command_normal_world',
        thresholds_N=[0.,.01,.05],threshold_scope='force reporting sensitivity, not qualified acoustic contact thresholds',
        physical_force_qualification_unchanged=True,flatness_qualification=False,training_eligible=False,
        limitation='Rigid contact solver active sectors are not skin contact area or acoustic seal. 46*sin(theta) is a tangent-plane geometric reference, not measured opposite-rim gap. No acoustic threshold is inferred from this run.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'hold_20hz.json').write_text(json.dumps(timeline,allow_nan=False)+'\n')
    print(json.dumps(dict(run_id=run.name,points=len(summary),physics_rows=count,
                         flatness_qualification=False,output=str(output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('output');args=parser.parse_args()
    audit(args.run,args.output)

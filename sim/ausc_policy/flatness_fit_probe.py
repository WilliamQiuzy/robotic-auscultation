"""Rank existing current-geometry finite-head proposals by sampled ring gaps.

CPU diagnosis only. Reusing proposals here does not qualify an arm path,
contact, controller, or training demonstration. Raw anchors never move.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
if __package__ in (None,''):
    from contact_surface_gap import front_samples,ray_offsets,sha,verify_geometry_binding,verify_point_coverage
else:
    from .contact_surface_gap import front_samples,ray_offsets,sha,verify_geometry_binding,verify_point_coverage
from sim.ausc_policy.finite_head import component_frames,_observation


def sampled_gaps(local,mesh,center,rotation,compression):
    _,moving=component_frames(center,rotation,compression)
    world=local@moving[:3,:3].T+moving[:3,3]
    direction=-moving[:3,2]
    ends=np.r_[world-.05*direction,world+.05*direction]
    chosen=np.all(mesh.max(axis=1)>=ends.min(axis=0)-1e-9,axis=1)&np.all(mesh.min(axis=1)<=ends.max(axis=0)+1e-9,axis=1)
    return ray_offsets(world,direction,mesh[chosen])


def probe(run,output):
    started=time.monotonic();run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError('Retain completed diagnostics')
    case=run/'output/review';directory=case/'geometry'
    native=json.loads((case/'fresh_expert_result.json').read_text())
    verify_geometry_binding(directory,native)
    entries=json.loads((directory/'collision_geometry.json').read_text())['colliders']
    patient=[e for e in entries if e['path'].startswith('/World/Patient/')]
    ring=sorted([e for e in entries if '/DeviceContactRing/sector_' in e['path']],key=lambda e:e['path'])
    if len(ring)!=32 or not patient or any(e['body']!='world' or e['approximation']!='none' for e in patient):
        raise ValueError('Full ring and static patient triangles required')
    mesh=[];local=[]
    with np.load(directory/'collision_geometry.npz',allow_pickle=False) as arrays:
        for e in patient:
            v=arrays[e['key']+'_v'];f=arrays[e['key']+'_f'];t=np.asarray(e['world_from_body'])
            mesh.append((v@t[:3,:3].T+t[:3,3])[f])
        for e in ring:local.extend(front_samples(arrays[e['key']+'_v']))
    mesh=np.concatenate(mesh);local=np.asarray(local);points=[];bindings=[]
    for binding in native['plan_bindings']:
        path=Path(binding['file'])
        if sha(path)!=binding['sha256']:raise ValueError('Plan SHA changed')
        plan=json.loads(path.read_text());name=plan['name']
        source=path.with_name(name+'_fits.json');fits=json.loads(source.read_text())
        if fits['geometry_artifacts']!=native['geometry_artifacts']:raise ValueError('Proposal geometry differs')
        bindings.append(dict(file=str(source),sha256=sha(source)))
        rows=[]
        for index,c in enumerate(fits['candidates']):
            if c['passed'] is not True or not _observation(c['fit'])['passed']:raise ValueError('Rejected proposal in accepted list')
            gaps=sampled_gaps(local,mesh,c['center_world_m'],c['rotation_world'],fits['loaded_compression_m'])
            finite=gaps[np.isfinite(gaps)]
            rows.append(dict(index=index,missing_rays=int(np.isnan(gaps).sum()),
                min_gap_mm=float(finite.min()*1000) if finite.size else None,
                max_gap_mm=float(finite.max()*1000) if finite.size else None,
                median_gap_mm=float(np.median(finite)*1000) if finite.size else None,
                tilt_deg=c['normal_deviation_from_surface_deg'],roll_rad=c['roll_rad']))
        complete=[r for r in rows if r['missing_rays']==0 and r['min_gap_mm']>=-1e-4]
        best=min(complete,key=lambda r:r['max_gap_mm']) if complete else None
        points.append(dict(target=name,proposals=len(rows),complete_nonpenetrating=len(complete),best=best,
            all_proposal_metrics=rows))
        print(json.dumps(dict(target=name,best=best)),flush=True)
    verify_point_coverage(dict.fromkeys(p['target'] for p in points),native)
    verify_geometry_binding(directory,native)
    report=dict(schema='flatness_existing_fit_probe_v1',run_id=run.name,points=points,
        source_sha256=sha(__file__),gap_source_sha256=sha(Path(__file__).with_name('contact_surface_gap.py')),
        native_sha256=sha(case/'fresh_expert_result.json'),geometry_artifacts=native['geometry_artifacts'],
        proposal_bindings=bindings,wall_s=time.monotonic()-started,physics_executed=False,training_eligible=False,
        scope='Existing finite-head proposals, 96 samples each; best within saved sample set, not a global optimum or all-point clearance certificate. No path or new physical qualification.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('output');a=p.parse_args();probe(a.run,a.output)

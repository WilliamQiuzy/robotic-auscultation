"""Bounded CPU orientation refinement with unchanged finite-head safety gates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
if __package__ in (None,''):
    from contact_surface_gap import front_samples,sha,verify_geometry_binding
    from flatness_fit_probe import sampled_gaps
else:
    from .contact_surface_gap import front_samples,sha,verify_geometry_binding
    from .flatness_fit_probe import sampled_gaps
from sim.ausc_policy.finite_head import FiniteHeadFit
from sim.ausc_policy.fit_refinement import _basis,_orientation


class BudgetExpired(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('probe');p.add_argument('output')
    p.add_argument('--targets',nargs='+',default=[s+str(i) for i in range(1,6) for s in 'LR']);a=p.parse_args()
    run=Path(a.run);probe=json.loads(Path(a.probe).read_text());output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous refinement')
    case=run/'output/review';directory=case/'geometry';native=json.loads((case/'fresh_expert_result.json').read_text())
    verify_geometry_binding(directory,native)
    metadata={**native,**native['geometry_artifacts'],'freshly_generated':True}
    fit=FiniteHeadFit(directory,metadata)
    entries=json.loads((directory/'collision_geometry.json').read_text())['colliders'];mesh=[];local=[]
    with np.load(directory/'collision_geometry.npz',allow_pickle=False) as arrays:
        for e in entries:
            if e['path'].startswith('/World/Patient/'):
                v=arrays[e['key']+'_v'];f=arrays[e['key']+'_f'];t=np.asarray(e['world_from_body'])
                mesh.append((v@t[:3,:3].T+t[:3,3])[f])
        for e in sorted([e for e in entries if '/DeviceContactRing/sector_' in e['path']],key=lambda e:e['path']):
            local.extend(front_samples(arrays[e['key']+'_v']))
    mesh=np.concatenate(mesh);local=np.asarray(local)
    plans={b['name']:b for b in native['plan_bindings']};rows=[]
    output.mkdir(parents=True,exist_ok=False)
    for target in a.targets:
        binding=plans[target];source=Path(binding['file'])
        if sha(source)!=binding['sha256']:raise ValueError('Native plan changed')
        plan=json.loads(source.read_text());fits_path=source.with_name(target+'_fits.json')
        expected=next(b['sha256'] for b in probe['proposal_bindings'] if b['file']==str(fits_path))
        if sha(fits_path)!=expected:raise ValueError('Proposals changed')
        proposals=json.loads(fits_path.read_text());initial=next(r for r in probe['points'] if r['target']==target)['best']
        c=proposals['candidates'][initial['index']];anchor=np.asarray(c['raw_anchor_world_m']);normal=np.asarray(plan['normal_world'])
        x,y=_basis(normal);n=np.asarray(c['normal_world']);azimuth=float(np.arctan2(n@y,n@x)%(2*np.pi))
        seed=np.array([c['normal_deviation_from_surface_deg'],azimuth,c['roll_rad']]);start=time.monotonic();evaluations=[];best=None
        def objective(angles):
            nonlocal best
            if len(evaluations)>=180 or time.monotonic()-start>=120:raise BudgetExpired()
            rotation=_orientation(normal,x,y,*angles)
            if rotation is None:return 1.
            found=fit.first_contact(anchor,normal,rotation,.0025)
            row=dict(angles=np.asarray(angles).tolist(),passed=found['passed'],reason=found.get('reason'))
            cost=1.
            if found['passed']:
                gaps=sampled_gaps(local,mesh,found['center_world_m'],rotation,.0025)
                if np.isfinite(gaps).all() and gaps.min()>=-1e-7:
                    cost=float(gaps.max());row.update(min_gap_mm=float(gaps.min()*1000),max_gap_mm=cost*1000)
                    if best is None or cost<best['max_gap_mm']/1000:
                        best={**row,'first_contact':found,'rotation_world':rotation.tolist(),
                              'gaps_mm':(gaps.reshape(32,3)*1000).tolist()}
            evaluations.append(row)
            return cost
        expired=False;seen=set()
        try:
            objective(seed)
            # Wide interval optimizers jump across rejected safety islands.
            # Evaluate a bounded local stencil at two scales instead.
            for scale in (1.,.5):
                origin=np.asarray(best['angles'] if best is not None else seed)
                for dt in (-2.,0.,2.):
                    for da in (-15.,-7.5,0.,7.5,15.):
                        for dr in (-10.,-5.,0.,5.,10.):
                            angles=origin+scale*np.array([dt,np.deg2rad(da),np.deg2rad(dr)])
                            angles[0]=np.clip(angles[0],0,40);angles[1:]%=2*np.pi
                            key=tuple(np.round(angles,10))
                            if key in seen:continue
                            seen.add(key);objective(angles)
        except BudgetExpired:expired=True
        row=dict(target=target,initial_best=initial,best=best,evaluations=len(evaluations),
            wall_s=time.monotonic()-start,budget_expired=expired,
            stencil_complete=not expired,
            scope='Bounded local search only; no global infeasibility claim; original anchor, clearance, preload and contact-search limits unchanged.')
        rows.append(row)
        (output/(target+'_evaluations.json')).write_text(json.dumps(evaluations,indent=2,allow_nan=False)+'\n')
        (output/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('target','evaluations','wall_s','budget_expired')}|{'best_max_gap_mm':None if best is None else best['max_gap_mm']}),flush=True)
    verify_geometry_binding(directory,native)
    report=dict(schema='flatness_bounded_refinement_v1',targets=rows,source_sha256=sha(__file__),
        geometry_artifacts=native['geometry_artifacts'],probe_sha256=sha(a.probe),
        search='Two local stencils; tilt offsets +/-2 deg, azimuth +/-15 deg, roll +/-10 deg, then half scale; no global optimality claim',
        budgets=dict(max_calls_per_target=180,max_wall_s_per_target=120),physics_executed=False,training_eligible=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

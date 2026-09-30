"""Read-only plots and independent impulse accounting for a completed v2 run."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .contact_geometry_audit import audit_contacts, group
from .geometry_epoch import GEOMETRY_ID, qualify_contact
from .retainer_contact import ALL_FORCE_GROUPS, CONTACT_SURFACE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();case=args.run/'output/review'
    owner=json.loads((args.run/'result.json').read_text())
    result=json.loads((case/'fresh_expert_result.json').read_text())
    trace=case/'fresh_expert/fresh_expert_trace.jsonl'
    rows=[json.loads(line) for line in trace.open()]
    assert owner['completed'] and owner['gpu_released']
    assert result['geometry_id']==GEOMETRY_ID and result['contact_surface']==CONTACT_SURFACE
    assert rows and len(rows)==result['physics_steps']
    assert all(row['contact_surface']==CONTACT_SURFACE for row in rows)
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==result['trace_sha256']
    times=np.asarray([row['time_s'] for row in rows])
    assert np.allclose(times,np.arange(1,len(rows)+1)/240,atol=1e-7,rtol=0)
    values=np.asarray([[row[k+'_N'] for k in ALL_FORCE_GROUPS] for row in rows])
    assert np.isfinite(values).all() and (values>=0).all()
    raw=np.zeros_like(values)
    for i,row in enumerate(rows):
        for point in row['contact_points']:
            label=group(point['colliders'],revised=True)
            if label is None:continue
            impulse=np.asarray(point['impulse_Ns'],float)
            assert impulse.shape==(3,) and np.isfinite(impulse).all()
            raw[i,ALL_FORCE_GROUPS.index(label)]+=np.linalg.norm(impulse)*240
    assert np.allclose(values,raw,atol=1e-9,rtol=0)
    releases=[]
    for target in dict.fromkeys(row['target'] for row in rows):
        indices=[i for i,row in enumerate(rows)
                 if row['target']==target and row.get('subphase')=='withdraw_settle']
        if not indices:continue
        i=indices[-1];row=rows[i]
        releases.append(dict(target=target,physics_row=i,time_s=row['time_s'],
            last_component_forces_N=dict(zip(ALL_FORCE_GROUPS,values[i].tolist())),
            last_total_force_N=float(values[i].sum()),hall_N=row['hall_N'],
            compression_m=row['compression_m'],
            all_settle_rows_total_below_0p2N=bool((values[indices].sum(axis=1)<.2).all()),
            last_sample_force_and_compression_released=bool(values[i].sum()<.2
                and row['hall_N']<.2 and row['compression_m']<=.00015)))
    summary=dict(source_run=str(args.run),source_trace_sha256=result['trace_sha256'],
        physics_rows=len(rows),duration_s=float(times[-1]),
        phase_duration_s={k:v/240 for k,v in Counter(row['phase'] for row in rows).items()},
        component_force_peak_N=dict(zip(ALL_FORCE_GROUPS,values.max(axis=0).tolist())),
        independent_contact_gate=qualify_contact(rows),
        raw_impulse_accounting_max_error_N=float(abs(raw-values).max()),
        hold_accounting=audit_contacts(rows) if any(r['phase']=='HOLD' for r in rows) else None,
        recorded_physical_validation=result['physical_validation'],
        recorded_success=result['success'],raw_capture_complete=result.get('raw_capture_complete'),
        recorded_failure_reason=result.get('failure_reason'),recorded_errors=result.get('errors',[]),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.run/'result.json',case/'fresh_expert_result.json',Path(__file__))},
        point_results=result['points'],learned_policy=False,training_eligible=False,
        independent_seven_channel_release_samples=releases,
        release_scope='Raw-data recheck of recorded withdraw_settle endpoint; does not claim the runtime used a revised release gate',
        scope='Recount of completed actual expert run; no new physics, student, acoustic or generalization qualification')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,1,figsize=(10,7.8),sharex=True)
    axes[0].axhspan(3.5,4.5,color='#bdd9c8',alpha=.35,label='Original contact band')
    axes[0].plot(times,values[:,1],color='#a95224',label='C retainer (PhysX)')
    axes[0].plot(times,values[:,0],color='#147d8c',label='B membrane (PhysX)')
    axes[0].plot(times,[r['hall_N'] for r in rows],color='#545b6d',ls='--',label='Hall force proxy')
    axes[0].set(ylabel='Force (N)');axes[0].legend(ncol=2,frameon=False)
    axes[1].plot(times,values[:,2:].sum(axis=1),color='#ac3838',label='All forbidden components')
    axes[1].set(ylabel='Forbidden force (N)');axes[1].legend(frameon=False)
    axes[2].plot(times,[r['compression_m']*1000 for r in rows],color='#53477a')
    axes[2].axhline(6.,ls='--',color='#ac3838',label='Original stroke limit')
    axes[2].set(ylabel='Compression (mm)',xlabel='Simulation time (s)');axes[2].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Recessed membrane + contact retainer | actual R3 expert run',x=.09,ha='left')
    fig.text(.09,.02,'Raw PhysX impulses, independent components. Expert mechanics only; no learned policy or acoustic validation.',fontsize=9)
    fig.tight_layout(rect=(.015,.04,.99,.95))
    for extension in ('svg','png'):fig.savefig(args.output/('contact.'+extension),dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('point_results','hold_accounting')},indent=2))


if __name__=='__main__':main()

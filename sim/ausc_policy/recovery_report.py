"""Summarize SHA-bound actual stationary expert trials without requalifying data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def load(root,run_id):
    root=root.resolve()
    if Path(run_id).name!=run_id: raise ValueError('Invalid run identity')
    manifest=json.loads((root/'MIRROR_SHA256.json').read_text())
    for name,binding in manifest.items():
        path=root/name
        if (root not in path.resolve().parents or path.stat().st_size!=binding['bytes']
                or sha(path)!=binding['sha256']): raise ValueError('Changed evidence: '+name)
    def bound(name):
        relative=run_id+'/'+name
        if relative not in manifest: raise ValueError('Required unbound evidence: '+relative)
        return root/relative
    def read(name): return json.loads(bound(name).read_text())
    owner=read('result.json');complete=read('output/COMPLETE.json')
    if not (owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
            and owner['stage']['exit_code']==0 and complete['completed']
            and owner['config']['experiment']=='recovery_run'):
        raise ValueError('Completed owned expert pilot required')
    summaries=[];traces=[]
    for entry in complete['cases']:
        name=entry['case_id'];prefix='output/'+name+'/'
        if name not in ('nominal','joint3-plus','joint3-minus','act609','history20','history120','history220'): raise ValueError('Unexpected case')
        outcome=read(prefix+'outcome.json');result=read(prefix+'result.json')
        if (not result['execution_completed'] or not outcome['expert_demonstration']
                or outcome['learned_policy'] or outcome['training_dataset_eligible']):
            raise ValueError('Unexpected evaluation ownership')
        rows=[json.loads(line) for line in bound(prefix+'expert_trace.jsonl').open()]
        times=np.asarray([r['time_s'] for r in rows])
        if not len(rows) or not np.allclose(np.diff(times),1/240,rtol=0,atol=1e-8):
            raise ValueError('Physics timeline not consecutive')
        oracle=outcome.get('oracle_receipt',{});idx=oracle.get('query_physics_row')
        statistics=None
        if idx is not None:
            if not 239<=idx<len(rows) or abs(times[idx]-oracle['query_time_s'])>1e-8:
                raise ValueError('Query does not bind actual history')
            window=rows[idx-239:idx+1];q=np.asarray([r['q'] for r in window])
            dq=np.asarray([r['dq'] for r in window]);clock=np.asarray([r['time_s'] for r in window])
            statistics=dict(window_s=float(clock[-1]-clock[0]),
                reported_velocity_peak_rad_s=float(abs(dq).max()),
                position_difference_velocity_peak_rad_s=float(abs(np.diff(q,axis=0)/np.diff(clock)[:,None]).max()),
                position_range_rad=float(np.ptp(q,axis=0).max()))
        dynamics=read(prefix+'measured_dynamics.json')
        components=('diaphragm_N','lip_N','contact_rim_N','carriage_other_N','frame_N','forbidden_N')
        hold=[r for r in rows if r['phase']=='HOLD']
        role_timeline=[]
        for i,r in enumerate(rows):
            role=r.get('oracle_role','UNRECORDED')
            if not role_timeline or role_timeline[-1]['role']!=role:
                role_timeline.append(dict(role=role,first_physics_row=i,start_time_s=r['time_s']))
            role_timeline[-1].update(last_physics_row=i,end_time_s=r['time_s'])
        summaries.append(dict(run_id=run_id,code_commit=owner['code_commit'],case_id=name,
            case_passed=result['case_passed'],physics_steps=len(rows),simulation_end_s=float(times[-1]),
            sensor_frames=outcome['sensor_frames'],query=oracle,held_state_statistics=statistics,
            max_simulator_hall_reference_N=max(r['hall_N'] for r in rows),
            verified_front_contact_hold_s=max((p.get('verified_front_contact_hold_s',0.) for p in outcome['points']),default=0.),
            centered_acceleration_rad_s2=dynamics['centered']['acceleration_rad_s2']['max'],
            centered_jerk_rad_s3=dynamics['centered']['jerk_rad_s3']['max'],
            observed_supervisor_replay_passed=result['observed_supervisor_replay_passed'],
            moving_recovery=outcome.get('moving_recovery'),
            role_timeline=role_timeline,
            physical_contact_component_peak_N={k:max(r[k] for r in rows) for k in components},
            hold_component_mean_N={k:float(np.mean([r[k] for r in hold])) if hold else None for k in components},
            best_physical_audit_hold_s=max(r['audit_hold_s'] for r in rows),
            physical_audit_accepted=result['physics_audit_accepted'],failure_reason=outcome.get('failure_reason'),
            m0_passed=False,learned_policy=False,training_dataset_eligible=False,
            trace_sha256=sha(bound(prefix+'expert_trace.jsonl')),
            outcome_sha256=sha(bound(prefix+'outcome.json'))))
        traces.append(rows)
    return summaries,traces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summaries,traces=load(args.evidence,args.run_id)
    args.output.mkdir(parents=True,exist_ok=False)
    moving=any(r['moving_recovery'] is not None for r in summaries)
    report=dict(scope=('Actual R4 moving expert takeover after student command replay' if moving
        else 'Actual R4 stationary expert pilot')+'; single body, no policy/dataset qualification',
        m0_passed=False,cases=summaries)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    x=np.arange(len(summaries));width=.32
    reported=[r['held_state_statistics']['reported_velocity_peak_rad_s'] for r in summaries]
    difference=[r['held_state_statistics']['position_difference_velocity_peak_rad_s'] for r in summaries]
    axes[0].bar(x-width/2,reported,width,label='Reported API dq',color='#ac614c')
    bars=axes[0].bar(x+width/2,difference,width,label='Consecutive measured q / time',color='#36777d')
    axes[0].bar_label(bars,labels=[f'{v:.3g}' for v in difference],padding=3,fontsize=9)
    axes[0].axhline(.005,color='#777',linestyle='--',lw=1,label='Stationary pose-velocity gate')
    axes[0].set(xticks=x,xticklabels=[r['case_id'] for r in summaries],ylabel='Peak velocity (rad/s)',
        title='Last second before each real recovery query: held positions vs reported velocity')
    axes[0].legend(fontsize=9)
    for row,trace in zip(summaries,traces):
        sample=trace[::12]
        axes[1].plot([r['time_s'] for r in sample],[r['hall_N'] for r in sample],label=row['case_id'])
    axes[1].axhspan(3.5,4.5,color='#36777d',alpha=.12,label='Original target force band')
    axes[1].set(xlabel='Simulator time (s)',ylabel='Simulator Hall force reference (N)',
        title='Actual recorded force; independent contact and supervisor audits remain required')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle(('Moving expert takeover' if moving else 'Stationary expert recovery')+
        ' diagnostics — not a learned-policy test',fontsize=13)
    for suffix in ('svg','png'):fig.savefig(args.output/('recovery.'+suffix),dpi=150)
    plt.close(fig)
    if len(traces)==1:
        trace=traces[0];t=np.asarray([r['time_s'] for r in trace])
        fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True,
                              gridspec_kw=dict(height_ratios=[3,1]))
        axes[0].plot(t,[r['lip_N'] for r in trace],color='#36777d',label='Physical front lip')
        axes[0].plot(t,[r['hall_N'] for r in trace],color='#a97a38',lw=1,label='Simulator Hall reference')
        axes[0].plot(t,[r['diaphragm_N'] for r in trace],color='#ac614c',ls='--',label='Physical diaphragm')
        held=[r['time_s'] for r in trace if r['phase']=='HOLD']
        if held:axes[0].axvspan(min(held),max(held)+1/240,color='#36777d',alpha=.15,label='Recorded HOLD')
        axes[0].set(ylabel='Force (N)',title='Contact load is carried by the lip; diaphragm force remains zero')
        axes[0].legend(loc='upper left',fontsize=9)
        labels={'STUDENT_COMMAND_REPLAY':'Student prefix replay','EXPERT_BRAKE':'Brake',
                'EXPERT_RECOVERY':'Expert recovery','MEASURED_ANALYTIC_L3':'Analytic contact / hold / release',
                'FINAL_RETURN':'Return home'}
        palette=['#9ba7b5','#c8a974','#7da7ac','#91b1a0','#c2cad3']
        for i,role in enumerate(summaries[0]['role_timeline']):
            start,end=role['start_time_s']-1/240,role['end_time_s']
            axes[1].broken_barh([(start,end-start)],(i-.35,.7),facecolors=palette[i%len(palette)])
        axes[1].set(yticks=range(len(summaries[0]['role_timeline'])),
            yticklabels=[labels.get(r['role'],r['role']) for r in summaries[0]['role_timeline']],
            xlabel='Actual simulator time (s)',xlim=(0,t[-1]))
        axes[1].invert_yaxis();axes[1].tick_params(axis='y',labelsize=8)
        for ax in axes:ax.spines[['right','top']].set_visible(False)
        fig.suptitle('Actual expert recovery — no learned policy or acoustic acceptance',fontsize=13)
        for suffix in ('svg','png'):fig.savefig(args.output/('contact.'+suffix),dpi=150)
        plt.close(fig)
    print(json.dumps(dict(cases=len(summaries),passed=sum(r['case_passed'] for r in summaries),m0_passed=False)))


if __name__=='__main__':main()

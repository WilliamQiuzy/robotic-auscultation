"""Report the fixed paired models, full TRAIN errors and actual R4 rollouts."""
import argparse
from pathlib import Path
import numpy as np
from . import act_baseline as a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();owner=a.read_json(args.probe/'result.json');complete=a.read_json(args.probe/'output/COMPLETE.json')
    a.need(owner['completed'] and owner['gpu_released'] and complete['completed']
        and complete['optimizer_updates']==0 and complete['train_windows']==8685,'Completed fixed-weight full TRAIN probe required')
    cases=[a.read_json(root/'summary.json') for root in (args.control,args.history)]
    for case,record in zip(cases,complete['records']):
        replay=record['actual_replay'][0]
        a.need(case['variant']==record['variant'] and case['run_id']==replay['run_id']
            and case['accepted_queries']==replay['accepted_queries'] and replay['maximum_error_rad']==0
            and case['same_frame_input_max_error']==0,'Case and exact model replay differ')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(14,5.4));colors=['#5c728f','#0e887f']
    for root,label,color in zip((args.control,args.history),('A: current only','B: causal history'),colors):
        values=a.read_json(root/'geometry.json');t=[x['time_s'] for x in values]
        for ax,key in zip(axes,('gap_mm','tangent_mm','axis_deg')):
            y=[x[key] for x in values];ax.plot(t,y,label=label,color=color,linewidth=2)
            ax.scatter(t[-1],y[-1],color=color,s=35,zorder=3)
    for ax,label,target in zip(axes,('Normal gap (mm)','Tangential error (mm)','Tool-axis angle (deg)'),(25,3,5)):
        ax.axhline(target,color='#a64b40',ls=':',lw=1,label='Nominal / gate reference')
        ax.set_xlabel('Actual simulation time (s)');ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Better offline fit did not produce a successful rollout',x=.06,ha='left',fontsize=16,fontweight='bold')
    fig.text(.06,.025,'Same final 2,000-update models and R4 scene. A stopped at 108.1 s; B at 18.1 s.\nBoth: no contact / hold. Saved sensor geometry; no expert takeover, no cross-body claim.',fontsize=10,color='#536171')
    fig.subplots_adjust(top=.83,bottom=.21,left=.07,right=.98,wspace=.33)
    for ext in ('png','svg'):fig.savefig(args.output/f'rollouts.{ext}',dpi=150,facecolor='white')
    a.save_json(args.output/'summary.json',dict(cases=cases,full_train=complete,
        native_queries_replayed=sum(c['accepted_queries'] for c in cases),m0_passed=False,
        independent_bodies=1,trials_per_variant=1,probe_owner_sha256=a.sha256(args.probe/'result.json')))


if __name__=='__main__':main()

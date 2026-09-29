"""Plot final paired history fit receipts, never select intermediate weights."""
import argparse
from pathlib import Path
from . import act_baseline as a


def summarize(root):
    owner=a.read_json(root/'result.json');complete=a.read_json(root/'output/COMPLETE.json')
    a.need(owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
        and owner['stage']['exit_code']==0 and complete['completed'] and complete['identical_initial_weights'],
        'Completed paired experiment required')
    rows={v['variant']:v for v in complete['variants']}
    a.need(set(rows)=={'repeat_current','causal_history'} and all(v['maximum_reload_error_rad']==0 for v in rows.values()),
        'Both final reloaded checkpoints required')
    return dict(run_id=root.name,owner_sha256=a.sha256(root/'result.json'),
        complete_sha256=a.sha256(root/'output/COMPLETE.json'),variants=rows,
        relative_improvement={k:1-rows['causal_history']['metrics']['metrics']['act'][k]/rows['repeat_current']['metrics']['metrics']['act'][k]
            for k in ('full_chunk_mae_rad','h0_mae_rad','median_sample_mae_rad')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit32',type=Path,required=True);p.add_argument('--train16',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();records=[summarize(args.fit32),summarize(args.train16)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    args.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6));colors=['#96a1b0','#0e887f'];x=np.arange(2);width=.32
    for ax,metric,title in zip(axes,('full_chunk_mae_rad','h0_mae_rad'),('Full H10 action block','First action')):
        for v,(variant,label) in enumerate((('repeat_current','A: repeated current'),('causal_history','B: causal history'))):
            values=[r['variants'][variant]['metrics']['metrics']['act'][metric]*1000 for r in records]
            bars=ax.bar(x+(v-.5)*width,values,width,color=colors[v],label=label)
            ax.bar_label(bars,labels=[f'{value:.3f}' for value in values],padding=5,fontsize=11)
        ax.set_xticks(x,['TRAIN32 fit\n512 updates / arm','TRAIN16 fit\n2,000 updates / arm'])
        ax.set_ylabel('Mean absolute joint error (mrad)');ax.set_title(title,loc='left',fontweight='bold')
        ax.set_ylim(0,ax.get_ylim()[1]*1.17);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Does measured motion history help ACT fit?',x=.06,ha='left',fontweight='bold',fontsize=17)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.96,.915),ncol=2,frameon=False)
    fig.text(.06,.035,'Same 32 TRAIN probe observations, seed, architecture and budget within each pair.\nFinal checkpoints only. One body; no held-out or closed-loop success evidence.',fontsize=10,color='#536171')
    fig.subplots_adjust(top=.79,bottom=.20,left=.07,right=.97,wspace=.28)
    for extension in ('svg','png'):fig.savefig(args.output/f'paired_fit.{extension}',dpi=150,facecolor='white')
    a.save_json(args.output/'summary.json',dict(records=records,units='rad',checkpoint_selection='fixed final only',
        m0_passed=False,closed_loop_evaluated=False,heldout_opened=False))


if __name__=='__main__':main()

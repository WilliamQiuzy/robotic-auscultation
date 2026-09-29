"""Export a static review figure from measured reports, with failed gates visible."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.results;models=json.loads((root/'model_comparison.json').read_text())['splits']['test']
    original=json.loads((root/'probes/L3_physics/report.json').read_text())['measured']
    gentle=json.loads((root/'probes/L3_live_gentle_v2/measured_dynamics.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11.6,7.2),layout='constrained')
    fig.suptitle('Contact and VLA repair review',fontsize=18,fontweight='bold')
    colors=['#8792a2','#207c8a','#d8a64a']
    def bars(ax,labels,values,title,unit):
        rects=ax.bar(labels,values,color=colors[:len(values)],width=.58)
        ax.bar_label(rects,fmt='%.2f',padding=4,fontsize=10)
        ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.set_ylabel(unit)
        ax.set_ylim(0,max(values)*1.26);ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True,alpha=.16);ax.set_axisbelow(True)
    bars(axes[0,0],['Original L3','Gentle L3'],
        [original['centered']['jerk_rad_s3']['max'],gentle['centered']['jerk_rad_s3']['max']],
        'Expert motion: centered jerk','rad/s^3')
    axes[0,0].axhline(20,color='#bd4747',ls='--',lw=1,label='Research target: 20')
    axes[0,0].legend(frameon=False,fontsize=9)
    bars(axes[0,1],['Old VLA','State repair','Phase mean'],
        [models['old']['mean_translation_error_mm'],models['new']['mean_translation_error_mm'],models['phase_mean']['mean_translation_error_mm']],
        'L5: mean translation error','mm; lower is better')
    bars(axes[1,0],['Old VLA','State repair'],
        [models['old']['p95_translation_error_mm'],models['new']['p95_translation_error_mm']],
        'L5: 95th-percentile error','mm; lower is better')
    bars(axes[1,1],['Old VLA','State repair'],[models['old_bound_pass']*100,models['new_bound_pass']*100],
        'L5: proposal-bound pass rate','%; higher is better')
    axes[1,1].set_ylim(0,100)
    fig.supxlabel('Unfiltered expert positions at 240 Hz; candidate armature is uncalibrated.\n'
        'VLA: 25 paired diagnostic windows on one previously inspected layout. Not a deployment qualification.',fontsize=9)
    fig.savefig(root/'repair_review.png',dpi=180);fig.savefig(root/'repair_review.pdf')
    plt.close(fig)


if __name__=='__main__':main()

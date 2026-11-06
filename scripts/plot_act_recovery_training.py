"""Plot the completed first ACT treatment from public scalar evidence only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(evidence,output):
    rows=evidence['curves'];selected=evidence['selected_step'];output=Path(output)
    if output.exists():raise FileExistsError('Use a new figure path')
    plt.rcParams.update({'font.size':13,'text.color':'#17313b','axes.labelcolor':'#17313b',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b4c5c8'})
    fig,axes=plt.subplots(1,2,figsize=(14.8,7.5),gridspec_kw={'width_ratios':[1.25,1]})
    fig.patch.set_facecolor('#f7faf8')
    for ax in axes:
        ax.set_facecolor('#f7faf8');ax.set_axisbelow(True);ax.grid(axis='y',color='#dce5e5',linewidth=.8)
    a,b=axes
    a.plot([r['step'] for r in rows],[r['train_mae_rad']*1000 for r in rows],color='#8ba5ad',label='TRAIN',linewidth=2)
    a.plot([r['step'] for r in rows],[r['selection_val_mae_rad']*1000 for r in rows],color='#087b70',label='Selection VAL',linewidth=2)
    a.axvline(selected,color='#b95d18',linestyle='--',linewidth=1.2)
    a.set(xlabel='Optimizer step',ylabel='Valid raw target MAE (mrad)',xlim=(0,20000),ylim=(0,42))
    a.set_xticks([0,5000,10000,15000,20000],['0','5k','10k','15k','20k'])
    a.set_title('Complete 20,000 step schedule',loc='left',fontweight='bold',pad=20)
    a.legend(loc='lower right',frameon=False)
    a.text(.04,.96,f'Selected at {selected:,}\nEarliest full VAL minimum',transform=a.transAxes,va='top',fontsize=12,color='#9d4d11')
    metrics=evidence['selected_metrics']['validation']
    names=['act','hold','constant_velocity'];values=[metrics[n]['full_chunk_mae_rad']*1000 for n in names]
    b.bar(range(3),values,color=['#087b70','#8ba5ad','#b4c5c8'],width=.62)
    b.set_xticks(range(3),['ACT','Hold','Constant\nvelocity'])
    b.set(ylabel='Valid raw target MAE (mrad)',ylim=(0,185))
    b.set_title('Same 2,345 selection VAL origins',loc='left',fontweight='bold',pad=20)
    for i,v in enumerate(values):b.text(i,v+4,f'{v:.2f}',ha='center',fontsize=13)
    score=metrics['act']['full_chunk_mae_rad']*1000
    fig.suptitle(f'Selected ACT: {score:.2f} mrad VAL · 0 / 4 native passes',x=.07,y=.99,ha='left',fontsize=24,fontweight='bold')
    fig.subplots_adjust(left=.075,right=.98,bottom=.21,top=.79,wspace=.37)
    fig.text(.075,.085,'Original Isaac development qualification: 0 / 4. Fresh 32 scenario cohort unexposed at this stage.',fontsize=13,color='#17313b')
    fig.text(.075,.035,'State only ACT L2 + analytic L3. Target space error is distinct from tracking, contact and physical acceptance.',fontsize=11,color='#536a74')
    fig.savefig(output,dpi=150,bbox_inches='tight',pad_inches=.12);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plot(json.loads(Path(a.evidence).read_text()),a.output)

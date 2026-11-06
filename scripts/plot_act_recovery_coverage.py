"""Plot the public scalar diagnosis. No patient arrays or simulation required."""
import argparse
import json
from pathlib import Path


def plot(evidence, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data=json.loads(Path(evidence).read_text())
    assert data['schema']=='ausc_act_recovery_diagnosis_v1'
    splits=('train','validation');old=data['legacy_subset']['comparisons'];new=data['coverage']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#354553','text.color':'#162c38',
        'axes.edgecolor':'#bac7cb','figure.facecolor':'#f7faf8','axes.facecolor':'#f7faf8'})
    fig,axes=plt.subplots(1,2,figsize=(10.4,5.2),gridspec_kw={'wspace':.35})
    fig.subplots_adjust(top=.76,bottom=.15)
    x=np.arange(2);width=.32
    for i,(label,color,values) in enumerate([
        ('Original full H20','#8ca4ad',[old[s]['legacy_windows'] for s in splits]),
        ('Real prefixes + mask','#0b766a',[new[s]['windows'] for s in splits])]):
        bars=axes[0].bar(x+(i-.5)*width,values,width,color=color,label=label)
        axes[0].bar_label(bars,labels=[f'{n:,}' for n in values],padding=4,fontsize=10)
    axes[0].set_xticks(x,['TRAIN','Selection VAL']);axes[0].set_ylim(0,10500)
    axes[0].set_ylabel('Valid observation origins')
    axes[0].set_title('Same 18 qualified\nnative demonstrations',loc='left',fontweight='bold',pad=18)
    axes[0].legend(frameon=False,fontsize=9)
    for i,(color,values) in enumerate([
        ('#8ca4ad',[data['original_tangent_minimum_m'][s]*1000 for s in splits]),
        ('#0b766a',[new[s]['tangent_minimum_m']*1000 for s in splits])]):
        bars=axes[1].bar(x+(i-.5)*width,values,width,color=color)
        axes[1].bar_label(bars,labels=[f'{n:.2f}' for n in values],padding=4,fontsize=10)
    axes[1].axhline(2,color='#b56b35',ls='--',lw=1)
    axes[1].text(.98,.98,'Dashed line: necessary 2 mm condition',transform=axes[1].transAxes,
        ha='right',va='top',color='#9e592a',fontsize=8)
    axes[1].set_xticks(x,['TRAIN','Selection VAL']);axes[1].set_ylim(0,6.8)
    axes[1].set_ylabel('Closest encoded tangential distance (mm)')
    axes[1].set_title('The export reaches\nthe handoff edge',loc='left',fontweight='bold',pad=18)
    fig.text(.125,.96,'Repairing a label coverage gap',ha='left',fontsize=17,fontweight='bold')
    fig.text(.125,.015,'Unavailable futures are masked, never STOP/ALIGN labels. Two real history frames remain.\n'
        'Data coverage is not policy success. Normal gap, rotation, speed and contact still require native tests.',
        fontsize=8.4,color='#52646d')
    fig.savefig(output,dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();plot(args.evidence,args.output)

"""Standalone scientific summary of a completed offline task-training run."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run):
    run=Path(run)
    report=json.loads((run/'evaluation/report.json').read_text())
    best=json.loads((run/'model/best.json').read_text())
    rows=[]
    for line in (run/'training.log').read_text().splitlines():
        try:value=json.loads(line)
        except ValueError:continue
        if isinstance(value,dict) and 'train_loss' in value:rows.append(value)
    if len(rows)!=1000 or rows[-1]['step']!=1000:
        raise ValueError('complete planned 1000-step run required for final figure')
    result=report['splits']['test']
    phases=[p for p in ('FREE','APPROACH','CONTACT','HOLD','RETRACT') if p in result['baseline_by_phase']]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.5),layout='constrained')
    ax=axes[0,0]
    x=np.array([r['step'] for r in rows]);y=np.array([r['train_loss'] for r in rows])
    ax.plot(x,y,color='#527aab',alpha=.22,lw=.6,label='Training batches')
    moving=np.convolve(y,np.ones(40)/40,mode='valid')
    ax.plot(x[39:],moving,color='#245587',lw=1.7,label='40-step mean')
    val=[r for r in rows if 'validation_loss' in r]
    ax.plot([r['step'] for r in val],[r['validation_loss'] for r in val],'o-',color='#be6c2c',label='Full validation set')
    ax.axvline(best['step'],color='#626262',ls=':',lw=1)
    ax.set(xlabel='Optimizer step',ylabel='Normalized flow-matching loss',title='A  Optimization and checkpoint selection',yscale='log')
    ax.legend(frameon=False,fontsize=8)
    x=np.arange(len(phases));width=.36
    for ax,key,title,ylabel in [
        (axes[0,1],'mean_translation_error_mm','B  Held-out L5 pose-command error','Translation error (mm)'),
        (axes[1,0],'mean_force_error_N','C  Held-out L5 force-reference error','Force reference error (N)'),
    ]:
        learned=[result['baseline_by_phase'][p][key] for p in phases]
        constant=[result['descriptive_baselines_by_phase'][p]['training_phase_mean'][key] for p in phases]
        ax.bar(x-width/2,learned,width,color='#245587',label='Trained pi0.5')
        ax.bar(x+width/2,constant,width,color='#bec9d6',label='Train-only phase mean')
        ax.set(xticks=x,xticklabels=phases,ylabel=ylabel,title=title)
        ax.tick_params(axis='x',labelsize=8);ax.legend(frameon=False,fontsize=8)
    names=['rgb','depth_geometry','hall','robot','instruction'];labels=['RGB','Depth geometry','Hall','Joints','Instruction']
    effect=[result['intervention_prediction_change'][name]['mean_translation_error_mm'] for name in names]
    ax=axes[1,1];ax.barh(labels,effect,color='#568676');ax.invert_yaxis()
    ax.set(xlabel='Change in predicted translation (mm)',title='D  Same-noise input interventions')
    ax.text(0,-.28,'Off-manifold channel probes; not a causal importance ranking.',transform=ax.transAxes,fontsize=8,color='#555555')
    fig.suptitle(f'Initial multimodal auscultation VLA | selected step {best["step"]}\nOne body / one layout; offline prediction only',fontsize=15,fontweight='bold')
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(run/'training_evaluation.png',dpi=170,facecolor='white')
    fig.savefig(run/'training_evaluation.pdf',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    plot(parser.parse_args().run)

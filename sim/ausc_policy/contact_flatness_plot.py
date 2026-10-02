"""Plot bounded CPU audit summaries; no scene rendering or raw export."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('flatness','gaps','fits','output'):p.add_argument(key)
    a=p.parse_args();flat=json.loads(Path(a.flatness).read_text());gaps=json.loads(Path(a.gaps).read_text());fits=json.loads(Path(a.fits).read_text())
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    names=[r['target'] for r in flat['points']]
    if names!=[r['target'] for r in gaps['points']] or names!=[r['target'] for r in fits['points']]:raise ValueError('Point order differs')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True);x=np.arange(10)
    axes[0].bar(x-.18,[r['metrics']['target_normal_angle_deg']['mean'] for r in flat['points']],.36,label='Tool vs target surface normal',color='#286d8c')
    axes[0].bar(x+.18,[r['metrics']['normal_resultant_angle_deg']['mean'] for r in flat['points']],.36,label='Tool vs normal force resultant',color='#c88937')
    axes[0].set_ylabel('Angle (degrees)');axes[0].set_xticks(x,names);axes[0].legend(loc='upper left',fontsize=10)
    axes[0].set_title('v10: actual contact does not establish a flat seal',loc='left',weight='bold')
    v=[r['gap_mm']['max'] for r in gaps['points']];b=[r['best']['max_gap_mm'] for r in fits['points']]
    axes[1].bar(x-.18,v,.36,color='#b44b45',label='Actual v10: worst sampled gap during HOLD')
    axes[1].bar(x+.18,b,.36,color='#658b65',label='Best saved CPU fit: no new physics or arm path')
    for i,value in enumerate(v):axes[1].text(i-.18,value+.3,f'{value:.1f}',ha='center',fontsize=9)
    axes[1].set_xticks(x,names);axes[1].set_ylabel('Patientward surface gap (mm)');axes[1].legend(loc='upper right',fontsize=9)
    axes[1].set_title('96 ring samples per native pose; 90 HOLD poses per point',loc='left')
    fig.suptitle('Approved 5 mm retainer / 1 mm recessed membrane · CPU audit of preserved Isaac data',fontsize=12)
    fig.savefig(output/'angles_and_gaps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
    force=np.asarray([r['sector_mean_force_N'] for r in flat['points']])
    gap=np.asarray([np.asarray(r['middle_frame']['sector_inner_center_outer_gap_mm'])[:,1] for r in gaps['points']])
    im=axes[0].imshow(force,aspect='auto',cmap='viridis',vmin=0)
    axes[0].set_title('Average normal load by C-ring sector during HOLD',loc='left',weight='bold');fig.colorbar(im,ax=axes[0],label='N')
    im=axes[1].imshow(gap,aspect='auto',cmap='magma',vmin=0)
    axes[1].set_title('Actual ring centre-sample gap: middle HOLD frame',loc='left',weight='bold');fig.colorbar(im,ax=axes[1],label='mm')
    for ax in axes:
        ax.set_yticks(np.arange(10),names);ax.set_xticks(np.arange(0,32,2));ax.set_xlabel('Sector index (0-31, cyclic)')
    fig.suptitle('Rigid solver load distribution is not skin contact area or acoustic quality',fontsize=12)
    fig.savefig(output/'ring_footprint.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

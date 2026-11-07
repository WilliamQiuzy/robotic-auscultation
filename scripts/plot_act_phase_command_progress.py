"""Plot scalar CPU command endpoints; does not load a simulator or patient data."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 x=json.loads(a.input.read_text());rows=x['rows'];assert len(rows)==6
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(16,6.4),dpi=100);fig.set_facecolor('#f5f4ee')
 positions=np.arange(6);labels=[f"{r['time_s']:g}s"+(' *' if r['reserved_diagnostic_only'] else '') for r in rows]
 fields=[('tangent_mm','Tangential error (mm)'),('gap_error_mm','Normal gap error (mm)')]
 for ax,(field,title) in zip(axes,fields):
  ax.set_facecolor('#f5f4ee');initial=np.array([r['initial_command'][field] for r in rows])
  for offset,key,label,color in [(-.23,'actual_student','ACT6 command end','#d45d43'),(0,'initial','Command at query','#7b8174'),(.23,'oracle_teacher','Oracle command end','#16705b')]:
   values=initial if key=='initial' else np.array([r['variants'][key]['end_command'][field] for r in rows])
   ax.bar(positions+offset,values,width=.21,color=color,label=label,zorder=3)
  ax.axhline(0,color='#999e93',linewidth=.7);ax.grid(axis='y',alpha=.2,zorder=0)
  ax.set_xticks(positions,labels);ax.set_ylabel(title);ax.set_xlabel('Actual query time in the failed TRAIN L3 rollout')
  ax.set_title(title,fontweight='bold',loc='left',pad=12)
 axes[0].legend(frameon=False,fontsize=10,loc='upper right')
 fig.suptitle('The expert advances where ACT6 cycles',fontsize=24,fontweight='bold',x=.06,ha='left',y=.99)
 fig.text(.06,.89,'CPU replay of 0.5 seconds from each authentic retained state; no new physics or contact result.',fontsize=13)
 fig.text(.06,.02,'601 available ACT commands match Native bit for bit; all 720 oracle prefix commands match the original proposal.\n* Final adjacent pair reserved for diagnosis. Zero-relative HOLD abstains under the original safety check in four of six states.',fontsize=11)
 fig.subplots_adjust(left=.06,right=.98,bottom=.23,top=.77,wspace=.25)
 a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,facecolor=fig.get_facecolor());plt.close(fig)
if __name__=='__main__':main()

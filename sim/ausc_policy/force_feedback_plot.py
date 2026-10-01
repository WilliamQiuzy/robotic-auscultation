"""Plot compact audited summaries on Linux, with no raw-data transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();before=json.loads(a.before.read_text());after=json.loads(a.after.read_text())
fig,axes=plt.subplots(2,2,figsize=(12,7.2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for ax,summary,title in [(axes[0,0],before,'v8: early Hall dwell'),(axes[0,1],after,'Repaired: wait for settled Hall load')]:
    s=summary['R2_20hz_series'];hold=[r['time_s'] for r in s if r['phase']=='HOLD'];assert hold
    start=hold[0];end=hold[-1]
    x=np.array([r['time_s']-start for r in s]);mask=(x>=-1.5)&(x<end-start+1.)
    ax.axhspan(3.5,4.5,color='#dceee5',label='Original normal-load gate')
    ax.axvspan(0,end-start,color='#cbd5e1',alpha=.35,label='Controller HOLD')
    ax.plot(x[mask],np.array([r['Hall_N'] for r in s])[mask],color='#6b5ca5',ls='--',label='Hall axial proxy')
    ax.plot(x[mask],np.array([r['normal_N'] for r in s])[mask],color='#007f79',label='Raw normal contact')
    ax.set(title=title,xlabel='Seconds from first HOLD sample',ylabel='Force (N)',ylim=(3.1,4.6));ax.grid(alpha=.18)
axes[0,0].legend(loc='lower right',fontsize=8)
s=after['R2_20hz_series'];rows=[r for r in s if r['phase']=='HOLD'];x=[r['time_s']-rows[0]['time_s'] for r in rows]
ax=axes[1,0]
for field,label,color in [('Hall_N','Hall axial proxy','#6b5ca5'),('normal_axial_N','Normal impulse, axial projection','#007f79'),('combined_axial_N','Normal + friction, axial projection','#c05d28')]:
    ax.plot(x,[r[field] for r in rows],label=label,color=color)
ax.set(title='R2: distinguish force definitions',xlabel='Seconds in HOLD',ylabel='Force along sensor axis (N)');ax.legend(fontsize=8);ax.grid(alpha=.18)
ax=axes[1,1];b={r['target']:r['continuous_contact_s'] for r in before['points']};names=[r['target'] for r in after['points']];x=np.arange(len(names))
ax.bar(x-.18,[b[t] for t in names],width=.36,color='#a5a9b1',label='v8')
ax.bar(x+.18,[r['continuous_contact_s'] for r in after['points']],width=.36,color='#007f79',label='Repaired')
ax.axhline(3,color='#aa3434',ls='--',label='Unchanged 3 s minimum');ax.set_xticks(x,names)
ax.set(title='Longest uninterrupted raw normal-load dwell',ylabel='Seconds in 3.5-4.5 N');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18)
fig.suptitle('Hall feedback repair | actual Isaac physics and independent impulse audit',fontsize=14)
fig.text(.02,.015,'Expert mechanics on one nominal body. Sensor formula and physical acceptance gates unchanged. No learned-policy or real-hardware claim.',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.96));a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=160);plt.close(fig)

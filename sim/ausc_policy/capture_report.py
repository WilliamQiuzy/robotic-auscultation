"""Plot the completed 35-node geometry screen; no physical acceptance claim."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.run/'output/COMPLETE.json'
    data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!='2294967d79500dd684443468c6aec1b0dc1c4b0e36becc6e73b7e1f93da68002':
        raise ValueError('Requires the frozen completed geometry screen')
    report=json.loads(data)
    if not report['completed'] or report['case_count']!=35 or report['physics_executed']:
        raise ValueError('Wrong geometry scope')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    a.output.mkdir(parents=True,exist_ok=True)
    green,red='#257e73','#bb514d'
    fig,axes=plt.subplots(1,2,figsize=(13,10),constrained_layout=True)
    for ax,kind,title in zip(axes,['translation','rotation'],['Position nodes (mm)','Orientation nodes (mrad)']):
        rows=[r for r in report['cases'] if r['case']['kind'] in (kind,'center')]
        for y,row in enumerate(rows):
            c=row['case'];passed=row['full_curve_geometry_passed']
            text='center' if c['kind']=='center' else f"{'tangent ' if c['axis']<2 else 'normal '}{c['axis']} : {c['value']*1000:+g}"
            ax.scatter([0],[y],s=100,color=green if passed else red,zorder=3)
            ax.text(.10,y,text,va='center',fontsize=10)
            ax.text(1.6,y,'PASS' if passed else 'REFUSED',va='center',color=green if passed else red,fontsize=9)
        ax.set(xlim=(-.12,2.35),ylim=(len(rows)+1.2,-1))
        ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',pad=12)
    fig.suptitle('R4 direct contact entry: geometry only\n35/35 IK + free-space; 20/35 original entry + full normal curve',fontsize=15)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=green,label='Original geometry gates pass'),
                        Line2D([0],[0],marker='o',color='none',markerfacecolor=red,label='Original geometry gate refuses')],
               loc='lower center',ncol=2,frameon=False)
    fig.savefig(a.output/'geometry.svg');fig.savefig(a.output/'geometry.png',dpi=140)
    (a.output/'summary.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(data).hexdigest(),
        counts=report['counts'],case_count=35,physics_executed=False,capture_success_rate=None,
        limitation='Center shown in both panels; denominator remains 35 unique nodes. Boundary numerical residuals are not rounded to pass.'),indent=2)+'\n')
    print(json.dumps(report['counts']))


if __name__=='__main__': main()

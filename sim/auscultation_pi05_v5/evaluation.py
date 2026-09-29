"""Offline action prediction metrics. Never establishes closed-loop success.

State ablations replace the named normalized channels by their training-range
midpoints. They may be off-manifold: they diagnose numerical input dependence,
not useful learned sensor fusion or causal real-world performance by themselves.
"""
import numpy as np
from sim.auscultation_vla_v2.chunks import PHASES

STATE_INTERVENTIONS={
    'hall':(20,21),
    'robot':tuple(range(14)),
    'depth_geometry':tuple(range(14,20))+tuple(range(22,32)),
}


def training_phase_means(episodes):
    """Descriptive constant-action baseline; no held-out fitting or execution."""
    if not episodes or any(e['split']!='train' for e in episodes):
        raise ValueError('phase means require training episodes only')
    values={}
    for episode in episodes:
        actions=np.asarray(episode['actions'],float);phases=np.asarray(episode['phases'])
        if (actions.ndim!=2 or actions.shape[1]!=7 or len(actions)==0
                or phases.shape!=(len(actions),) or not np.isfinite(actions).all()
                or not set(phases)<=PHASES):
            raise ValueError('invalid training action/phase rows')
        for phase in set(phases):
            values.setdefault(str(phase),[]).append(actions[phases==phase])
    return {phase:np.concatenate(rows).mean(axis=0) for phase,rows in values.items()}


def intervene_state(normalized_state,name):
    state=np.asarray(normalized_state,float)
    if state.shape!=(32,) or not np.isfinite(state).all() or name not in STATE_INTERVENTIONS:
        raise ValueError('invalid named state intervention')
    result=state.copy();result[list(STATE_INTERVENTIONS[name])]=0.
    return result


def balanced_indices(phases,*,per_phase=5):
    if type(per_phase) is not int or per_phase<1 or not phases or not set(phases)<=PHASES:
        raise ValueError('invalid evaluation phase selection')
    selected=[]
    for phase in sorted(set(phases)):
        available=np.flatnonzero(np.array(phases)==phase)
        positions=np.rint(np.linspace(0,len(available)-1,min(per_phase,len(available)))).astype(int)
        selected.extend(available[positions].tolist())
    return sorted(selected)


def action_metrics(prediction,reference):
    p=np.asarray(prediction,float);r=np.asarray(reference,float)
    if p.shape!=r.shape or p.ndim!=3 or p.shape[1:]!=(10,7) or len(p)==0 or not np.isfinite([p,r]).all():
        raise ValueError('paired finite N x 10 x 7 action arrays required')
    error=np.abs(p-r)
    return dict(chunks=len(p),mean_joint_error_rad=float(error.mean()),
        p95_joint_error_rad=float(np.quantile(error,.95)),max_joint_error_rad=float(error.max()),
        per_joint_mae_rad=error.mean(axis=(0,1)).tolist(),
        first_action_joint_mae_rad=float(error[:,0].mean()),
        first_action_max_joint_error_rad=float(error[:,0].max()))

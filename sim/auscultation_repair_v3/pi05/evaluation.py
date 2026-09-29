"""Offline action prediction metrics. Never establishes closed-loop success.

State ablations replace the named normalized channels by their training-range
midpoints. They may be off-manifold: they diagnose numerical input dependence,
not useful learned sensor fusion or causal real-world performance by themselves.
"""
import numpy as np
from scipy.spatial.transform import Rotation
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


def mask_inactive_force(actions,phase):
    """Explicitly remove an unused channel; never clip motion or active force.

    Continuous samplers cannot emit an exact zero reliably. Outside CONTACT
    and HOLD this channel is defined as inactive by the task contract. Record
    this deterministic semantic mask separately from the raw model prediction.
    The resulting proposal still requires all motion/execution safety checks.
    """
    a=np.asarray(actions,float)
    if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all() or phase not in PHASES:
        raise ValueError('invalid action sequence or phase')
    result=a.copy()
    if phase in {'FREE','APPROACH','RETRACT'}:result[:,6]=0.
    return result


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
    difference=p-r
    translation=np.linalg.norm(difference[:,:,:3],axis=-1)*1000
    rotation=(Rotation.from_rotvec(p[:,:,3:6].reshape(-1,3))*
              Rotation.from_rotvec(r[:,:,3:6].reshape(-1,3)).inv()).magnitude().reshape(p.shape[:2])
    force=np.abs(difference[:,:,6])
    return dict(chunks=len(p),mean_translation_error_mm=float(translation.mean()),
        p95_translation_error_mm=float(np.quantile(translation,.95)),
        mean_rotation_error_rad=float(rotation.mean()),
        mean_force_error_N=float(force.mean()),
        first_action_translation_error_mm=float(translation[:,0].mean()),
        first_action_rotation_error_rad=float(rotation[:,0].mean()),
        first_action_force_error_N=float(force[:,0].mean()))

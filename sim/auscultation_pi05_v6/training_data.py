"""Pure data preparation gates, not evidence that demonstrations are qualified."""
import numpy as np
import hashlib
from pathlib import Path
from functools import lru_cache
from sim.auscultation_vla_v2.chunks import PHASES
from .state_encoding import StateCodec
from .action_encoding import encode as encode_actions, decode as decode_actions, action_encoding_binding


@lru_cache(maxsize=1)
def physical_codec():
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    kin=OpenArmKinematics()
    return StateCodec(kin.lower_limits,kin.upper_limits)


def encoding_binding():
    from sim.auscultation_expert_layout_v1.kinematics import DEFAULT_URDF_PATH
    from .task_observation import observation_binding
    return dict(codec_sha256=hashlib.sha256(Path(__file__).with_name('state_encoding.py').read_bytes()).hexdigest(),
                robot_urdf_sha256=hashlib.sha256(Path(DEFAULT_URDF_PATH).read_bytes()).hexdigest(),
                observation_contract=observation_binding())

MODALITIES = dict(rgb='camera', depth='camera_metric_optical_z', hall='adc_voltage',
                  robot='measured_q_dq_fk', target='rgbd_surface_tracking', audio='absent')
ANCESTRY = ('body_source', 'layout_family', 'motion_family')


def validate_modalities(declaration):
    if declaration != MODALITIES:
        raise ValueError('Required RGB-D, Hall ADC and measured robot modalities differ')


def validate_groups(records, *, heldout_axes):
    """IDs must describe ancestry, not new random IDs for copies of a scene."""
    if not records or not heldout_axes or not set(heldout_axes) <= set(ANCESTRY):
        raise ValueError('Declare existing ancestry axes before splitting')
    seen, membership = set(), {key: {} for key in ANCESTRY}
    for record in records:
        for key in ('episode_id','split',*ANCESTRY):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError('Missing group identity: '+key)
        if record['episode_id'] in seen or record['split'] not in {'train','validation','test'}:
            raise ValueError('Duplicate episode or unknown split')
        seen.add(record['episode_id'])
        for key in ANCESTRY:
            membership[key].setdefault(record[key],set()).add(record['split'])
    for key in heldout_axes:
        if any(len(splits)>1 for splits in membership[key].values()):
            raise ValueError('Cross-split ancestry leakage: '+key)
    if 'train' not in {r['split'] for r in records}:
        raise ValueError('Missing train group')
    return dict(episode_count=len(seen),heldout_axes=list(heldout_axes),
        unseen_body_evaluation=('body_source' in heldout_axes
            and any(r['split']=='test' for r in records)))


def _matrix(value, width, name):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or value.shape[1]!=width or len(value)==0 or not np.isfinite(value).all():
        raise ValueError('Invalid '+name+' matrix')
    return value


def fit_normalization(episodes):
    if not episodes or any(e['split']!='train' for e in episodes):
        raise ValueError('Normalization may fit training episodes only')
    ids=[e['episode_id'] for e in episodes]
    if len(set(ids))!=len(ids):raise ValueError('Repeated training episode')
    state=[];actions=[]
    for episode in episodes:
        x=_matrix(episode['state'],41,'state');a=_matrix(episode['actions'],7,'actions')
        if len(x)!=len(a):raise ValueError('State/action pairing mismatch')
        state.append(x);actions.append(a)
    # Engineering minimum spans prevent constant/near-constant channels from
    # magnifying ADC noise. These are not fitted sensor-accuracy estimates.
    state_floor=np.array([.1]*14+[.02]*3+[.1]*3+[.1,.2,.1]+[.02]*3+[.1]*6)
    # Dimensionless logit span, fitted only on training labels. The underlying
    # physical joint commands remain byte-identical in the acquired dataset.
    action_floor=np.full(7,.001)
    def stats(values,floor):
        values=np.concatenate(values)
        low,high=np.quantile(values,[.01,.99],axis=0)
        span=np.maximum(high-low,floor);center=(high+low)/2
        return dict(low=(center-span/2).tolist(),high=(center+span/2).tolist(),
                    quantiles=[.01,.99],minimum_span=floor.tolist())
    codec=physical_codec()
    for x in state:codec.encode(x)
    by_phase={}
    for phase in sorted(PHASES):
        selected=[e['actions'][np.asarray(e['phases'])==phase] for e in episodes
                  if np.any(np.asarray(e['phases'])==phase)]
        if not selected:raise ValueError('Missing training action phase '+phase)
        by_phase[phase]=dict(stats([encode_actions(value) for value in selected],action_floor),
            action_encoding=action_encoding_binding(),units='dimensionless_joint_logit')
    return dict(schema='wam_physical_task_state_phase_logit_joint_actions_v1',fit_episode_ids=ids,
        state=codec.spec(),actions_by_phase=by_phase,encoding_binding=encoding_binding(),
        action_encoding_binding=action_encoding_binding())


def action_stats(normalization,phase):
    if normalization.get('action_encoding_binding')!=action_encoding_binding():
        raise ValueError('Action encoding source binding changed')
    if phase not in PHASES or phase not in normalization.get('actions_by_phase',{}):
        raise ValueError('Unknown or unfitted action phase')
    return normalization['actions_by_phase'][phase]


def phase_balanced_indices(phases,count,rng,episode_ids=None):
    if not phases or not set(phases)<=PHASES or type(count) is not int or count<1:
        raise ValueError('Invalid sampling request')
    ids=list(range(len(phases))) if episode_ids is None else list(episode_ids)
    if len(ids)!=len(phases):raise ValueError('Episode sampling identity mismatch')
    groups={}
    for i,(phase,episode) in enumerate(zip(phases,ids)):
        groups.setdefault(phase,{}).setdefault(episode,[]).append(i)
    ordered=sorted(groups);result=[]
    for _ in range(count):
        group=groups[ordered[int(rng.integers(len(ordered)))]]
        episodes=list(group);indices=group[episodes[int(rng.integers(len(episodes)))]]
        result.append(indices[int(rng.integers(len(indices)))])
    return np.asarray(result,dtype=np.int64)


def _range(stats, width):
    low=np.asarray(stats['low'],float);high=np.asarray(stats['high'],float)
    if low.shape!=(width,) or high.shape!=(width,) or not np.isfinite([low,high]).all() or np.any(high<=low):
        raise ValueError('Invalid normalization range')
    return low,high


def normalize(value,stats):
    if stats.get('schema')=='wam_physical_asinh_task_state_4096_v1':
        codec=physical_codec()
        if stats!=codec.spec():raise ValueError('State codec specification mismatch')
        return codec.encode(value)
    value=np.asarray(value,dtype=np.float32)
    if value.ndim<1 or not np.isfinite(value).all():raise ValueError('Invalid normalization input')
    if 'action_encoding' in stats:
        if stats['action_encoding']!=action_encoding_binding():raise ValueError('Action encoding differs')
        value=encode_actions(value)
    low,high=_range(stats,value.shape[-1])
    return (2*(value-low)/(high-low)-1).astype(np.float32)


def denormalize(value,stats):
    if 'schema' in stats:raise ValueError('Use the versioned codec to decode state tokens')
    value=np.asarray(value,dtype=np.float32)
    if value.ndim<1 or not np.isfinite(value).all():raise ValueError('Invalid normalized input')
    low,high=_range(stats,value.shape[-1])
    latent=((value+1)/2*(high-low)+low).astype(np.float32)
    if 'action_encoding' in stats:
        if stats['action_encoding']!=action_encoding_binding():raise ValueError('Action encoding differs')
        return decode_actions(latent)
    return latent


def full_chunks(times,phases,targets,actions,*,horizon=10,period_s=.05):
    """Use float64 seconds without padding, interpolation or cross-stage actions.

    Acquire/store timestamps as float64: upcasting previously rounded float32
    values cannot recover their timing precision. Keep the fixed cadence gate.
    """
    t=np.asarray(times);a=_matrix(actions,7,'actions')
    if t.dtype.kind!='f' or t.dtype.itemsize!=8:
        raise ValueError('Chunk timestamps must use float64 seconds')
    if (type(horizon) is not int or horizon<1 or not np.isfinite(period_s) or period_s<=0
            or t.shape!=(len(a),) or len(phases)!=len(t) or len(targets)!=len(t)
            or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<=0)
            or not set(phases)<=PHASES or any(not isinstance(x,str) or not x for x in targets)):
        raise ValueError('Invalid chunk time/stage/target contract')
    result=[]
    for start in range(len(t)-horizon+1):
        stop=start+horizon
        if (len(set(phases[start:stop]))!=1 or len(set(targets[start:stop]))!=1
                or not np.allclose(np.diff(t[start:stop]),period_s,atol=1e-7,rtol=0)):
            continue
        result.append(dict(start=start,actions=a[start:stop].astype(np.float32).copy()))
    return result

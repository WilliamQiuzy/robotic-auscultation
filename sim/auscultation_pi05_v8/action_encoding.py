"""Paired affine fixed-observation residual codecs in physical radians.

A delegates every operation to frozen V7. B changes only FREE/APPROACH fitting
from pooled horizon to independent horizon/joint cells. This does not bound an
output: the unchanged absolute-joint and physical execution gates still apply.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v7 import action_encoding as parent

A='A_pooled_control'
B='B_horizon_free_approach'
MODES=(A,B)
CHANGED_PHASES=('FREE','APPROACH')
SCHEMA='wam_paired_fixed_observation_horizon_normalization_v1'
VERSION='fixed_observed_q_paired_horizon_delta_rad_v1'
HORIZON=10
MINIMUM_FULL_SPAN_RAD=.002
ROUNDTRIP_TOLERANCE_RAD=2e-6
PARENT_CODEC_SHA256='709e0d67631e3164678198c7e50d9abf702421183e1ea3cb60d1e6c44817b885'
PARENT_ACTION_CONTRACT_SHA256='d21e73ec9e530b0a3785debb9e78cd42648f4de45880221317ba982aa44b7596'


def _need(condition,message):
    if not condition:raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _mode(mode):
    _need(isinstance(mode,str) and mode in MODES,'Explicit paired codec mode required')
    return mode


def recipe(mode):
    return 'pi05_v8_horizon_normalization_pair_v1_'+_mode(mode)


def action_encoding_binding(mode):
    _mode(mode)
    old=parent.action_encoding_binding()
    _need(old['source_sha256']==PARENT_CODEC_SHA256 and old['original_absolute_action_contract_sha256']==PARENT_ACTION_CONTRACT_SHA256,'Frozen parent codec/action contract changed')
    return dict(schema='wam_paired_horizon_action_encoding_binding_v1',version=VERSION,mode=mode,recipe=recipe(mode),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_binding=old,
        changed_phases=list(CHANGED_PHASES) if mode==B else [],horizon=HORIZON,action_width=7,units='rad',
        fit_axis='phase_horizon_joint_for_FREE_APPROACH' if mode==B else 'phase_joint_pooled_horizon',
        quantile_method='linear',quantiles=[.01,.99],minimum_full_span_rad=MINIMUM_FULL_SPAN_RAD,
        formula='delta[h,j]=absolute_command[h,j]-observation_q[j]; one original measured anchor for all10',
        action_clipping=False,bounded_by_codec=False,intermediate_dtype='float64_cpu',encoded_dtype='float32',decoded_dtype='float32',
        roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD)


def fit_stats(episodes,mode):
    """Use each explicit TRAIN start once; parent validates numeric/phase inputs.

    Cadence, target transitions and excluded derivation labels are established by
    the unchanged qualified parent loader. This pure numeric API never generates
    starts, treats sample draws as fit weights, or claims acquisition quality.
    """
    binding=action_encoding_binding(mode)
    pooled=parent.fit_stats(episodes)
    by_phase=copy.deepcopy(pooled['actions_by_phase'])
    if mode==B:
        for phase in CHANGED_PHASES:
            values=[np.asarray(e['actions'][int(s):int(s)+HORIZON],np.float64)-np.asarray(e['state'][int(s),:7],np.float64)
                    for e in episodes for s in e['starts'] if str(e['phases'][int(s)])==phase]
            if not values:continue
            values=np.stack(values)
            low,high=np.quantile(values,[.01,.99],axis=0,method='linear')
            center=(low+high)/2;span=np.maximum(high-low,MINIMUM_FULL_SPAN_RAD)
            by_phase[phase]=dict(low=(center-span/2).tolist(),high=(center+span/2).tolist(),
                observed_min=values.min(axis=0).tolist(),observed_max=values.max(axis=0).tolist(),
                window_count=len(values),residual_count=len(values)*HORIZON,units='rad',fit_axis='window_only_separate_horizon_joint')
    result=dict(schema=SCHEMA,version=VERSION,recipe=recipe(mode),mode=mode,horizon=HORIZON,units='rad',
        quantiles=[.01,.99],quantile_method='linear',minimum_full_span_rad=MINIMUM_FULL_SPAN_RAD,
        fit_episode_ids=pooled['fit_episode_ids'],fit_window_count=sum(len(e['starts']) for e in episodes),
        fit_windows_sha256=_digest([dict(episode_id=e['episode_id'],starts=list(map(int,e['starts']))) for e in episodes]),
        fitting_weight='each explicitly supplied original TRAIN window exactly once',
        action_encoding_binding=binding,parent_pooled_stats=pooled,actions_by_phase=by_phase)
    result['normalization_sha256']=_digest(result)
    return result


def validate_stats(stats,*,mode,expected_sha256=None):
    """Reject legacy/cross-arm/tampered statistics; caller may pin exact norm hash."""
    binding=action_encoding_binding(mode)
    _need(isinstance(stats,dict) and stats.get('schema')==SCHEMA and stats.get('version')==VERSION
          and stats.get('mode')==mode and stats.get('recipe')==recipe(mode),'Wrong paired normalization mode/schema/recipe')
    _need(stats.get('action_encoding_binding')==binding,'Paired encoding or parent source changed')
    _need(stats.get('horizon')==10 and stats.get('units')=='rad' and stats.get('minimum_full_span_rad')==.002
          and stats.get('quantiles')==[.01,.99] and stats.get('quantile_method')=='linear','Paired normalization numeric contract differs')
    actual=_digest({k:v for k,v in stats.items() if k!='normalization_sha256'})
    _need(stats.get('normalization_sha256')==actual,'Paired normalization integrity differs')
    _need(expected_sha256 is None or expected_sha256==actual,'Expected normalization belongs to another fit/mode')
    pooled=stats.get('parent_pooled_stats',{})
    _need(pooled.get('action_encoding_binding')==parent.action_encoding_binding(),'Parent normalization source differs')
    _need(stats['fit_episode_ids']==pooled.get('fit_episode_ids'),'Parent fit identities differ')
    _need(set(stats['actions_by_phase'])==set(pooled.get('actions_by_phase',{})),'Parent fitted phases differ')
    for phase,own in stats['actions_by_phase'].items():
        if mode==A or phase not in CHANGED_PHASES:
            _need(own==pooled['actions_by_phase'][phase],'Untouched phase differs from pooled parent')
    return actual


def normalization_digest(stats,*,mode):
    return validate_stats(stats,mode=mode)


def _range(phase,stats,mode):
    validate_stats(stats,mode=mode)
    _need(isinstance(phase,str) and phase in parent.action_contract.PHASES and phase in stats['actions_by_phase'],'Unknown or unfitted phase')
    shape=(10,7) if mode==B and phase in CHANGED_PHASES else (7,)
    own=stats['actions_by_phase'][phase]
    low=parent._array(own.get('low'),shape,'paired phase low');high=parent._array(own.get('high'),shape,'paired phase high')
    _need(own.get('units')=='rad' and np.all(high>low),'Invalid paired normalization range')
    return low,high


def encode_chunk(absolute_commands,observation_q,phase,stats,*,mode):
    low,high=_range(phase,stats,mode)
    if mode==A or phase not in CHANGED_PHASES:
        return parent.encode_chunk(absolute_commands,observation_q,phase,stats['parent_pooled_stats'])
    commands=parent._array(absolute_commands,(10,7),'absolute action chunk')
    anchor=parent._array(observation_q,(7,),'observation q anchor')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        result=(2*((commands-anchor)-low)/(high-low)-1).astype(np.float32)
    _need(np.isfinite(result).all(),'Nonfinite encoded action chunk')
    return result


def decode_chunk(normalized_chunk,observation_q,phase,stats,*,mode):
    low,high=_range(phase,stats,mode)
    if mode==A or phase not in CHANGED_PHASES:
        return parent.decode_chunk(normalized_chunk,observation_q,phase,stats['parent_pooled_stats'])
    normalized=parent._array(normalized_chunk,(10,7),'normalized action chunk')
    anchor=parent._array(observation_q,(7,),'observation q anchor')
    with np.errstate(over='ignore',invalid='ignore'):
        result=((normalized+1)/2*(high-low)+low+anchor).astype(np.float32)
    _need(np.isfinite(result).all(),'Nonfinite decoded absolute action chunk')
    return result

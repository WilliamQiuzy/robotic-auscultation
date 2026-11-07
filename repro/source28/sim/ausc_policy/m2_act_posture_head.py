"""Explicit ACT output parameterization; only the target joint box is guaranteed.

This deterministic smooth constraint transform is part of the model definition,
active in the same forward graph for training and inference. It is not vanilla
unbounded ACT and does not qualify dynamics, mesh clearance, force or task gates.
"""
from copy import deepcopy

import numpy as np

from . import m2_given_contact_runtime as rt

need=rt.need
SCHEMA='m2_act_smooth_posture_head_v1'
SCHEMA_V2='m2_act_smooth_posture_head_v2'
INIT_SCHEMA='m2_act_zero_relative_final_linear_initialization_v1'
INSET_RAD=.001
SOFTNESS_RAD=.001
SOFTNESS_RAD_V2=.010
PROFILE_SOURCE=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act-auxiliary-development-20261106-v1/smokes/p001120-l3.json',
    bytes=109449,sha256='cc5eda59b9b12f92c19a6d97065db761a19de64513f10ca697483bcaf802a2e7')


def make_spec(profile,sources,normalization,*,version=1):
    """Bind the entire original posture profile and real native normalization."""
    need(type(version) is int and version in (1,2),'Explicit supported head version required')
    value=dict(schema=SCHEMA_V2 if version==2 else SCHEMA,parameterization='smooth_separable_absolute_joint_box',
        physical_posture_profile=deepcopy(profile),profile_sources=deepcopy(sources),
        inset_rad=INSET_RAD,softness_rad=SOFTNESS_RAD_V2 if version==2 else SOFTNESS_RAD,normalization=deepcopy(normalization),
        guarantee='target_separable_joint_box_only',input='current_normalized_observation.state',
        output='normalized_fixed_observed_origin_relative_H20',additional_trainable_parameters=0)
    validate_spec(value)
    return value


def validate_spec(value):
    schema=value.get('schema') if isinstance(value,dict) else None
    softness=SOFTNESS_RAD_V2 if schema==SCHEMA_V2 else SOFTNESS_RAD
    need(isinstance(value,dict) and schema in (SCHEMA,SCHEMA_V2) and
         value.get('inset_rad')==INSET_RAD and value.get('softness_rad')==softness,
         'Explicit versioned ACT head with unchanged 1mrad inset required')
    p=value.get('physical_posture_profile',{});low=np.asarray(p.get('joint_lower_rad'),float);high=np.asarray(p.get('joint_upper_rad'),float)
    need(p.get('schema')=='low_elbow_posture_v1' and low.shape==high.shape==(7,) and
         np.isfinite(low).all() and np.isfinite(high).all() and np.all(high-low>2*INSET_RAD),
         'Original finite nonempty posture joint box required')
    sources=value.get('profile_sources');need(isinstance(sources,list) and bool(sources),'Bound original posture sources required')
    for ref in sources:rt._reference_shape(ref)
    rt.validate_normalization(value.get('normalization',{}))
    expected=dict(schema=schema,parameterization='smooth_separable_absolute_joint_box',
        physical_posture_profile=p,profile_sources=sources,inset_rad=INSET_RAD,softness_rad=softness,
        normalization=value['normalization'],guarantee='target_separable_joint_box_only',
        input='current_normalized_observation.state',output='normalized_fixed_observed_origin_relative_H20',
        additional_trainable_parameters=0)
    need(value==expected,'ACT head definition metadata changed')
    return value


def pinned_spec(normalization,*,version=1):
    config=rt.read_bound(PROFILE_SOURCE)
    return make_spec(config['posture_limits'],[PROFILE_SOURCE],normalization,version=version)


def validate_bound_spec(value,normalization,*,version=1):
    need(validate_spec(value)==pinned_spec(normalization,version=version),'Exact original posture source/profile and real native stats required')
    return value


def bounded_targets(x,low,high,tau):
    """Symmetric softplus difference, with safe inactive branches at extremes.

    Each branch sees only its half-domain; this avoids inf-inf and cancellation
    even for maximum finite float inputs. No target projection is performed.
    """
    import torch
    from torch.nn.functional import softplus
    midpoint=(low+high)/2
    left=torch.minimum(x,midpoint);right=torch.maximum(x,midpoint)
    lower=low+tau*(softplus((left-low)/tau)-softplus((left-high)/tau))
    upper=high-tau*(softplus((high-right)/tau)-softplus((low-right)/tau))
    return torch.where(x<=midpoint,lower,upper)


def _transform(raw,batch,spec):
    import torch
    need(raw.ndim==3 and tuple(raw.shape[1:])==(20,7),'Single ACT forward H20x7 output required')
    observed=batch['observation.state']
    need(tuple(observed.shape)==(len(raw),7),'Current normalized observed q required')
    stats=spec['normalization'];p=spec['physical_posture_profile']
    tensor=lambda value:torch.as_tensor(value,dtype=raw.dtype,device=raw.device)
    q=observed*tensor(stats['state_std'][:7])+tensor(stats['state_mean'][:7])
    mean=tensor(stats['action_mean']);std=tensor(stats['action_std'])
    absolute=raw*std+mean+q[:,None,:]
    bounded=bounded_targets(absolute,tensor(p['joint_lower_rad'])+spec['inset_rad'],
        tensor(p['joint_upper_rad'])-spec['inset_rad'],spec['softness_rad'])
    return (bounded-q[:,None,:]-mean)/std,absolute,bounded


def transform(raw,batch,spec):
    """Current observed q plus the one raw ACT prediction, never action labels."""
    return _transform(raw,batch,spec)[0]


def install(policy,spec):
    """Install once on official policy.model; no new parameters/state dict keys."""
    validate_spec(spec);need(not hasattr(policy,'_posture_head_spec'),'ACT posture head already installed; install exactly once')
    policy._posture_head_spec=deepcopy(spec)
    policy._posture_head_diagnostics=dict(forward_calls=0,prehead_min_rad=None,prehead_max_rad=None,max_displacement_rad=0.)
    def hook(model,args,kwargs,output):
        batch=args[0] if args else kwargs['batch']
        need(isinstance(output,tuple) and len(output)==2,'Official ACT forward output required')
        normalized,absolute,bounded=_transform(output[0],batch,policy._posture_head_spec)
        # Detached diagnostics from this same pass; never run the model again.
        d=policy._posture_head_diagnostics;d['forward_calls']+=1
        lo=float(absolute.detach().min().cpu());hi=float(absolute.detach().max().cpu())
        d['prehead_min_rad']=lo if d['prehead_min_rad'] is None else min(d['prehead_min_rad'],lo)
        d['prehead_max_rad']=hi if d['prehead_max_rad'] is None else max(d['prehead_max_rad'],hi)
        d['max_displacement_rad']=max(d['max_displacement_rad'],float((bounded-absolute).detach().abs().max().cpu()))
        return normalized,output[1]
    policy._posture_head_handle=policy.model.register_forward_hook(hook,with_kwargs=True)
    return policy


def diagnostics(policy):
    return deepcopy(policy._posture_head_diagnostics)


def audit_targets(relative,mask,histories,spec):
    """Reject unrepresentable labels; retain labels verbatim and ignore padding."""
    validate_spec(spec);a=np.asarray(relative);pad=np.asarray(mask);h=np.asarray(histories)
    need(a.ndim==3 and a.shape[1:]==(20,7) and pad.shape==a.shape[:2] and pad.dtype==bool and
         h.shape==(len(a),2,17),'Explicit fixed-origin labels/mask/H2 required')
    absolute=(a+h[:,-1,None,:7])[~pad];need(len(absolute)>0 and np.isfinite(absolute).all(),'Finite valid target labels required')
    p=spec['physical_posture_profile'];low=np.asarray(p['joint_lower_rad'])+spec['inset_rad'];high=np.asarray(p['joint_upper_rad'])-spec['inset_rad']
    margin=float(min(np.min(absolute-low),np.min(high-absolute)))
    need(margin>0,'Oracle/native labels must be representable strictly inside the model inset box; do not clip labels')
    return dict(valid_cells=len(absolute),minimum_inset_margin_rad=margin,labels_clipped=False)


def validate_runtime_head(complete,config):
    """Shared DEV/paired check. Original physical profile/gates remain exact."""
    if complete.get('schema') not in ('m2_act_auxiliary_act_pilot_v2','m2_act_auxiliary_act_pilot_v3','m2_act_auxiliary_act_pilot_v4'):return
    spec=complete.get('protocol',{}).get('posture_head',{})
    if complete['schema'] in ('m2_act_auxiliary_act_pilot_v3','m2_act_auxiliary_act_pilot_v4'):validate_bound_spec(spec,spec.get('normalization',{}),version=2)
    else:validate_bound_spec(spec,spec.get('normalization',{}))
    need(config.get('posture_limits')==spec['physical_posture_profile'],
         'Runtime original posture profile must exactly match the trained model head')


def initialization_spec(spec):
    validate_spec(spec);stats=spec['normalization']
    return dict(schema=INIT_SCHEMA,training_only=True,layer='model.action_head',weight=0.,
        bias=(-np.asarray(stats['action_mean'])/np.asarray(stats['action_std'])).tolist(),
        other_parameters='unchanged original MODEL_SEED initialization',model_seed=rt.MODEL_SEED)


def initialize_training(policy,spec):
    """Once, before optimization only: zero raw relative action at every cell.

    All other seeded backbone parameters remain unchanged. Inference installation
    and checkpoint loading never call this function or reset learned parameters.
    """
    import torch
    need(not hasattr(policy,'_posture_training_initialization'),'Training final linear already initialized; exactly once required')
    value=initialization_spec(spec);layer=policy.model.action_head
    need(isinstance(layer,torch.nn.Linear) and layer.out_features==7 and layer.bias is not None,
         'Original official final action linear layer required')
    with torch.no_grad():
        layer.weight.zero_();layer.bias.copy_(torch.as_tensor(value['bias'],dtype=layer.bias.dtype,device=layer.bias.device))
    policy._posture_training_initialization=deepcopy(value)
    return value

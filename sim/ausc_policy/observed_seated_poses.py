"""Bounded observed-SMPL numeric pose candidates, not seated/physical labels.

The real CLI is Linux CPU only. It forwards each identity's own observed
body_pose, global_orient and transl without authoring angles, rotating/placing
meshes or leveling the torso. Canonical shape is optional and explicitly bound
to frozen identity quality. Original per-frame betas are always retained.

Classification uses actual posed SMPL hip/knee/ankle geometry in the measured
anatomical frame. This is rotation-invariant local geometry: some squats or
bends may pass, source gravity is unverified, and seat support, placement, IK,
clothing, reachability and physics are not evaluated. All qualifications remain
false. Parameters and anchors are written as JSON; mesh arrays are never saved.

Each identity gets one deterministic hash-ranked frame pool across all its
sequences, with a hard evaluation cap including model/geometry failures.
``none_found_in_bounded_scan`` never asserts that the identity has no seated
pose. Injected tiny CPU models are supported solely for contract tests; actual
SMPL/assets/cost and anatomical validity require a separate Linux CPU pilot.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np

from . import body_placement, m2_identity, surface_anchors

SCHEMA = 'observed_smpl_numeric_pose_candidates_v1'
PARAMETERS = {'global_orient':3, 'body_pose':69, 'betas':10, 'transl':3}
FLAGS = dict(training_eligible=False, physics_qualified=False, placement_qualified=False,
             expert_qualified=False, seat_support_qualified=False, ik_qualified=False)
LIMITATIONS = ('Numeric anatomical geometry candidate only: some squat or bending poses may pass; '
    'source gravity and semantic seated labels are unverified. No seat support, global placement, '
    'torso leveling, clothing, reachability, IK or physical qualification is established.')


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _hash(value,label):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Explicit SHA256 required for '+label)
    return value


def _bound_json(path,expected,label):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=_hash(expected,label):raise ValueError('Source hash changed: '+label)
    return json.loads(data)


def _identity(value):
    if not isinstance(value,str) or not re.fullmatch(r'p\d+',value):raise ValueError('Invalid identity')
    return value


def require_linux_cli():
    if sys.platform!='linux':raise RuntimeError('Real observed-pose generation requires Linux CPU; no Mac dataset/model reads')


def classify_numeric_seated(joints):
    """Local anatomical leg geometry only; never a gravity or seat-support test."""
    values=np.asarray(joints)
    if (values.ndim!=2 or values.shape[1]!=3 or len(values)<24 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):raise ValueError('Finite actual SMPL joint geometry required')
    values=values.astype(float,copy=False)
    frame=body_placement.anatomical_frame(values)
    local=(values-values[0])@frame
    pitch=[];flexion=[];forward=[]
    for hip,knee,ankle in ((1,4,7),(2,5,8)):
        thigh=local[knee]-local[hip];shin=local[ankle]-local[knee]
        lengths=np.array([np.linalg.norm(thigh),np.linalg.norm(shin)])
        if np.any(lengths<1e-8):raise ValueError('Degenerate hip/knee/ankle geometry')
        a,b=thigh/lengths[0],shin/lengths[1]
        pitch.append(float(np.degrees(np.arcsin(np.clip(a[1],-1,1)))))
        flexion.append(float(np.degrees(np.arccos(np.clip(a@b,-1,1)))))
        forward.append(float(thigh[2]))
    passed=max(np.abs(pitch))<=45. and min(flexion)>=35. and max(flexion)<=145. and min(forward)>0.
    return dict(numeric_criterion_passed=bool(passed),thigh_pitch_deg=pitch,knee_flexion_deg=flexion,
        thigh_forward_m=forward,torso_axis_in_source_coordinates=frame[:,1].tolist(),
        criterion=dict(max_abs_thigh_pitch_deg=45.,min_knee_flexion_deg=35.,max_knee_flexion_deg=145.,
                       both_thighs_forward=True),coordinate_basis='anatomical left/torso-up/front; source gravity unverified')


class _SMPLCPUAdapter:
    def __init__(self,path):
        require_linux_cli()
        import torch
        import smplx
        torch.set_num_threads(2)
        self.torch=torch
        self.model=smplx.SMPL(model_path=str(path),gender='neutral',batch_size=1,num_betas=10).to('cpu').eval()

    def __call__(self,**inputs):
        torch=self.torch
        tensors={key:torch.tensor(value,dtype=torch.float32,device='cpu').reshape(1,-1)
                 for key,value in inputs.items()}
        with torch.no_grad():result=self.model(**tensors,return_verts=True)
        return dict(vertices=result.vertices[0].cpu().numpy(),joints=result.joints[0].cpu().numpy(),
                    faces=np.asarray(self.model.faces))


def _identity_sources(split,quality,split_sha,expected_split,selected):
    splits=split.get('splits',{})
    if set(splits)!={'train','validation','test'} or expected_split not in splits:
        raise ValueError('Explicit train/validation/test frozen split required')
    all_ids=[]
    for identities in splits.values():
        if not isinstance(identities,list):raise ValueError('Invalid frozen identity split')
        all_ids.extend(_identity(value) for value in identities)
    if len(set(all_ids))!=len(all_ids):raise ValueError('Identity leaks across frozen splits or repeats')
    if quality.get('identity_split_sha256')!=split_sha:raise ValueError('Quality references a different split hash')
    eligible=quality.get('eligible_shape_betas',{})
    quarantined=quality.get('quarantined_identities',[])
    if (not isinstance(eligible,dict) or not isinstance(quarantined,list) or
            not set(eligible)<=set(all_ids) or not set(quarantined)<=set(all_ids) or
            set(eligible)&set(quarantined)):
        raise ValueError('Invalid shape-eligible/quarantined identity quality')
    for value in eligible.values():m2_identity.canonical_shape([value])
    if selected is None:requested=sorted(splits[expected_split])
    else:
        if (not isinstance(selected,(list,tuple)) or not selected or len(set(selected))!=len(selected) or
                any(identity not in splits[expected_split] for identity in selected)):
            raise ValueError('Selected identities must be unique and belong to expected split')
        if any(identity not in eligible for identity in selected):raise ValueError('Selected identity is not shape eligible')
        requested=sorted(selected)
    return requested,{identity:np.asarray(eligible[identity],float) for identity in requested if identity in eligible},set(all_ids)


def _inventory_records(inventory,root,all_ids):
    if not isinstance(inventory,list):raise ValueError('Source inventory must be the frozen sequence list')
    records=[];seen=set()
    for record in inventory:
        identity=_identity(record['identity']);name=record['path']
        if not isinstance(name,str) or '\\' in name:raise ValueError('Invalid source inventory path')
        path=PurePosixPath(name)
        match=re.fullmatch(r'(p\d+)_a\d+',path.parts[0]) if len(path.parts)==2 else None
        if (path.is_absolute() or '..' in path.parts or str(path)!=name or match is None or
                path.parts[1]!='smpl_params.npz' or identity!=match[1] or identity not in all_ids or name in seen):
            raise ValueError('Inventory path/identity mismatch, duplicate or escape')
        if type(record['frames']) is not int or record['frames']<=0:raise ValueError('Positive observed frame count required')
        _hash(record['sha256'],'sequence')
        source=root.joinpath(*path.parts)
        if source.is_symlink() or source.parent.is_symlink() or root not in source.resolve().parents:
            raise ValueError('Sequence path escapes source root or is symlinked')
        seen.add(name);records.append(dict(record,source_path=source))
    return sorted(records,key=lambda item:item['path'])


def _load_sequence(record,canonical_beta):
    path=record['source_path']
    if _sha(path)!=record['sha256']:raise ValueError('Sequence hash changed: '+record['path'])
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:np.asarray(archive[key]).copy() for key in PARAMETERS}
    for key,width in PARAMETERS.items():
        value=arrays[key]
        if (value.shape!=(record['frames'],width) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()):
            raise ValueError('Invalid/nonfinite observed '+key+' in '+record['path'])
    if _sha(path)!=record['sha256']:raise ValueError('Sequence hash changed while loading')
    # Reuse the frozen inventory's cross-sequence shape consistency criterion;
    # do not silently trust a canonical beta from a different identity.
    m2_identity.canonical_shape([canonical_beta,np.median(arrays['betas'],axis=0)])
    return arrays


def generate_candidates(*,source_root,model_path,model_sha256,inventory_path,inventory_sha256,
                        split_path,split_sha256,quality_path,quality_sha256,anchor_path,anchor_sha256,
                        output,expected_split,candidates_per_identity=3,max_evaluations_per_identity=256,
                        seed=20260930,selected_identities=None,use_canonical_betas=True,model_adapter=None):
    """Generate bounded numeric candidates; no model/dataset use occurs on import.

    ``selected_identities`` enables an explicit small same-split cost pilot.
    If omitted, the requested denominator is the entire named split; excluded
    shape identities are reported. Each identity's evaluation cap is global
    across its sequences and includes failed forwards/invalid geometry.
    """
    if model_adapter is None:require_linux_cli()
    if (type(candidates_per_identity) is not int or not 1<=candidates_per_identity<=64 or
            type(max_evaluations_per_identity) is not int or
            not candidates_per_identity<=max_evaluations_per_identity<=4096 or
            type(seed) is not int or not 0<=seed<2**63 or type(use_canonical_betas) is not bool):
        raise ValueError('Bounded integer candidate/evaluation limits and explicit seed/shape option required')
    if _sha(model_path)!=_hash(model_sha256,'model'):raise ValueError('Model hash changed')
    inventory=_bound_json(inventory_path,inventory_sha256,'inventory')
    split=_bound_json(split_path,split_sha256,'split');quality=_bound_json(quality_path,quality_sha256,'quality')
    anchors=_bound_json(anchor_path,anchor_sha256,'anchor').get('anchors',{})
    if set(anchors)!=body_placement.TARGET_NAMES:raise ValueError('Exactly ten bound surface anchors required')
    requested,eligible,all_ids=_identity_sources(split,quality,split_sha256,expected_split,selected_identities)
    root=Path(source_root).resolve(strict=True)
    records=_inventory_records(inventory,root,all_ids)
    output=Path(output)
    if output.exists():raise FileExistsError('New candidate output directory required')
    bindings=dict(model_sha256=model_sha256,inventory_sha256=inventory_sha256,split_sha256=split_sha256,
                  quality_sha256=quality_sha256,anchor_sha256=anchor_sha256,
                  code_sha256={name:_sha(path) for name,path in dict(generator=__file__,
                    identity=m2_identity.__file__,anatomical_frame=body_placement.__file__,
                    surface_anchors=surface_anchors.__file__).items()})
    adapter=model_adapter if model_adapter is not None else _SMPLCPUAdapter(model_path)
    if not callable(adapter):raise ValueError('Callable CPU model adapter required')
    candidates=[];statuses=[]
    for identity in requested:
        if identity not in eligible:
            statuses.append(dict(identity=identity,status='skipped_shape_ineligible',evaluated_frames=0,
                                 available_frames=0,candidates=0,rejection_counts={},**FLAGS));continue
        own=[record for record in records if record['identity']==identity]
        pool=((record,index) for record in own for index in range(record['frames']))
        def rank(item):
            record,index=item
            return hashlib.sha256(f'{seed}\0{identity}\0{record["path"]}\0{index}'.encode()).digest()
        chosen=heapq.nsmallest(max_evaluations_per_identity,pool,key=rank)
        # Verify complete selected sequence arrays before consuming any frame;
        # a NaN in an unchosen frame cannot be silently hidden by sampling.
        arrays={record['path']:_load_sequence(record,eligible[identity]) for record in own
                if any(candidate['path']==record['path'] for candidate,index in chosen)}
        evaluated=accepted=0;rejected=Counter()
        for record,index in chosen:
            if accepted>=candidates_per_identity:break
            observed={key:arrays[record['path']][key][index].copy() for key in PARAMETERS}
            parameters={key:value.copy() for key,value in observed.items()}
            if use_canonical_betas:parameters['betas']=eligible[identity].copy()
            evaluated+=1
            try:body=adapter(**{key:value.copy() for key,value in parameters.items()})
            except Exception:rejected['model_forward_error']+=1;continue
            try:
                metrics=classify_numeric_seated(body['joints'])
                if not metrics['numeric_criterion_passed']:
                    rejected['numeric_criterion_failed']+=1;continue
                anchored=surface_anchors.evaluate_anchors(body['vertices'],body['faces'],anchors)
            except (ValueError,KeyError,TypeError,IndexError):
                rejected['invalid_geometry_or_anchors']+=1;continue
            candidates.append(dict(name=f'{identity}_numeric_candidate_{accepted:03d}',identity=identity,
                split=expected_split,source=dict(identity=identity,sequence_path=record['path'],
                    sequence_sha256=record['sha256'],frame_index=index),
                observed_parameters={key:value.tolist() for key,value in observed.items()},
                model_parameters={key:value.tolist() for key,value in parameters.items()},
                shape_source=dict(kind='frozen_identity_quality_canonical_beta' if use_canonical_betas else 'observed_frame_beta',
                                  identity=identity,quality_sha256=quality_sha256),
                bindings=bindings,metrics=metrics,anchors=anchored,
                global_placement_applied=False,torso_leveling_applied=False,
                injected_model_adapter=model_adapter is not None,**FLAGS))
            accepted+=1
        statuses.append(dict(identity=identity,status='numeric_candidates_found' if accepted else 'none_found_in_bounded_scan',
            evaluated_frames=evaluated,available_frames=sum(row['frames'] for row in own),
            bounded_pool_frames=len(chosen),candidates=accepted,rejection_counts=dict(rejected),**FLAGS))
    # The byte bindings remain explicit even when an injected test adapter is
    # used. Source bindings alone never establish real SMPL/anatomical accuracy.
    if _sha(model_path)!=model_sha256:raise ValueError('Model hash changed during evaluation')
    summary=dict(schema=SCHEMA,expected_split=expected_split,requested_identities=requested,
        denominators=dict(requested=len(requested),shape_eligible=len(eligible),selected=len(eligible),
                          skipped=len(requested)-len(eligible)),candidates=len(candidates),by_identity=statuses,
        parameters=dict(seed=seed,candidates_per_identity=candidates_per_identity,
                        max_evaluations_per_identity=max_evaluations_per_identity,use_canonical_betas=use_canonical_betas),
        selection='deterministic SHA256-ranked frames across each own identity; one global per-identity evaluation cap',
        bindings=bindings,injected_model_adapter=model_adapter is not None,
        model_input_dtype='adapter-defined CPU fixture' if model_adapter is not None else 'float32 CPU SMPL',
        coordinates='source SMPL global_orient and transl preserved; no global placement or torso leveling',
        limitations=LIMITATIONS,**FLAGS)
    manifest=dict(schema=SCHEMA,bindings=bindings,candidates=candidates,limitations=LIMITATIONS,
                  injected_model_adapter=model_adapter is not None,**FLAGS)
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    summary['candidate_manifest_sha256']=hashlib.sha256(manifest_bytes).hexdigest()
    summary_bytes=(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    output.mkdir(parents=True,exist_ok=False)
    (output/'candidate_manifest.json').write_bytes(manifest_bytes)
    (output/'summary.json').write_bytes(summary_bytes)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    for name in ('model','inventory','split','quality','anchor'):
        parser.add_argument('--'+name,dest=name+'_path',type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-split',choices=('train','validation','test'),required=True)
    parser.add_argument('--selected-identity',dest='selected_identities',action='append')
    parser.add_argument('--candidates-per-identity',type=int,default=3)
    parser.add_argument('--max-evaluations-per-identity',type=int,default=256)
    parser.add_argument('--seed',type=int,default=20260930)
    parser.add_argument('--observed-betas',dest='use_canonical_betas',action='store_false')
    args=parser.parse_args(argv);require_linux_cli()
    summary=generate_candidates(**vars(args))
    print(json.dumps({key:summary[key] for key in ('schema','denominators','candidates','candidate_manifest_sha256')},indent=2))


if __name__=='__main__':main()

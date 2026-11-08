"""Frozen observed selection-VAL subset and float64 TCP candidate selection.

Teacher FK is computed once, before optimization. Candidate verification reads
the same stored Official raw predictions, never a rounded CPU re-inference.
The score is diagnostic model selection, not a training loss or physical gate.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import m2_given_contact_runtime as rt
from . import m2_given_contact_expanded_pilot_train as expanded
from . import m2_measured_recovery_data as measured
from . import m2_act_refinement_data as refinement
from . import tool_protrusion
from sim.auscultation_expert_layout_v1 import kinematics, tool_frames

need=rt.need
ARRAY_KEYS=measured.ARRAY_KEYS
BASE29=deepcopy(measured.BASE29)
BASE_COUNTS=deepcopy(measured.BASE_COUNTS)
NORMALIZATION=deepcopy(refinement.NORMALIZATION)
SUBSET_SCHEMA='m2_act_refinement_observed_validation_subset_v9'
SCORE_SCHEMA='m2_act_refinement_task_space_score_v9'
SELECTION_SCHEMA='m2_act_refinement_task_space_selection_v9'
CANDIDATE_STEPS=list(range(500,20001,500))
TARGETS=('R3','L3')
RULE='observed_actual_POINT_prepose_error_le_30mm'
METRIC=dict(translation_scale_m=.001,rotation_scale_rad=float(np.deg2rad(4)),
    reduction='per_target_valid_pose_cell_mean_then_max_R3_L3',compression_m=0.,tcp_offset_m=.132,
    dtype='float64',padded_cells='excluded',raw_H20_joint_MAE='diagnostic_only')


def _geometry_sources():
    return dict(kinematics=rt.reference(kinematics.__file__),urdf=rt.reference(kinematics.DEFAULT_URDF_PATH),
        geometry=rt.reference(rt.geometry.__file__),protrusion=rt.reference(tool_protrusion.__file__),
        tool_frames=rt.reference(tool_frames.__file__))


def _kinematics():
    return kinematics.OpenArmKinematics(contact_frame=rt.geometry.contact_frame)


def _require_kinematics(kin):
    need(type(kin) is kinematics.OpenArmKinematics and kin.urdf_path==kinematics.DEFAULT_URDF_PATH and
        kin.contact_frame is rt.geometry.contact_frame,'Exact bound float64 FK URDF and geometry_epoch.contact_frame required')
    return kin


def _sources(header):
    episodes=[e for e in header['episodes'] if e.get('split')=='validation'];cursor=0;seen=set();result=[]
    for e in episodes:
        lo,hi=e['split_sample_range'];eid=e['episode_id']
        need(lo==cursor and type(hi) is int and hi>lo and hi-lo==e['windows'] and eid not in seen,
            'Complete unique contiguous original selection-VAL episode ranges required')
        config=rt.read_bound(e['config']);run=rt.read_bound(e['run_result'])
        route=config.get('m2_given_contact_control',{});meta=route.get('metadata',{})
        need(all(meta.get(k)==e[k] for k in ('subject_id','split','target_id','episode_id')) and route.get('episode_id')==eid,
            'Selection-VAL actual configuration identity differs')
        rt.geometry.require_current(meta)
        points=[p for p in run.get('points',[]) if p.get('target')==e['target_id']]
        need(len(points)==1,'Exactly one actual selection-VAL POINT required')
        point=points[0];gap=point.get('phase_config',{}).get('precontact_gap_m')
        need(type(gap) in (int,float) and np.isfinite(gap) and gap>0 and point.get('checked_prepose_gap_m')==gap,
            'Actual POINT phase/checked PREPOSE gap must agree; default phase gap forbidden')
        source=dict(episode=eid,body=e['subject_id'],target=e['target_id'],split_sample_range=[lo,hi],
            actual_gap_m=float(gap),config=deepcopy(e['config']),report=deepcopy(e['run_result']),geometry=deepcopy(meta))
        if 'given_reference' in meta:
            reference=rt.pose.require_route(meta);fresh=reference.snapshot()['sources']['fresh_plan']
            ref=rt.reference(fresh['file']);plan=rt.read_bound(ref)
            need(ref['sha256']==fresh['sha256'] and plan.get('name',plan.get('target'))==e['target_id'] and
                plan.get('free_approach_start_gap_m')==gap,'Selection source actual gap/canonical plan differs')
            source['canonical_plan']=ref
        result.append(source);seen.add(eid);cursor=hi
    need(cursor==header['counts']['validation'],'Every original selection-VAL window required')
    return result


def _samples(samples):
    rt.given.require_samples(samples)
    n=len(samples['states']);shapes=dict(states=(n,17),histories=(n,2,17),actions=(n,20,7),origins=(n,7),
        velocities=(n,7),observation_times=(n,),action_is_pad=(n,20))
    need(all(np.asarray(samples[k]).shape==shape for k,shape in shapes.items()) and
        all(np.asarray(samples[k]).dtype.kind in 'fiu' and np.isfinite(samples[k]).all() for k in ARRAY_KEYS if k!='action_is_pad') and
        np.array_equal(samples['states'],samples['histories'][:,1]) and
        np.array_equal(samples['origins'],samples['states'][:,:7]),'Complete finite original H2/fixed-origin arrays required')
    mask=rt.shared.valid_action_mask(samples)
    need(mask.shape==(n,20) and mask.any(axis=1).all(),'Every original window needs actual unpadded future cells')
    return {k:measured.array_proof(samples[k]) for k in ARRAY_KEYS}


def _population(dataset_ref,samples):
    need(dataset_ref==BASE29,'Only unchanged frozen Base29 selection-VAL allowed')
    header=rt.read_bound(dataset_ref)
    need(header.get('counts')==BASE_COUNTS and header.get('normalization')==NORMALIZATION,
        'Original8373 TRAIN/2345 selection-VAL and unchanged2173B normalization required')
    rt.validate_normalization(rt.read_bound(header['normalization']))
    sources=_sources(header);proofs=_samples(samples);n=header['counts']['validation']
    need(len(samples['states'])==n,'Complete original native selection-VAL array population required')
    index=rt.read_bound(header['index']);items=[w for w in index.get('samples',[]) if w.get('split')=='validation']
    need(len(items)==n and [w.get('sample_index') for w in items]==list(range(n)),
        'Exact ordered original selection-VAL indices required')
    valid=rt.shared.valid_action_mask(samples);rows=[]
    for source in sources:
        lo,hi=source['split_sample_range']
        for i in range(lo,hi):
            need(items[i].get('episode_id')==source['episode'] and
                items[i].get('window',{}).get('valid_future_cells')==int(valid[i].sum()),'Selection index source/mask differs')
            error=float(np.linalg.norm(np.asarray(samples['states'][i,7:10],dtype=np.float64)+[0,0,source['actual_gap_m']]))
            if source['target'] in TARGETS and error<=.030:
                rows.append(dict(index=i,episode=source['episode'],target=source['target'],actual_gap_m=source['actual_gap_m'],
                    observed_prepose_error_m=error,valid_cells=int(valid[i].sum())))
    counts={t:sum(r['target']==t for r in rows) for t in TARGETS}
    need(all(n>0 for n in counts.values()),'Both R3 and L3 observed30mm selection subsets must be nonempty')
    cells=[[r['index'],k] for r in rows for k in np.flatnonzero(valid[r['index']]).tolist()]
    return dict(schema=SUBSET_SCHEMA,dataset=deepcopy(dataset_ref),index=deepcopy(header['index']),
        validation_arrays=deepcopy(header['arrays']['validation']),normalization=deepcopy(header['normalization']),
        array_proofs=proofs,normalization_sha256=rt.digest(rt.read_bound(header['normalization'])),
        sources=sources,original_window_order_sha256=rt.digest(items),indices=[r['index'] for r in rows],rows=rows,
        target_window_counts=counts,pose_cells=cells,selection_rule=RULE,metric=deepcopy(METRIC),
        geometry_sources=_geometry_sources(),selector_source=rt.reference(__file__),
        label_protocol=rt.shared.MASKED_LABEL_PROTOCOL,subset_uses='observations_and_target_identity_only',
        outcome_used=False,training_labels_used_for_subset=False,known4_used=False,fresh_used=False)


def _pose(kin,q):
    q=np.asarray(q,dtype=np.float64)
    need(q.shape==(7,) and np.isfinite(q).all() and np.all(q>=kin.lower_limits) and np.all(q<=kin.upper_limits),
        'Nonfinite/out-of-range unpadded joint prediction or target')
    pose=np.asarray(kin.tcp(q,compression=0.),dtype=np.float64)
    need(pose.shape==(4,4) and np.isfinite(pose).all(),'Finite float64 TCP transform required')
    return pose


def freeze_subset(dataset_ref,output):
    """Call before optimizer construction; returns the immutable SUBSET ref."""
    need(dataset_ref==BASE29,'Only unchanged Base29 native selection-VAL allowed')
    output=Path(output).resolve();need(not output.exists(),'Fresh pre-optimizer subset output required')
    prepared=expanded.load_dataset(dataset_ref);samples=prepared['validation'];value=_population(dataset_ref,samples)
    kin=_require_kinematics(_kinematics());teacher=np.asarray([_pose(kin,np.asarray(samples['origins'][i],dtype=np.float64)+
        np.asarray(samples['actions'][i,k],dtype=np.float64)) for i,k in value['pose_cells']],dtype=np.float64)
    need(value['geometry_sources']==_geometry_sources(),'FK/geometry sources changed during teacher cache computation')
    need(value==_population(dataset_ref,samples),'Source/index/arrays changed during teacher cache computation')
    output.mkdir(parents=True)
    with (output/'TEACHER_TCP.npy').open('xb') as stream:np.save(stream,teacher,allow_pickle=False)
    value.update(teacher_cache=rt.reference(output/'TEACHER_TCP.npy'),teacher_cache_proof=measured.array_proof(teacher))
    return rt.write_new(output/'SUBSET.json',value)


def _bound_subset(subset,samples):
    need(isinstance(subset,dict) and set(subset)=={'path','bytes','sha256'} and
        type(subset.get('bytes')) is int and subset['bytes']>0,'Full independently frozen subset file reference required; inline subsets forbidden')
    rt._reference_shape(subset);path=Path(subset['path'])
    need(path==path.resolve(),'Canonical frozen subset path required')
    refinement._regular_unaliased(path)
    value=rt.read_bound(subset)
    expected=_population(value.get('dataset'),samples)
    need(set(value)==set(expected)|{'teacher_cache','teacher_cache_proof'} and
        all(value.get(k)==v for k,v in expected.items()),'Frozen observed indices/gaps/geometry/source/array/mask subset differs')
    with rt.bound_path(value['teacher_cache']).open('rb') as stream:teacher=np.load(stream,allow_pickle=False)
    need(teacher.dtype==np.float64 and teacher.shape==(len(value['pose_cells']),4,4) and np.isfinite(teacher).all() and
        measured.array_proof(teacher)==value['teacher_cache_proof'],'Frozen teacher FK cache changed')
    return value,teacher


def score(predictions,samples,subset,*,kin=None):
    """Score against the independently frozen full SUBSET reference.

    Runtime FK is always a fresh internal canonical instance. The retained
    keyword must be None; mutable caller-provided FK instances are forbidden.
    """
    need(kin is None,'External kinematics instances forbidden; use the internal canonical FK')
    value,teacher=_bound_subset(subset,samples);prediction=np.asarray(predictions)
    need(prediction.shape==samples['actions'].shape and prediction.dtype.kind in 'fiu',
        'Complete original ordered H20 joint prediction array required')
    # Match runtime.save_predictions exactly. Expanding Official float32 values
    # to float64 is lossless; neither rounding nor a new model inference occurs.
    prediction=np.ascontiguousarray(prediction,dtype='<f8')
    valid=rt.shared.valid_action_mask(samples)
    need(np.isfinite(prediction[valid]).all(),'Nonfinite unpadded predictions cannot be omitted')
    kin=_require_kinematics(_kinematics())
    sums={t:0. for t in TARGETS};counts={t:0 for t in TARGETS};targets={r['index']:r['target'] for r in value['rows']}
    # Validate every original unpadded predicted pose, including rows outside the
    # observed subset. An invalid cell is never dropped to improve a candidate.
    for i,k in zip(*np.nonzero(valid)):
        q=np.asarray(samples['origins'][i],dtype=np.float64)+np.asarray(prediction[i,k],dtype=np.float64)
        need(np.isfinite(q).all() and np.all(q>=kin.lower_limits) and np.all(q<=kin.upper_limits),
            'Out-of-range unpadded prediction anywhere in native validation')
    for (i,k),truth in zip(value['pose_cells'],teacher):
        pose=_pose(kin,np.asarray(samples['origins'][i],dtype=np.float64)+np.asarray(prediction[i,k],dtype=np.float64))
        translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))/.001
        angle=float(Rotation.from_matrix(pose[:3,:3]@truth[:3,:3].T).magnitude())/METRIC['rotation_scale_rad']
        cell=max(translation,angle);need(np.isfinite(cell),'Nonfinite TCP score forbidden')
        target=targets[i];sums[target]+=cell;counts[target]+=1
    means={t:sums[t]/counts[t] for t in TARGETS}
    diagnostic=float(np.mean(np.abs(np.asarray(prediction[valid],dtype=np.float64)-np.asarray(samples['actions'][valid],dtype=np.float64))))
    need(value['geometry_sources']==_geometry_sources(),'FK/geometry sources changed during scoring')
    return dict(schema=SCORE_SCHEMA,task_score=max(means.values()),target_means=means,target_sums=sums,valid_cells=counts,
        raw_joint_mae_rad=diagnostic,subset_sha256=rt.digest(value),metric=deepcopy(METRIC),
        prediction_proof=measured.array_proof(prediction))


def select_candidate(rows):
    need(isinstance(rows,list) and [r.get('step') for r in rows]==CANDIDATE_STEPS,
        'All forty ordered future candidate steps required')
    for row in rows:
        result=row.get('score',row);number=result.get('task_score')
        need(type(number) in (int,float) and np.isfinite(number) and number>=0,'Finite nonnegative task score for every candidate required')
    return min(rows,key=lambda r:((r.get('score',r))['task_score'],r['step']))


def verify_candidates(selection):
    """Recompute all40 from the SAME stored Official prediction arrays/cache.

    `candidates` are ordered {step, predictions, score}; predictions may be a
    full immutable .npy reference or the original runtime.save_predictions
    binding. Exact dict/float equality is intentional, with no tolerance window.
    """
    need(selection.get('schema')==SELECTION_SCHEMA and selection.get('dataset')==BASE29,
        'Explicit V9 frozen task-space selection required')
    prepared=expanded.load_dataset(selection['dataset']);samples=prepared['validation']
    subset,_=_bound_subset(selection['subset'],samples);rows=selection.get('candidates')
    select_candidate(rows)
    for row in rows:
        binding=row.get('predictions',{})
        if 'reference' in binding:
            prediction=rt.load_predictions(binding,window_order_sha256=subset['original_window_order_sha256'])
        else:prediction=np.load(rt.bound_path(binding),allow_pickle=False)
        actual=score(prediction,samples,selection['subset'])
        need(actual==row.get('score'),'Stored Official candidate task score/proofs differ on exact recomputation')
    chosen=select_candidate(rows)
    need(selection.get('selected_step')==chosen['step'],'Selected step differs from exact earliest task minimum')
    return dict(selected_step=chosen['step'],task_score=chosen['score']['task_score'],verified_candidates=len(rows))

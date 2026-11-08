"""Fixed eight TRAIN live ACT prefixes followed by one teacher intervention.

The route is data collection only. A collection bootstrap and a physically
qualified capture never admit a learned controller to any benchmark.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import numpy as np

from . import m2_given_contact_runtime as rt, m2_given_contact_recovery as recovery
from . import m2_act_refinement_warm_start as warm, m2_given_contact_pilot_deployment as pipe
from . import m2_given_contact_dataset as data, m2_act_refinement_data as refinement
from . import m2_contract as contract, m2_stage as stage, m2_given_contact_handoff as handoff
from .m2_given_contact_controller import GivenContactExpert, GivenContactBoundary, GivenContactACT, GIVEN_TRAJECTORY_LIMITS
from .m2_given_contact_pose import GivenPoseProvider, require_route
from .retainer_contact import ALL_FORCE_GROUPS
from .m2_act_controller import M2ACTController
from .fresh_expert import FreshExpert
from sim.auscultation_expert_layout_v1.controller import DT, EpisodeRejected
from sim.auscultation_pi05_v6.execution import FinishedStationaryRoundoffError

need=rt.need
MODULE=__name__
ROUTE_KEY='m2_act_onpolicy_teacher_collection'
DEPLOYMENT_SCHEMA='m2_act_onpolicy_teacher_collection_deployment_v1'
PREPARATION_SCHEMA='m2_act_onpolicy_teacher_collection_prepared_v1'
MANIFEST_SCHEMA='m2_act_onpolicy_teacher_collection_manifest_v1'
BINDING_SCHEMA='m2_act_onpolicy_teacher_collection_binding_v1'
PROTOCOL_SCHEMA='m2_act_onpolicy_teacher_collection_cpu_pipe_v1'
CPU_EXIT_SCHEMA='m2_act_onpolicy_teacher_collection_cpu_exit_v1'
BOOTSTRAP_SIDECAR_SCHEMA='m2_act_onpolicy_teacher_collection_bootstrap_sidecar_v1'
BOOTSTRAP_FILENAME='collection_cpu_bootstrap.json'
SIDECAR_SCHEMA='m2_act_onpolicy_teacher_collection_native_v1'
LIVE_SCHEMA='m2_act_onpolicy_live_state_certificate_v1'
CAPTURE_BOUNDARY_SCHEMA='m2_act_onpolicy_capture_boundary_v1'
ADMISSION_SCHEMA='m2_act_onpolicy_teacher_collection_data_admission_v1'
_B='/opt/wam-runtime/fast-wam'
PREPARED=dict(path=_B+'/diagnostics/act10-dart-cpu-preparation-20261107-v2/COMPLETE.json',bytes=67617,
    sha256='6483d1022a54a3f6ea898b5475e695c91ea28ca413b8b31e04db009025c7acbd')
CPU_PLAN=dict(path=_B+'/diagnostics/act10-dart-cpu-preparation-20261107-v2/CPU_PLAN.json',bytes=33041,
    sha256='bf9dbe74ae835ecdc088c04b40c395f726e354f85fac20283960835a3f69b201')
SLOT_INDICES=(2,3,8,9,14,15,16,17)
_NOMINAL_HASHES=(
    (108923,'e149f7deed0422f80527ec9fa7daaa144ee9169f635ae22fa3ee7fc5c73f2336'),
    (108955,'d7b0c90373bbb34bb0f18b8c33677a2177dafd099be5a84fee2d156bd14ad692'),
    (108934,'afb7b2ff367254e87f22d66c4dc7d357df778b987e8fc3f01e721da20c72b46b'),
    (108950,'d8fd718a0c547842c13171b0828c31731f874db498c1ae10c06e470daea6a8b8'),
    (108943,'b8d7580b44b8cf0358adc76b943ae838cf8c170bfac4cb572c6cf4e62c866ee7'),
    (108954,'cb0f9855a0e5ed4358999e0e5ec036d9b164efb7ef99238e87acd6115d542fda'),
    (108950,'193592fcf7ad5bc2f4f81db3ae3ca0330e74c0ef47f5139c867d447d946c7f5b'),
    (108948,'4dad1ba677d56a328ca58eaa6d38a3d6418a6eb278cf448973cb457bbe0650c3'))
FLAGS=dict(collection_only=True,benchmark_act_eligible=False,expert_intervention=True,simulation_only=True,
    formal_acceptance=False,training_started=False,model_qualified=False)


def planned_requests():
    return [deepcopy(refinement.dart.planned_requests()[i]) for i in SLOT_INDICES]


def nominal_references():
    return [dict(path=_B+'/diagnostics/act10-dart-cpu-preparation-20261107-v1/nominal-configs/dart-'+str(i).zfill(2)+'.json',
        bytes=size,sha256=digest) for i,(size,digest) in zip(SLOT_INDICES,_NOMINAL_HASHES)]


def _prepared(ref):
    need(ref==PREPARED,'Exact original Source37 PREPARED full reference required before any payload read')
    value=rt.read_bound(ref);rows=value.get('candidates')
    need(value.get('schema')=='act10_dart_source37_cpu_authoring_v2' and value.get('complete') is True and
        value.get('planned')==20 and value.get('prepared')==17 and value.get('cpu_plan')==CPU_PLAN and
        isinstance(rows,list) and len(rows)==20,'Original fixed20/prepared17 Source37 preparation required')
    for row,request in zip(rows,refinement.dart.planned_requests()):
        need(row.get('request')==request and row.get('status') in ('prepared','cpu_unavailable'),
            'Original all20 TRAIN slot order/status required')
    for i,request,ref in zip(SLOT_INDICES,planned_requests(),nominal_references()):
        need(rows[i].get('status')=='prepared' and rows[i].get('request')==request and
            rows[i].get('nominal_source_config')==ref,'Exact prepared fixed8 original nominal references required')
    plan=refinement.dart.validate_precollection(rt.read_bound(CPU_PLAN))
    need([r['status'] for r in rows]==[r['status'] for r in plan['slots']],'Original CPU denominator changed')
    return value


def sources():
    root=Path(__file__).resolve().parents[2]
    names=('m2_act_onpolicy_collection','m2_act_refinement_data','m2_act_refinement_warm_start',
        'm2_given_contact_dataset','m2_given_contact_controller','m2_given_contact_pose','m2_given_contact_intent',
        'm2_given_contact_handoff','m2_given_contact_recovery','m2_given_contact_runtime','m2_given_contact_pilot_deployment',
        'm2_given_contact_deployment','m2_stop_residual_trajectory','m2_act_controller','m2_act','m2_contract','m2_stage',
        'fresh_expert','geometry_qualification','expert_capture','training_capture','m2_given_contact_evaluation')
    files={n+'.py':rt.reference(importlib.import_module('.'+n,__package__).__file__) for n in names}
    commit=(root/'CODE_COMMIT').read_text().strip() if (root/'CODE_COMMIT').exists() else subprocess.check_output(
        ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    return dict(root=str(root),code_commit=commit,files=files,
        source_manifest=rt.reference(root/'SOURCE_MANIFEST.json') if (root/'SOURCE_MANIFEST.json').exists() else None)


# One seven-DOF cache-recalculation row contains the actual event's four
# complete input snapshots plus its live interpolator. CPU producer evidence
# reaches 17,933 bytes; this format cap remains inside the existing byte budgets.
STOP_CAPTURE_RECORD_BYTES=65_536


def _capture_source_bindings():
    from . import m2_given_contact_evaluation as neutral
    return dict(collection_native_capture=rt.reference(__file__),neutral_native_snapshots=rt.reference(neutral.__file__))


def _projection(original,dep):
    result=deepcopy(original);route=result['m2_given_contact_control'];episode=dep['episode_id']
    need(episode!=route['episode_id'],'Collection requires new episode identity')
    route['episode_id']=episode;route['metadata']['episode_id']=episode
    capture=result['fresh_expert']['raw_multimodal_capture'];capture['episode_id']=episode
    capture['compact']['metadata']=deepcopy(route['metadata'])
    capture['compact']['code_bindings'].update({name:ref['sha256'] for name,ref in _capture_source_bindings().items()})
    if 'compact_scene_storage' in result:result['compact_scene_storage']['batch_id']=episode
    return result


def _validate_deployment(dep):
    need(isinstance(dep,dict) and set(dep)=={'schema','prepared','request','source_config','initialization','episode_id',
        'source_episode_id','metadata','source_inventory','runtime_config_sha256','model',*FLAGS} and
        dep.get('schema')==DEPLOYMENT_SCHEMA and all(dep.get(k) is v for k,v in FLAGS.items()),
        'Explicit collection-only deployment required')
    need(dep['request'] in planned_requests() and dep['prepared']==PREPARED,'Only original fixed8 TRAIN slots permitted')
    index=planned_requests().index(dep['request']);need(dep['source_config']==nominal_references()[index],
        'Exact fixed8 original nominal configuration reference required')
    meta=dep['metadata'];request=dep['request']
    need(all(meta.get(k)==request[k] for k in ('subject_id','split','target_id')) and meta.get('episode_id')==dep['episode_id'] and
        isinstance(dep['episode_id'],str) and dep['episode_id'].startswith('act11-onpolicy-'+request['slot_id']+'-'),
        'Collection TRAIN identity/order/episode differs')
    rt.shared.require_membership('train',meta['subject_id']);_prepared(dep['prepared'])
    initialization=warm.validate_receipt(dep['initialization'])
    need(dep['source_inventory']==sources() and dep['model']==dict(schema=warm.COLLECTION_SCHEMA,
        checkpoint=initialization['state_dict'],benchmark_model=False),'Collection bootstrap/current source differs')
    original=rt.read_bound(dep['source_config'])
    need(not any(k in original for k in (ROUTE_KEY,recovery.ROUTE_KEY,'m2_act_deployment','m2_given_contact_deployment',
        'm2_given_contact_pilot_deployment','m2_given_contact_expanded_pilot_deployment','m2_given_contact_evaluation')),
        'Original nominal teacher config cannot contain learned/recovery/benchmark routes')
    old=original.get('m2_given_contact_control',{});oldmeta=old.get('metadata',{})
    need(old.get('episode_id')==dep['source_episode_id'] and all(oldmeta.get(k)==request[k] for k in ('split','subject_id','target_id')),
        'Original nominal TRAIN identity differs')
    ref=require_route(meta);ref.verify_sources()
    patient=rt.pose.bound_json(ref.snapshot()['sources']['patient_receipt'])
    need(patient.get('identity')==request['subject_id'] and patient.get('split')=='train' and
        patient.get('proposal')==request['subject_id']+'_'+request['pose_id'],'Original actual body/pose differs')
    expected=_projection(original,dep)
    need(meta==expected['m2_given_contact_control']['metadata'] and dep['runtime_config_sha256']==rt.digest(expected),
        'Collection original metadata/projection differs')
    need(expected.get('physics')==recovery.PHYSICS and expected.get('isaac_python')==recovery.NATIVE_PYTHON and
        expected.get('geometry_id')==rt.geometry.GEOMETRY_ID and expected.get('geometry_source_sha256')==rt.geometry.GEOMETRY_SOURCE_SHA256,
        'Original Native60/64-iteration physics/geometry required')
    capture=expected['fresh_expert']['raw_multimodal_capture']
    need(capture.get('enabled') is True and capture.get('profile')=='compact' and capture.get('episode_id')==dep['episode_id'] and
        capture['compact']['metadata']==meta and expected['fresh_expert']['plans']==[ref.snapshot()['sources']['fresh_plan']] and
        expected['m2_given_contact_control'].get('handoff_policy_id')==handoff.POLICY_ID and meta.get('handoff_policy_id')==handoff.POLICY_ID,
        'Unchanged attached compact capture/reference/interior handoff required')
    from . import m2_stop_residual_trajectory as residual
    residual.native_options(expected,metadata=meta)
    return expected,initialization,ref


def preflight(config,*,output=None):
    dep=config.get(ROUTE_KEY);expected,initialization,ref=_validate_deployment(dep)
    need(config==dict(expected,**{ROUTE_KEY:dep}),'Collection altered original scene/physics/camera/config projection')
    return dict(schema=BINDING_SCHEMA,configuration=deepcopy(config),deployment=deepcopy(dep),metadata=deepcopy(dep['metadata']),
        reference=ref,initialization=initialization,config_sha256=rt.digest(config))


def validate_bootstrap(packet,binding):
    value=binding['initialization'];dep=binding['deployment']
    need(isinstance(packet,dict) and set(packet)=={'schema','collection_only','benchmark_model','optimizer_created','initialization',
        'normalization','posture_head','tensor_proofs','parent_model','historical_optimization','source_evidence','equivalence'} and
        packet.get('schema')==warm.COLLECTION_SCHEMA and packet.get('collection_only') is True and
        packet.get('benchmark_model') is False and packet.get('optimizer_created') is False and packet.get('initialization')==dep['initialization'] and
        all(packet.get(k)==value[k] for k in ('normalization','posture_head','tensor_proofs','parent_model','historical_optimization')) and
        packet.get('source_evidence',{}).get('parent')==warm.SOURCE41 and
        packet.get('source_evidence',{}).get('current',{}).get('helper')==rt.reference(warm.__file__),
        'Verified collection-only Source41/current bootstrap receipt required')
    source=packet['source_evidence'];current=source.get('current',{});software=current.get('software',{})
    need(set(source)=={'parent','current'} and set(current)=={'software','helper'} and isinstance(software,dict) and
        set(software)=={'schema','vendor','dependencies','python','prefix','python_version','source_bindings'} and
        software.get('schema')==rt.RUNTIME_SCHEMA and software.get('prefix')==rt.VENDOR_ROOT+'/.venv' and
        isinstance(software.get('dependencies'),list) and bool(software['dependencies']) and
        software.get('python_version') and software.get('source_bindings')=={
            str(Path(m.__file__).resolve()):rt.shared.sha256(m.__file__) for m in (rt.shared,rt.given,rt.pose,rt.handoff,rt)},
        'Complete original collection CPU software/current source inventory required')
    vendor=software.get('vendor',{});files=vendor.get('files',{})
    need(set(vendor)=={'root','upstream_commit','files'} and vendor.get('root')==rt.VENDOR_ROOT and
        vendor.get('upstream_commit')==rt.UPSTREAM_COMMIT and isinstance(files,dict) and
        len(files)==rt.VENDOR_PYTHON_FILES+2 and sum(n.endswith('.py') for n in files)==rt.VENDOR_PYTHON_FILES and
        {'pyproject.toml','uv.lock'}<=set(files),'Complete pinned collection CPU vendor inventory required')
    for name,ref in files.items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and
            ref.get('path')==str(Path(rt.VENDOR_ROOT)/name),'Collection CPU vendor file origin differs')
        rt._reference_shape(ref)
    eq=packet.get('equivalence',{})
    need(eq.get('schema')==warm.EQUIVALENCE_SCHEMA and eq.get('passed') is True and eq.get('collection_only') is True and
        eq.get('optimizer_created') is False and eq.get('initialization')==dep['initialization'] and
        eq.get('initial_tensor_proofs')==value['tensor_proofs'],'Actual imported54 tensor/equivalence proof required')
    return packet


def _bootstrap_artifact(ref,expected):
    need(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'} and type(ref.get('bytes')) is int and
        ref['bytes']>0 and ref.get('path')==str(expected),'Exact canonical collection bootstrap artifact location/full reference required')
    path=Path(ref['path'])
    need(path.is_absolute() and path.resolve()==path and all(not p.is_symlink() for p in (path,*path.parents)),
        'Collection bootstrap artifact aliases forbidden')
    info=path.lstat();need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'Regular unaliased collection bootstrap artifact required')
    return rt.read_bound(ref)


def load_bootstrap(ref,binding,*,output):
    """Hydrate the full receipt from its one exact case-output sidecar."""
    output=Path(output).resolve();value=_bootstrap_artifact(ref,output/BOOTSTRAP_FILENAME)
    need(isinstance(value,dict) and set(value)=={'schema','deployment','source_inventory','bootstrap'} and
        value.get('schema')==BOOTSTRAP_SIDECAR_SCHEMA and value.get('source_inventory')==binding['deployment']['source_inventory'],
        'Typed full collection bootstrap sidecar/source inventory required')
    deployment=_bootstrap_artifact(value['deployment'],output/'pilot_cpu_deployment.json')
    need(deployment==binding['deployment'],'Collection bootstrap sidecar bound deployment differs')
    return validate_bootstrap(value['bootstrap'],binding)


def validate_binding(binding):
    need(isinstance(binding,dict) and binding.get('schema')==BINDING_SCHEMA,'Authenticated collection binding required')
    actual=preflight(binding['configuration'])
    need(all(binding.get(k)==actual[k] for k in ('deployment','metadata','initialization','config_sha256')) and
        binding['reference'].sha256==actual['reference'].sha256,'Stale/wrong collection binding')
    if 'bootstrap' in binding:validate_bootstrap(binding['bootstrap'],actual)
    return binding


def prepare(prepared_ref,initialization_ref,output):
    prepared=_prepared(prepared_ref);initialization=warm.validate_receipt(initialization_ref)
    output=Path(output).resolve();need(not output.exists(),'New collection preparation directory required')
    inventory=sources();configs=[];stamp=rt.digest(dict(prepared=prepared_ref,initialization=initialization_ref,output=str(output)))[:12]
    # All eight identities above were admitted before reading nominal configs.
    for request,ref in zip(planned_requests(),nominal_references()):
        original=rt.read_bound(ref);episode='act11-onpolicy-'+request['slot_id']+'-'+stamp
        dep=dict(schema=DEPLOYMENT_SCHEMA,prepared=deepcopy(prepared_ref),request=request,source_config=ref,
            initialization=deepcopy(initialization_ref),episode_id=episode,
            source_episode_id=original['m2_given_contact_control']['episode_id'],metadata={},source_inventory=inventory,
            runtime_config_sha256='',model=dict(schema=warm.COLLECTION_SCHEMA,checkpoint=initialization['state_dict'],benchmark_model=False),**FLAGS)
        projected=_projection(original,dep);dep['metadata']=deepcopy(projected['m2_given_contact_control']['metadata'])
        dep['runtime_config_sha256']=rt.digest(projected);config=dict(projected,**{ROUTE_KEY:dep});preflight(config)
        configs.append((request,config))
    output.mkdir(parents=True);jobs=[];rows=[]
    for request,config in configs:
        ref=rt.write_new(output/(request['slot_id']+'.json'),config)
        jobs.append(dict(id=request['slot_id'],config=ref['path'],sha256=ref['sha256']))
        rows.append(dict(request=request,config=ref,episode_id=config[ROUTE_KEY]['episode_id']))
    batch=rt.write_new(output/'batch.json',dict(experiment='isaac_batch',base=configs[0][1]['base'],parallelism=1,
        time_limit_seconds=sum(c['time_limit_seconds'] for _,c in configs)+300,jobs=jobs,**FLAGS))
    return rt.write_new(output/'PREPARED.json',dict(schema=PREPARATION_SCHEMA,complete=True,prepared=prepared_ref,
        initialization=initialization_ref,source_inventory=inventory,fixed_denominator=8,attempts=rows,batch=batch,**FLAGS))


def validate_live_collection_query(query,reference,collection_binding):
    """Same original numeric predicates, admitting actual live FREE or APPROACH."""
    binding=validate_binding(collection_binding)
    witness=binding.get('live_witness',{})
    need(witness.get('schema')==LIVE_SCHEMA and witness.get('query')==query,
        'Query differs from certified actual live H2/origins/clock')
    need(query.get('metadata')==binding['metadata'] and reference.sha256==binding['reference'].sha256,
        'Live collection query/reference/binding differs')
    q=contract._array(query['measured_q'],(7,),'same-state measured q')
    dq=contract._array(query['measured_dq'],(7,),'same-state measured dq')
    history=contract._array(query['history'],(2,17),'real recovery history')
    times=contract._array(query['history_times'],(2,),'real recovery clocks')
    context=query['context'];stamp=query['time_s'];row=query['physics_row']
    need(isinstance(context,(tuple,list)) and len(context)==4,'Same-state context malformed')
    metadata=query['metadata'];need(metadata.get('split')=='train','Same-state recovery is TRAIN only')
    rt.shared.require_membership('train',metadata.get('subject_id'))
    need(require_route(metadata).sha256==reference.sha256 and context[0]==metadata['episode_id'],
        'Same-state metadata/reference/episode differs')
    need(query['reference_sha256']==reference.sha256 and type(context[2]) is int and context[2]==0 and
        context[3] in ('FREE','APPROACH') and context[1]==reference.snapshot()['target_id'],
        'Same-state reference/live FREE or APPROACH context differs')
    need(type(row) is int and row>=12 and row%12==0 and abs(stamp-row/240.)<=1e-8 and
        abs(times[1]-stamp)<=1e-8 and abs(times[1]-times[0]-.05)<=1e-8 and np.array_equal(history[-1,:7],q),
        'Same-state history/clock differs')
    need(np.max(abs(dq))<=.175,'Same-state measured velocity exceeds original limit')
    for frame in history:
        encoded=contract.encode_state(frame[:7],reference.pose_base,1.,metadata=metadata)
        need(np.allclose(encoded,frame,atol=1e-10,rtol=0),'Same-state physically inconsistent state17 history')
    scalars=np.asarray([query[k] for k in ('hall_N','physical_contact_N','compression_m')],float)
    need(np.isfinite(scalars).all() and 0<=scalars[0]<.2 and 0<=scalars[1]<.2 and scalars[2]<=.00015,
        'Recovery requires actual released noncontact state')
    return q


def propose_live_collection_recovery(query,trajectory,command_history,destination,*,reference,collection_binding,planner,verify_command):
    """Read-only final-prepose teacher oracle; no replay context or state reset."""
    result=dict(kind='abstain',physics_executed=False,training_eligible=False,query=deepcopy(query))
    original=handoff.input_snapshot(trajectory);output=handoff.output_snapshot(trajectory)
    objects=tuple(id(v) for v in (trajectory,trajectory.input,trajectory.output,trajectory.generator))
    try:
        measured=validate_live_collection_query(query,reference,collection_binding)
        witness=collection_binding['live_witness']
        need(witness.get('input')==original and witness.get('output')==output and
            np.array_equal(witness.get('command_history'),command_history) and
            witness.get('object_ids')==dict(zip(('trajectory','input','output','generator'),objects)),
            'Certified live retained command/history/input/output/objects differ')
        goal=contract._array(destination,(7,),'authenticated FINAL destination')
        canonical=np.asarray(rt.pose.bound_json(reference.snapshot()['sources']['fresh_plan']).get('transfer_q'),float)
        need(canonical.ndim==2 and canonical.shape[1:]==(7,) and len(canonical)>=2 and np.isfinite(canonical).all() and
            np.array_equal(goal,canonical[-1]),'Live collection destination must be authenticated FINAL prepose')
        retained=np.asarray(original['fields']['current_position']);deadline=time.monotonic()+recovery.MAX_PLAN_CPU_S
        for q in (retained,measured,goal):verify_command(q)
        route,report=planner(measured.copy(),goal.copy(),deadline)
        need(time.monotonic()<deadline and route is not None,'Collection recovery planner abstained or deadline: '+str(report))
        route=np.asarray(route,float)
        need(route.ndim==2 and route.shape[1:]==(7,) and len(route)>=2 and np.isfinite(route).all() and
            np.array_equal(route[0],measured) and np.array_equal(route[-1],goal),'Recovery route endpoints differ')
        planner_route=route.copy();attempts=[];preview=None
        for candidate in recovery.recovery_route_candidates(retained,route,all_joints=True,goal_near_detours=True):
            candidate=candidate[np.r_[True,np.max(abs(np.diff(candidate,axis=0)),axis=1)>1e-12]]
            if len(candidate)==1:candidate=np.vstack([candidate,candidate])
            try:
                recovery._preview(trajectory,command_history,candidate[1:],lambda q:None,settle=True,deadline=deadline)
            except FinishedStationaryRoundoffError as error:
                attempts.append(dict(numerical_proposal_accepted=False,reason=str(error),details=deepcopy(getattr(error,'details',{})),physics_executed=False))
                continue
            preview=recovery._preview(trajectory,command_history,candidate[1:],verify_command,settle=True,deadline=deadline)
            route=candidate;break
        result['numeric_proposal_rejections']=attempts
        need(preview is not None,'Every collection recovery numerical proposal was rejected')
        result.update(kind='recovery_proposal',route=route.tolist(),planner_route=np.asarray(planner_route).tolist(),planner=report,preview=preview,
            destination_q=goal.tolist(),query_sha256=rt.digest(query))
    except (ValueError,KeyError,TypeError,RuntimeError,TimeoutError) as error:
        result.update(kind='abstain',reason=type(error).__name__+': '+str(error),details=deepcopy(getattr(error,'details',{})))
    finally:
        need(handoff.input_snapshot(trajectory)==original and handoff.output_snapshot(trajectory)==output and
            tuple(id(v) for v in (trajectory,trajectory.input,trajectory.output,trajectory.generator))==objects,
            'Collection teacher preview changed live trajectory/input/output/objects')
    return result


class OnPolicyTeacherCollection(GivenContactExpert):
    collection_route=ROUTE_KEY

    def __init__(self,*args,predictor,bootstrap_receipt,bootstrap_ref,collection_binding,**kwargs):
        binding=validate_binding(collection_binding);validate_bootstrap(bootstrap_receipt,binding)
        env=args[0] if args else kwargs.get('env')
        need(Path(env.out).name=='fresh_expert','Original case-output/fresh_expert controller location required')
        need(load_bootstrap(bootstrap_ref,binding,output=Path(env.out).parent)==bootstrap_receipt,
            'Actual collection bootstrap packet/reference differs')
        need(callable(predictor) and kwargs.get('action_provider') is None,'Verified predictor and original analytic L3 required')
        self.collection_binding=binding;self.bootstrap_receipt=bootstrap_receipt;self.bootstrap_ref=deepcopy(bootstrap_ref)
        self.predictor=predictor;self.policy_receipt=deepcopy(bootstrap_receipt);self.l2_queries=[];self.l2_actions=[]
        self._collection_events=[];self._intervention=None;self._intervention_attempts=0;self._teacher_completed=False
        # Normal Expert constructor sees the same already-attached capture.
        capture=getattr(args[0] if args else kwargs.get('env'),'raw_capture',None)
        super().__init__(*args,**kwargs)
        need(self.metadata==binding['metadata'] and self.reference.sha256==binding['reference'].sha256,
            'Actual collection constructor reference differs')
        from .expert_capture import CompactExpertCapture
        need(isinstance(capture,CompactExpertCapture) and self.env.raw_capture is capture and capture.writer is not None and
            capture.camera is not None and not capture._closed and capture.metadata==self.metadata,
            'Initialized compact capture must remain attached unchanged through normal Expert construction')
        need(self.joint_speed_limit==.175 and self.action_provider is None,'Original .175 speed and analytic action provider required')
        need(all(capture.binding['compact']['code_bindings'].get(name)==ref['sha256'] and
            capture.sources.get(ref['path'])==ref['sha256'] for name,ref in _capture_source_bindings().items()),
            'Actual collection capture must bind its capture and canonical snapshot source files')
        self._collection_capture=capture;self._collection_config_digest=binding['config_sha256']
        self._collection_stop_capture=None

    def _observation(self):
        if self.control_owner=='ACT_L2':return M2ACTController._observation(self)
        return self._reference_observation()

    def _query(self,observation):
        self.predictor.context=dict(observed_at_s=float(observation['observed_at_s']),physics_row=len(self.env.rows),
            context=list(observation['context']),source_id=observation['source_id'])
        try:return M2ACTController._query(self,observation)
        finally:self.predictor.context=None

    def _execute_action(self,context):return GivenContactACT._execute_action(self,context)

    def _wait_history_cell(self,observation):
        # The legacy helper chooses observations by isinstance(M2ACTController).
        # This data-only Expert subclass instead follows the actual owner.
        if self.control_owner=='TEACHER_L2' and self._handoff_ready(observation):return 'handoff'
        if len(self.env.rows)%12==0:
            self._issue_intent(np.asarray(self.trajectory.input.target_position),observation['context'],'given_l2_history_wait')
        while True:
            self._advance_tick(self._dispatcher().current['execution_role']);observation=self._observation()
            if self.control_owner=='TEACHER_L2' and self._handoff_ready(observation):return 'handoff'
            if len(self.env.rows)%12==0:return 'continue'

    def _issue_intent(self,target,context,role,*,stop=False):
        setter=super()._issue_intent(target,context,role,stop=stop)
        kind=('learner_prefix' if self.control_owner=='ACT_L2' else 'recovery_teacher') if not stop and role=='given_l2_motion' else 'excluded'
        self._collection_events.append(dict(intent_id=setter['intent_id'],setter_sha256=setter['setter_sha256'],
            physics_row_start=setter['physics_row_start'],physics_row_end_exclusive=setter['physics_row_start'],kind=kind,completed=False))
        return setter

    def _tick(self,command,subphase,active=False,recovery=False):
        if self.control_owner in ('ACT_L2','TEACHER_L2','L2_STOP'):self._deadline()
        start=len(self.env.rows)
        try:return super()._tick(command,subphase,active=active,recovery=recovery)
        finally:
            if len(self.env.rows)>start:
                finish_collection_tick(self,start)
                if self._collection_events and getattr(self,'_current_intent_audit',None) is not None:
                    event=self._collection_events[-1];event['physics_row_end_exclusive']=len(self.env.rows)
                    event['completed']=len(self.env.rows)-event['physics_row_start']==12

    def _query_state(self):
        observation=self._observation();state=self.env.state();history,times=self._history.snapshot()
        pose=np.eye(4);pose[:3,:3]=self.env.base[:3,:3].T@np.asarray(state['rotation_world']);pose[:3,3]=state['tcp_base']
        return dict(measured_q=np.asarray(state['q']).tolist(),measured_dq=np.asarray(state['dq']).tolist(),
            measured_pose_base=contract._pose(pose,'actual full TCP').tolist(),history=history.tolist(),history_times=times.tolist(),
            time_s=float(state['time_s']),physics_row=len(self.env.rows),reference_sha256=self.reference.sha256,
            context=list(observation['context']),hall_N=float(state['hall_N']),physical_contact_N=float(sum(self.env.last_contacts.values())),
            compression_m=float(state['compression_m']),metadata=deepcopy(self.metadata))

    def _live_snapshot(self):
        return dict(schema=LIVE_SCHEMA,query=self._query_state(),input=handoff.input_snapshot(self.trajectory),output=handoff.output_snapshot(self.trajectory),
            object_ids={k:id(v) for k,v in dict(trajectory=self.trajectory,input=self.trajectory.input,output=self.trajectory.output,
                generator=self.trajectory.generator).items()},command_history=[deepcopy(r['q_command']) for r in self.env.rows[-3:]],
            l2_started_at=self._l2_started_at,history_id=id(self._history),capture_id=id(self.env.raw_capture),
            prefix=None if self.prefix is None else dict(object_id=id(self.prefix),index=self.prefix.index,
                absolute=self.prefix.absolute.tolist(),context=list(self.prefix.context)))

    def _eligible(self,observation):
        if len(self.env.rows)%12 or not isinstance(self._history,stage.CausalHistory) or len(self._history.frames)!=2:return False
        history,times=self._history.snapshot()
        if abs(times[-1]-float(self.env.state()['time_s']))>1e-8:return False
        error=float(np.linalg.norm(history[-1,7:10]+[0.,0.,self.manager.config.precontact_gap_m]))
        return error<=.012 or float(self.env.state()['time_s'])-self._l2_started_at>=40.

    def _intervene(self):
        need(self._intervention_attempts==0,'Only one FIRST eligible live intervention is permitted')
        self._intervention_attempts+=1;self._deadline()
        before=self._live_snapshot();query=before['query'];history=np.asarray(before['command_history'],float)
        goal=np.asarray(rt.pose.bound_json(self.reference.snapshot()['sources']['fresh_plan'])['transfer_q'],float)[-1]
        proposal=propose_live_collection_recovery(query,self.trajectory,history,goal,reference=self.reference,
            collection_binding=dict(self.collection_binding,live_witness=before),planner=lambda start,goal,deadline:self._plan_inter_point_bridge(start,goal,deadline),
            verify_command=lambda q:recovery.GivenContactRecoveryExpert._verify_recovery_command(self,q))
        after=self._live_snapshot();need(after==before,'Oracle changed actual measured/live history/limiter/time/objects')
        self._intervention=dict(physics_row=query['physics_row'],time_s=query['time_s'],query=deepcopy(query),
            live_before=before,live_after=after,proposal=deepcopy(proposal),l2_started_at_s=self._l2_started_at,
            l2_deadline_s=self._l2_started_at+120.,transition=['ACT_L2','TEACHER_L2'],saved_state_injected=False,
            history_reset=False,limiter_reset=False,world_reset=False)
        need(proposal['kind']=='recovery_proposal','Collection teacher abstained: '+str(proposal.get('reason')))
        self.prefix=None;self.control_owner='TEACHER_L2'
        # Same CausalHistory, provider stage latch, q/v/a and simulation clock.
        for waypoint in np.asarray(proposal['route'],float)[1:]:
            self._deadline()
            recovery._preview(self.trajectory,np.asarray([r['q_command'] for r in self.env.rows[-3:]],float),[waypoint],
                lambda q:recovery.GivenContactRecoveryExpert._verify_recovery_command(self,q),settle=True)
            if GivenContactExpert._teacher_waypoint(self,waypoint):self._teacher_completed=True;return True
        raise EpisodeRejected('Collection teacher final prepose never reached measured original handoff')

    def _move_to_prepose(self,row):
        self._point_plan=row;self._validate_target_reference(row);canonical,index=self._canonical(row)
        self.canonical_transfer=canonical.copy();self.canonical_stage_index=index
        receipt=dict(target=self.target.name,l2_coarse_reached=False,l2_handoff_success=False,expert_l2_fallback=False,l3_started=False,l3_completed=False)
        self.l2_handoffs.append(receipt)
        try:
            need(self.sequence_index==0 and np.max(abs(self.q))<=.03 and np.max(abs(self.env.state()['q']))<=.03,
                'Collection requires original single-target home')
            self.control_owner='PREPARATION';self._hold(1.,'given_initial_settle');need(self._stationary(),'No real stationary native history')
            self.trajectory=self.trajectory_factory(self.q,dt=DT,**self._trajectory_limits())
            self.control_owner='ACT_L2';self.actor_phase='FREE';self.prefix=None
            self._l2_started_at=float(self.env.state()['time_s']);self._history=stage.CausalHistory(self._l2_started_at)
            receipt.update(start_time_s=self._l2_started_at,start_physics_row=len(self.env.rows))
            while self.control_owner=='ACT_L2':
                self._deadline();observation=self._observation()
                if self._eligible(observation):self._intervene();break
                if self.prefix is None or not self.prefix.remaining or self.prefix.context!=observation['context']:
                    self.prefix=None
                    if len(self._history.frames)!=2:self._wait_history_cell(observation);continue
                    self._query(observation)
                self._execute_action(observation['context'])
                # The unchanged executor may return off-grid on an actual
                # coarse/context event. Complete its held cell to the first
                # eligible H2/grid boundary inside this owned L2 loop.
                while len(self.env.rows)%12:
                    self._advance_tick(self._dispatcher().current['execution_role']);self._observation()
            need(self._teacher_completed,'Exactly one completed teacher intervention required')
            while True:
                self._deadline();observation=self._reference_observation()
                need(self._handoff_ready(observation),'Teacher measured handoff lost')
                parked=self._stop_for_handoff(observation['context'])
                if parked is None:
                    GivenContactExpert._teacher_waypoint(self,canonical[-1]);continue
                receipt.update(l2_handoff_success=True,entry=parked,parked_entry=parked)
                return canonical
        except Exception as error:
            receipt['failure_reason']=str(error);self.prefix=None;self.control_owner='STOPPED';raise
        finally:receipt['end_physics_row']=len(self.env.rows)

    def close_bridge(self):
        receipt=self.predictor.close();self.policy_receipt['cpu_child_exit']=receipt;return receipt

    def run(self,points):
        result=None;exit_receipt=None
        try:
            result=super().run(points)
            if not (self._intervention_attempts==1 and self._teacher_completed):
                result.update(success=False,failure_reason='No complete single collection intervention')
            result.update(learned_policy=True,expert_demonstration=False,learner_prefix_executed=bool(self.l2_queries),
                expert_intervention=self._intervention_attempts==1,benchmark_act_eligible=False,collection_bootstrap=self.bootstrap_ref,
                l2_queries=deepcopy(self.l2_queries),l2_actions=deepcopy(self.l2_actions),collection_route=ROUTE_KEY)
            return result
        finally:
            exit_receipt=self.close_bridge()
            value=dict(schema=SIDECAR_SCHEMA,completed=bool(result and result.get('success') is True and self._teacher_completed),
                episode_id=self.episode_id,reference_sha256=self.reference.sha256,config_sha256=self.collection_binding['config_sha256'],
                deployment=self.collection_binding['deployment'],bootstrap=self.bootstrap_ref,native_steps=len(self.env.rows),
                events=self._collection_events,intervention=self._intervention,interventions=self._intervention_attempts,
                learner_prefix_executed=bool(self.l2_queries),cpu_child_exit=exit_receipt,**FLAGS)
            ref=rt.write_new(self.env.out/'onpolicy_collection.json',value)
            if result is not None:result.update(onpolicy_collection=ref,cpu_child_exit=exit_receipt)


def _authenticated_controller(controller):
    need(isinstance(controller,OnPolicyTeacherCollection) and controller.collection_binding.get('schema')==BINDING_SCHEMA and
        controller.collection_binding['config_sha256']==controller._collection_config_digest and
        rt.digest(controller.collection_binding['configuration'])==controller._collection_config_digest and
        controller.env.raw_capture is controller._collection_capture,'Authenticated collection route/capture changed')


def capture_evidence(controller,evidence):
    """Dedicated source attribution before the existing capture finish seam."""
    _authenticated_controller(controller);result=deepcopy(evidence)
    result['collection_controller']=controller
    owner=controller.control_owner
    result['clean_command']['source_id']=('onpolicy_collection_act_native_command_v1' if owner=='ACT_L2' else
        'onpolicy_collection_teacher_native_command_v1' if owner=='TEACHER_L2' else 'onpolicy_collection_original_native_command_v1')
    need(controller._collection_stop_capture is None,'Unfinished collection STOP capture cannot be reused')
    if owner=='L2_STOP':
        from . import m2_given_contact_evaluation as neutral
        raw=deepcopy(getattr(controller,'_current_intent_audit',None))
        need(raw is not None and raw['setter']['control_owner']=='L2_STOP' and raw['setter']['offgrid_stop'] is True and
            raw['setter']['execution_role']=='given_l2_park' and raw['setter']['context']==
            [controller.episode_id,controller.metadata['target_id'],0,'APPROACH'],
            'Actual collection STOP setter/context required before physics')
        witness=dict(observed=neutral._observed(controller.env),interpolator=neutral._interpolator(controller),
            env_id=id(controller.env),capture_id=id(controller.env.raw_capture),
            raw_intent=raw,command=deepcopy(result['clean_command']),
            stop_residual_adapter_tick=deepcopy(getattr(controller,'_current_stop_residual_audit',None)))
        controller._collection_stop_capture=witness
        result['collection_stop_capture']=witness
    return result


def finish_stop_capture(controller,witness,*,rows_before,before,after,clean_command):
    """Readbacks around the one original physics tick, after its limiter step."""
    _authenticated_controller(controller)
    if controller.control_owner!='L2_STOP':
        need(witness is None and controller._collection_stop_capture is None,'STOP capture crossed actual owner')
        return {}
    need(witness is not None and witness is controller._collection_stop_capture,'Missing/stale actual collection STOP capture')
    controller._collection_stop_capture=None
    from . import m2_given_contact_evaluation as neutral, m2_stop_residual_trajectory as residual
    env=controller.env;observed=neutral._observed(env);prior=witness['observed']
    need(witness['env_id']==id(env) and witness['capture_id']==id(env.raw_capture) and len(env.rows)==rows_before+1 and
        prior['physics_row']==rows_before and observed['physics_row']==rows_before+1,
        'Collection STOP capture requires exactly one original native tick')
    row=env.rows[-1];command=contract._array(clean_command['q_command'],(7,),'actual STOP command')
    need(observed['time_s']==row['time_s'] and abs(observed['time_s']-prior['time_s']-1/240.)<=1e-8 and
        np.asarray(observed['q'],dtype=np.float64).tobytes()==contract._array(row['q'],(7,),'actual STOP measured q').tobytes() and
        command.tobytes()==contract._array(row['q_command'],(7,),'actual STOP applied command').tobytes() and
        witness['command']==clean_command and clean_command['issued_at_s']==prior['time_s'],
        'Actual collection STOP clock/sensors/issued command differs')
    state=env.state()
    need(contract._array(state['dq'],(7,),'actual STOP measured dq').tobytes()==
        contract._array(row['dq'],(7,),'actual STOP native row dq').tobytes(),
        'Actual collection STOP sensor dq differs from native row')
    interpolator=witness['interpolator'];raw=witness['raw_intent']
    need(rt.digest(interpolator)==rt.digest(neutral._interpolator(controller)) and
        raw==before.get('raw_intent')==after.get('raw_intent')==getattr(controller,'_current_intent_audit',None) and
        before.get('actual_control_owner')==after.get('actual_control_owner')=='L2_STOP' and
        interpolator['planning_profile']=='stop' and
        command.tobytes()==contract._array(interpolator['input']['fields']['current_position'],(7,),'live STOP position').tobytes() and
        command.tobytes()==contract._array(interpolator['output']['new_position'],(7,),'live STOP output position').tobytes() and
        np.array_equal(raw['live_target_q'],interpolator['input']['fields']['target_position']) and
        np.array_equal(raw['live_target_after_step_q'],interpolator['input']['fields']['target_position']),
        'Original collection STOP limiter/state/objects/owner/target changed inside physics tick')
    evidence=witness['stop_residual_adapter_tick'];result=dict(interpolator=deepcopy(interpolator))
    need(evidence==getattr(controller,'_current_stop_residual_audit',None),'Actual STOP event changed inside physics tick')
    preview=controller.l2_handoffs[-1].get('parking_preview',{}).get('stop_residual_adapter')
    if preview is not None and getattr(controller,'_parking_tail_audit',None) is None:
        need(evidence is not None,'Actual residual STOP tick lacks its pre-physics event')
    if evidence is not None:
        event=controller.trajectory.last_stop_residual_event();objects=controller.trajectory.last_stop_residual_object_check()
        need(evidence==dict(adapter_id=residual.ADAPTER_ID,event=event,event_sha256=handoff.digest(event),objects=objects) and
            event['input_after']==interpolator['input'] and event['output']==interpolator['output'] and
            event['command_q']==command.tolist() and objects['after']==dict(trajectory=interpolator['trajectory_object'],
                **interpolator['objects']),'Actual collection STOP event/digest/live interpolator differs')
        result['stop_residual_adapter_tick']=deepcopy(evidence)
    return result


def finish_collection_tick(controller,start):
    _authenticated_controller(controller)
    need(len(controller.env.rows)==start+1,'Collection attribution requires exactly one original native tick')
    row=controller.env.rows[-1];owner=controller.control_owner
    row.update(control_owner=owner,actual_control_owner=owner,learned_policy=owner=='ACT_L2',
        expert_demonstration=owner=='TEACHER_L2',collection_route=ROUTE_KEY,benchmark_act_eligible=False,
        command_source_id='onpolicy_collection_'+owner.lower()+'_v1')


class CPUBridge(pipe.CPUBridge):
    """The one existing bounded pipe child, with collection-specific adapter."""
    def __init__(self,deployment,*,output):
        _validate_deployment(deployment)
        super().__init__(deployment,output=output,adapter=sys.modules[__name__])
        try:
            need('bootstrap' not in self.runtime,'Full collection bootstrap forbidden on bounded IPC')
            binding=preflight(dict(_validate_deployment(deployment)[0],**{ROUTE_KEY:deployment}))
            self.bootstrap_ref=deepcopy(self.runtime.get('bootstrap_ref'))
            self.bootstrap=load_bootstrap(self.bootstrap_ref,binding,output=output)
        except BaseException:self.close();raise


def make_request(dep,sequence,q,target,confidence,**kwargs):
    return pipe.make_request(dep,sequence,q,target,confidence,protocol_schema=PROTOCOL_SCHEMA,
        binding_extra=dict(collection_bootstrap=dep['initialization'],prepared=PREPARED,slot=dep['request']['slot_id']),**kwargs)


def validate_request(request,dep,*,sequence,previous=None):
    return pipe.validate_request(request,dep,sequence=sequence,previous=previous,protocol_schema=PROTOCOL_SCHEMA,
        binding_extra=dict(collection_bootstrap=dep['initialization'],prepared=PREPARED,slot=dep['request']['slot_id']))


def decode_response(response,request):
    return pipe.decode_response(response,request,protocol_schema=PROTOCOL_SCHEMA,checkpoint_schema=warm.COLLECTION_SCHEMA)


def make_controller(*args,config,output,**kwargs):
    binding=preflight(config,output=output);bridge=None
    try:
        bridge=CPUBridge(binding['deployment'],output=output);packet=bridge.bootstrap
        binding['bootstrap']=validate_bootstrap(packet,binding);binding['bootstrap_ref']=deepcopy(bridge.bootstrap_ref);meta=binding['metadata']
        provider=GivenPoseProvider(binding['reference'],meta['episode_id'],meta)
        return OnPolicyTeacherCollection(*args,predictor=bridge,bootstrap_receipt=packet,bootstrap_ref=bridge.bootstrap_ref,collection_binding=binding,
            metadata=meta,episode_id=meta['episode_id'],estimate_provider=provider,**kwargs)
    except BaseException:
        if bridge is not None:bridge.close()
        raise


def serve(deployment_ref):
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'Owned official collection CPU child must hide CUDA')
    dep=rt.read_bound(deployment_ref);_validate_deployment(dep)
    with redirect_stdout(sys.stderr):torch,policy,packet=warm.make_collection_policy(dep['initialization'])
    binding=preflight(dict(_validate_deployment(dep)[0],**{ROUTE_KEY:dep}))
    validate_bootstrap(packet,binding)
    output=Path(deployment_ref['path']).parent
    need(deployment_ref==rt.reference(output/'pilot_cpu_deployment.json'),'Exact collection CPU deployment location required')
    bootstrap_ref=rt.write_new(output/BOOTSTRAP_FILENAME,dict(schema=BOOTSTRAP_SIDECAR_SCHEMA,
        deployment=deepcopy(deployment_ref),source_inventory=deepcopy(dep['source_inventory']),bootstrap=packet))
    load_bootstrap(bootstrap_ref,binding,output=output)
    def send(value):
        payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n'
        need(len(payload)<=131072,'Bounded collection CPU response required');sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()
    send(dict(schema=PROTOCOL_SCHEMA,ready=True,deployment_sha256=deployment_ref['sha256'],checkpoint=dep['model']['checkpoint'],
        bootstrap_ref=bootstrap_ref,pid=os.getpid(),parent_pid=os.getppid(),source_inventory=sources(),execution=rt.execution_settings(torch,policy,device='cpu')))
    sequence=0;previous=None
    while True:
        raw=sys.stdin.buffer.readline(32769)
        if not raw:return
        need(len(raw)<=32768 and raw.endswith(b'\n'),'Bounded complete collection CPU request required')
        request=pipe.wire_decode(raw);sequence+=1;validate_request(request,dep,sequence=sequence,previous=previous)
        history=np.asarray(request['history'],float)
        with redirect_stdout(sys.stderr):relative=rt.infer(torch,policy,dict(states=history[None,1],histories=history[None]),packet['normalization'],device='cpu')[0]
        send(pipe.response_for(request,relative,protocol_schema=PROTOCOL_SCHEMA));previous=request


def admission(ledger):
    ordered=isinstance(ledger,list) and len(ledger)==8 and [e.get('request') for e in ledger]==planned_requests()
    qualified=[];identities=[]
    if ordered:
        qualified=[e for e in ledger if e.get('status')=='qualified' and type(e.get('windows')) is int and e['windows']>0 and
            e.get('infrastructure_contamination') is False]
        try:identities=[refinement.dart._reset(e) for e in qualified]
        except (ValueError,KeyError,TypeError):identities=[]
    eligible=ordered and len(qualified)==8 and len(identities)==len(set(identities))==8
    return dict(schema=ADMISSION_SCHEMA,data_optimization_eligible=bool(eligible),fixed_attempt_denominator=8,
        minimum_success_percent=95,minimum_qualified_episodes=8,qualified_episodes=len(qualified),all_eight_required=True,
        benchmark_act_eligible=False,model_qualified=False,formal_acceptance=False)


def validate_manifest(manifest):
    need(isinstance(manifest,dict) and set(manifest)=={'schema','prepared','initialization','producer_source','attempts'} and
        manifest.get('schema')==MANIFEST_SCHEMA and manifest.get('prepared')==PREPARED,'Typed fixed8 collection manifest required')
    attempts=manifest['attempts'];need(isinstance(attempts,list) and len(attempts)==8,'All original eight attempts required, no replacement')
    for attempt,request in zip(attempts,planned_requests()):
        need(set(attempt)=={'request','episode','outcome','reason'} and attempt.get('request')==request and
            attempt.get('outcome') in ('collected','physical_negative','preview_rejected','infrastructure_error','unavailable') and
            isinstance(attempt.get('reason'),str),'Exact fixed8 TRAIN request/order/outcome required')
        row=attempt['episode']
        if row is not None:need(all(row.get(k)==request[k] for k in ('split','subject_id','pose_id','target_id')),'No VAL/fresh/foreign episode permitted')
        need(row is not None or attempt['outcome'] in ('infrastructure_error','unavailable'),'Native outcome requires original capture declaration')
    return attempts


def manifest_headers(manifest,base):
    attempts=validate_manifest(manifest);_prepared(manifest['prepared']);warm.validate_receipt(manifest['initialization'])
    need(manifest['producer_source']==sources(),'Exact current collection producer inventory required')
    # Identity metadata across ALL eight precedes any native payload opening.
    for a in attempts:
        row=a['episode']
        if row is None:continue
        config,_=data._bound(row['config'],base);dep=config.get(ROUTE_KEY,{})
        need(dep.get('request')==a['request'] and dep.get('initialization')==manifest['initialization'] and dep.get('prepared')==manifest['prepared'],
            'Actual configured fixed8 slot/bootstrap differs')
        meta=config.get('m2_given_contact_control',{}).get('metadata',{})
        need(all(meta.get(k)==a['request'][k] for k in ('split','subject_id','target_id')),'All collection metadata must be original TRAIN')
        if a['outcome']=='collected':
            compact=row.get('artifacts',{}).get('compact/manifest.json',{})
            value,_=data._bound(dict(compact,path=str(data._path(row['capture_root'],base)/'compact/manifest.json')),base)
            need(value.get('metadata')==meta,'Actual compact TRAIN metadata differs before payload reads')
    return attempts


def read_episode(attempt,base,manifest):
    validate_manifest(manifest);row=attempt['episode'];data._declarations([row],base)
    config,_=data._bound(row['config'],base);binding=preflight(config);context=data._metadata(row,base)
    run,actual,freeze=data._native_evidence(row,base,context[2],physical=False,_collection_context=binding)
    need(actual==config,'Actual collection config differs')
    child,_=data._bound(row['child_result'],base)
    need(child.get('physical_pass') is True,'Actual collection child physical pass required')
    sample,receipt,meta=data.decode_onpolicy_collection_capture(row,base,context,run,config,freeze,collection_binding=binding)
    audit=data.read_native_capture_audit(context[0],rt.shared._json(context[0]/'receipt.json'),transport=row['capture_audit'])
    receipt.update(schema='m2_act_onpolicy_physical_data_receipt_v1',cumulative_l2=cumulative_l2(run,audit),
        qualification_scope='actual_collection_teacher_data_only',benchmark_act_eligible=False,learner_prefix_executed=True,
        expert_intervention=True,collection_deployment=deepcopy(binding['deployment']),collection_sidecar=run['onpolicy_collection'])
    return sample,receipt,meta


def cumulative_l2(run,audit):
    """Original120s includes actual learner prefix, intervention and parking."""
    rows=[i for i,e in enumerate(audit) if e.get('actual_control_owner') in ('ACT_L2','TEACHER_L2')]
    handoffs=run.get('l2_handoffs',[])
    need(rows and len(handoffs)==1,'Exactly one actual cumulative collection L2 interval required')
    first=rows[0];end=handoffs[0].get('parking_end_physics_row')
    need(handoffs[0].get('start_physics_row')==first and abs(handoffs[0].get('start_time_s',-1)-first/240.)<=1e-8 and
        type(end) is int and first<end<=len(audit) and
        all(e.get('actual_control_owner') in ('ACT_L2','TEACHER_L2','L2_STOP') for e in audit[first:end]),
        'Original cumulative collection start/ownership/parking interval differs')
    duration=(end-first)/240.;need(duration<=120.,'Original cumulative collection L2 through parking exceeds120s')
    return dict(start_physics_row=first,parking_end_physics_row=end,duration_s=duration,maximum_s=120.,
        prefix_steps=sum(e.get('actual_control_owner')=='ACT_L2' for e in audit[first:end]))


def read_negative(attempt,base,manifest):
    validate_manifest(manifest);need(attempt in manifest['attempts'] and attempt['outcome'] in ('physical_negative','preview_rejected'),
        'Explicit fixed8 terminal negative required')
    row=attempt['episode'];config,_=data._bound(row['config'],base);binding=preflight(config)
    run,actual,freeze=data._native_evidence(row,base,binding['metadata'],physical=False,_collection_context=binding)
    child,_=data._bound(row['child_result'],base)
    need(actual==config and child.get('physical_pass') is False and run.get('success') is False and
        run.get('physical_validation') is False and bool(run.get('failure_reason') or run.get('phase_failure_diagnostics') or
        run.get('collision_failure_diagnostics') or any(h.get('failure_reason') for h in run.get('l2_handoffs',[]))),
        'Original completed source/owner-bound physical negative required')
    return dict(run_result=row['run_result'],child_result=row['child_result'],owner_terminal=row['owner_terminal'],
        source_freeze=row['source_freeze'],classification=attempt['outcome'],positive_teacher_data=False)


def validate_capture_context(run,config,freeze,binding):
    validate_binding(binding)
    need(run.get('executed_controller_class')==MODULE+'.OnPolicyTeacherCollection' and run.get('collection_route')==ROUTE_KEY and
        run.get('learner_prefix_executed') is True and run.get('expert_intervention') is True and
        run.get('benchmark_act_eligible') is False and run.get('learned_policy') is True and run.get('expert_demonstration') is False and
        run.get('success') is True and run.get('physical_validation') is True and run.get('errors')==[] and
        run.get('controller_motion_complete') is True and run.get('compact_capture_complete') is True,
        'Complete mixed collection data route required; never a pure expert or ACT benchmark')
    sidecar_ref=run.get('onpolicy_collection');rt._reference_shape(sidecar_ref)
    sidecar_path=Path(sidecar_ref['path']);output=sidecar_path.parent.parent
    sidecar=_bootstrap_artifact(sidecar_ref,output/'fresh_expert/onpolicy_collection.json')
    bootstrap_ref=run.get('collection_bootstrap')
    need(sidecar.get('bootstrap')==bootstrap_ref,'Run/collection sidecar bootstrap references differ')
    binding['bootstrap']=load_bootstrap(bootstrap_ref,binding,output=output)
    binding['bootstrap_ref']=deepcopy(bootstrap_ref)
    actual=run.get('runtime_sources_sha256',{});current=binding['deployment']['source_inventory']
    need(freeze.get('code_commit')==current['code_commit'] and all(actual.get(r['path'])==r['sha256'] for r in current['files'].values()),
        'Actual collection producer/current source inventory differs')
    return binding


def capture_roles(sidecar,audit,metadata,binding):
    need(sidecar.get('schema')==SIDECAR_SCHEMA and sidecar.get('completed') is True and sidecar.get('interventions')==1 and
        sidecar.get('native_steps')==len(audit) and sidecar.get('episode_id')==metadata['episode_id'] and
        sidecar.get('reference_sha256')==metadata['reference_sha256'] and sidecar.get('config_sha256')==binding['config_sha256'] and
        sidecar.get('deployment')==binding['deployment'] and sidecar.get('learner_prefix_executed') is True and
        sidecar.get('benchmark_act_eligible') is False,'Complete actual collection sidecar required')
    need(sidecar.get('bootstrap')==binding.get('bootstrap_ref'),'Collection sidecar bootstrap differs from admitted case output')
    load_bootstrap(sidecar['bootstrap'],binding,output=Path(binding['bootstrap_ref']['path']).parent)
    exit=sidecar.get('cpu_child_exit',{})
    need(exit.get('schema')==CPU_EXIT_SCHEMA and exit.get('normal_exit') is True and exit.get('reaped') is True and
        exit.get('exit_code')==0 and exit.get('signals')==[],'Normally exited original collection CPU child required')
    intervention=sidecar.get('intervention',{});query=intervention.get('query',{})
    validate_live_collection_query(query,binding['reference'],dict(binding,live_witness=intervention.get('live_before',{})))
    boundary=query['physics_row'];need(intervention.get('physics_row')==boundary and intervention.get('transition')==['ACT_L2','TEACHER_L2'] and
        intervention.get('live_before')==intervention.get('live_after') and intervention.get('live_before',{}).get('query')==query and
        intervention.get('l2_deadline_s')==intervention.get('l2_started_at_s',float('nan'))+120. and
        query['time_s']<intervention['l2_deadline_s'] and all(intervention.get(k) is False for k in
        ('saved_state_injected','history_reset','limiter_reset','world_reset')),'Actual same-state single boundary/reset/clock differs')
    before=[r for r in audit[:boundary] if r.get('raw_intent') is not None]
    after=[r for r in audit[boundary:] if r.get('raw_intent') is not None]
    need(before and after and any(r['raw_intent']['setter']['execution_role']=='given_l2_motion' for r in before) and
        all(r['actual_control_owner']=='ACT_L2' for r in before) and
        all(r['actual_control_owner'] in ('TEACHER_L2','L2_STOP') for r in after) and audit[boundary]['actual_control_owner']=='TEACHER_L2',
        'Actual learner then teacher ownership transition differs')
    proof=dict(schema=refinement.ROLE_SCHEMA,boundary_row=boundary,events=sidecar.get('events'))
    refinement.completed_roles(audit,proof['events'],boundary_row=boundary)
    return boundary,proof


def _recorded_collection_boundary(event,physics_row,metadata,config_sha256):
    """Mandatory raw-capture witness, separate from the oracle sidecar."""
    captured=event.get('onpolicy_collection_boundary')
    need(isinstance(captured,dict) and set(captured)=={'schema','source_id','physics_row','time_s','context',
        'actual_control_owner','reference_sha256','config_sha256','measured_q','measured_dq','measured_pose_base',
        'input','output','object_ids','capture_id','history_id'} and captured.get('schema')==CAPTURE_BOUNDARY_SCHEMA and
        captured.get('source_id')=='actual_native_post_step_collection_boundary' and captured.get('physics_row')==physics_row and
        captured.get('time_s')==event.get('time_s') and abs(captured['time_s']-physics_row/240.)<=1e-8 and
        captured.get('actual_control_owner')==event.get('actual_control_owner')=='ACT_L2' and
        captured.get('reference_sha256')==metadata['reference_sha256'] and captured.get('config_sha256')==config_sha256,
        'Mandatory independent recorded live collection boundary missing, stale or malformed')
    context=captured.get('context')
    need(isinstance(context,list) and len(context)==4 and context[:3]==[metadata['episode_id'],metadata['target_id'],0] and
        context[3] in ('FREE','APPROACH'),'Independent recorded live collection context differs')
    return captured


def verify_native_tcp_consistency(kin,q,compression,pose_base,world_from_base):
    """Check cross-representation consistency, after independent byte binding.

    Native wrist position/quaternion and joints are float32 readbacks; analytic
    URDF FK is float64. Budget four-term transform dots over seven origin/motion
    pairs plus world/base conversion: n=4*(2*7+2)=64, u=2**-24,
    gamma=n*u/(1-n*u). This is an explicit conservative representation envelope,
    NOT a proven PhysX solver error bound or a physical acceptance tolerance.
    Translation scales with chain/tool reach and world/base cancellation;
    orientation has its own radian ceiling (2*gamma), never a 4x4 mixed-unit atol.
    Only the source-bound authored base is permitted at the reader call site.
    """
    q=contract._array(q,(7,),'native consistency measured q')
    compression=contract._scalar(compression,'native consistency signed compression',minimum=-np.inf)
    actual=contract._pose(pose_base,'native consistency full TCP')
    base=contract._pose(world_from_base,'native consistency authored world_from_base')
    links=kin.link_transforms(q)  # Retain the existing strict URDF FK q domain.
    translations=np.asarray([value[:3,3] for value in links.values()])
    chain_length=float(np.linalg.norm(np.diff(translations,axis=0),axis=1).sum())
    contact=kin.contact_frame()  # Geometry's command/IK compression domain stays strict.
    expected=links[kin.wrist_link]@contact
    # Exactly env.state's signed measured-slider algebra. Do not clamp the raw
    # residual, change capture/H2, or pass a negative sensor value to geometry.
    expected[:3,3]-=expected[:3,2]*compression
    reach=chain_length+abs(float(contact[2,3])-compression)
    base_origin=float(np.linalg.norm(base[:3,3]))
    need(len(kin.joint_names)==7 and reach<=1. and base_origin<=2.,
        'Native TCP consistency exceeds the seven-joint / 1m reach / 2m world-base numerical domain')
    u=2.**-24;n=4*(2*len(kin.joint_names)+2);gamma=n*u/(1-n*u)
    position_tolerance=gamma*(reach+2*base_origin)
    rotation_tolerance=2*gamma
    position_error=float(np.linalg.norm(actual[:3,3]-expected[:3,3]))
    relative=expected[:3,:3].T@actual[:3,:3]
    skew=np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])
    rotation_error=float(np.arctan2(np.linalg.norm(skew)/2,(np.trace(relative)-1)/2))
    need(position_error<=position_tolerance and rotation_error<=rotation_tolerance,
        'Source-bound native TCP disagrees with signed-compression joint FK representation envelope')
    return dict(schema='m2_act_onpolicy_native_tcp_consistency_v1',position_error_m=position_error,
        position_tolerance_m=position_tolerance,rotation_error_rad=rotation_error,rotation_tolerance_rad=rotation_tolerance,
        units=dict(position='m',rotation='rad'),float32_unit_roundoff=u,transform_roundoff_gamma=gamma,
        chain_length_m=chain_length,world_base_origin_norm_m=base_origin,
        native_roundoff_scope='representation consistency only; not PhysX accuracy or physical acceptance')


def verify_captured_boundary(sidecar,audit,records,metadata,binding):
    """Reconstruct the first eligible grid, real H2 and executed teacher stream."""
    boundary,proof=capture_roles(sidecar,audit,metadata,binding)
    intervention=sidecar['intervention'];query=intervention['query'];before=intervention['live_before']
    reference=binding['reference'];config=binding['configuration']
    hand=rt.pose.bound_json(reference.snapshot()['sources']['fresh_plan'])
    gap=min(.04,float(hand['free_approach_start_gap_m']));eligible=[];started=intervention['l2_started_at_s']
    selected=[]
    for stamp in (query['time_s']-.05,query['time_s']):
        rows=[r for r in records if abs(float(r['observed_at_s'])-stamp)<=1e-8]
        need(len(rows)==1,'Actual collection boundary H2 frame missing/duplicate');selected.append(rows[0])
    need(np.array_equal(query['history'],[r['state17'] for r in selected]) and
        np.array_equal(query['measured_q'],selected[-1]['measured_q']),'Actual boundary differs from captured H2/measured origin')
    captured=_recorded_collection_boundary(audit[boundary-1],boundary,metadata,binding['config_sha256'])
    need(captured['time_s']==query['time_s'] and captured['context']==query['context'],
        'Certified live collection context/time differs from independent native capture')
    for name,shape in (('measured_q',(7,)),('measured_dq',(7,)),('measured_pose_base',(4,4))):
        recorded=contract._array(captured[name],shape,'independent captured '+name)
        certified=contract._array(query[name],shape,'certified collection '+name)
        need(recorded.tobytes()==certified.tobytes(),'Certified live '+name+' differs from independent native capture')
    need(rt.digest(captured['input'])==rt.digest(before.get('input')) and
        rt.digest(captured['output'])==rt.digest(before.get('output')) and captured['object_ids']==before.get('object_ids') and
        captured['capture_id']==before.get('capture_id') and captured['history_id']==before.get('history_id'),
        'Certified live input/output/objects/history differ from independent native capture')
    output=captured['output'];inp=captured['input']
    need(isinstance(output,dict) and set(output)=={'new_position','new_velocity','new_acceleration','time'} and
        type(output.get('time')) in (int,float) and np.isfinite(output['time']) and output['time']>=0,
        'Independent captured live output is malformed')
    for position,derivative in (('position',None),('velocity','velocity'),('acceleration','acceleration')):
        actual_output=contract._array(output['new_'+position],(7,),'actual live output '+position)
        actual_input=contract._array(inp['fields']['current_'+position],(7,),'actual live input '+position)
        need(actual_output.tobytes()==actual_input.tobytes(),'Independent live input/output '+position+' disagree')
        if derivative is not None:
            need(actual_output.tobytes()==contract._array(inp[derivative],(7,),'actual limiter '+derivative).tobytes(),
                'Independent live limiter/output '+position+' disagree')
    actual=audit[boundary-1]['measurements']
    need(query['hall_N']==actual['hall_N'] and query['compression_m']==actual['compression_m'] and
        query['physical_contact_N']==sum(actual[k+'_N'] for k in ALL_FORCE_GROUPS) and
        float(np.max(abs(np.asarray(query['measured_dq']))))==actual['max_abs_joint_velocity_rad_s'],
        'Boundary actual released force/compression/dq maximum witness differs')
    contexts={};causal=stage.CausalStage()
    for i,event in enumerate(audit[:boundary]):
        if event.get('actual_control_owner')!='ACT_L2':continue
        for side,stamp in (('controller_before',i/240.),('controller_after',(i+1)/240.)):
            actor=event[side]['actor']
            stage.validate_actor(actor,stamp,source=rt.pose.NATIVE_SOURCE,reference_sha256=reference.sha256)
            causal.update(actor['estimated_tcp_distance_m'],stamp)
        if (i+1)%12==0:
            measured=_recorded_collection_boundary(event,i+1,metadata,binding['config_sha256'])
            need(measured['context'][3]==causal.phase,'Independent capture phase differs from actual measured causal latch')
            contexts[i+1]=measured['context']
    for i in range(1,len(records)):
        row=12*(i+1)
        if row>boundary:break
        stamp=float(records[i]['observed_at_s'])
        if stamp<started+.05-1e-8:continue
        # WAIT labels remain excluded. Their measured native distances still
        # advance the original monotonic latch, so use the two independent
        # recorded grid contexts to reconstruct the real causal H2 reset.
        if row not in contexts or row-12 not in contexts or contexts[row]!=contexts[row-12]:continue
        encoded=np.asarray(records[i]['state17'])
        if np.linalg.norm(encoded[7:10]+[0.,0.,gap])<=.012 or stamp-started>=40.:eligible.append(row)
    need(eligible and min(eligible)==boundary,'Intervention was not the FIRST eligible actual 20Hz/H2 boundary')
    commands=np.asarray([r['q_command'] for r in audit[boundary-3:boundary]],float)
    need(np.array_equal(before.get('command_history'),commands) and
        np.array_equal(before['input']['fields']['current_position'],commands[-1]) and before.get('l2_started_at')==started and
        before.get('history_id') is not None and before.get('capture_id') is not None and
        set(before.get('object_ids',{}))=={'trajectory','input','output','generator'} and
        len(set(before['object_ids'].values()))==4,'Retained actual q/v/a/history/capture/object certificate differs')
    proposal=intervention.get('proposal',{});preview=proposal.get('preview',{});goal=np.asarray(hand['transfer_q'],float)[-1]
    need(proposal.get('kind')=='recovery_proposal' and proposal.get('query')==query and proposal.get('query_sha256')==rt.digest(query) and
        np.array_equal(proposal.get('destination_q'),goal) and preview.get('input_state')==before['input'] and
        preview.get('input_sha256')==rt.digest(before['input']) and preview.get('all_240hz_commands_verified') is True and
        preview.get('physics_executed') is False,'Actual authenticated FINAL oracle preview differs')
    route=np.asarray(proposal.get('route'),float);planned=np.asarray(proposal.get('planner_route'),float)
    need(planned.ndim==2 and planned.shape[1:]==(7,) and len(planned)>=2 and
        np.array_equal(planned[0],query['measured_q']) and np.array_equal(planned[-1],goal) and
        route.ndim==2 and route.shape[1:]==(7,) and len(route)>=2 and
        np.array_equal(route[0],commands[-1]) and np.array_equal(route[-1],goal),'Measured/retained/final oracle route endpoints differ')
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    kin=OpenArmKinematics(contact_frame=rt.geometry.contact_frame)
    native_tcp=verify_native_tcp_consistency(kin,query['measured_q'],query['compression_m'],query['measured_pose_base'],
        reference.snapshot()['world_from_base'])
    clone=handoff.restore_trajectory(before['input']);history=commands.copy();route_index=1;end=boundary
    for i in range(boundary,len(audit)):
        event=audit[i];raw=event.get('raw_intent')
        if event.get('actual_control_owner')!='TEACHER_L2':break
        need(raw is not None,'Actual teacher tick lacks a native setter')
        setter=raw['setter'];role=setter['execution_role']
        if raw['native_tick_in_cell']==0:
            value=setter['target_q']
            if role=='given_l2_motion':
                while route_index<len(route) and not np.array_equal(route[route_index],value):route_index+=1
                need(route_index<len(route),'Actual teacher setter differs from ordered accepted route')
            else:need(role=='given_l2_history_wait' and np.array_equal(value,clone.input.target_position),'Teacher wait changed live target')
            clone.set_target(value)
        command=clone.step()
        need(command.astype(np.float64).tobytes()==np.asarray(event['q_command'],dtype=np.float64).tobytes(),
            'Actual retained-q/v/a teacher native replay differs')
        values=recovery.command_derivatives(history,command)
        need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
            'Actual teacher original derivative seam failed')
        history=np.vstack([history[1:],command]);end=i+1
    need(end>boundary,'No actually executed teacher intervention')
    return dict(boundary_row=boundary,actual_h2_bound=True,first_eligible_boundary=True,
        teacher_native_end=end,retained_qva_replayed=True,final_destination_bound=True,native_tcp_consistency=native_tcp)


def tag_result(result,controller,config):
    result.update(collection_route=ROUTE_KEY,collection_only=True,learned_policy=True,expert_demonstration=False,
        benchmark_act_eligible=False,expert_intervention=bool(controller and controller._intervention_attempts==1),
        learner_prefix_executed=bool(controller and controller.l2_queries),configured_collection=deepcopy(config[ROUTE_KEY]))
    if controller is not None:
        result.update(executed_controller_class=MODULE+'.OnPolicyTeacherCollection',collection_bootstrap=controller.bootstrap_ref,
            l2_queries=deepcopy(controller.l2_queries),l2_actions=deepcopy(controller.l2_actions),cpu_child_exit=controller.policy_receipt.get('cpu_child_exit'))
    return result


def qualify(manifest_ref,output):
    return refinement.export_onpolicy(manifest_ref,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--serve',action='store_true')
    parser.add_argument('--deployment',type=Path,required=True);parser.add_argument('--deployment-sha256',required=True)
    args=parser.parse_args();ref=rt.reference(args.deployment);need(args.serve and ref['sha256']==args.deployment_sha256,'Exact collection deployment required')
    serve(ref)


if __name__=='__main__':main()

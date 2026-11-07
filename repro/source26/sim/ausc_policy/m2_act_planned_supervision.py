"""Pure CPU ideal-tracking command supervision on AVAILABLE canonical scenes.

Generated H2 frames and clocks are explicitly synthetic. The original retained
Ruckig q/v/a, full 240Hz collision/posture/FD preview and finite bounds are used;
no simulator, contact rollout, run_probe or fabricated owner marker is involved.
"""
from copy import deepcopy
import hashlib
import time

import numpy as np

from . import m2_act_auxiliary_data as data
from . import m2_given_contact_recovery as recovery
from .m2_stage import CausalStage

rt=data.rt
need=rt.need


def planned_windows(preview,metadata):
    """Convert a checked preview into causal H2/raw-setter windows.

    Startup frames are q0 at generated times 0 and .05. Input i subsequently
    uses the preceding completed 12-tick endpoint. No applied endpoint is ever
    a target label. Context changes discard the crossing history and truncate
    the preceding future prefix to that context.
    """
    reference=rt.pose.require_route(metadata)
    need(reference.snapshot()['available_at_s']==0.,'Generated startup requires a reference available at time zero')
    need(preview.get('all_240hz_commands_verified') is True and preview.get('physics_executed') is False,
         'Unexecuted full 240Hz preview required')
    cells=np.asarray(preview['cells'],float);commands=np.asarray(preview['commands'],float)
    q0=rt.shared.contract._array(preview['input_state']['fields']['current_position'],(7,),'generated initial command')
    need(cells.ndim==2 and cells.shape[1:]==(7,) and 0<len(cells)<=2400 and
         commands.shape==(12*len(cells),7) and np.isfinite(cells).all() and np.isfinite(commands).all(),
         'Bounded aligned raw setter/command preview required')
    endpoints=commands[11::12]
    current=np.vstack([q0,endpoints[:-1]])
    encoded=np.asarray([rt.shared.contract.encode_state(q,reference.pose_base,1.,metadata=metadata) for q in current])
    stage=CausalStage();phases=[stage.update(float(np.linalg.norm(s[7:10])),(i+1)/20.) for i,s in enumerate(encoded)]
    arrays={k:[] for k in data.ARRAY_KEYS};windows=[]
    for i in range(len(cells)):
        if i and phases[i-1]!=phases[i]:continue
        end=i+1
        while end<min(i+20,len(cells)) and phases[end]==phases[i]:end+=1
        count=end-i;relative=np.zeros((20,7));relative[:count]=cells[i:end]-current[i]
        arrays['source_histories'].append(np.stack([encoded[max(0,i-1)],encoded[i]]))
        arrays['source_times'].append([i/20.,(i+1)/20.])
        arrays['relative_setters'].append(relative);arrays['setter_is_pad'].append(np.arange(20)>=count)
        arrays['source_indices'].append(0)
        windows.append(dict(input_index=i,phase=phases[i]))
    return {k:np.asarray(v,dtype=bool if k=='setter_is_pad' else np.int64 if k=='source_indices' else float)
        for k,v in arrays.items()},windows


def _scene(scene,target,metadata,config_ref,allowed,*,cpu_fixture=False):
    # Gate both scene and routed identity before reading prepared/plan payloads.
    data._train(scene.get('subject_id'),scene.get('split'),allowed)
    data._train(metadata.get('subject_id'),metadata.get('split'),allowed)
    need(scene['subject_id']==metadata['subject_id'] and metadata.get('target_id')==target and
         isinstance(scene.get('pose_id'),str) and bool(scene['pose_id']),'Prepared source identity/pose/target differs')
    rt.geometry.require_current(metadata);reference=rt.pose.require_route(metadata)
    slot=scene['plans'][target]
    need(set(slot)=={'plan'} and slot.get('plan') is not None,'AVAILABLE CPU-prepared canonical plan required')
    plan=rt.read_bound(slot['plan']);config=rt.read_bound(config_ref)
    prepared=rt.read_bound(scene['prepared_complete']);patient=rt.read_bound(scene['patient_receipt'])
    need(prepared.get('completed') is True and prepared.get('physics_executed') is False and
         prepared.get('training_eligible') is False and prepared.get('source_sha256')==rt.geometry.GEOMETRY_SOURCE_SHA256 and
         prepared.get('patient')==patient,'Actual completed CPU-installed patient required')
    rt.geometry.require_current(config);rt.geometry.require_current(plan);reference.verify_sources()
    snap=reference.snapshot();plan_bound=snap['sources']['fresh_plan'];patient_bound=snap['sources']['patient_receipt']
    need(rt.reference(plan_bound['file'])==slot['plan'] and rt.reference(patient_bound['file'])==scene['patient_receipt'] and
         config.get('patient_receipt')==patient_bound and patient.get('identity')==metadata['subject_id'] and
         plan.get('name')==target and plan.get('success') is True and
         plan.get('patient_provenance',{}).get('binding')==patient_bound,
         'Installed patient, reference and canonical target bindings differ')
    need(config.get('collision_profile')=='full' and config.get('posture_limits')==plan.get('posture_limits') and
         np.array_equal(config.get('world_from_left_base'),plan.get('world_from_left_base')) and
         np.array_equal(config.get('world_from_left_base'),snap['world_from_base']),
         'Original full collision profile/posture/base required')
    artifacts=rt.geometry.require_geometry_artifacts(config['collision_directory'],config)
    need(artifacts==plan.get('geometry_artifacts'),'Canonical geometry artifact binding differs')
    need(cpu_fixture or 'geometry_source' in config,'Original approved geometry source binding required')
    if 'geometry_source' in config:
        need(rt.reference(config['geometry_source'])['sha256']==rt.geometry.GEOMETRY_SOURCE_SHA256,
             'Approved geometry source bytes differ')
    transfer=np.asarray(plan.get('transfer_q'),float)
    need(transfer.ndim==2 and transfer.shape[1:]==(7,) and len(transfer)>=2 and np.isfinite(transfer).all(),
         'Finite canonical transfer_q required')
    return reference,plan,config,transfer,artifacts


def _original_verifier(config,plan,reference):
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits,PostureChecker
    from .contact_collision import ContactCollisionChecker
    kin=OpenArmKinematics(contact_frame=rt.geometry.contact_frame)
    limits=PostureLimits.from_dict(config['posture_limits'],kin)
    need(limits.to_dict()==plan['posture_limits'],'Original full posture limits differ')
    base=np.asarray(config['world_from_left_base'],float)
    checker=PostureChecker(ContactCollisionChecker(kin,base,config['collision_directory'],config,profile='full'),base,limits)
    def verify(q):
        need(limits.evaluate(kin,base,q,contact=False)['clear'],'Original posture rejected planned command')
        result=checker.check(q,clearance=.003,allow_patient_contact=False,compression=0.)
        need(result.get('clear') is True and np.isfinite(result.get('minimum_distance_m',np.nan)) and
             result['minimum_distance_m']>=.003-1e-9,'Original collision rejected planned command')
    return verify


def generate_planned_scene(scene,target_id,*,metadata,plan_config_ref,allowed_train_ids,artifact_output,
                           trajectory_factory=None,verify_factory=None):
    """Return {samples, slot}; rejection retains the declared body/pose/target slot.

    Defaults use the installed SDK/FCL and original JointTrajectory. Explicit
    factories support unqualified CPU fixtures and are recorded in source
    metadata. Run production defaults only on the prepared Linux CPU runtime.
    artifact_output must be a fresh per-case directory. Its separate raw setter
    archive retains the complete schedule, including omitted transition cells.
    """
    injected_factories=[name for name,factory in
        (('trajectory_factory',trajectory_factory),('verify_factory',verify_factory)) if factory is not None]
    slot=dict(slot_id='planned:'+scene['subject_id']+':'+scene['pose_id']+':'+target_id,
        subject_id=scene['subject_id'],pose_id=scene['pose_id'],target_id=target_id,
        kind='planned_ideal_tracking',status='rejected',**data.FLAGS)
    try:
        reference,plan,config,transfer,geometry=_scene(scene,target_id,metadata,plan_config_ref,allowed_train_ids,
            cpu_fixture=verify_factory is not None)
        if trajectory_factory is None:
            from sim.auscultation_pi05_v6.execution import JointTrajectory
            trajectory_factory=JointTrajectory
        verify=(verify_factory or _original_verifier)(config,plan,reference)
        trajectory=trajectory_factory(transfer[0],dt=rt.handoff.DT,**rt.handoff.PLANNING_LIMITS)
        verify(transfer[0])
        preview=recovery._preview(trajectory,np.repeat(transfer[:1],3,axis=0),transfer[1:],verify,settle=True,
            trajectory_factory=trajectory_factory,deadline=time.monotonic()+recovery.MAX_PLAN_CPU_S)
        arrays,windows=planned_windows(preview,metadata)
        raw_schedule=data.write_planned_schedule(artifact_output,preview,metadata,scene['pose_id'],allowed_train_ids)
        bindings=dict(source=scene['plans'][target_id]['plan'],geometry=plan_config_ref,
            prepared=scene['prepared_complete'],patient=scene['patient_receipt'],generator=rt.reference(__file__),
            numerical_preview=rt.reference(recovery.__file__),retained_dynamics=rt.reference(rt.handoff.__file__),
            raw_schedule=raw_schedule)
        source=data.make_source('planned_ideal_tracking',metadata,scene['pose_id'],bindings,windows)
        summary={k:deepcopy(v) for k,v in preview.items() if k not in ('commands','cells')}
        summary.update(raw_setter_cells=len(preview['cells']),raw_setter_sha256=hashlib.sha256(
            np.asarray(preview['cells'],dtype='<f8').tobytes()).hexdigest())
        source.update(verification_profile='injected_cpu_fixture_unqualified' if injected_factories else 'original_full_cpu_preview',
            injected_factories=injected_factories,
            geometry_artifacts=geometry,planning_limits=deepcopy(rt.handoff.PLANNING_LIMITS),
            preview=summary,history_clock='generated_20hz; startup q0 repeated; no measured history')
        samples=data.bind_samples(arrays,[source],allowed_train_ids)
        # Reverify original geometry and references after the bounded CPU preview.
        need(rt.geometry.require_geometry_artifacts(config['collision_directory'],config)==geometry,'Geometry changed during planned preview')
        reference.verify_sources()
        slot.update(status='accepted',source_id=samples['sources'][0]['source_id'])
        return dict(samples=samples,slot=slot)
    except (ValueError,KeyError,TypeError,RuntimeError,TimeoutError) as error:
        slot['rejection']=type(error).__name__+': '+str(error)
        return dict(samples=None,slot=slot)

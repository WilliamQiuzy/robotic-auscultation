"""Pure hash-bound M2 pilot configuration builders; never launch or qualify.

Both builders take preparation_config, prepared_complete and patient_receipt
references with {file,sha256}. COMPLETE's parent owns candidate_physics.usdc
and geometry/. Metadata identity/split preflight precedes body/scene/collision
payload hashing; this is not a claim that no TEST metadata path was accessed.
Returned {config,receipt} is JSON serializable. Caller writes it and invokes
the existing geometry_plan or geometry_qualification CLI separately.

The deterministic position perturbation is a synthetic fixed 4 mm bias, visible
in the actual state17 target input. It is not an empirical noise calibration.
The requested gt_plus_measured_noise contract label does not change that fact.
No orientation/normal noise, filtering, new runtime controls or force changes.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import numpy as np

from . import m2_contract as contract
from .geometry_epoch import GEOMETRY_ID,GEOMETRY_SOURCE_SHA256,require_current
from .m2_capture import APPROVED_CAMERA_SHA256
from .m2_sensor_probe import CALIBRATION_SHA256

# Resolve shared assets from the actual imported contract, including isolated
# overlays whose new helper file lives outside the frozen source checkout.
ROOT=Path(contract.__file__).resolve().parents[2]
CAMERA=ROOT/'sim/camera_framing_20260923/recommended_camera.json'
CALIBRATION=ROOT/'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
FLAGS=dict(physics_executed=False,physical_validation=False,physics_qualified=False,
           training_eligible=False,data_accepted=False,learned_policy=False)


def need(ok,message):
    if not ok:raise ValueError(message)


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def reference(path):
    path=Path(path).resolve();return dict(file=str(path),sha256=hash_file(path))


def bound_json(ref):
    need(isinstance(ref,dict) and set(ref)=={'file','sha256'} and
         isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
         'Explicit file/SHA256 reference required')
    data=Path(ref['file']).read_bytes()
    need(hashlib.sha256(data).hexdigest()==ref['sha256'],'Hash-bound metadata changed: '+ref['file'])
    return json.loads(data)


def _prepared(preparation_config,prepared_complete,patient_receipt,*,formal_preparation=None):
    if formal_preparation is not None:
        from .m2_given_contact_evaluation import authorize_prepared
        authorize_prepared(formal_preparation,preparation_config,prepared_complete,patient_receipt)
    prep=bound_json(preparation_config);complete=bound_json(prepared_complete);patient=bound_json(patient_receipt)
    require_current(prep);spec=prep.get('patient_asset',{});require_current(spec)
    need(prep.get('prepare_only') is True and complete.get('completed') is True and
         complete.get('geometry_id')==GEOMETRY_ID and complete.get('source_sha256')==GEOMETRY_SOURCE_SHA256,
         'Completed current prepare-only scene required')
    need(complete.get('patient')==patient and patient.get('non_patient_scene_unchanged') is True,
         'Installed patient receipt differs from prepared scene')
    # Trusted hashes authenticate these metadata bytes; all identity checks run
    # before any body NPZ, installed USD or exported collision payload is read.
    split=patient.get('split');subject=patient.get('identity')
    need((split in ('train','validation') if formal_preparation is None else split=='test') and spec.get('expected_split','train')==split and
         subject not in contract.QUARANTINED_IDENTITIES,'TRAIN/VALIDATION identity split only; quarantined IDs forbidden')
    split_ref=dict(file=spec['identity_split'],sha256=contract.IDENTITY_SPLIT_SHA256)
    splits=bound_json(split_ref)['splits']
    need([name for name,ids in splits.items() if subject in ids]==[split] and
         patient.get('identity_split_sha256')==contract.IDENTITY_SPLIT_SHA256,'Frozen identity split membership differs')
    directory=Path(spec['directory']).resolve()
    manifest=bound_json(dict(file=str(directory/'asset_manifest.json'),sha256=spec['manifest_sha256']))
    summary=bound_json(dict(file=str(directory/'summary.json'),sha256=spec['summary_sha256']))
    reconstructed=False
    if ('reconstruction_source' in spec or
            summary.get('schema')=='m2_reconstructed_observed_seated_retarget_v1'):
        from .observed_reconstruction import validate_reconstruction
        validate_reconstruction(spec,summary)
        need(patient.get('reconstruction_source')==spec['reconstruction_source'] and
             patient.get('reconstruction_cohort')==spec['reconstruction_cohort'] and
             patient.get('original_depth_plane_bytes_recovered') is False,
             'Installed reconstruction source receipt differs')
        reconstructed=True
    need(summary.get('identity_split_sha256')==contract.IDENTITY_SPLIT_SHA256 and
         (summary.get('schema')=='m2_observed_seated_retarget_v1' or reconstructed),'Observed retarget identity provenance required')
    rows=[r for r in manifest if r.get('file')==spec['file']]
    need(len(rows)==1,'Body manifest identity must be unique');body=rows[0]
    need(body.get('identity')==subject and body.get('split')==split and
         body.get('sha256')==patient.get('asset_sha256') and body.get('mesh_scale')==1. and
         body.get('numeric_candidate_screen_passed') is True,'Body identity/receipt mismatch')
    from .patient_scene import layout_translation
    need(np.allclose(patient.get('layout_translation_world_m'),layout_translation(spec),rtol=0,atol=1e-12),
         'Installed patient layout differs from preparation config')
    body_path=(directory/spec['file']).resolve()
    need(body_path.parent==directory,'Body path escapes authenticated manifest')
    root=Path(prepared_complete['file']).resolve().parent
    need(Path(patient_receipt['file']).resolve()==root/'patient_receipt.json','Patient receipt must belong to prepared scene')
    if 'scene_dependency_inventory' in complete:
        from . import scene_store
        inventory=scene_store.validate_prepared_dependency_inventory(complete)
        need(inventory['primary']==scene_store.source_ref(root/'candidate_physics.usdc'),
             'Layered primary must be the exact authenticated prepared scene path/bytes/SHA')
    need(hash_file(body_path)==body['sha256'],'Body payload hash changed')
    need(hash_file(prep['geometry_source'])==GEOMETRY_SOURCE_SHA256,'Original approved geometry source changed')
    need(hash_file(root/'candidate_physics.usdc')==complete.get('effective_scene_sha256'),'Prepared scene payload changed')
    artifacts=complete.get('geometry_artifacts',{})
    for name,key in [('collision_geometry.json','collision_geometry_sha256'),('collision_geometry.npz','collision_arrays_sha256')]:
        need(hash_file(root/'geometry'/name)==artifacts.get(key),'Exported geometry payload changed: '+name)
    base=contract._pose(complete['world_from_left_base'],'actual robot base')
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    PostureLimits.from_dict(prep['posture_limits'],OpenArmKinematics())
    provenance=dict(preparation_config=deepcopy(preparation_config),prepared_complete=deepcopy(prepared_complete),
        patient_receipt=deepcopy(patient_receipt),subject_id=subject,split=split,
        identity_split_sha256=contract.IDENTITY_SPLIT_SHA256,body=reference(body_path),
        effective_scene=reference(root/'candidate_physics.usdc'),geometry_artifacts=deepcopy(artifacts),
        payload_identity_preflight=True,test_payloads_opened=formal_preparation is not None,
        test_metadata_path_nonaccess_claim=False,**FLAGS)
    return prep,complete,patient,root,base,provenance


def _target(patient,name):
    need(name in [s+str(i) for s in 'LR' for i in range(1,6)],'One explicit target required')
    rows=[row for row in patient['targets'] if row.get('name')==name]
    need(len(rows)==1,'Installed target missing or duplicated');target=deepcopy(rows[0])
    need(target.get('contact_plane_radius_m')==.023,'Target requires actual 23 mm contact plane')
    for key in ('normal_world','contact_plane_normal_world'):
        value=contract._array(target[key],(3,),key)
        need(abs(np.linalg.norm(value)-1.)<=1e-8,'Invalid installed target plane normal')
    contract._array(target['position_world_m'],(3,),'target point')
    return target


def _profile(value):
    need(value in ('full','certified_temporal_v1'),'Unsupported collision profile')
    return value


def build_geometry_plan(*,preparation_config,prepared_complete,patient_receipt,target,
                        plane_tilt_limit_deg=3.,fallback_reason=None,seeds_per_candidate=12,
                        per_gap_seconds=180.,collision_profile='full',planning_plane_margin_deg=None,formal_preparation=None):
    """Optional strict-plane reserve proposes geometry, never physical margin."""
    prep,complete,patient,root,base,receipt=_prepared(preparation_config,prepared_complete,patient_receipt,
        formal_preparation=formal_preparation)
    row=_target(patient,target)
    need(type(plane_tilt_limit_deg) in (int,float) and plane_tilt_limit_deg in (3.,10.),'Only 3 or explicit 10 degree plane bound')
    need(plane_tilt_limit_deg==3. or isinstance(fallback_reason,str) and bool(fallback_reason.strip()),
         '10 degree fallback requires an explicit recorded reason')
    from .finite_head import planning_plane_policy
    policy=planning_plane_policy(plane_tilt_limit_deg,planning_plane_margin_deg)
    need(type(seeds_per_candidate) is int and 1<=seeds_per_candidate<=12 and
         type(per_gap_seconds) in (int,float) and np.isfinite(per_gap_seconds) and 0<per_gap_seconds<=1800,
         'Finite bounded planner work required')
    config={key:deepcopy(prep[key]) for key in ('geometry_id','geometry_source','geometry_source_sha256','freshly_generated','posture_limits')}
    config.update(collision_directory=str(root/'geometry'),world_from_left_base=base.tolist(),
        **complete['geometry_artifacts'],patient_receipt=deepcopy(patient_receipt),targets=[row],
        loaded_compression_m=.0025,fresh_pre_gap_search_m=[.025,.03],seeds_per_candidate=seeds_per_candidate,
        per_gap_seconds=float(per_gap_seconds),plane_tilt_limit_deg=float(plane_tilt_limit_deg),
        collision_profile=_profile(collision_profile))
    receipt.update(schema='m2_pilot_geometry_config_v1',target=target,tilt_relaxed=plane_tilt_limit_deg==10.,
        fallback_reason=fallback_reason,cli_module='sim.ausc_policy.geometry_plan',collision_profile=collision_profile)
    if policy is not None:
        config['planning_plane_margin_deg']=policy['margin_deg']
        receipt['planning_plane_policy']=policy
    return dict(config=config,receipt=receipt)


def build_given_contact_expert(*, preparation_config, prepared_complete, patient_receipt, fresh_plan,
                               episode_id, available_at_s=0., **options):
    """Reuse checked geometry/config assembly before capture, with a NEW task.

    This reads no old capture/data/weights. Shared assembly validates the same
    asset/fit/clearance/force settings; the unused noise intent is removed before
    returning a runnable configuration. Only the new teacher may acquire it.
    """
    from . import m2_given_contact_pose as given, m2_given_contact_controller as controller, m2_given_contact_intent as intent, m2_given_contact_handoff as handoff
    need('noise_seed' not in options and 'target_confidence' not in options,
         'Known-goal task has no noise seed or accuracy-confidence parameter')
    assembled = build_compact_expert(preparation_config=preparation_config, prepared_complete=prepared_complete,
        patient_receipt=patient_receipt, fresh_plan=fresh_plan, episode_id=episode_id, noise_seed=0, **options)
    config, receipt = assembled['config'], assembled['receipt']
    plan = bound_json(fresh_plan); patient = bound_json(patient_receipt)
    target = _target(patient, plan['name'])
    ref = given.build_given_reference(dict(target, manifest_ref=deepcopy(patient_receipt)),
        fresh_plan, patient_receipt, plan['world_from_left_base'], available_at_s)
    capture = config['fresh_expert']['raw_multimodal_capture']; settings = capture['compact']
    meta = given.route_metadata(settings['metadata'], ref)
    meta['handoff_policy_id']=handoff.POLICY_ID
    settings.update(metadata=meta, estimated_target_base=ref.pose_base.tolist(), target_confidence=1.,
                    target_estimate_source_id='given-contact-'+ref.sha256)
    codes = settings['code_bindings']; codes.pop('noise_model', None)
    codes.update(target_estimator=hash_file(given.__file__), action_source=hash_file(controller.__file__),
                 raw_intent_dispatcher=hash_file(intent.__file__),interior_handoff=hash_file(handoff.__file__))
    config['m2_given_contact_control'] = dict(schema=given.TASK_ID, metadata=deepcopy(meta), episode_id=episode_id,
        handoff_policy_id=handoff.POLICY_ID)
    receipt.pop('target_estimate_noise', None)
    receipt['source_bindings'].pop('noise_model', None)
    receipt['source_bindings'].update(target_estimator=reference(given.__file__), action_source=reference(controller.__file__),
                                    raw_intent_dispatcher=reference(intent.__file__),interior_handoff=reference(handoff.__file__))
    receipt.update(schema='m2_given_contact_compact_expert_config_v2', task_id=given.TASK_ID,
        action_semantics_id=intent.ACTION_SEMANTICS_ID,
        observation_contract=given.observation_contract(), given_reference=ref.snapshot(),
        reference_sha256=ref.sha256,handoff_policy_id=handoff.POLICY_ID,
        teacher_boundary='first original coarse observation; fresh interior trigger, cloned stop preview, actual settling then L3_ALIGN',
        translation_noise='absent only in explicitly named known-goal task', physical_qualification=False,
        learned_policy=False, production_act_ready=False)
    from .expert_capture import capture_settings
    capture_settings(config['fresh_expert'])
    return assembled


def target_estimate(target,base,seed):
    """Plane frame z points inward; choose roll deterministically, add 4mm bias."""
    need(type(seed) is int and 0<=seed<2**63,'Nonnegative deterministic noise seed required')
    z=-np.asarray(target['contact_plane_normal_world'],float)
    reference_axis=np.array([1.,0,0]) if abs(z[0])<.9 else np.array([0.,1,0])
    x=reference_axis-z*np.dot(reference_axis,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    world=np.eye(4);world[:3,:3]=np.column_stack([x,y,z]);world[:3,3]=target['position_world_m']
    clean=np.linalg.inv(base)@world
    direction=np.random.Generator(np.random.PCG64(seed)).normal(size=3)
    bias=.004*direction/np.linalg.norm(direction)
    estimate=clean.copy();estimate[:3,3]+=bias
    receipt=dict(schema='m2_fixed_visible_position_bias_v1',seed=seed,prng='numpy_PCG64',
        magnitude_m=.004,position_bias_base_m=bias.tolist(),clean_target_base=clean.tolist(),
        estimated_target_base=estimate.tolist(),normal_noise_applied=False,orientation_noise_applied=False,
        applied_to_model_input=True,fixed_per_episode=True,empirically_calibrated_noise=False,
        description='Synthetic fixed-magnitude position bias; no measured normal-noise claim')
    return estimate,receipt


def build_compact_expert(*,preparation_config,prepared_complete,patient_receipt,fresh_plan,
                         episode_id,noise_seed,camera_file=CAMERA,calibration_file=CALIBRATION,
                         target_confidence=1.,max_frames=5000,collision_profile='full',scene_store=None,
                         retainer_witness_contract=None,formal_preparation=None):
    prep,complete,patient,root,base,receipt=_prepared(preparation_config,prepared_complete,patient_receipt,
        formal_preparation=formal_preparation)
    plan=bound_json(fresh_plan);require_current(plan)
    need(plan.get('success') is True,'A successful fresh geometric plan is required')
    target=_target(patient,plan.get('name'));bound=plan.get('patient_provenance',{})
    need(bound.get('binding')==patient_receipt and bound.get('identity')==patient['identity'] and
         bound.get('asset_sha256')==patient['asset_sha256'],'Fresh plan installed patient binding mismatch')
    need(plan.get('geometry_artifacts')==complete['geometry_artifacts'] and
         np.allclose(plan['world_from_left_base'],base,atol=1e-9,rtol=0) and
         plan.get('posture_limits')==prep['posture_limits'],'Fresh plan geometry/base/posture changed')
    need(plan.get('loaded_compression_m')==.0025 and plan.get('free_approach_start_gap_m') in (.025,.03),
         'Fresh plan must retain .0025 compression and approved pre-gap')
    need(np.allclose(plan['position_world'],target['position_world_m'],rtol=0,atol=1e-10) and
         np.allclose(plan['normal_world'],target['normal_world'],rtol=0,atol=1e-10),'Fresh plan target differs from installed body')
    from . import fresh_expert,m2_capture
    plane=fresh_expert.contact_plane_contract(plan)
    need(plane is not None and np.allclose(plane['contact_plane_normal_world'],target['contact_plane_normal_world'],
         rtol=0,atol=1e-10),'Fresh plan 23 mm plane differs from installed body')
    need(isinstance(episode_id,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',episode_id) is not None,
         'Explicit bounded episode identity required')
    need(type(max_frames) is int and 1<=max_frames<=6000,'Compact maximum frames must be 1..6000')
    confidence=contract._scalar(target_confidence,'target confidence',maximum=1.)
    need(hash_file(camera_file)==APPROVED_CAMERA_SHA256 and hash_file(calibration_file)==CALIBRATION_SHA256,
         'Approved camera/Hall calibration bytes changed')
    estimate,noise=target_estimate(target,base,noise_seed)
    metadata=dict(geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,freshly_generated=True,
        **complete['geometry_artifacts'],contract_id=contract.CONTRACT_ID,state_dim=17,horizon=20,
        observation_contract=contract.observation_contract(),
        execution_prefix=10,control_hz=20,joint_unit='rad',position_unit='m',target_pose_frame='openarm_left_link0',
        target_source='gt_plus_measured_noise',episode_id=episode_id,target_id=target['name'],
        subject_id=patient['identity'],split=patient['split'],identity_split_sha256=contract.IDENTITY_SPLIT_SHA256,
        source_kind='isaac_live_physics',replay=False,patient_asset_sha256=patient['asset_sha256'],
        installed_patient_receipt_sha256=patient_receipt['sha256'],physical_qualification=False,
        artifact_hashes_verified=False,**FLAGS)
    source_files=dict(native_backend=Path(m2_capture.__file__),state_adapter=Path(contract.__file__),
        target_estimator=Path(__file__),noise_model=Path(__file__),
        action_source=Path(fresh_expert.__file__),body=Path(receipt['body']['file']))
    codes={key:hash_file(path) for key,path in source_files.items()}
    compact=dict(metadata=metadata,code_bindings=codes,max_frames=max_frames,limit_bytes=100_000_000,
        estimated_target_base=estimate.tolist(),target_confidence=confidence,
        target_estimate_source_id='m2-fixed4mm-seed-'+str(noise_seed)+'-'+patient_receipt['sha256'][:16])
    capture=dict(enabled=True,profile='compact',episode_id=episode_id,camera_file=str(Path(camera_file).resolve()),
        camera_sha256=APPROVED_CAMERA_SHA256,calibration_file=str(Path(calibration_file).resolve()),
        calibration_sha256=CALIBRATION_SHA256,compact=compact)
    config=deepcopy(prep);config.pop('prepare_only',None)
    # Explicitly discard preparation-only experiment controls and nominal-target
    # declarations; preserve the exact physics and force settings themselves.
    for key in ('patient_preview','m2_sensor_probe','m2_render_settling_probe','contact_probe'):
        config.pop(key,None)
    need('m2_act_deployment' not in config,'Expert capture cannot be attached to a learned-policy configuration')
    config.update(targets=[target],raw_target_provenance=dict(source='installed patient surface anchor and 23 mm local plane',
        patient_receipt=deepcopy(patient_receipt),identity=patient['identity'],asset_sha256=patient['asset_sha256'],
        target_id=target['name'],nominal_template_targets_used=False))
    config['fresh_expert']=dict(config['fresh_expert'],plans=[deepcopy(fresh_plan)],
        raw_multimodal_capture=capture,collision_profile=_profile(collision_profile))
    from .expert_capture import capture_settings
    if retainer_witness_contract is not None:
        from .retainer_witness import load_contract
        load_contract(retainer_witness_contract)
        config['fresh_expert']['native_retainer_audit']=deepcopy(retainer_witness_contract)
        metadata['native_retainer_audit']=deepcopy(retainer_witness_contract)
        receipt['native_retainer_audit']=deepcopy(retainer_witness_contract)
    capture_settings(config['fresh_expert'])
    receipt.update(schema='m2_pilot_compact_expert_config_v2',fresh_plan=deepcopy(fresh_plan),
        observation_contract=contract.observation_contract(),
        target_estimate_noise=noise,source_bindings={k:reference(p) for k,p in source_files.items()},
        camera=reference(camera_file),hall_calibration=reference(calibration_file),
        metadata_is_acquisition_intent_not_execution_evidence=True,cli_module='sim.ausc_policy.geometry_qualification',
        collision_profile=collision_profile,compact_max_frames=max_frames,total_byte_cap=100_000_000)
    if scene_store is not None:
        need(isinstance(scene_store,dict) and set(scene_store) in
             ({'root','batch_id'},{'root','batch_id','sdk_mdl_manifest'}),
             'Explicit scene store root/batch_id and optional hash-bound SDK MDL manifest required')
        if 'sdk_mdl_manifest' in scene_store:
            need(isinstance(scene_store['sdk_mdl_manifest'],dict),'Explicit SDK MDL manifest reference required')
        from .scene_store import build_binding
        binding=build_binding(preparation_config=preparation_config,prepared_complete=prepared_complete,
            patient_receipt=patient_receipt,store_root=scene_store['root'],batch_id=scene_store['batch_id'],
            retainer_witness_contract=retainer_witness_contract,sdk_mdl_manifest=scene_store.get('sdk_mdl_manifest'),
            formal_preparation=formal_preparation)
        config['compact_scene_storage']=binding;receipt['compact_scene_storage']=deepcopy(binding)
    return dict(config=config,receipt=receipt)

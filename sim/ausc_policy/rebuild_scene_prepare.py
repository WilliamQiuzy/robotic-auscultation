"""Bounded CPU scene preparation/planning for the new reconstructed cohort.

Run inside the original CPU owner, with CUDA_VISIBLE_DEVICES empty. No Kit,
simulation, old formal population gate or GPU job is launched. All requested
subject/pose/target slots survive failures in the ordered scene manifest.
The default purpose excludes frozen evaluation; that cohort requires an explicit
frozen_evaluation request and remains ineligible for training.
"""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys

from . import observed_reconstruction as sources
from . import m2_pilot_config as builder
from .geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from .m2_capture import APPROVED_CAMERA_SHA256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'm2_rebuild_scene_manifest_v1'
REQUEST_SCHEMA = 'm2_rebuild_scene_prepare_request_v1'
PHYSICS = dict(position_iterations=64, velocity_iterations=1, armature_kg_m2=.02)
FLAGS = dict(physics_executed=False, physical_validation=False, training_eligible=False)


def need(ok, message):
    if not ok: raise ValueError(message)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reference(path):
    ref = sources.reference(path)
    return dict(path=ref['file'], bytes=Path(ref['file']).stat().st_size, sha256=ref['sha256'])


def read(path):
    return json.loads(Path(path).read_bytes())


def validate_targets(targets):
    names = {s+str(i) for s in 'LR' for i in range(1, 6)}
    need(isinstance(targets, list) and 2 <= len(targets) <= 4 and
         len(targets) == len(set(targets)) and set(targets) <= names,
         'Explicit two to four unique retained target IDs required')
    return list(targets)


def preparation_config(template, spec, *, base, asset_root, native_python,
                       ld_library_path, approved_geometry):
    need(template.get('geometry_id') == GEOMETRY_ID and
         template.get('geometry_source_sha256') == GEOMETRY_SOURCE_SHA256 and
         template.get('freshly_generated') is True, 'Approved 5mm/recess1mm geometry template required')
    expert = template.get('fresh_expert', {})
    need(expert.get('passive_preload_N') == 1.5 and
         expert.get('passive_preload_model') == {'mode': 'single_explicit_v1'},
         'Original single explicit 1.5N preload required')
    need(not any(k.startswith(('m2_given_', 'm2_act_')) for k in template),
         'Use the nominal preparation template, without old deployment/capture routes')
    configured = deepcopy(template)
    configured.update(base=str(base), asset_root=str(asset_root), isaac_python=str(native_python),
                      ld_library_path=str(ld_library_path), geometry_source=str(approved_geometry),
                      physics=dict(PHYSICS), prepare_only=True, patient_asset=deepcopy(spec))
    configured['fresh_expert']['plans'] = []
    configured['fresh_expert'].pop('raw_multimodal_capture', None)
    for key in ('patient_preview', 'm2_sensor_probe', 'm2_render_settling_probe', 'contact_probe'):
        configured.pop(key, None)
    return configured


def run_cpu(command, logfile, timeout, *, ld_library_path=None):
    need(isinstance(timeout, (int, float)) and 0 < timeout <= 7200, 'Finite CPU child timeout required')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if ld_library_path is not None: env['LD_LIBRARY_PATH'] = str(ld_library_path)
    # Preserve the original owner marker; never manufacture ownership here.
    need(env.get('AUSC_OWNED_RUN') == '1', 'Original CPU owner environment required')
    with Path(logfile).open('x') as log:
        try:
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout, check=False)
            return dict(returncode=done.returncode, timed_out=False, command=command)
        except subprocess.TimeoutExpired:
            return dict(returncode=None, timed_out=True, command=command)
        except OSError as error:
            return dict(returncode=None, timed_out=False, command=command, error=str(error))


def plan_slot(target, output, process):
    output = Path(output)
    plan = output / (target+'_fresh_plan.json')
    complete = output / 'COMPLETE.json'
    attempt = dict(target=target, process=process, **FLAGS)
    if plan.is_file(): attempt['retained_plan'] = reference(plan)
    if complete.is_file(): attempt['complete'] = reference(complete)
    reason = 'planning_process_failed'
    if process.get('timed_out'): reason = 'planning_timeout'
    elif process.get('returncode') == 0 and complete.is_file():
        data = read(complete)
        rows = [r for r in data.get('targets', []) if r.get('name') == target]
        if data.get('completed') is True and len(rows) == 1 and data.get('physics_executed') is False:
            attempt['geometry_path_found'] = rows[0].get('geometry_path_found') is True
            reason = 'geometry_path_not_found'
            if attempt['geometry_path_found'] and plan.is_file():
                result = read(plan)
                if result.get('name') == target and result.get('success') is True:
                    attempt['status'] = 'planned'
                    return dict(plan=reference(plan)), attempt
                reason = 'fresh_plan_not_successful_or_target_mismatch'
        else: reason = 'planning_completion_invalid'
    attempt.update(status='unavailable', reason=reason)
    return dict(status='unavailable', reason=reason), attempt


def validate_scope(cohorts, purpose):
    need(purpose in ('capture', 'frozen_evaluation'), 'Explicit capture or frozen_evaluation purpose required')
    allowed = {'frozen_evaluation'} if purpose == 'frozen_evaluation' else {'train', 'selection_validation'}
    need(cohorts and len(cohorts) == len(set(cohorts)) and set(cohorts) <= allowed,
         'frozen_evaluation purpose permits only frozen_evaluation; capture excludes frozen evaluation')


def load_scenes(complete_ref, cohorts, *, purpose='capture'):
    validate_scope(cohorts, purpose)
    complete = sources.bound_json(complete_ref)
    need(complete.get('schema') == 'm2_observed_reconstruction_complete_v1' and
         complete.get('completed') is True and complete.get('training_eligible') is False,
         'Completed new unqualified reconstruction required')
    bundle_ref = complete['reconstruction_source']
    bundle = sources.bound_json(bundle_ref)
    need(bundle.get('schema') == sources.BUNDLE_SCHEMA, 'New reconstruction source bundle required')
    entries = {row['cohort']['name']: row for row in complete['cohorts']}
    fixed = {row['name']: row for row in bundle['cohorts']}
    identities = [identity for row in bundle['cohorts'] for identity in row['identities']]
    need(len(fixed) == len(bundle['cohorts']) and set(fixed) <= {'train', 'selection_validation', 'frozen_evaluation'} and
         len(identities) == len(set(identities)), 'Reconstruction source cohorts overlap identities or duplicate roles')
    need(len(entries) == len(complete['cohorts']), 'Duplicate reconstructed cohort entries')
    need(set(cohorts) <= set(entries), 'Requested reconstruction cohort is absent')
    scenes, bound = [], [complete_ref, bundle_ref]
    for name in cohorts:
        entry = entries[name]
        need(entry['cohort'] == fixed.get(name), 'Reconstruction completion shrank or changed the source cohort')
        summary = sources.bound_json(entry['summary'])
        specs = sources.bound_json(entry['patient_specs'])
        bound.extend([entry['summary'], entry['patient_specs']])
        expected_split = 'train' if name == 'train' else 'validation'
        need(entry['cohort']['split'] == expected_split, 'Reconstructed cohort split differs')
        for spec in specs:
            need(spec.get('reconstruction_cohort') == name and spec.get('expected_split') == expected_split and
                 spec.get('reconstruction_source') == bundle_ref, 'Patient specification/source cohort differs')
            sources.validate_reconstruction(spec, summary)
            manifest_ref = dict(file=str(Path(spec['directory'])/'asset_manifest.json'), sha256=spec['manifest_sha256'])
            rows = [r for r in sources.bound_json(manifest_ref) if r['file'] == spec['file']]
            need(len(rows) == 1, 'Unique reconstructed asset required')
            row = rows[0]
            pose = row['source_pose']
            pose_id = Path(pose['sequence_path']).parent.name+f'_f{pose["frame_index"]:04d}'
            scenes.append(dict(subject_id=row['identity'], split=expected_split, cohort=name,
                               pose_id=pose_id, patient_spec=spec,
                               numeric_candidate_screen_passed=row.get('numeric_candidate_screen_passed') is True))
            bound.append(manifest_ref)
        expected = {(identity, sequence+f'_f{frame:04d}') for identity in entry['cohort']['identities']
                    for _, sequence, frame, _ in sources.SOURCES}
        actual = [(r['subject_id'], r['pose_id']) for r in scenes if r['cohort'] == name]
        need(len(actual) == len(set(actual)) and set(actual) == expected,
             'Every requested subject must retain both observed poses')
    need(len({s['subject_id'] for s in scenes}) == sum(len(entries[n]['cohort']['identities']) for n in cohorts),
         'Scene cohorts overlap identities')
    return scenes, bundle_ref, bound


def verify_preparation(directory, spec):
    directory = Path(directory)
    complete = read(directory/'COMPLETE.json')
    patient = read(directory/'patient_receipt.json')
    profile = read(directory/'physics_profile.json')
    camera = read(directory/'approved_camera_restoration.json')
    preload = read(directory/'passive_preload_saved_usd_readback.json')
    need(complete.get('completed') is True and complete.get('source_sha256') == GEOMETRY_SOURCE_SHA256 and
         complete.get('physics_executed') is False and complete.get('training_eligible') is False and
         complete.get('patient') == patient, 'CPU preparation completion invalid')
    need(profile.get('profile') == PHYSICS and camera.get('camera_sha256') == APPROVED_CAMERA_SHA256 and
         camera.get('restored_user_approved_pose') is True, 'Parking solver or approved down7 camera differs')
    need(preload.get('mode') == 'single_explicit_v1' and preload.get('nominal_preload_N') == 1.5 and
         preload.get('explicit_effort_N') == -1.5 and preload.get('drive', {}).get('target_position_m') == 0.,
         'Saved single explicit preload readback differs')
    need(patient.get('reconstruction_source') == spec['reconstruction_source'] and
         patient.get('reconstruction_cohort') == spec['reconstruction_cohort'] and
         patient.get('original_depth_plane_bytes_recovered') is False,
         'Prepared patient lost reconstruction provenance')


def prepare_scene(source, *, template, base, asset_root, native_python, planner_python,
                  ld_library_path, approved_geometry, output, targets, prepare_seconds, per_gap_seconds):
    """One complete scene; children inherit the original CPU owner's session."""
    attempts = []
    scene_id = source['subject_id']+'-'+source['pose_id']
    directory = output/scene_id
    row = {key: source[key] for key in ('subject_id', 'split', 'cohort', 'pose_id')}
    row.update(preparation_config=None, prepared_complete=None, patient_receipt=None,
               plans={target: dict(status='unavailable', reason='preparation_unavailable') for target in targets})
    try:
        directory.mkdir()
        need(source['numeric_candidate_screen_passed'], 'Reconstructed asset failed its numeric screen')
        spec = source['patient_spec']
        configured = preparation_config(template, spec, base=base, asset_root=asset_root,
            native_python=native_python, ld_library_path=ld_library_path, approved_geometry=approved_geometry)
        config_path = directory/'preparation.json'
        save(config_path, configured)
        row['preparation_config'] = reference(config_path)
        prepared = directory/'prepared'
        process = run_cpu([str(native_python), '-B', '-m', 'sim.ausc_policy.geometry_qualification',
            '--config', str(config_path), '--output', str(prepared), '--cpu-prepare-only'],
            directory/'prepare.log', prepare_seconds, ld_library_path=ld_library_path)
        attempts.append(dict(scene_id=scene_id, stage='prepare', process=process, **FLAGS))
        need(process.get('returncode') == 0 and not process.get('timed_out'), 'CPU scene preparation process failed')
        verify_preparation(prepared, spec)
        refs = dict(preparation_config=builder.reference(config_path),
            prepared_complete=builder.reference(prepared/'COMPLETE.json'),
            patient_receipt=builder.reference(prepared/'patient_receipt.json'))
        row.update(prepared_complete=reference(prepared/'COMPLETE.json'), patient_receipt=reference(prepared/'patient_receipt.json'))
        for target in targets:
            plan_dir = directory/('plan_'+target)
            try:
                built = builder.build_geometry_plan(**refs, target=target, collision_profile='full',
                    per_gap_seconds=per_gap_seconds, plane_tilt_limit_deg=3., planning_plane_margin_deg=.25)
                plan_config = directory/(target+'_plan_config.json')
                save(plan_config, built['config'])
                process = run_cpu([str(planner_python), '-B', '-m', 'sim.ausc_policy.geometry_plan',
                    '--config', str(plan_config), '--output', str(plan_dir)],
                    directory/(target+'_plan.log'), 2*per_gap_seconds+120., ld_library_path=ld_library_path)
                slot, attempt = plan_slot(target, plan_dir, process)
            except Exception as error:
                slot = dict(status='unavailable', reason=type(error).__name__+': '+str(error))
                attempt = dict(target=target, status='unavailable', reason=slot['reason'], **FLAGS)
            row['plans'][target] = slot
            attempts.append(dict(scene_id=scene_id, stage='plan', **attempt))
    except Exception as error:
        reason = type(error).__name__+': '+str(error)
        row.update(preparation_error=reason, prepared_complete=None, patient_receipt=None)
        row['plans'] = {target: dict(status='unavailable', reason=reason) for target in targets}
        attempts.append(dict(scene_id=scene_id, stage='prepare_unavailable', reason=reason, **FLAGS))
    return row, attempts


def prepare_and_plan(*, complete_ref, template_ref, approved_geometry, base, asset_root,
                     native_python, planner_python, ld_library_path, output, targets,
                     cohorts, prepare_seconds=600., per_gap_seconds=180., purpose='capture', parallelism=1):
    need(sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
         os.environ.get('AUSC_OWNED_RUN') == '1', 'Original Linux CPU owner with empty CUDA visibility required')
    need(type(parallelism) is int and 1 <= parallelism <= 3, 'CPU parallelism must be an integer from 1 through 3')
    validate_scope(cohorts, purpose)
    targets = validate_targets(targets)
    need(30 <= prepare_seconds <= 1800 and 1 <= per_gap_seconds <= 1800, 'Bounded preparation/planner budgets required')
    template = sources.bound_json(template_ref)
    geometry_ref = sources.reference(approved_geometry)
    need(geometry_ref['sha256'] == GEOMETRY_SOURCE_SHA256, 'Approved geometry bytes differ')
    scenes, bundle_ref, bindings = load_scenes(complete_ref, cohorts, purpose=purpose)
    bindings.extend([template_ref, geometry_ref, sources.reference(builder.CAMERA)])
    need(bindings[-1]['sha256'] == APPROVED_CAMERA_SHA256, 'Approved down7 camera bytes differ')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results, attempts = [], []
    worker = partial(prepare_scene, template=template, base=base, asset_root=asset_root,
        native_python=native_python, planner_python=planner_python, ld_library_path=ld_library_path,
        approved_geometry=approved_geometry, output=output, targets=targets,
        prepare_seconds=prepare_seconds, per_gap_seconds=per_gap_seconds)
    # Only this owner thread writes progress; map preserves input order even if
    # later scenes finish first. Each scene runs one CPU subprocess at a time.
    context = ThreadPoolExecutor(max_workers=parallelism) if parallelism > 1 else nullcontext()
    with context as pool:
        ordered = pool.map(worker, scenes) if pool is not None else map(worker, scenes)
        for row, scene_attempts in ordered:
            results.append(row); attempts.extend(scene_attempts)
            save(output/'progress.json', dict(purpose=purpose, parallelism=parallelism, scenes=results, attempts=attempts, **FLAGS))
    for binding in bindings: sources.verify_ref(binding)
    available = sum('plan' in slot for row in results for slot in row['plans'].values())
    manifest = dict(schema=SCHEMA, purpose=purpose, parallelism=parallelism, targets=targets, subject_ids=list(dict.fromkeys(s['subject_id'] for s in scenes)),
        reconstruction_source=reference(bundle_ref['file']), reconstruction_complete=reference(complete_ref['file']),
        template=reference(template_ref['file']), approved_geometry=reference(approved_geometry),
        physics=dict(PHYSICS), collision_profile='full', plane_tilt_limit_deg=3., planning_plane_margin_deg=.25,
        scenes=results, attempts=attempts, requested_scenes=len(scenes), requested_target_slots=len(scenes)*len(targets),
        available_plans=available, unavailable_target_slots=len(scenes)*len(targets)-available,
        all_requested_plans_available=available==len(scenes)*len(targets), **FLAGS)
    save(output/'scene_manifest.json', manifest)
    completion = dict(schema='m2_rebuild_scene_prepare_complete_v1', completed=True, purpose=purpose, parallelism=parallelism,
        scene_manifest=reference(output/'scene_manifest.json'), requested_target_slots=manifest['requested_target_slots'],
        available_plans=available, **FLAGS)
    save(output/'COMPLETE.json', completion)
    return completion


def run_request(request, output):
    need(request.get('schema') == REQUEST_SCHEMA and request.get('experiment') == 'rebuild_scene_prepare',
         'Explicit CPU scene preparation request required')
    purpose = request.get('purpose', 'capture')
    cohorts = request.get('cohorts', ['frozen_evaluation'] if purpose == 'frozen_evaluation' else ['train', 'selection_validation'])
    validate_scope(cohorts, purpose)
    sources.verify_ref(request['approved_geometry'])
    return prepare_and_plan(complete_ref=request['reconstruction_complete'], template_ref=request['template'],
        approved_geometry=Path(request['approved_geometry']['file']), base=request['base'],
        asset_root=request['asset_root'], native_python=request['native_python'],
        planner_python=request['planner_python'], ld_library_path=request['ld_library_path'],
        output=output, targets=request['targets'], cohorts=cohorts, purpose=purpose, parallelism=request.get('parallelism', 1),
        prepare_seconds=request.get('prepare_seconds',600.), per_gap_seconds=request.get('per_gap_seconds',180.))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--complete', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--template', type=Path)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--native-python', type=Path)
    parser.add_argument('--planner-python', type=Path)
    parser.add_argument('--ld-library-path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', nargs='+')
    parser.add_argument('--cohorts', nargs='+')
    parser.add_argument('--purpose', choices=('capture', 'frozen_evaluation'))
    parser.add_argument('--parallelism', type=int)
    parser.add_argument('--prepare-seconds', type=float)
    parser.add_argument('--per-gap-seconds', type=float)
    args=parser.parse_args()
    if args.config is not None:
        need(all(value is None for value in (args.complete,args.sha256,args.template,args.geometry,args.base,
            args.asset_root,args.native_python,args.planner_python,args.ld_library_path,args.targets,
            args.cohorts,args.prepare_seconds,args.per_gap_seconds,args.purpose,args.parallelism)),
            'Use owner config or direct arguments, without mixed overrides')
        print(json.dumps(run_request(sources.bound_json(sources.reference(args.config)),args.output),indent=2))
        return
    need(all(value is not None for value in (args.complete,args.sha256,args.template,args.geometry,args.base,
         args.asset_root,args.native_python,args.planner_python,args.ld_library_path,args.targets)),
         'Complete direct CPU preparation arguments or owner --config required')
    result=prepare_and_plan(complete_ref=dict(file=str(args.complete.resolve()), sha256=args.sha256),
        template_ref=sources.reference(args.template), approved_geometry=args.geometry.resolve(),
        base=args.base.resolve(), asset_root=args.asset_root.resolve(), native_python=args.native_python.resolve(),
        planner_python=args.planner_python.resolve(), ld_library_path=args.ld_library_path,
        output=args.output, targets=args.targets,
        cohorts=args.cohorts or (['frozen_evaluation'] if args.purpose == 'frozen_evaluation' else ['train','selection_validation']),
        purpose=args.purpose or 'capture', parallelism=1 if args.parallelism is None else args.parallelism,
        prepare_seconds=600. if args.prepare_seconds is None else args.prepare_seconds,
        per_gap_seconds=180. if args.per_gap_seconds is None else args.per_gap_seconds)
    print(json.dumps(result, indent=2))


if __name__=='__main__': main()

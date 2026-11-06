"""New, opt-in observed M2 reconstruction; never restores old depth-plane bytes.

Linux authoring reuses frozen SMPL/observed parameters, canonical shape betas,
saved diagnostic joints, exact stool placement, 23 mm patches and full FCL.
The saved-joint rigid fit recovers orientation only, under the original level
floor assumption. No source depth, gravity, support, IK or physics is inferred.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import m2_retarget as retarget
from .geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from .observed_placement import SMPL_PARENTS

SCHEMA = 'm2_reconstructed_observed_seated_retarget_v1'
BUNDLE_SCHEMA = 'm2_observed_reconstruction_sources_v1'
REQUEST_SCHEMA = 'm2_observed_reconstruction_request_v1'
SAVED_JOINTS_SHA256 = '81c7b2db756caa58908f287e7ebe4a1950218a1c8d2122e90e8c7caaed74f712'
SOURCES = (('p000474', 'p000474_a001230', 0, '13161189d84b7d5c1141b0d7a8d713c31a4e295583105318adabbdeb553d3b4a'),
           ('p000476', 'p000476_a001231', 23, 'd8f5defe492f744f429c7515b1fb2a45c2e1fdb4f642aec35bffac47fe0ff953'))
FLAGS = dict(physics_qualified=False, training_eligible=False, seat_support_qualified=False,
             ik_qualified=False, static_scene_qa_passed=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(file=str(path), sha256=retarget.sha(path))


def verify_ref(ref):
    need(isinstance(ref, dict) and set(ref) == {'file', 'sha256'} and
         isinstance(ref['file'], str) and isinstance(ref['sha256'], str) and
         re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Explicit file/SHA256 reference required')
    need(retarget.sha(ref['file']) == ref['sha256'], 'Reconstruction source hash changed: ' + ref['file'])


def bound_json(ref):
    verify_ref(ref)
    payload = Path(ref['file']).read_bytes()
    need(hashlib.sha256(payload).hexdigest() == ref['sha256'], 'Source changed during JSON read')
    return json.loads(payload)


def recover_orientation(source_joints, saved_world_joints):
    """Proper rigid fit; reject changed bone geometry, reflections and bad rank."""
    source = np.asarray(source_joints, dtype=float)
    world = np.asarray(saved_world_joints, dtype=float)
    need(source.shape == world.shape == (24, 3) and np.isfinite(source).all() and
         np.isfinite(world).all(), 'Two finite 24-joint arrays required')
    x = source - source.mean(0)
    y = world - world.mean(0)
    need(np.linalg.matrix_rank(x, tol=1e-8) == 3 and np.linalg.matrix_rank(y, tol=1e-8) == 3,
         'Degenerate joint geometry cannot identify a rigid orientation')
    u, singular, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ u.T
    need(np.linalg.det(rotation) > .999999999, 'Reflection is not a proper rigid transform')
    translation = world.mean(0) - rotation @ source.mean(0)
    residual = np.linalg.norm(source @ rotation.T + translation - world, axis=1)
    children = np.arange(1, 24)
    before = np.linalg.norm(source[children] - source[SMPL_PARENTS[1:]], axis=1)
    after = np.linalg.norm(world[children] - world[SMPL_PARENTS[1:]], axis=1)
    bone_error = float(np.max(np.abs(before - after)))
    need(float(residual.max()) <= 2e-5 and bone_error <= 2e-5,
         'Saved joints do not preserve source rigid geometry or bone lengths')
    normal = rotation.T @ np.array([0., 0., 1.])
    return dict(schema='m2_floor_orientation_from_saved_joints_v1',
                rotation_world_from_source=rotation.tolist(), translation_world_m=translation.tolist(),
                normal_source=normal.tolist(), max_joint_residual_m=float(residual.max()),
                max_bone_length_error_m=bone_error, residual_limit_m=2e-5,
                bone_length_limit_m=2e-5, singular_values=singular.tolist(),
                original_depth_plane_bytes_recovered=False, floor_offset_recovered=False,
                source_gravity_verified=False, floor_semantics_reverified=False,
                method='proper rigid Procrustes on preserved diagnostic joints; level floor assumed')


def select_cohorts(split, quality, requests, *, seed):
    need(type(seed) is int and 0 <= seed < 2**63, 'Explicit bounded integer seed required')
    need(set(split) == {'train', 'validation', 'test'} and isinstance(requests, list) and requests,
         'Frozen split and explicit cohorts required')
    all_ids = [i for values in split.values() for i in values]
    need(len(all_ids) == len(set(all_ids)), 'Frozen split identities overlap')
    eligible = quality['eligible_shape_betas']
    quarantined = set(quality.get('quarantined_identities', []))
    used, names, result = set(), set(), []
    roles = {'train': 'train', 'selection_validation': 'validation', 'frozen_evaluation': 'validation'}
    for request in requests:
        name, expected = request.get('name'), request.get('split')
        need(name in roles and roles[name] == expected and name not in names,
             'Unique explicit TRAIN/VALIDATION cohort role required; TEST forbidden')
        need(set(request) in ({'name', 'split', 'count'}, {'name', 'split', 'identity_ids'}),
             'Use cohort count or explicit identity_ids')
        pool = [i for i in split[expected] if i in eligible and i not in quarantined and i not in used]
        if 'identity_ids' in request:
            chosen = request['identity_ids']
            need(isinstance(chosen, list) and chosen and len(chosen) == len(set(chosen)),
                 'Unique nonempty cohort identities required')
            need(not (set(chosen) & used), 'Cohort identities must be disjoint; overlap rejected')
            need(set(chosen) <= set(pool), 'Cohort identity is outside eligible frozen split')
        else:
            count = request['count']
            need(type(count) is int and 1 <= count <= len(pool), 'Insufficient disjoint eligible identities')
            chosen = sorted(pool, key=lambda i: hashlib.sha256(f'{seed}:{name}:{i}'.encode()).digest())[:count]
        used.update(chosen)
        names.add(name)
        result.append(dict(name=name, split=expected, identities=list(chosen)))
    return result


def _expected_inputs():
    values = {k: retarget.FROZEN[k] for k in ('model', 'split', 'quality', 'anchors', 'geometry', 'geometry_arrays')}
    values.update({identity: sha for identity, _, _, sha in SOURCES})
    values['saved_joints'] = SAVED_JOINTS_SHA256
    return values


def _checked_inputs(inputs):
    need(set(inputs) == set(_expected_inputs()), 'Closed reconstruction input inventory required')
    for name, expected in _expected_inputs().items():
        need(inputs[name]['sha256'] == expected, 'Frozen reconstruction input differs: ' + name)
    for ref in inputs.values():
        verify_ref(ref)
    split = bound_json(inputs['split'])
    quality = bound_json(inputs['quality'])
    need(quality['identity_split_sha256'] == inputs['split']['sha256'], 'Identity quality/split mismatch')
    return split, quality


def code_bindings():
    paths = [__file__, retarget.__file__]
    paths.extend(inspect.getmodule(function).__file__ for function in
                 (retarget._SMPLCPUAdapter, retarget.transform_observed_arrays,
                  retarget.anatomical_frame, retarget.evaluate_anchors))
    return {Path(path).name: retarget.sha(path) for path in paths}


def _validate_observed_sources(bundle, split, quality):
    """Recheck exact frame parameters and independently recompute the rigid audit.

    The model-joint values are newly authored model outputs, not old authority.
    Their input/code hashes and actual generation remain explicitly recorded.
    """
    sources = bundle.get('observed_sources', [])
    need(len(sources) == len(SOURCES), 'Closed two-observed-source inventory required')
    saved = bound_json(bundle['inputs']['saved_joints'])
    for source, (identity, sequence, frame, digest) in zip(sources, SOURCES):
        expected = dict(identity=identity, sequence_path=sequence + '/smpl_params.npz',
                        sequence_sha256=digest, frame_index=frame)
        need(source.get('source') == expected and identity in split['splits']['train'],
             'Reconstruction observed source identity/frame changed')
        with np.load(io.BytesIO(Path(bundle['inputs'][identity]['file']).read_bytes()), allow_pickle=False) as z:
            observed = {key: np.array(z[key][frame], dtype=float) for key in ('body_pose', 'global_orient', 'transl', 'betas')}
        for key, value in observed.items():
            need(np.array_equal(source['observed_parameters'][key], value), 'Reconstruction changed original observed frame: ' + key)
        parameters = retarget.retarget_parameters(source, quality['eligible_shape_betas'][identity])
        need(all(np.array_equal(source['model_parameters'][key], value) for key, value in parameters.items()),
             'Reconstruction source model parameters changed')
        rows = [r for r in saved['records'] if r.get('sequence') == sequence and r.get('source_frame') == frame and r.get('identity') == identity]
        need(len(rows) == 1, 'Saved source joint record is not unique')
        audit = recover_orientation(source['source_model_joints'], rows[0]['body_joints_world_m'])
        need(source.get('orientation') == audit and source.get('training_eligible') is False and
             source.get('physics_qualified') is False, 'Recovered orientation audit or qualification changed')
        document = bound_json(source['orientation_reference'])
        need(document.get('schema') == 'm2_reconstructed_floor_orientation_receipt_v1' and
             document.get('audit') == audit and document.get('source') == expected and
             document.get('saved_joints_report') == bundle['inputs']['saved_joints'] and
             document.get('original_depth_plane_bytes_recovered') is False,
             'New orientation receipt/source binding differs')


def validate_reconstruction(spec, summary, record=None):
    """Authenticate this explicit new route before patient payload admission."""
    need(summary.get('schema') == SCHEMA, 'New reconstruction summary schema required')
    need('reconstruction_source' in spec and 'reconstruction_cohort' in spec,
         'New reconstruction requires explicit opt-in source and cohort')
    source_ref = spec['reconstruction_source']
    need(summary.get('reconstruction_source') == source_ref and
         summary.get('reconstruction_cohort') == spec['reconstruction_cohort'], 'Reconstruction opt-in binding differs')
    bundle = bound_json(source_ref)
    need(bundle.get('schema') == BUNDLE_SCHEMA and bundle.get('geometry_id') == GEOMETRY_ID and
         bundle.get('geometry_source_sha256') == GEOMETRY_SOURCE_SHA256 and
         bundle.get('original_depth_plane_bytes_recovered') is False and
         bundle.get('physics_qualified') is False and bundle.get('training_eligible') is False,
         'Reconstruction source bundle authority or qualifications differ')
    split, quality = _checked_inputs(bundle['inputs'])
    _validate_observed_sources(bundle, split, quality)
    need(bundle.get('authoring_code') == code_bindings(),
         'Reconstruction authoring source changed')
    request = bound_json(bundle['request'])
    need(request.get('schema') == REQUEST_SCHEMA and request.get('inputs') == bundle['inputs'],
         'Reconstruction request/source mismatch')
    cohorts = select_cohorts(split['splits'], quality, request['cohorts'], seed=request['seed'])
    need(bundle.get('cohorts') == cohorts, 'Reconstruction cohort selection changed')
    selected = [c for c in cohorts if c['name'] == spec['reconstruction_cohort']]
    need(len(selected) == 1 and selected[0]['split'] == spec.get('expected_split'), 'Reconstruction cohort split differs')
    need(spec.get('anchor_inventory') == bundle['inputs']['anchors']['file'] and
         spec.get('identity_split') == bundle['inputs']['split']['file'], 'Reconstruction patient source paths differ')
    need(summary.get('identity_split_sha256') == retarget.FROZEN['split'] and
         summary.get('anchor_inventory_sha256') == retarget.FROZEN['anchors'] and
         summary.get('manifest_sha256') == spec.get('manifest_sha256'), 'Reconstruction summary hashes differ')
    records = bound_json(dict(file=str(Path(spec['directory']) / 'asset_manifest.json'), sha256=spec['manifest_sha256']))
    matches = [r for r in records if r.get('file') == spec.get('file')]
    need(len(matches) == 1 and (record is None or record == matches[0]), 'Unique reconstruction asset required')
    row = matches[0]
    need(row.get('identity') in selected[0]['identities'] and row.get('split') == selected[0]['split'] and
         row.get('reconstruction_source') == source_ref and row.get('source_cohort') == selected[0]['name'],
         'Reconstruction asset identity/source mismatch')
    need(Path(row['provenance_file']).name == row['provenance_file'], 'Reconstruction provenance escapes asset directory')
    provenance = bound_json(dict(file=str(Path(spec['directory']) / row['provenance_file']), sha256=row['provenance_sha256']))
    sources = [s for s in bundle['observed_sources'] if s['source'] == row.get('source_pose')]
    need(len(sources) == 1 and provenance.get('pose_source') == sources[0]['source'] and
         provenance.get('observed_parameters') == sources[0]['observed_parameters'] and
         provenance.get('reconstruction_source') == source_ref and
         provenance.get('floor_orientation_reconstruction') == sources[0]['orientation'] and
         provenance.get('original_depth_plane_bytes_recovered') is False and
         provenance.get('training_eligible') is False and provenance.get('physics_qualified') is False,
         'Observed reconstruction provenance differs')
    parameters = retarget.retarget_parameters(sources[0], quality['eligible_shape_betas'][row['identity']])
    for key, value in parameters.items():
        need(np.array_equal(np.asarray(provenance['retargeted_parameters'][key]), value),
             'Reconstruction authored or changed observed parameters: ' + key)
    return bundle


def default_request(source_root, cohorts, seed=20261105):
    root = Path(source_root)
    runs = root / 'results/ausc_policy'
    paths = dict(model=root / 'auscultation/data/smpl/SMPL_NEUTRAL.pkl',
                 split=runs / 'm2_identity_5mm_v2/identity_split.json',
                 quality=runs / 'm2_identity_5mm_v2/identity_quality.json',
                 anchors=runs / 'asset_inventory/inventory.json',
                 saved_joints=runs / 'observed_source_seat_pilot_v3/summary.json',
                 geometry=runs / 'retainer_r3_single_preload_v3/output/review/geometry/collision_geometry.json',
                 geometry_arrays=runs / 'retainer_r3_single_preload_v3/output/review/geometry/collision_geometry.npz')
    paths.update({identity: root / 'auscultation/data/humman/smpl_params' / sequence / 'smpl_params.npz'
                  for identity, sequence, _, _ in SOURCES})
    return dict(schema=REQUEST_SCHEMA, inputs={name: reference(path) for name, path in paths.items()},
                cohorts=cohorts, seed=seed)


def author(request_ref, output):
    need(sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
         'Real reconstruction authoring requires Linux CPU and CUDA_VISIBLE_DEVICES empty')
    request = bound_json(request_ref)
    need(request.get('schema') == REQUEST_SCHEMA, 'Explicit reconstruction request required')
    inputs = request['inputs']
    split, quality = _checked_inputs(inputs)
    cohorts = select_cohorts(split['splits'], quality, request['cohorts'], seed=request['seed'])
    anchors = bound_json(inputs['anchors'])['anchors']
    saved = bound_json(inputs['saved_joints'])
    need(saved.get('training_eligible') is False and len(saved.get('records', [])) == 2,
         'Preserved two-source unqualified diagnostic required')
    from .observed_seated_poses import _SMPLCPUAdapter, classify_numeric_seated
    adapter = _SMPLCPUAdapter(inputs['model']['file'])
    observed_sources = []
    for identity, sequence, frame, digest in SOURCES:
        need(identity in split['splits']['train'] and identity in quality['eligible_shape_betas'],
             'Observed source must remain eligible TRAIN')
        with np.load(io.BytesIO(Path(inputs[identity]['file']).read_bytes()), allow_pickle=False) as z:
            observed = {key: np.array(z[key][frame], dtype=float) for key in ('body_pose', 'global_orient', 'transl', 'betas')}
        candidate = dict(observed_parameters={k: v.tolist() for k, v in observed.items()})
        parameters = retarget.retarget_parameters(candidate, quality['eligible_shape_betas'][identity])
        body = adapter(**parameters)
        metrics = classify_numeric_seated(body['joints'])
        need(metrics['numeric_criterion_passed'], 'Preserved observed pose failed numeric seated screen')
        rows = [r for r in saved['records'] if r.get('sequence') == sequence and r.get('source_frame') == frame and r.get('identity') == identity]
        need(len(rows) == 1 and rows[0].get('split') == 'train', 'Saved diagnostic source identity/frame differs')
        orientation = recover_orientation(body['joints'][:24], rows[0]['body_joints_world_m'])
        observed_sources.append(dict(source=dict(identity=identity, sequence_path=sequence + '/smpl_params.npz',
                                                sequence_sha256=digest, frame_index=frame),
                                     observed_parameters=candidate['observed_parameters'],
                                     model_parameters={k: v.tolist() for k, v in parameters.items()},
                                     numeric_seated_screen=metrics, orientation=orientation,
                                     source_model_joints=body['joints'][:24].tolist(), saved_joints_report=inputs['saved_joints'],
                                     source_frame='HuMMan Point kinect_color_000', **FLAGS))
    metadata = bound_json(inputs['geometry'])
    with np.load(io.BytesIO(Path(inputs['geometry_arrays']['file']).read_bytes()), allow_pickle=False) as z:
        external = {key: np.array(z[key], copy=True) for key in z.files}
    screen = retarget.ExternalSurfaceScreen(metadata, external)
    seat = screen.geometry['/World/Stool/seat'][0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    retarget.write_json(output / 'SOURCE_REQUEST.json', request)
    for source in observed_sources:
        orientation_file = output / (Path(source['source']['sequence_path']).parent.name + '_orientation.json')
        retarget.write_json(orientation_file, dict(schema='m2_reconstructed_floor_orientation_receipt_v1',
            source=source['source'], saved_joints_report=inputs['saved_joints'], audit=source['orientation'],
            normal_source=source['orientation']['normal_source'], original_depth_plane_bytes_recovered=False,
            floor_offset_recovered=False, source_gravity_verified=False,
            description='New rigid-orientation reconstruction from saved diagnostic joints; no original depth-plane bytes or renewed semantic review.'))
        source['orientation_reference'] = reference(orientation_file)
    bundle = dict(schema=BUNDLE_SCHEMA, geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                  inputs=inputs, request=reference(output / 'SOURCE_REQUEST.json'), cohorts=cohorts,
                  observed_sources=observed_sources, original_depth_plane_bytes_recovered=False,
                  authoring_code=code_bindings(), **FLAGS)
    retarget.write_json(output / 'RECONSTRUCTION.json', bundle)
    bundle_ref = reference(output / 'RECONSTRUCTION.json')
    results = []
    for cohort in cohorts:
        directory = output / cohort['name']
        directory.mkdir()
        records = []
        for identity in cohort['identities']:
            for source in observed_sources:
                pose = source['source']
                label = f'{identity}_{Path(pose["sequence_path"]).parent.name}_f{pose["frame_index"]:04d}'
                parameters = retarget.retarget_parameters(source, quality['eligible_shape_betas'][identity])
                body = adapter(**parameters)
                need(body['vertices'].shape == (6890, 3) and body['faces'].shape == (13776, 3), 'Unexpected SMPL topology')
                binding = dict(method='floor_normal_assuming_level', source_plane_sha256=source['orientation_reference']['sha256'],
                               source_report_sha256=inputs['saved_joints']['sha256'], source_frame=source['source_frame'])
                asset = retarget.place_retarget(body['vertices'], body['faces'], body['joints'], anchors,
                                               seat, source['orientation']['normal_source'], binding)
                placement = asset.pop('placement')
                patches = {str(name): retarget.fit_contact_patch(asset['vertices'], asset['faces'], position, normal,
                           seed_vertex_ids=anchors[str(name)]['smpl_vertex_indices'])
                           for name, position, normal in zip(asset['anchor_names'], asset['anchor_positions'], asset['anchor_normals'])}
                asset['contact_patch_normals'] = np.array([patches[str(n)]['normal_world'] for n in asset['anchor_names']])
                numeric = screen.check(asset['vertices'], asset['faces'])
                need(numeric['surface_pairs_completed'] == 22 and len(numeric['pairs']) == 22, 'Full external screen incomplete')
                passed = bool(numeric['surface_intersections'] == 0 and numeric['floor_min_z_m'] >= -1e-6 and
                              abs(placement['seat_gap_m'] - 1e-6) <= 1e-7)
                np.savez_compressed(directory / (label + '.npz'), **asset)
                detail = dict(identity=identity, pose_source=pose, observed_parameters=source['observed_parameters'],
                              retargeted_parameters=parameters, reconstruction_source=bundle_ref,
                              floor_orientation_reconstruction=source['orientation'], placement=placement,
                              contact_patches=patches, external_static_screen=numeric,
                              original_depth_plane_bytes_recovered=False, **FLAGS)
                retarget.write_json(directory / (label + '.json'), detail)
                records.append(dict(identity=identity, split=cohort['split'], proposal=label, file=label + '.npz',
                                    sha256=retarget.sha(directory / (label + '.npz')), provenance_file=label + '.json',
                                    provenance_sha256=retarget.sha(directory / (label + '.json')), mesh_scale=1.,
                                    source_pose=pose, reconstruction_source=bundle_ref, source_cohort=cohort['name'],
                                    numeric_candidate_screen_passed=passed, external_surface_intersections=numeric['surface_intersections'],
                                    seat_gap_m=placement['seat_gap_m'], floor_min_z_m=numeric['floor_min_z_m'], **FLAGS))
        retarget.write_json(directory / 'asset_manifest.json', records)
        summary = dict(schema=SCHEMA, geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                       freshly_generated=True, identity_split_sha256=retarget.FROZEN['split'],
                       identity_quality_sha256=retarget.FROZEN['quality'], anchor_inventory_sha256=retarget.FROZEN['anchors'],
                       smpl_neutral_sha256=retarget.FROZEN['model'], manifest_sha256=retarget.sha(directory / 'asset_manifest.json'),
                       reconstruction_source=bundle_ref, reconstruction_cohort=cohort['name'], selected_identities=cohort['identities'],
                       target_identity_split=cohort['split'], generated_assets=len(records),
                       numeric_screen_passed_assets=sum(r['numeric_candidate_screen_passed'] for r in records),
                       original_depth_plane_bytes_recovered=False, **FLAGS)
        retarget.write_json(directory / 'summary.json', summary)
        specs = [dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256, freshly_generated=True,
                      directory=str(directory), file=r['file'], manifest_sha256=summary['manifest_sha256'],
                      summary_sha256=retarget.sha(directory / 'summary.json'), expected_split=cohort['split'],
                      identity_split=inputs['split']['file'], anchor_inventory=inputs['anchors']['file'],
                      reconstruction_source=bundle_ref, reconstruction_cohort=cohort['name']) for r in records]
        retarget.write_json(directory / 'patient_specs.json', specs)
        for spec in specs: validate_reconstruction(spec, summary)
        results.append(dict(cohort=cohort, summary=reference(directory / 'summary.json'), patient_specs=reference(directory / 'patient_specs.json')))
    _checked_inputs(inputs)
    verify_ref(request_ref)
    result = dict(schema='m2_observed_reconstruction_complete_v1', completed=True, reconstruction_source=bundle_ref,
                  cohorts=results, original_depth_plane_bytes_recovered=False, **FLAGS)
    retarget.write_json(output / 'COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--make-request', action='store_true')
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--train', type=int, default=2)
    parser.add_argument('--selection-val', type=int, default=0)
    parser.add_argument('--frozen-val', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    if args.make_request:
        need(args.input_root is not None and args.request is None and not args.output.exists(), 'New request output and input root required')
        counts = [('train', 'train', args.train), ('selection_validation', 'validation', args.selection_val),
                  ('frozen_evaluation', 'validation', args.frozen_val)]
        need(all(count >= 0 for _, _, count in counts), 'Cohort counts cannot be negative')
        cohorts = [dict(name=name, split=split, count=count) for name, split, count in counts if count]
        request = default_request(args.input_root, cohorts, args.seed)
        _checked_inputs(request['inputs'])
        select_cohorts(bound_json(request['inputs']['split'])['splits'], bound_json(request['inputs']['quality']), cohorts, seed=args.seed)
        retarget.write_json(args.output, request)
    else:
        need(args.request is not None, 'Authoring request required')
        print(json.dumps(author(reference(args.request), args.output), indent=2))


if __name__ == '__main__':
    main()

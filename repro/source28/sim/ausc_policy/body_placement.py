"""Rigid numerical SMPL seating and QA; no IK, physics, rendering or labels.

The seat top and its horizontal footprint are configuration inputs in metres.
An anatomical frame is measured from SMPL hips and pelvis-to-neck joints, not
assumed from the NPZ filename. The pelvis is centered horizontally on the seat;
the lowest buttock-region vertex over its footprint is translated to seat top.
This is a repeatable placement convention, not a solved equilibrium. Foot and
buttock vertex regions are explicitly nearest-joint proxies, not segmentation.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from sim.ausc_policy.surface_anchors import evaluate_anchors


TARGET_NAMES = {f'{side}{level}' for side in ('L', 'R') for level in range(1, 6)}


@dataclass(frozen=True)
class SeatConfig:
    center_world_m: tuple
    up_world: tuple
    front_world: tuple
    half_extents_m: tuple
    thickness_m: float
    floor_z_m: float
    geometry_tolerance_m: float
    table_bounds_world_m: tuple
    camera_world_from_optical: tuple
    camera_intrinsics: dict
    max_thigh_pitch_deg: float = 45.
    min_knee_flexion_deg: float = 35.
    max_knee_flexion_deg: float = 145.


def _unit(value, description):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Invalid ' + description)
    length = np.linalg.norm(value)
    if length < 1e-8:
        raise ValueError('Degenerate ' + description)
    return value / length


def _points(value, description, minimum=1):
    array = np.asarray(value, dtype=float)
    if (array.ndim != 2 or array.shape[1] != 3 or len(array) < minimum
            or not np.isfinite(array).all()):
        raise ValueError('Invalid ' + description)
    return array


def _validate_seat(seat):
    center = np.asarray(seat.center_world_m, float)
    _unit(seat.up_world, 'seat up')
    up = np.asarray(seat.up_world, float)
    front = np.asarray(seat.front_world, float)
    _unit(front, 'seat front')
    if (not np.allclose(np.linalg.norm(up), 1., atol=1e-10, rtol=0.)
            or not np.allclose(np.linalg.norm(front), 1., atol=1e-10, rtol=0.)
            or abs(up @ front) > 1e-10):
        raise ValueError('Seat axes must be unit and orthogonal')
    if not np.allclose(up, [0., 0., 1.], atol=1e-10, rtol=0.):
        raise ValueError('Current floor and exported seat bounds require world Z up')
    # Strip only the roundoff admitted by the strict checks. Never pass scaled
    # or weakly nonorthogonal axes into the claimed rigid world transform.
    up = np.array([0., 0., 1.])
    front = _unit(front - up * (up @ front), 'seat front')
    half = np.asarray(seat.half_extents_m, float)
    scalar = np.asarray([seat.thickness_m, seat.geometry_tolerance_m,
                         seat.floor_z_m, seat.max_thigh_pitch_deg,
                         seat.min_knee_flexion_deg, seat.max_knee_flexion_deg], float)
    if (center.shape != (3,) or not np.isfinite(center).all()
            or half.shape != (2,) or not np.isfinite(half).all() or np.any(half <= 0)
            or not np.isfinite(scalar).all() or scalar[0] <= 0 or scalar[1] < 0
            or not 0 < scalar[3] < 90 or not 0 <= scalar[4] < scalar[5] <= 180):
        raise ValueError('Invalid seat dimensions or numerical QA thresholds')
    table = _points(seat.table_bounds_world_m, 'table bounds', 2)
    if table.shape != (2, 3) or np.any(table[1] <= table[0]):
        raise ValueError('Invalid table bounds')
    camera = np.asarray(seat.camera_world_from_optical, float)
    if (camera.shape != (4, 4) or not np.isfinite(camera).all()
            or not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-10)
            or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(camera[:3, :3]), 1., atol=1e-8)):
        raise ValueError('Invalid camera rigid transform')
    intrinsics = seat.camera_intrinsics
    values = np.array([intrinsics[k] for k in ('fx', 'fy', 'cx', 'cy', 'width', 'height')], float)
    if not np.isfinite(values).all() or np.any(values[[0, 1, 4, 5]] <= 0):
        raise ValueError('Invalid camera intrinsics')
    return center, up, front, half, table, camera


def anatomical_frame(joints):
    """Columns are anatomical left, torso up, front; SMPL joint order required."""
    joints = _points(joints, 'SMPL joints', 24)
    left = _unit(joints[1] - joints[2], 'hip axis')
    torso = joints[12] - joints[0]
    up = _unit(torso - left * (left @ torso), 'torso axis')
    front = _unit(np.cross(left, up), 'front axis')
    return np.column_stack((left, up, front))


def place_body(body, seat, anchors, *, clothing='none'):
    """Return arrays and explicit QA for one existing posed mesh without scaling.

    ``anchors`` is the hash-bound V6A barycentric inventory used by the source
    assets. Stored normals must agree with actual face winding; arbitrary sign
    fixes are not allowed. ``existing_shirt`` is deliberately unsupported until
    a fitted clothing mesh and its correspondence can be supplied separately.
    """
    if clothing != 'none':
        raise ValueError('Fitted clothing collision surface unavailable; use clothing=none')
    center, up, front, half, table, camera = _validate_seat(seat)
    vertices = _points(body['vertices'], 'vertices', 3)
    joints = _points(body['joints'], 'SMPL joints', 24)
    faces = np.asarray(body['faces'])
    names = np.asarray(body['anchor_names'])
    if (names.shape != (10,) or set(names.tolist()) != TARGET_NAMES
            or set(anchors) != TARGET_NAMES):
        raise ValueError('Expected exactly ten V6A targets')
    source_anchors = evaluate_anchors(vertices, faces, anchors)
    expected_positions = np.array([source_anchors[name]['position_m'] for name in names])
    expected_normals = np.array([source_anchors[name]['normal'] for name in names])
    positions = _points(body['anchor_positions'], 'anchor positions', 10)
    normals = _points(body['anchor_normals'], 'anchor normals', 10)
    if positions.shape != (10, 3) or not np.allclose(positions, expected_positions, atol=1e-7, rtol=0):
        raise ValueError('Stored anchor position differs from actual surface')
    if (normals.shape != (10, 3) or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-7)
            or not np.allclose(normals, expected_normals, atol=1e-7, rtol=0)):
        raise ValueError('Stored anchor normal differs from triangle winding or is not unit')

    source_frame = anatomical_frame(joints)
    world_frame = np.column_stack((np.cross(up, front), up, front))
    rotation = world_frame @ source_frame.T
    local_vertices = (vertices - joints[0]) @ source_frame
    local_joints = (joints - joints[0]) @ source_frame
    # Nearest joint explicitly provides a coarse, deterministic anatomical proxy.
    nearest_joint = np.argmin(np.sum((vertices[:, None] - joints[None, :24]) ** 2, axis=2), axis=1)
    footprint = ((np.abs(local_vertices[:, 0]) <= half[0])
                 & (np.abs(local_vertices[:, 2]) <= half[1]))
    support = (footprint & np.isin(nearest_joint, [0, 1, 2])
               & (local_vertices[:, 1] <= np.mean(local_joints[1:3, 1])))
    if not np.any(support):
        raise ValueError('No buttock-region vertices over seat footprint; seat support absent')
    bottom = float(np.min(local_vertices[support, 1]))
    translation = center - rotation @ joints[0] - bottom * up
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    placed_vertices = vertices @ rotation.T + translation
    placed_joints = joints @ rotation.T + translation
    placed_positions = positions @ rotation.T + translation
    placed_normals = normals @ rotation.T
    # Re-evaluate actual world-space triangle winding independently of transport.
    world_anchors = evaluate_anchors(placed_vertices, faces, anchors)
    actual_normals = np.array([world_anchors[name]['normal'] for name in names])
    winding_error = float(np.max(np.abs(actual_normals - placed_normals)))
    if winding_error > 1e-8:
        raise ValueError('World triangle normal transport failed')

    reasons = []
    thigh_pitch = []
    knee_flexion = []
    forward_reach = []
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
        thigh = local_joints[knee] - local_joints[hip]
        shin = local_joints[ankle] - local_joints[knee]
        thigh_unit = _unit(thigh, 'thigh bone')
        shin_unit = _unit(shin, 'shin bone')
        thigh_pitch.append(float(np.degrees(np.arcsin(np.clip(thigh_unit[1], -1, 1)))))
        knee_flexion.append(float(np.degrees(np.arccos(np.clip(thigh_unit @ shin_unit, -1, 1)))))
        forward_reach.append(float(thigh[2]))
    if (max(np.abs(thigh_pitch)) > seat.max_thigh_pitch_deg or min(forward_reach) <= 0
            or min(knee_flexion) < seat.min_knee_flexion_deg
            or max(knee_flexion) > seat.max_knee_flexion_deg):
        reasons.append('pose_not_seated')

    tol = seat.geometry_tolerance_m
    mesh_min = placed_vertices.min(axis=0)
    mesh_max = placed_vertices.max(axis=0)
    floor_gap = float(mesh_min[2] - seat.floor_z_m)
    if floor_gap < -tol:
        reasons.append('floor_penetration')
    foot_gaps = []
    foot_counts = []
    for ankle, toe in ((7, 10), (8, 11)):
        region = np.isin(nearest_joint, [ankle, toe])
        foot_counts.append(int(region.sum()))
        if not region.any():
            raise ValueError('Foot mesh proxy absent')
        foot_gaps.append(float(placed_vertices[region, 2].min() - seat.floor_z_m))

    seat_heights = local_vertices[:, 1] - bottom
    seat_interior = (footprint & (seat_heights < -tol)
                     & (seat_heights > -seat.thickness_m + tol))
    if seat_interior.any():
        reasons.append('seat_envelope_penetration_unresolved')
    # A strictly positive AABB gap proves separation. Contact and any overlap,
    # including overlap smaller than the screening tolerance, remain unresolved.
    axis_gaps = np.maximum(np.maximum(table[0] - mesh_max, mesh_min - table[1]), 0.)
    table_gap = float(np.linalg.norm(axis_gaps))
    overlap = np.minimum(mesh_max, table[1]) - np.maximum(mesh_min, table[0])
    table_overlaps = bool(np.all(overlap >= 0.))
    if table_overlaps:
        reasons.append('table_aabb_overlap_unresolved')

    camera_to_pelvis = camera[:3, 3] - placed_joints[0]
    back_cosine = float(_unit(camera_to_pelvis, 'camera direction') @ -front)
    target_to_camera = camera[:3, 3] - placed_positions
    distances = np.linalg.norm(target_to_camera, axis=1)
    if np.any(distances < 1e-8):
        raise ValueError('Anchor at camera origin')
    normal_cosines = np.sum(placed_normals * target_to_camera / distances[:, None], axis=1)
    if back_cosine <= 0 or np.any(normal_cosines <= 0):
        reasons.append('back_not_facing_camera')
    optical = (placed_positions - camera[:3, 3]) @ camera[:3, :3]
    intr = seat.camera_intrinsics
    uv = np.full((10, 2), np.nan)
    positive = optical[:, 2] > 1e-8
    uv[positive, 0] = intr['fx'] * optical[positive, 0] / optical[positive, 2] + intr['cx']
    uv[positive, 1] = intr['fy'] * optical[positive, 1] / optical[positive, 2] + intr['cy']
    in_frame = (positive & (uv[:, 0] >= 0) & (uv[:, 0] < intr['width'])
                & (uv[:, 1] >= 0) & (uv[:, 1] < intr['height']))
    if not in_frame.all():
        reasons.append('targets_outside_camera_frustum')

    qa = dict(
        numerical_gate_passed=not reasons, rejection_reasons=reasons,
        training_eligible=False, expert_qualified=False, physics_qualified=False,
        clothing='none', clothing_targets_status='unavailable_no_fitted_clothing_mesh',
        seat_support_status='buttock_vertex_proxy_on_seat_envelope_not_physics',
        foot_support_status=('unverified_feet_above_floor' if max(foot_gaps) > tol
                             else 'unverified_floor_contact_or_penetration'),
        table_collision_status=('unverified_conservative_aabb_overlap' if table_overlaps
                                else 'separated_by_mesh_aabb_lower_bound'),
        visibility_status='projection_and_normal_test_only_occlusion_unverified',
        metrics=dict(
            full_mesh_bounds_world_m=[mesh_min.tolist(), mesh_max.tolist()],
            pelvis_world_m=placed_joints[0].tolist(),
            seat_support_gap_m=float(np.min(seat_heights[support])),
            buttock_support_vertex_count=int(support.sum()),
            seat_envelope_penetrating_vertex_count=int(seat_interior.sum()),
            foot_min_floor_clearances_m=foot_gaps, foot_proxy_vertex_counts=foot_counts,
            ankle_floor_clearances_m=(placed_joints[[7, 8], 2] - seat.floor_z_m).tolist(),
            mesh_min_floor_clearance_m=floor_gap,
            floor_align_vertical_delta_m=-floor_gap,
            floor_align_seat_support_gap_m=-floor_gap,
            table_aabb_separation_lower_bound_m=table_gap,
            thigh_pitch_deg=thigh_pitch, knee_flexion_deg=knee_flexion,
            thigh_forward_displacement_m=forward_reach,
            camera_back_hemisphere_cosine=back_cosine,
            anchor_camera_normal_cosines=normal_cosines.tolist(),
            target_count=10, in_camera_frustum_count=int(in_frame.sum()),
            target_uv_px=[row.tolist() if good else None for row, good in zip(uv, positive)],
            target_optical_depth_m=optical[:, 2].tolist(),
            maximum_normal_transport_error=winding_error,
            maximum_unit_normal_error=float(np.max(np.abs(np.linalg.norm(placed_normals, axis=1) - 1.)))),
        limitations=[
            'No exact triangle collision, self collision, robot collision, IK or reachability check.',
            'Seat footprint is the exported AABB envelope; exact stool shape/contact unverified.',
            'Buttock and foot subsets are nearest SMPL joint proxies, not validated anatomy masks.',
            'Torso-up is pelvis-to-neck; rigid alignment makes that measured axis upright.',
            'Source seated pilot arms remain T-pose-like and are not a usable patient arm posture.',
            'No dynamic settling, balance, foot support, camera occlusion or clothing fit verified.',
            'Pose angle bounds and geometric tolerance are numerical screens, not clinical validation.',
        ])
    return dict(vertices=placed_vertices, faces=faces.copy(), joints=placed_joints,
                anchor_names=names.copy(), anchor_positions=placed_positions,
                anchor_normals=placed_normals, world_from_smpl=transform,
                source_anatomical_frame=source_frame, qa=qa)


def _world_bounds(collider):
    bounds = _points(collider['bounds_local'], 'collider bounds', 2)
    matrix = np.asarray(collider['world_from_body'], float)
    if (bounds.shape != (2, 3) or np.any(bounds[1] < bounds[0])
            or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-8)):
        raise ValueError('Invalid collider rigid transform or bounds')
    corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1]))))
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return np.array([world.min(axis=0), world.max(axis=0)])


def seat_config_from_scene(geometry, camera, *, front_world, geometry_tolerance_m):
    """Read layout-only export; never read q, paths or nominal target labels.

    The caller explicitly supplies the approved patient's front axis. Shoe
    bounds must agree with its sign. This loader supports the existing Z-up,
    axis-aligned stool; arbitrary rotated seats need a richer geometry export.
    """
    colliders = {value['path']: value for value in geometry['colliders']}
    stool_transform = np.asarray(colliders['/World/Stool/seat']['world_from_body'], float)
    if (stool_transform.shape != (4, 4) or not np.isfinite(stool_transform).all()
            or not np.allclose(stool_transform[:3, :3], np.eye(3), atol=1e-10, rtol=0.)):
        raise ValueError('Unsupported rotated or tilted stool; horizontal axis-aligned seat required')
    bounds = {key: _world_bounds(value) for key, value in colliders.items()
              if key in ('/World/Stool/seat', '/World/Table/table_low', '/World/GroundPlane/geom')
              or key.startswith('/World/Patient/')}
    seat = bounds['/World/Stool/seat']
    ground = bounds['/World/GroundPlane/geom']
    if not np.isclose(ground[0, 2], ground[1, 2], atol=1e-8):
        raise ValueError('Expected horizontal flat floor')
    front = _unit(front_world, 'scene patient front')
    if not (np.allclose(front, [0, 1, 0], atol=1e-10)
            or np.allclose(front, [0, -1, 0], atol=1e-10)):
        raise ValueError('Current scene loader requires patient front along world Y')
    center = np.array([*seat.mean(axis=0)[:2], seat[1, 2]])
    shoes = [value for key, value in bounds.items() if 'sneaker' in key.lower()]
    if not shoes:
        raise ValueError('Patient shoe reference missing; verify front axis explicitly')
    shoe_center = np.mean(shoes, axis=(0, 1))
    if (shoe_center - center) @ front <= 0:
        raise ValueError('Patient front disagrees with nominal shoe placement')
    config = SeatConfig(
        center_world_m=tuple(center), up_world=(0., 0., 1.), front_world=tuple(front),
        half_extents_m=tuple((seat[1, :2] - seat[0, :2]) / 2.),
        thickness_m=float(seat[1, 2] - seat[0, 2]), floor_z_m=float(ground[1, 2]),
        geometry_tolerance_m=geometry_tolerance_m,
        table_bounds_world_m=bounds['/World/Table/table_low'].tolist(),
        camera_world_from_optical=camera['world_from_optical'], camera_intrinsics=camera['intrinsics'])
    _validate_seat(config)
    patient = [value for key, value in bounds.items() if key.startswith('/World/Patient/')]
    shirts = {key: value.tolist() for key, value in bounds.items() if 'shirt' in key.lower()}
    reference = dict(
        seat_bounds_world_m=seat.tolist(),
        nominal_patient_bounds_world_m=[np.min(patient, axis=(0, 1)).tolist(),
                                       np.max(patient, axis=(0, 1)).tolist()],
        nominal_shoe_min_floor_clearance_m=float(min(value[0, 2] for value in shoes) - ground[1, 2]),
        front_axis_evidence='Caller supplied world Y axis, sign checked against nominal shoe bounds.',
        pelvis_alignment='New SMPL pelvis XY centered on stool; original scene has no exported pelvis joint.',
        existing_shirt_status='nominal_reference_only_not_fitted',
        existing_shirt_bounds_world_m=shirts,
        camera_variant=camera.get('render_variant', 'unidentified_camera_configuration'),
        old_targets_paths_and_q_used=False,
    )
    return config, reference


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_pilot(*, source_dir, output_dir, inventory_path, identity_split_path,
              geometry_path, camera_path, front_world, geometry_tolerance_m=.003):
    """Assess existing assets, place one seated candidate per frozen train identity.

    All source hashes and identity membership are checked before any output is
    created. Selection minimizes numerical rejection count, then conservative
    seat-envelope vertex overlaps, with pose name as a deterministic tie-break.
    Rejected selections remain explicitly rejected; nothing becomes training data.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    paths = dict(source_manifest=source_dir / 'asset_manifest.json',
                 source_summary=source_dir / 'summary.json', anchor_inventory=Path(inventory_path),
                 identity_split=Path(identity_split_path), geometry=Path(geometry_path),
                 camera=Path(camera_path), placement_module=Path(__file__))
    hashes = {key: _sha(path) for key, path in paths.items()}
    source_manifest = json.loads(paths['source_manifest'].read_text())
    source_summary = json.loads(paths['source_summary'].read_text())
    for field, key in (('identity_split_sha256', 'identity_split'),
                       ('anchor_inventory_sha256', 'anchor_inventory')):
        if source_summary[field] != hashes[key]:
            raise ValueError('Source ' + field + ' hash mismatch')
    split = json.loads(paths['identity_split'].read_text())['splits']
    train = set(split['train'])
    other = set().union(*(set(value) for key, value in split.items() if key != 'train'))
    if train & other:
        raise ValueError('Identity split overlaps train and held-out identities')
    identities = sorted({record['identity'] for record in source_manifest})
    if not identities or len(identities) != source_summary['identities']:
        raise ValueError('Source identity count mismatch')
    source_paths = {}
    for record in source_manifest:
        if record['identity'] not in train:
            raise ValueError('Pilot includes non-train identity')
        path = source_dir / record['file']
        if path.resolve().parent != source_dir.resolve() or path.name in source_paths:
            raise ValueError('Duplicate or non-local source asset path')
        if _sha(path) != record['sha256']:
            raise ValueError('Source asset hash mismatch: ' + record['file'])
        source_paths[path.name] = path
    anchors = json.loads(paths['anchor_inventory'].read_text())['anchors']
    config, reference = seat_config_from_scene(
        json.loads(paths['geometry'].read_text()), json.loads(paths['camera'].read_text()),
        front_world=front_world, geometry_tolerance_m=geometry_tolerance_m)
    assessed = []
    placed = {}
    for record in source_manifest:
        item = {key: record[key] for key in ('identity', 'pose', 'file', 'sha256')}
        with np.load(source_paths[record['file']], allow_pickle=False) as asset:
            try:
                result = place_body(asset, config, anchors)
                placed[record['file']] = result
                item['qa'] = result['qa']
                item['world_from_smpl'] = result['world_from_smpl'].tolist()
            except ValueError as error:
                item['qa'] = dict(numerical_gate_passed=False, training_eligible=False,
                                  rejection_reasons=['invalid_asset_or_absent_support'],
                                  error=str(error))
        assessed.append(item)
    selected = []
    for identity in identities:
        candidates = [item for item in assessed if item['identity'] == identity
                      and item['pose'] in ('seated_asset', 'seated_lean_asset')
                      and item['file'] in placed]
        if not candidates:
            raise ValueError('No placeable seated candidate for identity ' + identity)
        selected.append(min(candidates, key=lambda item: (
            len(item['qa']['rejection_reasons']),
            item['qa']['metrics']['seat_envelope_penetrating_vertex_count'], item['pose'], item['file'])))
    # No source or identity changes may slip in during assessment.
    for key, path in paths.items():
        if _sha(path) != hashes[key]:
            raise ValueError('Provenance input changed during assessment: ' + key)
    for record in source_manifest:
        if _sha(source_paths[record['file']]) != record['sha256']:
            raise ValueError('Source asset changed during assessment')
    output_dir.mkdir(parents=True, exist_ok=False)
    output_manifest = []
    for index, item in enumerate(selected):
        result = placed[item['file']]
        filename = f'body{index:02d}_{item["identity"]}_placed.npz'
        np.savez_compressed(output_dir / filename, **{key: value for key, value in result.items() if key != 'qa'})
        output_manifest.append(dict(
            identity=item['identity'], split='train', pose=item['pose'], file=filename,
            sha256=_sha(output_dir / filename), source_file=item['file'], source_sha256=item['sha256'],
            world_from_smpl=item['world_from_smpl'], clothing='none', qa=item['qa']))
    summary = dict(
        schema='m2_numerical_seating_pilot_v1', placed_identities=len(selected),
        assessed_assets=len(assessed), split='frozen train only',
        numerical_gate_passed_count=sum(item['qa']['numerical_gate_passed'] for item in selected),
        numerical_gate_rejected_count=sum(not item['qa']['numerical_gate_passed'] for item in selected),
        identity_split_sha256=hashes['identity_split'],
        smpl_neutral_sha256=source_summary['smpl_neutral_sha256'],
        anchor_inventory_sha256=hashes['anchor_inventory'],
        source_assets_unchanged=True, identity_shape_or_scale_changed=False,
        selection_policy='Fewest numerical rejections, fewest seat-envelope vertices, then pose/file name.',
        geometry_role='Existing patient/stool/table as layout-only reference; no old targets, q or paths.',
        placement_role='Rigid numeric candidate with nearest-joint support proxies, not physical seating.',
        clothes='none; existing nominal shirt is reference only, not transferred to diverse bodies',
        isaac_rendered=False, expert_generated=False, physics_qualified=False,
        reachability_qualified=False, training_eligible=False,
        provenance={key: dict(path=str(path.resolve()), sha256=hashes[key]) for key, path in paths.items()},
    )
    _write_json(output_dir / 'seat_config.json', dict(config=asdict(config), reference=reference))
    _write_json(output_dir / 'asset_manifest.json', output_manifest)
    _write_json(output_dir / 'candidate_assessments.json', assessed)
    _write_json(output_dir / 'summary.json', summary)
    rows = ['# M2 numerical placement pilot', '',
            f'{summary["numerical_gate_passed_count"]}/{len(selected)} selected identities pass numerical screens.',
            'None is physics-, reachability-, expert- or training-qualified.', '',
            '| Identity | Pose | Numeric gate | Feet above floor (m) | Seat-envelope vertices | Camera targets |',
            '|---|---|---|---|---|---|']
    for item in selected:
        qa = item['qa']
        metric = qa['metrics']
        rows.append(f'| {item["identity"]} | {item["pose"]} | '
                    f'{"pass" if qa["numerical_gate_passed"] else ", ".join(qa["rejection_reasons"])} | '
                    f'{metric["foot_min_floor_clearances_m"][0]:.4f}, {metric["foot_min_floor_clearances_m"][1]:.4f} | '
                    f'{metric["seat_envelope_penetrating_vertex_count"]} | {metric["in_camera_frustum_count"]}/10 |')
    rows += ['', 'The stool and camera are unchanged. The original nominal shoes are also elevated '
             f'({reference["nominal_shoe_min_floor_clearance_m"]:.4f} m); foot support remains unverified.',
             'Pelvis XY is explicitly aligned to seat center because the nominal export has no pelvis joint. '
             'Torso up derives from pelvis-to-neck and front from the oriented hip axis.',
             'Buttock and foot subsets use nearest SMPL joints. Seat overlap is a conservative envelope '
             'screen, not exact stool collision. Lowering feet to the floor would move the support '
             'surface below the seat by the recorded floor-alignment delta; it is not applied.',
             'The source seated pilot arms remain T-pose-like and are not a usable patient arm posture. '
             'Self collision, arms/robot collision, '
             'dynamic settling, camera occlusion and fitted clothing are unverified.',
             'All 24 source candidates, including rejected rest poses, are retained as QA records '
             'when the supplied source manifest contains the full pilot.', '']
    (output_dir / 'README.md').write_text('\n'.join(rows))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'output-dir', 'inventory-path', 'identity-split-path', 'geometry-path', 'camera-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--front-world', type=float, nargs=3, required=True)
    parser.add_argument('--geometry-tolerance-m', type=float, default=.003,
                        help='Numerical envelope screening tolerance, not validated physics uncertainty')
    args = parser.parse_args()
    print(json.dumps(run_pilot(**vars(args)), indent=2))


if __name__ == '__main__':
    main()

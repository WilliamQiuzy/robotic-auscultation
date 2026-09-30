"""Authored SMPL arm-pose candidates and local triangle screens, never physics.

The frozen identity betas, neutral model and V6A anchors remain unchanged.
All pose parameters are synthetic axis-angle values in SMPL joint order.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sim.ausc_policy import body_placement, surface_anchors
from sim.ausc_policy.body_placement import SeatConfig, anatomical_frame, place_body
from sim.ausc_policy.surface_anchors import evaluate_anchors


ARM_JOINTS = {'left': (16, 18, 20, 22), 'right': (17, 19, 21, 23)}


def synthetic_pose_proposals():
    proposals = []
    for name, shoulder_deg, elbow_deg, intent in (
            ('arms_down', 60., 20., 'Arms lowered with lateral clearance; no hand support claim.'),
            ('lap_low', 60., 50., 'Hands lowered toward outer lap; contact/support unverified.'),
            ('lap_forward_hover', 60., 70., 'Forearms forward above outer lap; intentional unsupported hover.')):
        pose = np.zeros((23, 3), np.float32)
        pose[0, 0] = pose[1, 0] = -1.25
        pose[3, 0] = pose[4, 0] = 1.4
        pose[2, 0] = .12
        pose[15, 2] = -np.deg2rad(shoulder_deg)
        pose[16, 2] = np.deg2rad(shoulder_deg)
        pose[15, 1] = -np.deg2rad(15.)
        pose[16, 1] = np.deg2rad(15.)
        pose[17, 1] = -np.deg2rad(elbow_deg)
        pose[18, 1] = np.deg2rad(elbow_deg)
        proposals.append(dict(
            name=name, intent=intent,
            pose_source='authored_synthetic_joint_rotations_not_HuMMan_motion',
            baseline='m2_surface_pilot_v1 seated_lean_asset lower body and spine',
            rotation_representation='SMPL axis-angle vectors; XYZ entries are not sequential Euler rotations.',
            body_pose_axis_angle_rad=pose.reshape(-1).tolist(),
            global_orient_axis_angle_rad=[0., 0., 0.], translation_m=[0., 0., 0.],
            explicit_arm_parameters=dict(
                left_shoulder_joint16_z_deg=-shoulder_deg,
                right_shoulder_joint17_z_deg=shoulder_deg,
                left_shoulder_joint16_y_deg=-15.,
                right_shoulder_joint17_y_deg=15.,
                left_elbow_joint18_y_deg=-elbow_deg,
                right_elbow_joint19_y_deg=elbow_deg,
                wrist_and_hand_rotations_rad=[0., 0., 0.]),
            mesh_scale=1., patient_posture_qualified=False))
    return proposals


def _segment_hits(start, end, triangles):
    """Vectorized finite segment / noncoplanar triangle intersection."""
    direction = end - start
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    determinant = np.sum(edge1 * h, axis=1)
    valid = np.abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1. / determinant[valid]
    relative = start - triangles[:, 0]
    u = inverse * np.sum(relative * h, axis=1)
    q = np.cross(relative, edge1)
    v = inverse * np.sum(direction * q, axis=1)
    t = inverse * np.sum(edge2 * q, axis=1)
    eps = 1e-8
    return valid & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)


def triangle_intersections(vertices, faces, first_face_ids, second_face_ids):
    """Screen nonadjacent triangles using local trimesh R-tree plus segment tests.

    Shared-vertex pairs are excluded as mesh seams. Coplanar AABB candidates
    stay unresolved. No signed open-mesh distance or FCL is used, and absence
    of intersections does not establish containment clearance or physical safety.
    """
    import trimesh
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Invalid triangle mesh')
    first = np.asarray(first_face_ids, dtype=int)
    second = np.asarray(second_face_ids, dtype=int)
    if (first.ndim != 1 or second.ndim != 1 or not len(first) or not len(second)
            or min(first.min(), second.min()) < 0 or max(first.max(), second.max()) >= len(faces)):
        raise ValueError('Empty or invalid triangle region')
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    selected = np.unique(np.r_[first, second])
    if np.any(lengths[selected] < 1e-12):
        raise ValueError('Degenerate screened triangle')
    normals[selected] /= lengths[selected, None]
    b_tris = tris[second]
    bounds = np.stack((b_tris.min(axis=1), b_tris.max(axis=1)), axis=1)
    tree = trimesh.util.bounds_tree(bounds)
    sample = []
    crossings = coplanar = shared = broadphase = 0
    for face_id in first:
        a = tris[face_id]
        nearby = np.array(list(tree.intersection(np.r_[a.min(axis=0), a.max(axis=0)])), dtype=int)
        if not len(nearby):
            continue
        other = second[nearby]
        broadphase += len(other)
        adjacent = np.any(faces[other, :, None] == faces[face_id][None, None, :], axis=(1, 2))
        shared += int(adjacent.sum())
        other = other[~adjacent]
        if not len(other):
            continue
        b = tris[other]
        same_plane = ((np.linalg.norm(np.cross(normals[other], normals[face_id]), axis=1) < 1e-8)
                      & (np.abs((b[:, 0] - a[0]) @ normals[face_id]) < 1e-9))
        coplanar += int(same_plane.sum())
        hit = np.zeros(len(other), dtype=bool)
        repeated = np.broadcast_to(a, b.shape)
        for edge in range(3):
            hit |= _segment_hits(repeated[:, edge], repeated[:, (edge + 1) % 3], b)
            hit |= _segment_hits(b[:, edge], b[:, (edge + 1) % 3], repeated)
        hit &= ~same_plane
        crossings += int(hit.sum())
        for other_id in other[hit]:
            if len(sample) < 10:
                sample.append([int(face_id), int(other_id)])
    return dict(
        first_region_face_count=len(first), second_region_face_count=len(second),
        broadphase_pair_count=broadphase, excluded_shared_vertex_pair_count=shared,
        intersecting_pair_count=crossings, coplanar_unresolved_pair_count=coplanar,
        sample_face_pairs=sample, numeric_intersection_screen_clear=(crossings == 0 and coplanar == 0),
        physics_verified=False, containment_verified=False,
        method='trimesh_Rtree_AABB_then_bidirectional_segment_triangle_tests_no_FCL',
        limitations='Shared-vertex seams excluded; coplanar candidates unresolved; no enclosed-volume or physics proof.')


def arm_joint_metrics(joints):
    """Measure arm lowering in the derived anatomical frame, invariant to placement."""
    joints = np.asarray(joints, float)
    frame = anatomical_frame(joints)
    local = (joints - joints[0]) @ frame
    downward, elbow_drop, wrist_height, hand_height, wrist_forward = [], [], [], [], []
    for shoulder, elbow, wrist, hand in ((16, 18, 20, 22), (17, 19, 21, 23)):
        upper = local[elbow] - local[shoulder]
        length = np.linalg.norm(upper)
        if length < 1e-8:
            raise ValueError('Degenerate upper arm')
        downward.append(float(np.degrees(np.arcsin(np.clip(-upper[1] / length, -1, 1)))))
        elbow_drop.append(float(-upper[1]))
        wrist_height.append(float(local[wrist, 1]))
        hand_height.append(float(local[hand, 1]))
        wrist_forward.append(float(local[wrist, 2]))
    return dict(upper_arm_downward_angle_deg=downward, shoulder_to_elbow_drop_m=elbow_drop,
                wrist_above_pelvis_m=wrist_height, hand_above_pelvis_m=hand_height,
                wrist_forward_of_pelvis_m=wrist_forward,
                arms_down_min_angle_threshold_deg=45.,
                arms_down_numeric_screen_passed=min(downward) >= 45.)


def evaluate_arm_geometry(vertices, faces, joints, lbs_weights):
    """Use dominant LBS joint regions, explicit triangle crossings and unsigned distance."""
    import trimesh
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    weights = np.asarray(lbs_weights, float)
    if (weights.shape != (len(vertices), 24) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1., atol=1e-5, rtol=0.)):
        raise ValueError('Invalid SMPL LBS weights')
    labels = weights.argmax(axis=1)
    regions = {}
    for side, joint_ids in ARM_JOINTS.items():
        regions[side] = np.flatnonzero(np.sum(np.isin(labels[faces], joint_ids), axis=1) >= 2)
    regions['body'] = np.setdiff1d(np.arange(len(faces)), np.r_[regions['left'], regions['right']])
    regions['thigh'] = np.flatnonzero(np.sum(np.isin(labels[faces], [1, 2, 4, 5]), axis=1) >= 2)
    if any(not len(ids) for ids in regions.values()):
        raise ValueError('Missing arm/body/thigh LBS region')
    comparisons = {}
    for name, first, second in (
            ('left_arm_vs_body', 'left', 'body'), ('right_arm_vs_body', 'right', 'body'),
            ('left_arm_vs_right_arm', 'left', 'right'),
            ('left_arm_vs_thigh', 'left', 'thigh'), ('right_arm_vs_thigh', 'right', 'thigh')):
        comparisons[name] = triangle_intersections(vertices, faces, regions[first], regions[second])
    distal = vertices[np.isin(labels, [18, 19, 20, 21, 22, 23])]
    if not len(distal):
        raise ValueError('Missing distal arm LBS region')
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces[regions['body']], process=False)
    thigh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces[regions['thigh']], process=False)
    _, body_distance, _ = trimesh.proximity.closest_point(body_mesh, distal)
    _, thigh_distance, _ = trimesh.proximity.closest_point(thigh_mesh, distal)
    joint_metrics = arm_joint_metrics(joints)
    clear = all(result['numeric_intersection_screen_clear'] for result in comparisons.values())
    return dict(
        numeric_arm_screen_passed=bool(clear and joint_metrics['arms_down_numeric_screen_passed']),
        region_method='dominant_SMPL_LBS_joint_majority_face_proxy',
        region_face_counts={key: len(value) for key, value in regions.items()},
        comparisons=comparisons, joint_metrics=joint_metrics,
        distal_arm_vertex_count=len(distal),
        distal_arm_to_nonarm_surface_min_m=float(body_distance.min()),
        distal_arm_to_thigh_surface_min_m=float(thigh_distance.min()),
        full_body_width_m=float(np.ptp((vertices - joints[0]) @ anatomical_frame(joints), axis=0)[0]),
        distance_status='unsigned_open_region_distance_not_penetration_or_support',
        patient_posture_qualified=False, physics_verified=False,
        limitations=[
            'Arm/body/thigh regions are dominant LBS joint proxies, not validated anatomical segmentation.',
            'Mixed seam triangles and excluded shared-vertex pairs can hide local fold/intersection defects.',
            'Unsigned distance to open region meshes does not distinguish interior from exterior.',
            'No robot collision, full-body containment, dynamic settling, balance or hand support validation.',
            'Triangle intersection screen is CPU numeric geometry, not an FCL or physics-engine result.',
        ])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_pilot(*, source_dir, identity_dir, inventory_path, model_path, seat_config_path, output_dir):
    """Regenerate three explicitly synthetic arm poses for the existing eight identities."""
    source_dir, identity_dir, output_dir = Path(source_dir), Path(identity_dir), Path(output_dir)
    paths = dict(source_manifest=source_dir / 'asset_manifest.json',
                 source_summary=source_dir / 'summary.json',
                 identity_split=identity_dir / 'identity_split.json',
                 identity_quality=identity_dir / 'identity_quality.json',
                 anchor_inventory=Path(inventory_path), smpl_neutral=Path(model_path),
                 seat_config=Path(seat_config_path), seated_pose_module=Path(__file__),
                 body_placement_module=Path(body_placement.__file__),
                 surface_anchors_module=Path(surface_anchors.__file__))
    hashes = {key: _sha(path) for key, path in paths.items()}
    source_summary = json.loads(paths['source_summary'].read_text())
    source_manifest = json.loads(paths['source_manifest'].read_text())
    quality = json.loads(paths['identity_quality'].read_text())
    split = json.loads(paths['identity_split'].read_text())['splits']
    for field, key in [('identity_split_sha256', 'identity_split'),
                       ('anchor_inventory_sha256', 'anchor_inventory'),
                       ('smpl_neutral_sha256', 'smpl_neutral')]:
        if source_summary[field] != hashes[key]:
            raise ValueError('Frozen source hash mismatch: ' + field)
    if quality['identity_split_sha256'] != hashes['identity_split']:
        raise ValueError('Identity quality split hash mismatch')
    ids = list(dict.fromkeys(item['identity'] for item in source_manifest))
    train = set(split['train'])
    heldout = set().union(*(set(value) for key, value in split.items() if key != 'train'))
    if len(ids) != 8 or source_summary['identities'] != 8 or not set(ids) <= train or train & heldout:
        raise ValueError('Expected exactly the eight frozen train identities without held-out overlap')
    betas = {}
    for identity in ids:
        beta = np.asarray(quality['eligible_shape_betas'][identity], float)
        if beta.shape != (10,) or not np.isfinite(beta).all():
            raise ValueError('Invalid frozen beta for ' + identity)
        betas[identity] = beta
    assets = {}
    baselines = {}
    for record in source_manifest:
        path = source_dir / record['file']
        if path.resolve().parent != source_dir.resolve() or path.name in assets:
            raise ValueError('Duplicate or non-local frozen source asset')
        if _sha(path) != record['sha256']:
            raise ValueError('Frozen source asset hash mismatch')
        assets[path.name] = path
        if record['pose'] == 'seated_lean_asset':
            if record['identity'] in baselines:
                raise ValueError('Duplicate source seated baseline')
            baselines[record['identity']] = record
    if set(baselines) != set(ids):
        raise ValueError('Each frozen identity needs its original seated_lean_asset baseline')
    anchors = json.loads(paths['anchor_inventory'].read_text())['anchors']
    seat_document = json.loads(paths['seat_config'].read_text())
    seat = SeatConfig(**seat_document['config'])
    import smplx
    import torch
    torch.set_num_threads(2)
    model = smplx.SMPL(model_path=str(paths['smpl_neutral']), gender='neutral', batch_size=1, num_betas=10)
    weights = model.lbs_weights.detach().cpu().numpy()
    proposals = synthetic_pose_proposals()
    generated = []
    baseline_checks = []
    baseline_pose = np.zeros((23, 3), np.float32)
    baseline_pose[0, 0] = baseline_pose[1, 0] = -1.25
    baseline_pose[3, 0] = baseline_pose[4, 0] = 1.4
    baseline_pose[2, 0] = .12
    with torch.no_grad():
        for number, identity in enumerate(ids):
            beta = torch.tensor(betas[identity], dtype=torch.float32).reshape(1, 10)
            baseline = model(betas=beta, body_pose=torch.tensor(baseline_pose).reshape(1, 69))
            reference = baselines[identity]
            with np.load(assets[reference['file']], allow_pickle=False) as asset:
                vertex_error = float(np.max(np.abs(baseline.vertices[0].numpy() - asset['vertices'])))
                joint_error = float(np.max(np.abs(baseline.joints[0].numpy() - asset['joints'])))
                if (vertex_error > 1e-6 or joint_error > 1e-6 or not np.array_equal(model.faces, asset['faces'])):
                    raise ValueError('Regenerated frozen beta/model baseline differs from source: ' + identity)
            baseline_checks.append(dict(identity=identity, source_file=reference['file'],
                                        source_sha256=reference['sha256'],
                                        maximum_vertex_error_m=vertex_error, maximum_joint_error_m=joint_error,
                                        same_topology=True, mesh_scale=1., beta_unchanged=True))
            for proposal in proposals:
                pose = np.asarray(proposal['body_pose_axis_angle_rad'], np.float32)
                value = model(betas=beta, body_pose=torch.tensor(pose).reshape(1, 69))
                vertices = value.vertices[0].numpy()
                joints = value.joints[0].numpy()
                anchored = evaluate_anchors(vertices, model.faces, anchors)
                body = dict(vertices=vertices, joints=joints, faces=model.faces,
                            anchor_names=np.array(list(anchored)),
                            anchor_positions=np.array([a['position_m'] for a in anchored.values()]),
                            anchor_normals=np.array([a['normal'] for a in anchored.values()]))
                placed = place_body(body, seat, anchors)
                # body_placement describes the old source pilot arms. This new
                # generator replaces those rotations; retain the qualification limit.
                placed['qa']['limitations'] = [
                    ('Source T-pose-like arm rotations were replaced by recorded synthetic rotations; '
                     'patient arm posture remains unqualified.') if 'T-pose-like' in text else text
                    for text in placed['qa']['limitations']]
                arms = evaluate_arm_geometry(placed['vertices'], placed['faces'], placed['joints'], weights)
                reasons = list(placed['qa']['rejection_reasons'])
                if not arms['joint_metrics']['arms_down_numeric_screen_passed']:
                    reasons.append('arms_not_sufficiently_lowered')
                for key, result in arms['comparisons'].items():
                    if not result['numeric_intersection_screen_clear']:
                        reasons.append(key + '_intersection_or_coplanar_unresolved')
                filename = f'body{number:02d}_{identity}_{proposal["name"]}.npz'
                arrays = {key: value for key, value in placed.items() if key != 'qa'}
                arrays.update(betas=beta.numpy()[0], body_pose_axis_angle_rad=pose,
                              global_orient_axis_angle_rad=np.zeros(3, np.float32),
                              source_translation_m=np.zeros(3, np.float32))
                record = dict(
                    identity=identity, split='train', proposal=proposal['name'], file=filename,
                    source_baseline_file=reference['file'], source_baseline_sha256=reference['sha256'],
                    source_beta_sha256=hashlib.sha256(json.dumps(betas[identity].tolist(), separators=(',', ':')).encode()).hexdigest(),
                    mesh_scale=1., synthetic_pose=True, human_motion_capture_pose=False,
                    numeric_candidate_screen_passed=not reasons, rejection_reasons=reasons,
                    placement_qa=placed['qa'], arm_geometry_qa=arms,
                    world_from_smpl=placed['world_from_smpl'].tolist(),
                    patient_posture_qualified=False, training_eligible=False,
                    expert_qualified=False, isaac_rendered=False)
                generated.append((record, arrays))
            print(f'Assessed {identity}: three authored arm-pose candidates.', flush=True)
    for key, path in paths.items():
        if _sha(path) != hashes[key]:
            raise ValueError('Provenance changed during candidate generation: ' + key)
    for record in source_manifest:
        if _sha(assets[record['file']]) != record['sha256']:
            raise ValueError('Frozen source asset changed during generation')
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for record, arrays in generated:
        np.savez_compressed(output_dir / record['file'], **arrays)
        record['sha256'] = _sha(output_dir / record['file'])
        manifest.append(record)
    summary = dict(
        schema='m2_authored_seated_arm_pose_pilot_v1', identities=8, proposals_per_identity=len(proposals),
        candidate_count=len(manifest), split='same eight frozen train identities',
        numeric_candidate_screen_passed_count=sum(item['numeric_candidate_screen_passed'] for item in manifest),
        identities_with_numeric_candidate=sum(any(item['numeric_candidate_screen_passed'] for item in manifest
                                                 if item['identity'] == identity) for identity in ids),
        synthetic_pose_source='Authored SMPL axis-angle rotations, not HuMMan observed motion.',
        identity_betas_unchanged=True, mesh_scale=1., source_assets_unchanged=True,
        baseline_reproduction=baseline_checks,
        identity_split_sha256=hashes['identity_split'], smpl_neutral_sha256=hashes['smpl_neutral'],
        anchor_inventory_sha256=hashes['anchor_inventory'],
        lbs_region_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        clothing='none; no fitted garment collision surface',
        numeric_screen_scope='SMPL placement plus LBS-region triangle intersections and unsigned distances.',
        patient_posture_qualified=False, physics_qualified=False, expert_qualified=False,
        reachability_qualified=False, isaac_rendered=False, training_eligible=False,
        provenance={key: dict(path=str(path.resolve()), sha256=hashes[key]) for key, path in paths.items()})
    _write_json(output_dir / 'pose_proposals.json', proposals)
    _write_json(output_dir / 'seat_config.json', seat_document)
    _write_json(output_dir / 'asset_manifest.json', manifest)
    _write_json(output_dir / 'summary.json', summary)
    rows = ['# Authored seated arm-pose numerical pilot', '',
            'Eight frozen train identities, three authored SMPL proposals each. '
            'These are synthetic joint rotations, not HuMMan observed poses.',
            f'{summary["numeric_candidate_screen_passed_count"]}/{len(manifest)} candidates pass the combined numerical screen; '
            f'{summary["identities_with_numeric_candidate"]}/8 identities have a passing candidate.',
            'No candidate is patient-posture-, physics-, expert-, reachability- or training-qualified.', '',
            '| Identity | Proposal | Numerical gate | Arm/body crossings L/R | Distal arm-to-thigh minimum (mm) | Arm lowering L/R (deg) |',
            '|---|---|---|---|---|---|']
    for item in manifest:
        arms = item['arm_geometry_qa']
        comparisons = arms['comparisons']
        angle = arms['joint_metrics']['upper_arm_downward_angle_deg']
        rows.append(f'| {item["identity"]} | {item["proposal"]} | '
                    f'{"pass" if item["numeric_candidate_screen_passed"] else "rejected"} | '
                    f'{comparisons["left_arm_vs_body"]["intersecting_pair_count"]}/'
                    f'{comparisons["right_arm_vs_body"]["intersecting_pair_count"]} | '
                    f'{1000 * arms["distal_arm_to_thigh_surface_min_m"]:.2f} | {angle[0]:.1f}/{angle[1]:.1f} |')
    rows += ['', 'All beta vectors, model bytes, V6A anchors and identity split are hash-bound. '
             'Regenerating each original seated-lean baseline verifies that shape and scale were preserved.',
             'The approved stool and camera remain unchanged. Feet and hands may remain unsupported; '
             'no foot-ground or hand-lap support is required or claimed.',
             'Collision screens use dominant SMPL skinning-joint regions, trimesh R-tree candidates and '
             'nonadjacent triangle segment tests. Shared-vertex pairs are excluded and coplanar candidates '
             'remain unresolved. Open-region distances are unsigned, not penetration or support evidence.',
             'Exact whole-body containment, shoulder skin folds, robot/stool collision, clothing, camera '
             'occlusion and physical stability still require downstream verification.', '']
    (output_dir / 'README.md').write_text('\n'.join(rows))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'identity-dir', 'inventory-path', 'model-path', 'seat-config-path', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot(**vars(args)), indent=2))


if __name__ == '__main__':
    main()

"""CPU export of the frozen USD/FCL geometry to legacy cuRobo v0.7.6.

cuRobo is a proposal generator. Its conservatively covered robot geometry is
checked against the original meshes/posture independently before execution.
This module does not import torch, Isaac Sim, CUDA, or alter the source package.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_expert_layout_v1.scene import ROBOT, FIXED, MOVING
from sim.auscultation_expert_layout_v1.tool_frames import wrist_from_diaphragm

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'sim/auscultation_expert_layout_v1'
PLANS = SOURCE / 'results/expert_v2/plans'
BASE_PATH = SOURCE / 'results/expert_v2/physics_all10/world_from_left_base.json'
JOINT_NAMES = tuple(f'openarm_left_joint{i}' for i in range(1, 8))
ARM_LINKS = tuple(f'openarm_left_link{i}' for i in range(8))
TOOL_LINKS = ('stetho_fixed', 'stetho_carriage', 'stetho_diaphragm')
FK_LINKS = ARM_LINKS + TOOL_LINKS


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def _rigid(value):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-9, rtol=0)):
        raise ValueError('expected a finite proper rigid transform')
    return transform


def points_in_base(points_world, world_from_base):
    """Column-vector convention: p_B = R_WB.T @ (p_W - t_WB)."""
    transform = _rigid(world_from_base)
    points = np.asarray(points_world, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('points must be finite Nx3')
    return (points - transform[:3, 3]) @ transform[:3, :3]


def tool_transforms():
    """Wrist-from-body transforms at released zero compression, in metres."""
    fixed = np.diag([1., -1., -1., 1.])
    fixed[2, 3] = .085
    return fixed, fixed.copy(), wrist_from_diaphragm(0.)


def maximum_uncovered_distance(points, spheres):
    points = np.asarray(points, float)
    if not spheres or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('coverage requires finite points and nonempty spheres')
    centers = np.asarray([s['center'] for s in spheres], float)
    radii = np.asarray([s['radius'] for s in spheres], float)
    worst = -float('inf')
    for start in range(0, len(points), 1024):
        delta = np.linalg.norm(points[start:start+1024, None] - centers[None], axis=2) - radii
        worst = max(worst, float(delta.min(axis=1).max()))
    return worst


def cover_convex_mesh(vertices, max_radius=.02):
    """Cover the ENTIRE convex hull volume using occupied grid-cell spheres.

    The source AABB is tiled without gaps. A cell is discarded only if its
    minimum signed distance to at least one convex-hull supporting plane is
    positive, proving that the entire cell is outside the hull. Every retained
    cell is contained in its half-diagonal sphere. Thus all source mesh points
    and enclosed volume are covered, including for a nonconvex source (whose
    convex hull is deliberately conservative). No sphere fitting/sample-only
    assertion is used as the coverage proof.
    """
    vertices = np.asarray(vertices, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or not np.isfinite(max_radius)
            or not .001 <= max_radius <= .05):
        raise ValueError('invalid vertices or sphere radius')
    extent = np.ptp(vertices, axis=0)
    if np.any(extent < 1e-10):
        raise ValueError('robot collider must have nonzero volume in all axes')
    equations = ConvexHull(vertices).equations
    count = np.maximum(1, np.ceil(extent / (2*max_radius/np.sqrt(3))).astype(int))
    if np.prod(count) > 200000:
        raise ValueError('sphere cover exceeds bounded cell budget')
    steps = extent / count
    axes = [vertices[:, axis].min() + (np.arange(count[axis])+.5)*steps[axis] for axis in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    support = np.abs(equations[:, :3]) @ (steps/2)
    keep = []
    for begin in range(0, len(centers), 512):
        chunk = centers[begin:begin+512]
        separated = np.any(chunk @ equations[:, :3].T + equations[:, 3] - support > 1e-10, axis=1)
        keep.extend(chunk[~separated])
    radius = float(np.linalg.norm(steps/2) + 1e-10)
    spheres = [{'center': c.tolist(), 'radius': radius} for c in keep]
    sampled = maximum_uncovered_distance(vertices, spheres)
    if sampled > 1e-9:
        raise ValueError('source vertices are not covered')
    return spheres, {
        'guarantee': 'entire_convex_hull_volume',
        'method': 'AABB cell tiling; hull supporting-plane separation; half-diagonal enclosing spheres',
        'max_radius_requested_m': float(max_radius), 'actual_radius_m': radius,
        'grid_counts': count.tolist(), 'grid_step_m': steps.tolist(),
        'source_aabb_m': [vertices.min(0).tolist(), vertices.max(0).tolist()],
        'total_cells': int(np.prod(count)), 'retained_cells': len(spheres),
        'source_vertex_count': len(vertices), 'vertex_max_uncovered_distance_m': sampled,
        'conservatism': 'Convexification and grid spheres can reject valid paths; they must not be shrunk to obtain success.',
    }


def _refine_self_cover(cells, hulls, joint_vectors, allowed_pairs, max_spheres=1023):
    """Split only enclosing cells responsible for known-pose false positives.

    Splitting never reduces geometric coverage: child boxes partition the
    parent, and only hull-plane-separated boxes can be discarded. No collision
    ignore is added and no fitted sphere is shrunk. The finite reference set is
    a fit diagnostic, not a claim about arbitrary configurations or paths.
    """
    kin = OpenArmKinematics()
    transforms = []
    for q in joint_vectors:
        frames = kin.link_transforms(q)
        frames.update({name: frames[ARM_LINKS[-1]] @ transform for name, transform in zip(TOOL_LINKS, tool_transforms())})
        transforms.append(frames)
    history = []
    for iteration in range(17):
        links = sorted({c['link'] for c in cells})
        indices = {link: np.array([i for i, c in enumerate(cells) if c['link'] == link]) for link in links}
        centers = {link: np.array([cells[i]['center'] for i in ids]) for link, ids in indices.items()}
        radii = {link: np.array([np.linalg.norm(cells[i]['half'])+1e-10 for i in ids]) for link, ids in indices.items()}
        split, overlaps, minimum = set(), 0, float('inf')
        for frames in transforms:
            transformed = {link: points @ frames[link][:3, :3].T + frames[link][:3, 3] for link, points in centers.items()}
            for i, a in enumerate(links):
                for b in links[i+1:]:
                    if frozenset((a, b)) in allowed_pairs:
                        continue
                    delta = np.linalg.norm(transformed[a][:, None]-transformed[b][None], axis=2)-radii[a][:, None]-radii[b][None]
                    minimum = min(minimum, float(delta.min()))
                    ai, bi = np.where(delta < 1e-6)
                    overlaps += len(ai)
                    for x, y in zip(ai, bi):
                        split.add(int(indices[a][x] if radii[a][x] >= radii[b][y] else indices[b][y]))
        history.append({'iteration': iteration, 'sphere_count': len(cells),
                        'overlapping_sphere_pairs_across_reference_poses': overlaps,
                        'minimum_sphere_gap_m': minimum, 'cells_to_split': len(split)})
        if not split:
            if len(cells) > max_spheres:
                raise ValueError('legacy self-collision kernel requires fewer than 1024 spheres')
            return cells, {'status': 'passed_reference_pose_fit', 'iterations': iteration,
                           'reference_pose_count': len(transforms), 'final_overlapping_pairs': 0,
                           'history': history, 'scope': 'ten canonical free preposes only; actual runtime endpoints/paths still require cuRobo and independent FCL checks'}
        if iteration == 16:
            raise ValueError('conservative sphere fitting did not clear bounded reference poses')
        refined = []
        for index, cell in enumerate(cells):
            if index not in split:
                refined.append(cell)
                continue
            half = cell['half'].copy()
            axis = int(np.argmax(half)); half[axis] /= 2
            equations = hulls[cell['key']]
            for sign in (-1, 1):
                center = cell['center'].copy(); center[axis] += sign*half[axis]
                separated = np.any(center @ equations[:, :3].T + equations[:, 3] - np.abs(equations[:, :3]) @ half > 1e-10)
                if not separated:
                    refined.append({**cell, 'center': center, 'half': half.copy()})
        cells = refined
        if len(cells) > max_spheres:
            raise ValueError('conservative sphere fitting exceeds native self-collision sphere budget; do not drop geometry')


def _pose(matrix):
    matrix = _rigid(matrix)
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [*matrix[:3, 3].tolist(), float(xyzw[3]), *xyzw[:3].tolist()]


def reference_fk(joint_vectors):
    kin = OpenArmKinematics()
    vectors = np.asarray(joint_vectors, float)
    if vectors.ndim != 2 or vectors.shape[1] != 7:
        raise ValueError('FK joint vectors must have shape Nx7')
    poses = []
    for q in vectors:
        frames = kin.link_transforms(q)
        wrist = frames[ARM_LINKS[-1]]
        frames.update({name: wrist @ local for name, local in zip(TOOL_LINKS, tool_transforms())})
        poses.append([_pose(frames[link]) for link in FK_LINKS])
    return {'joint_names': list(JOINT_NAMES), 'link_names': list(FK_LINKS), 'poses': poses,
            'frame': ARM_LINKS[0], 'pose_format': 'xyz_wxyz', 'units': 'metres_radians'}


def validate_backend_fk(joint_vectors, response, position_tolerance_m=2e-5, angle_tolerance_rad=2e-4):
    """Compare every arm/tool link with independent float64 NumPy FK."""
    if 'link_poses' in response:
        metadata = response.get('metadata', {})
        if (response.get('success') is not True or metadata.get('pose_frame') != 'robot_base'
                or metadata.get('quaternion_order') != 'wxyz'):
            raise ValueError('backend FK failed or frame/quaternion convention differs')
        if not isinstance(response['link_poses'], dict) or set(response['link_poses']) != set(FK_LINKS):
            raise ValueError('backend FK link set differs')
        response = {'joint_names': metadata.get('joint_names'), 'link_names': list(FK_LINKS),
                    'poses': np.stack([np.asarray(response['link_poses'][name], float) for name in FK_LINKS], axis=1)}
    if response.get('joint_names') != list(JOINT_NAMES):
        raise ValueError('backend joint order differs from left-arm contract')
    if response.get('link_names') != list(FK_LINKS):
        raise ValueError('backend FK links missing or reordered')
    expected = np.asarray(reference_fk(joint_vectors)['poses'])
    actual = np.asarray(response.get('poses'), float)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('backend FK pose array shape/nonfinite')
    lengths = np.linalg.norm(actual[:, :, 3:], axis=-1)
    if not np.allclose(lengths, 1., atol=1e-5, rtol=0):
        raise ValueError('backend FK quaternions are not normalized')
    position = float(np.linalg.norm(actual[:, :, :3]-expected[:, :, :3], axis=-1).max())
    dot = np.sum(actual[:, :, 3:]/lengths[..., None] * expected[:, :, 3:], axis=-1)
    angle = float((2*np.arccos(np.clip(np.abs(dot), 0., 1.))).max())
    if position > position_tolerance_m or angle > angle_tolerance_rad:
        raise ValueError(f'backend FK mismatch: position={position}, angle={angle}')
    return {'status': 'passed', 'max_position_error_m': position, 'max_orientation_error_rad': angle,
            'joint_vectors_checked': len(expected), 'links_checked': list(FK_LINKS),
            'position_tolerance_m': position_tolerance_m, 'orientation_tolerance_rad': angle_tolerance_rad}


def _derived_urdf(source_path, destination):
    """Retain the exact left7 FK chain; geometry lives in the exported spheres."""
    original = ET.parse(source_path).getroot()
    robot = ET.Element('robot', name='openarm_left_stetho_curobo_v1')
    for name in ARM_LINKS + TOOL_LINKS:
        ET.SubElement(robot, 'link', name=name)
    joints = {x.get('name'): x for x in original.findall('joint')}
    for name in JOINT_NAMES:
        joint = copy.deepcopy(joints[name])
        # Engineering command speed; positions/axes/origins are unchanged.
        joint.find('limit').set('velocity', '0.25')
        robot.append(joint)
    for name, parent, xyz, rpy in [
        ('stetho_fixed', ARM_LINKS[-1], '0 0 0.085', f'{np.pi:.17g} 0 0'),
        ('stetho_carriage', 'stetho_fixed', '0 0 0', '0 0 0'),
        ('stetho_diaphragm', ARM_LINKS[-1], '0 0 0.1262', '0 0 0'),
    ]:
        joint = ET.SubElement(robot, 'joint', name=name+'_fixed_joint', type='fixed')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(joint, 'origin', xyz=xyz, rpy=rpy)
    ET.indent(robot)
    ET.ElementTree(robot).write(destination, encoding='utf-8', xml_declaration=True)


def _obj(path, vertices, faces):
    with Path(path).open('w') as out:
        out.write('# Exact exported collider vertices in robot base coordinates; metres\n')
        for v in vertices:
            out.write('v '+' '.join(format(float(x), '.17g') for x in v)+'\n')
        for face in faces:
            out.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')


def export_bridge(output_dir, *, geometry_dir=PLANS, world_from_base_path=BASE_PATH,
                  arm_radius_m=.022, tool_radius_m=.011):
    """Export relocatable configs plus a complete, hash-bound collider ledger.

    Free-space only: passive compression is fixed to zero. The controller must
    establish force/stroke release before querying the planner. All robot links
    remain independently checked against every original static mesh by FCL.
    """
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('export destination must be empty; preserve prior configurations')
    if output.is_relative_to(SOURCE.resolve()):
        raise ValueError('cannot write into frozen source package')
    geometry = Path(geometry_dir).resolve()
    base_path = Path(world_from_base_path).resolve()
    sources = [SOURCE/'openarm_v10_follower_stetho.urdf', geometry/'collision_geometry.json',
               geometry/'collision_geometry.npz', base_path, SOURCE/'kinematics.py',
               SOURCE/'tool_frames.py', SOURCE/'collision.py', SOURCE/'scene.py',
               SOURCE/'layout_inputs.json', Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(PROJECT)): _digest(p) for p in sources}
    metadata = json.loads((geometry/'collision_geometry.json').read_text())
    base = _rigid(json.loads(base_path.read_text()))
    for missing in metadata['unsupported']:
        if missing != {'path': '/World/GroundPlane/collisionPlane', 'type': 'Plane'}:
            raise ValueError('unsupported collision geometry must be explicitly resolved: '+str(missing))
    output.mkdir(parents=True, exist_ok=True)
    (output/'world_meshes').mkdir()
    _derived_urdf(sources[0], output/'robot.urdf')
    spheres = {name: [] for name in ARM_LINKS + TOOL_LINKS[:2]}
    mapped = {ROBOT+'/'+name: name for name in ARM_LINKS}
    mapped.update({FIXED: 'stetho_fixed', MOVING: 'stetho_carriage'})
    world = {'mesh': {}, 'cuboid': {}}
    ledger, coverage, unsupported, omitted = [], {}, [], []
    cells, hulls, vertices_by_key, faces_by_key = [], {}, {}, {}
    reference_vectors = []
    for number in range(1, 6):
        for side in ('R', 'L'):
            path = PLANS/f'planned_paths_{side}{number}.json'
            source_hashes[str(path.relative_to(PROJECT))] = _digest(path)
            reference_vectors.append(json.loads(path.read_text())['points'][0]['transfer_q'][-1])
    arrays = np.load(geometry/'collision_geometry.npz', allow_pickle=False)
    for entry in metadata['colliders']:
        key, body = entry['key'], entry['body']
        vertices = np.asarray(arrays[key+'_v'], float)
        faces = np.asarray(arrays[key+'_f'], np.int64)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('invalid collider array: '+key)
        record = {'key': key, 'source_path': entry['path'], 'source_body': body,
                  'source_approximation': entry['approximation']}
        if body in mapped:
            link = mapped[body]
            result, proof = cover_convex_mesh(vertices, tool_radius_m if body in (FIXED, MOVING) else arm_radius_m)
            sample = np.concatenate([vertices, vertices[faces].mean(axis=1)], axis=0)
            proof['triangle_centroid_count'] = len(faces)
            proof['vertex_and_triangle_centroid_max_uncovered_distance_m'] = maximum_uncovered_distance(sample, result)
            if proof['vertex_and_triangle_centroid_max_uncovered_distance_m'] > 1e-9:
                raise ValueError('collider sample coverage failed: '+key)
            cells.extend({'center': np.array(s['center']), 'half': np.array(proof['grid_step_m'])/2,
                          'link': link, 'key': key} for s in result)
            hulls[key] = ConvexHull(vertices).equations
            vertices_by_key[key], faces_by_key[key] = vertices, faces
            coverage[key] = proof
            record.update(role='fixed_robot' if link == ARM_LINKS[0] else 'moving_robot',
                          robot_link=link)
        elif body == ROBOT+'/openarm_body_link0':
            # The represented fixed left_link0 is bolted into this parent body.
            # v0.7.6 has no per-robot-link/world-object allowed-pair mask; putting
            # this parent in the world creates permanent start collision. Keep
            # all original geometry in the required independent FCL gate.
            record.update(role='fcl_only_fixed_mount', reason='fixed left base overlaps its allowed parent; native world pair filtering unavailable',
                          independent_gate='original_full_mesh_FCL')
            omitted.append(copy.deepcopy(record))
        else:
            transform = _rigid(entry['world_from_body'])
            vertices_world = vertices @ transform[:3, :3].T + transform[:3, 3]
            vertices_base = points_in_base(vertices_world, base)
            relative = 'world_meshes/'+key+'.obj'
            _obj(output/relative, vertices_base, faces)
            world['mesh'][key] = {'file_path': relative, 'pose': [0., 0., 0., 1., 0., 0., 0.], 'scale': [1., 1., 1.]}
            record.update(role='world', world_object=key, vertices_baked_in=ARM_LINKS[0])
        ledger.append(record)
    # The source Plane is unbounded and omitted by the old triangle exporter.
    # Keep its actual c000 ground mesh AND add a thick conservative workspace
    # floor. This covers below-ground containment within the 40m source bounds.
    ground = next((e for e in metadata['colliders'] if e['path'] == '/World/GroundPlane/geom'), None)
    if metadata['unsupported'] and ground is None:
        raise ValueError('ground Plane lacks its documented mesh substitute')
    if metadata['unsupported']:
        floor_world = np.eye(4); floor_world[2, 3] = -1.
        world['cuboid']['ground_volume'] = {'dims': [40., 40., 2.], 'pose': _pose(np.linalg.inv(base) @ floor_world)}
        unsupported = [{**x, 'resolution': 'covered_by_exported_ground_mesh_plus_conservative_floor_cuboid',
                        'scope': 'finite 40x40m workspace; floor extends from world z=-2 to 0'} for x in metadata['unsupported']]
    ignores = {name: [] for name in spheres}
    pair_ledger = []
    for pair in metadata['allowed_body_pairs']:
        if pair[0] in mapped and pair[1] in mapped:
            a, b = mapped[pair[0]], mapped[pair[1]]
            if b not in ignores[a]:
                ignores[a].append(b)
                pair_ledger.append({'source_pair': pair, 'robot_pair': [a, b]})
    allowed = {frozenset(item['robot_pair']) for item in pair_ledger}
    cells, fit = _refine_self_cover(cells, hulls, reference_vectors, allowed)
    for record in ledger:
        if record['role'] not in ('moving_robot', 'fixed_robot'):
            continue
        key, link = record['key'], record['robot_link']
        selected = [cell for cell in cells if cell['key'] == key]
        fitted = [{'center': c['center'].tolist(), 'radius': float(np.linalg.norm(c['half'])+1e-10)} for c in selected]
        record['sphere_start_index'], record['sphere_count'] = len(spheres[link]), len(fitted)
        spheres[link].extend(fitted)
        samples = np.concatenate([vertices_by_key[key], vertices_by_key[key][faces_by_key[key]].mean(axis=1)], axis=0)
        gap = maximum_uncovered_distance(samples, fitted)
        if gap > 1e-9:
            raise ValueError('refined conservative cover lost source geometry: '+key)
        coverage[key].update(refined_sphere_count=len(fitted),
                             refined_vertex_and_triangle_centroid_max_uncovered_distance_m=gap,
                             final_cell_bounds_m=[[(c['center']-c['half']).tolist(), (c['center']+c['half']).tolist()] for c in selected])
    midpoint = (OpenArmKinematics().lower_limits + OpenArmKinematics().upper_limits)/2
    # Use a recorded checked posture, rather than claiming the joint midpoint is
    # collision-free. This is a regularizer/retract seed only, not an exemption.
    plan = json.loads((PLANS/'planned_paths_R1.json').read_text())
    retract = plan['points'][0]['transfer_q'][-1]
    source_hashes[str((PLANS/'planned_paths_R1.json').relative_to(PROJECT))] = _digest(PLANS/'planned_paths_R1.json')
    config = {'robot_cfg': {'kinematics': {
        'urdf_path': 'robot.urdf', 'asset_root_path': '.', 'use_usd_kinematics': False,
        'base_link': ARM_LINKS[0], 'ee_link': TOOL_LINKS[-1], 'link_names': list(FK_LINKS),
        'collision_link_names': list(spheres), 'collision_spheres': spheres,
        'collision_sphere_buffer': 0., 'self_collision_ignore': ignores,
        'self_collision_buffer': {name: 0. for name in spheres},
        'use_global_cumul': True, 'lock_joints': None, 'mesh_link_names': [], 'load_meshes': False,
        'cspace': {'joint_names': list(JOINT_NAMES), 'retract_config': retract,
                   'null_space_weight': [1.]*7, 'cspace_distance_weight': [1.]*7,
                   'max_acceleration': 2., 'max_jerk': 20.},
    }}}
    _write_json(output/'robot_config.json', config)
    _write_json(output/'world_config.json', world)
    _write_json(output/'sphere_coverage.json', coverage)
    # Valid URDF configurations, not claims that all poses are collision-free.
    fk_q = [midpoint.tolist(), retract]
    for i in range(7):
        q = midpoint.copy(); q[i] += .1; fk_q.append(q.tolist())
    _write_json(output/'fk_reference.json', {'joint_vectors': fk_q, **reference_fk(fk_q)})
    if any(_digest(PROJECT/path) != value for path, value in source_hashes.items()):
        raise ValueError('source changed during bridge export')
    outputs = {str(path.relative_to(output)): _digest(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {
        'schema': 'wam_curobo_bridge_v1', 'backend_api': 'NVlabs/curobo v0.7.6',
        'source_sha256': source_hashes, 'output_sha256': outputs,
        'world_from_left_base': base.tolist(), 'base_frame': ARM_LINKS[0],
        'joint_names': list(JOINT_NAMES), 'joint_units': 'rad', 'length_units': 'm', 'pose_format': 'xyz_wxyz',
        'collider_ledger': ledger, 'allowed_pair_ledger': pair_ledger,
        'unsupported_source_geometry': unsupported, 'intentionally_omitted_colliders': omitted,
        'sphere_count': sum(len(v) for v in spheres.values()),
        'self_collision_fit': fit,
        'static_right_arm_policy': 'all right-arm and finger meshes frozen at exported world_from_body; execution must verify unchanged poses',
        'passive_compression_m': 0.,
        'compression_scope': 'free-space proposals only after measured force/stroke release; compression/contact handled by independent controller',
        'urdf_derivation': 'exact source left7 origin/axis/position limits; engineering velocity 0.25rad/s; added fixed tool and membrane frames; no stale gripper meshes',
        'required_independent_validation': [
            'validate_backend_fk on every reference link before planner use',
            'original FCL mesh/posture check every proposed edge, retained-command connector and final 240Hz commands',
            'original collision exceptions only; no patient contact exception in free-space',
            'c002 mounted fixed parent is excluded from native world only; all moving-robot contacts against it require original FCL',
            'verify frozen right-arm/body/world poses against actual runtime before motion',
            'recorded PhysX full ten-point force, HOLD, release, posture and raw command/measured dynamics audits',
        ],
        'limits': [
            'Conservative hull/grid spheres can cause false-positive collision and planning failure.',
            'Original FCL unsigned triangle distance and 0.015rad edge sampling are not a continuous solid-volume collision proof.',
            'World patient meshes preserve original topology; no assertion that clinical body geometry is calibrated.',
            'The bridge does not assert a GPU FK or planning result; CPU export is not planner success.',
        ],
    }
    _write_json(output/'bridge_manifest.json', manifest)
    return {'robot_config_path': str(output/'robot_config.json'), 'world_config_path': str(output/'world_config.json'),
            'manifest_path': str(output/'bridge_manifest.json')}


def verify_bridge(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root/'bridge_manifest.json').read_text())
    if manifest.get('schema') != 'wam_curobo_bridge_v1' or manifest.get('joint_names') != list(JOINT_NAMES):
        raise ValueError('bridge schema or joint contract mismatch')
    outputs = manifest.get('output_sha256')
    source_hashes = manifest.get('source_sha256')
    required = {'robot.urdf', 'robot_config.json', 'world_config.json', 'sphere_coverage.json', 'fk_reference.json'}
    if not isinstance(outputs, dict) or not required.issubset(outputs):
        raise ValueError('bridge is missing required output hash anchors')
    actual_files = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and path.name != 'bridge_manifest.json'}
    if set(outputs) != actual_files:
        raise ValueError('bridge directory files do not match output hash anchors')
    required_source_names = {'openarm_v10_follower_stetho.urdf', 'collision_geometry.json',
                             'collision_geometry.npz', 'world_from_left_base.json', 'kinematics.py',
                             'tool_frames.py', 'collision.py', 'scene.py', 'layout_inputs.json', 'robot_bridge.py'}
    if not isinstance(source_hashes, dict) or not required_source_names.issubset({Path(p).name for p in source_hashes}):
        raise ValueError('bridge is missing required source hash anchors')
    if (not isinstance(manifest.get('sphere_count'), int) or not 0 < manifest['sphere_count'] < 1024
            or manifest.get('self_collision_fit', {}).get('final_overlapping_pairs') != 0):
        raise ValueError('bridge native self collision fit is unavailable')
    for relative, expected in manifest['output_sha256'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _digest(path) != expected:
            raise ValueError('bridge output hash mismatch: '+relative)
    for relative, expected in manifest['source_sha256'].items():
        path = (PROJECT/relative).resolve()
        if not path.is_relative_to(PROJECT) or not path.is_file() or _digest(path) != expected:
            raise ValueError('bridge source hash mismatch: '+relative)
    return {'status': 'passed', 'manifest_sha256': _digest(root/'bridge_manifest.json'),
            'outputs_checked': len(manifest['output_sha256']), 'sources_checked': len(manifest['source_sha256'])}


def load_bridge(directory):
    """Return (wrapped robot dict, world dict, manifest), with absolute paths."""
    verify_bridge(directory)
    root = Path(directory).resolve()
    robot = json.loads((root/'robot_config.json').read_text())
    world = json.loads((root/'world_config.json').read_text())
    kin = robot['robot_cfg']['kinematics']
    kin['urdf_path'] = str(root/kin['urdf_path'])
    kin['asset_root_path'] = str(root)
    for mesh in world['mesh'].values():
        mesh['file_path'] = str(root/mesh['file_path'])
    return robot, world, json.loads((root/'bridge_manifest.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm-radius', type=float, default=.022)
    parser.add_argument('--tool-radius', type=float, default=.011)
    args = parser.parse_args()
    print(json.dumps(export_bridge(args.output, arm_radius_m=args.arm_radius, tool_radius_m=args.tool_radius), indent=2))


if __name__ == '__main__':
    main()

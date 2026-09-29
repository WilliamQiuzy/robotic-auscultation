"""Calibrated RGB-D interface from actual saved poses, not visual policy training.

Pure projection helpers import on Mac. The capture CLI is Linux/Isaac only and
never advances physics or changes the immutable source episode.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
from .media_teardown import release_owned_render_resources, run_scoped_capture, require_complete_teardown

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CAMERA_CONFIG = PROJECT/'sim/camera_framing_20260923/recommended_camera.json'


def _calibration(intrinsics, world_from_optical):
    k = {key: float(intrinsics[key]) for key in ('fx', 'fy', 'cx', 'cy')}
    width, height = int(intrinsics['width']), int(intrinsics['height'])
    matrix = np.asarray(world_from_optical, float)
    if (not np.isfinite(list(k.values())).all() or min(k['fx'], k['fy'], width, height) <= 0
            or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-12)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-8)):
        raise ValueError('invalid pinhole calibration')
    return k, width, height, matrix


def scale_intrinsics(intrinsics, width, height):
    _calibration(intrinsics, np.eye(4))
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('invalid image dimensions')
    sx, sy = width/intrinsics['width'], height/intrinsics['height']
    return dict(fx=intrinsics['fx']*sx, fy=intrinsics['fy']*sy,
                cx=intrinsics['cx']*sx, cy=intrinsics['cy']*sy, width=width, height=height)


def project_points(points_world, intrinsics, world_from_optical):
    k, _, _, matrix = _calibration(intrinsics, world_from_optical)
    points = np.asarray(points_world, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('world points must be finite N x 3')
    optical = (points-matrix[:3, 3]) @ matrix[:3, :3]
    uv = np.full((len(points), 2), np.nan)
    visible = optical[:, 2] > 0.
    uv[visible, 0] = k['fx']*optical[visible, 0]/optical[visible, 2]+k['cx']
    uv[visible, 1] = k['fy']*optical[visible, 1]/optical[visible, 2]+k['cy']
    return uv, optical[:, 2]


def backproject(pixels, image_plane_depth_m, intrinsics, world_from_optical):
    k, _, _, matrix = _calibration(intrinsics, world_from_optical)
    uv, z = np.asarray(pixels, float), np.asarray(image_plane_depth_m, float)
    if uv.ndim != 2 or uv.shape[1] != 2 or z.shape != (len(uv),) or not np.isfinite(uv).all() or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError('finite positive image-plane depth and N x 2 pixels required')
    optical = np.column_stack([(uv[:, 0]-k['cx'])*z/k['fx'], (uv[:, 1]-k['cy'])*z/k['fy'], z])
    return optical @ matrix[:3, :3].T + matrix[:3, 3]


def target_depth_report(targets, depth, intrinsics, world_from_optical):
    _, width, height, _ = _calibration(intrinsics, world_from_optical)
    depth = np.asarray(depth)
    if depth.shape != (height, width):
        raise ValueError('depth image does not match calibration')
    result = []
    for name, position in targets.items():
        target = np.asarray(position, float)
        uv, z = project_points(target[None], intrinsics, world_from_optical)
        u, v = uv[0]; forward = bool(z[0] > 0)
        inside = bool(forward and 0 <= u < width and 0 <= v < height)
        pixel = np.rint(uv[0]).astype(int) if inside else None
        sampleable = inside and 0 <= pixel[0] < width and 0 <= pixel[1] < height
        measured = float(depth[pixel[1], pixel[0]]) if sampleable else None
        valid = bool(measured is not None and np.isfinite(measured) and 0 < measured < 100.)
        residual = measured-float(z[0]) if valid else None
        point = backproject(pixel[None], [measured], intrinsics, world_from_optical)[0] if valid else None
        exact = backproject(uv, z, intrinsics, world_from_optical)[0] if forward else None
        result.append(dict(name=name, target_world_m=target.tolist(), uv_px=uv[0].tolist() if forward else None,
            optical_depth_m=float(z[0]), inside=inside, sample_pixel_xy=None if pixel is None else pixel.tolist(),
            valid_depth=valid, measured_depth_m=measured if valid else None,
            depth_residual_m=residual, occluded_by_10mm=bool(valid and residual < -.01),
            depth_matches_within_5mm=bool(valid and abs(residual) <= .005),
            backprojected_world_m=point.tolist() if valid else None,
            measured_backprojection_residual_m=float(np.linalg.norm(point-target)) if valid else None,
            mathematical_roundtrip_residual_m=float(np.linalg.norm(exact-target)) if forward else None,
            sampling='nearest integer pixel; metric optical-Z depth; no hole filling or visibility substitution'))
    return result


def select_samples(rows, pose_rows, targets):
    indices = np.asarray(pose_rows)
    if (indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer)
            or indices[0] < 0 or indices[-1] >= len(rows) or np.any(np.diff(indices) <= 0)):
        raise ValueError('invalid saved-pose row indices')
    samples = []
    def add(kind, target, predicate, middle=False):
        options = [(k, int(r)) for k, r in enumerate(indices) if predicate(rows[int(r)])]
        if not options:
            raise ValueError('missing saved '+('HOLD' if kind == 'hold' else kind)+' pose for '+str(target))
        k, r = options[len(options)//2] if middle else options[-1]
        samples.append(dict(kind=kind, target=target, physics_row=r, pose_index=k,
                            simulation_time_s=float(rows[r]['time_s']), phase=rows[r]['phase'], subphase=rows[r]['subphase']))
    add('initial_home', targets[0], lambda r: r['target'] == targets[0] and r['subphase'] == 'settle')
    for target in targets:
        add('prepose', target, lambda r: r['target'] == target and r['subphase'] == 'align')
        add('hold', target, lambda r: r['target'] == target and r['phase'] == 'HOLD', middle=True)
        add('release', target, lambda r: r['target'] == target and r['subphase'] == 'normal_release_confirm')
    add('final_home', targets[-1], lambda r: r['target'] == targets[-1] and r['subphase'] == 'home_settle')
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not sys.platform.startswith('linux'):
        raise RuntimeError('RGB-D capture runs only on the authorized Linux Isaac host')
    import isaacsim
    from .run import graceful_simulation_shutdown
    from .contract import sha
    from sim.expert_quality_20260924.quality import verify_hashes
    saved_argv = sys.argv
    try:
        with graceful_simulation_shutdown(isaacsim) as lifecycle:
            output, manifest, provenance, hashes = _capture(args)
    finally:
        sys.argv = saved_argv
    if lifecycle['apps_created'] != 1:
        raise RuntimeError('capture did not return through exactly one initialized SimulationApp')
    require_complete_teardown(manifest.get('resource_teardown'), annotators=2, products=1)
    if any(sha(PROJECT/path) != digest for path, digest in hashes.items()):
        raise RuntimeError('capture source changed')
    if verify_hashes(args.episode.resolve(), provenance['artifact_sha256']):
        raise RuntimeError('bound episode changed during capture')
    manifest.update(source_files_unchanged=True, sample_count=len(manifest['samples']),
        capture_lifecycle=dict(apps_created=1, close_returned=True,
            completion_stage='post_app_close_pre_process_exit',
            exit_policy='controlled_os_exit_after_close'))
    with (output/'rgbd_manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    print('RGBD_INTERFACE_CAPTURE_COMPLETE', flush=True)
    return 0


def _capture(args):
    """Collect arrays and return only after the owned SimulationApp closes."""
    from .quality import validate_episode_provenance
    from .contract import sha
    from .provenance import implementation_sources
    episode, output = args.episode.resolve(), args.output.resolve()
    if output == episode or episode in output.parents:
        raise ValueError('capture output must be outside the immutable episode')
    provenance = validate_episode_provenance(episode)
    if provenance['status'] != 'passed':
        raise ValueError('episode provenance rejected: '+str(provenance.get('reasons')))
    outcome = json.loads((episode/'outcome.json').read_text())
    source = episode/'training_scene.usdc'
    if sha(source) != outcome.get('training_scene_sha256'):
        raise ValueError('scene differs from executed episode')
    camera = json.loads(CAMERA_CONFIG.read_text())
    if camera['downward_pitch_delta_deg'] != 7. or camera['camera_height_delta_m'] != .05:
        raise ValueError('approved camera selection changed')
    intrinsics = camera['intrinsics']; optical = np.asarray(camera['world_from_optical'])
    _calibration(intrinsics, optical)
    np.testing.assert_allclose(np.asarray(camera['world_from_camera_usd']) @ np.diag([1., -1., -1., 1.]), optical, atol=1e-12)
    width, height = int(intrinsics['width']), int(intrinsics['height'])
    if (width, height) != (1920, 1080):
        raise ValueError('capture uses the approved 1920 x 1080 calibration')
    source_paths = set(implementation_sources().values()) | {source, CAMERA_CONFIG, episode/'outcome.json',
        episode/'expert_trace.jsonl', episode/'expert_replay.npz'}
    hashes = {str(p.relative_to(PROJECT)): sha(p) for p in source_paths}
    rows = []
    with (episode/'expert_trace.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.append({k: row[k] for k in ('target', 'phase', 'subphase', 'time_s')})
    with np.load(episode/'expert_replay.npz', allow_pickle=False) as record:
        poses, indices, paths = (record[k].copy() for k in ('poses', 'row_indices', 'paths'))
    if poses.shape != (len(indices), len(paths), 7) or not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[:, :, 3:], axis=2), 1., atol=1e-4):
        raise ValueError('invalid recorded rigid-body poses')
    samples = select_samples(rows, indices, outcome['requested_points'])
    targets = {p['target']: p['world_target_m'] for p in outcome['points']}
    output.mkdir(parents=True, exist_ok=False)
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=width, height=height, renderer='RayTracedLighting', multi_gpu=False))
    manifest, teardown = run_scoped_capture(app, _capture_scope, episode, output, source,
        provenance, hashes, camera, intrinsics, optical, width, height, poses, paths, samples, targets)
    manifest['resource_teardown'] = teardown
    return output, manifest, provenance, hashes


def _capture_scope(app, errors, released, episode, output, source, provenance, hashes,
                   camera, intrinsics, optical, width, height, poses, paths, samples, targets):
    """All USD, stage, sensor and product locals end before outer app cleanup."""
    from .contract import sha
    rep = None
    owned_annotators, owned_products = [], []
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from PIL import Image
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
        for path in ('/Render', '/Replicator', '/Orchestrator'):
            if stage.GetPrimAtPath(path): stage.RemovePrim(path)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint): UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        stage.GetRootLayer().Export(str(output/'disposable_capture.usdc'))
        omni.usd.get_context().open_stage(str(output/'disposable_capture.usdc'))
        timeline = omni.timeline.get_timeline_interface(); timeline.stop()
        for _ in range(35): app.update()
        stage = omni.usd.get_context().get_stage(); cache = UsdGeom.XformCache()
        transforms = []; dynamic = set(map(str, paths))
        for path in paths:
            prim = stage.GetPrimAtPath(str(path))
            if not prim: raise ValueError('recorded body absent from stage')
            parent = prim.GetParent()
            while parent and not parent.IsPseudoRoot():
                if str(parent.GetPath()) in dynamic: raise ValueError('recorded bodies have dynamic parent')
                parent = parent.GetParent()
            inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
            transforms.append((xf.AddTransformOp(opSuffix='capturedPose'), inv))
        for path in (camera['camera_path'], '/World/RobotCamBody'):
            prim = stage.GetPrimAtPath(path)
            if not prim: raise ValueError('approved camera prim missing')
            inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='approved7deg').Set(Gf.Matrix4d(np.asarray(camera['world_from_camera_usd']).T)*inv)
        sensor = UsdGeom.Camera(stage.GetPrimAtPath(camera['camera_path']))
        def actual_intrinsics():
            f, ha, va, ho, vo = [float(attr.Get()) for attr in (sensor.GetFocalLengthAttr(), sensor.GetHorizontalApertureAttr(),
                sensor.GetVerticalApertureAttr(), sensor.GetHorizontalApertureOffsetAttr(), sensor.GetVerticalApertureOffsetAttr())]
            return dict(fx=f/ha*width, fy=f/va*height, cx=width/2-ho/ha*width, cy=height/2+vo/va*height, width=width, height=height)
        product = rep.create.render_product(camera['camera_path'], (width, height))
        owned_products.append(product)
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        owned_annotators.append(rgb)
        depth_ann = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
        owned_annotators.append(depth_ann)
        rgb.attach([product]); depth_ann.attach([product]); rep.orchestrator.set_capture_on_play(False)
        for _ in range(20): app.update()
        actual = actual_intrinsics()
        for key in intrinsics:
            if not np.isclose(actual[key], intrinsics[key], atol=1e-6, rtol=1e-8):
                raise ValueError('render optics differ from calibration: '+key)
        manifest = dict(schema='auscultation_saved_pose_rgbd_v1', episode=str(episode.relative_to(PROJECT)),
            source_sha256=hashes, episode_provenance=provenance, camera_config=str(CAMERA_CONFIG.relative_to(PROJECT)),
            intrinsics=actual, world_from_optical=optical.tolist(), world_from_camera_usd=camera['world_from_camera_usd'],
            depth_semantics='distance_to_image_plane: metric optical-Z, not radial range',
            coordinate_convention='optical +X right, +Y down, +Z forward; USD camera conversion diag(1,-1,-1,1)',
            scope='Calibrated sensor/goal-feature interface from actual saved physics poses. No live policy input and no RGB-D policy training.',
            raw_rgb_has_overlays=False, physics_advanced=False, samples=[])
        for capture_id, sample in enumerate(samples):
            for (op, inv), pose in zip(transforms, poses[sample['pose_index']]):
                matrix = Gf.Matrix4d(1.).SetRotate(Gf.Quatd(float(pose[3]), Gf.Vec3d(*map(float, pose[4:]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
                op.Set(matrix*inv)
            rep.orchestrator.step(rt_subframes=6, delta_time=0., pause_timeline=True)
            if timeline.is_playing() or abs(timeline.get_current_time()) > 1e-12:
                raise RuntimeError('physics advanced during sensor capture')
            pixels = np.asarray(rgb.get_data())[:, :, :3].copy()
            depth = np.asarray(depth_ann.get_data(), dtype=np.float32).copy()
            if pixels.shape != (height, width, 3) or pixels.std() < 5 or depth.shape != (height, width):
                raise RuntimeError('malformed synchronized RGB-D capture')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 100.)
            prefix = f'{capture_id:03d}_{sample["target"]}_{sample["kind"]}'
            files = dict(rgb=prefix+'_rgb.png', depth_m=prefix+'_depth_m.npy', valid_mask=prefix+'_valid_mask.npy')
            Image.fromarray(pixels.astype(np.uint8)).save(output/files['rgb'])
            np.save(output/files['depth_m'], depth, allow_pickle=False)
            np.save(output/files['valid_mask'], valid, allow_pickle=False)
            projections = target_depth_report(targets, depth, actual, optical)
            manifest['samples'].append(dict(sample, capture_id=capture_id,
                rgb_source_time_s=sample['simulation_time_s'], depth_source_time_s=sample['simulation_time_s'],
                capture_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                synchronization='both annotators on one render product after one completed orchestrator capture at a fixed recorded pose',
                files=files, file_sha256={name: sha(output/path) for name, path in files.items()},
                valid_depth_fraction=float(valid.mean()), target_projection=projections,
                all_targets_visible=all(p['depth_matches_within_5mm'] for p in projections)))
            print('RGBD_CAPTURE', capture_id+1, '/', len(samples), sample['target'], sample['kind'], flush=True)
    finally:
        released.update(release_owned_render_resources(rep, owned_annotators, owned_products, errors))
    return manifest


if __name__ == '__main__':
    from .runtime import exit_after_cleanup
    exit_after_cleanup(main())

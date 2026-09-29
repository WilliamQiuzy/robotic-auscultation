"""Reversible seated-layout camera comparison. Render only; no robot control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
MODULE = PROJECT / 'sim/auscultation_training'
EPISODE = MODULE / 'results/low_elbow_release_v1/physics_all10'
SOURCE = EPISODE / 'training_scene.usdc'
PLAN_DIR = MODULE / 'results/low_elbow_release_v1/plans'
VARIANTS = [('baseline', 0., 0.), ('farther_5cm', .05, 0.),
            ('lower_2cm', 0., -.02), ('farther_5cm_lower_2cm', .05, -.02),
            ('farther_10cm_lower_2cm', .10, -.02)]
POINTS = [side + str(level) for level in range(1, 6) for side in ('R', 'L')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'camera_comparison',
                        help='New output directory; existing directories are never overwritten.')
    parser.add_argument('--variants', nargs='+', choices=[v[0] for v in VARIANTS],
                        default=[v[0] for v in VARIANTS], help='One or more layout names; default: all five.')
    args = parser.parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        parser.error('--variants must not contain duplicate names')
    args.output = args.output.expanduser().resolve()
    return args


def annotation_font(image_font, size=28):
    """Use a readable installed font or Pillow's bundled default without apt."""
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                 'DejaVuSans.ttf'):
        try:
            return image_font.truetype(path, size), path
        except OSError:
            pass
    try:
        return image_font.load_default(size=size), 'Pillow bundled default'
    except TypeError:  # Pillow before load_default accepted a size argument.
        return image_font.load_default(), 'Pillow bundled bitmap default'


def patient_ground_state(stage):
    """Read authored visible control vertices, independently of render bounds."""
    from pxr import Usd, UsdGeom
    floor = stage.GetPrimAtPath('/World/GroundPlane/geom')
    if not floor:
        raise ValueError('Cannot validate patient feet without the existing ground mesh')
    cache = UsdGeom.XformCache()
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(floor).ComputeAlignedRange()
    floor_low, floor_high = float(r.GetMin()[2]), float(r.GetMax()[2])
    if not np.isfinite([floor_low, floor_high]).all() or abs(floor_high-floor_low) > 1e-6:
        raise ValueError('Patient ground check requires the existing horizontal ground plane')
    rows = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Patient')):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if str(imageable.ComputeVisibility()) == 'invisible' or str(imageable.ComputePurpose()) not in ('default', 'render'):
            continue
        vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError(f'Invalid patient geometry at {prim.GetPath()}')
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        minimum = float(world[:, 2].min())
        name = prim.GetName().lower()
        rows.append({'path': str(prim.GetPath()), 'minimum_z_m': minimum,
                     'ground_clearance_m': minimum-floor_high,
                     'penetration_m': max(0., floor_high-minimum),
                     'is_shoe_or_foot': any(token in name for token in ('sneaker', 'shoe', 'foot'))})
    feet = [row for row in rows if row['is_shoe_or_foot']]
    if not rows or not feet:
        raise ValueError('Cannot identify visible patient shoe/foot geometry for ground check')
    return {'ground_z_m': floor_high, 'patient_meshes': rows,
            'minimum_patient_clearance_m': min(row['ground_clearance_m'] for row in rows),
            'minimum_shoe_clearance_m': min(row['ground_clearance_m'] for row in feet),
            'scope': 'Authored visible control vertices versus flat ground; no physics or foot support validation.'}


def verify_no_new_patient_penetration(reference, current, tolerance_m=1e-6):
    if abs(reference['ground_z_m']-current['ground_z_m']) > tolerance_m:
        raise ValueError('Ground plane changed while checking the patient layout')
    old = {row['path']: row for row in reference['patient_meshes']}
    new = {row['path']: row for row in current['patient_meshes']}
    if old.keys() != new.keys():
        raise ValueError('Visible patient mesh set changed during layout comparison')
    for path, row in new.items():
        if row['penetration_m'] > old[path]['penetration_m']+tolerance_m:
            raise ValueError(f'Layout introduces/increases patient ground penetration: {path}')
    return True


def move_world(stage, path, delta):
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(path)
    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(prim)
    world.SetTranslateOnly(world.ExtractTranslation() + Gf.Vec3d(*map(float, delta)))
    local = world * cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp(opSuffix='layoutReview').Set(local)


def move_local_z(prim, dz):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    matrix = xf.GetLocalTransformation()
    v = matrix.ExtractTranslation()
    matrix.SetTranslateOnly(Gf.Vec3d(v[0], v[1], v[2] + dz))
    xf.ClearXformOpOrder()
    xf.AddTransformOp(opSuffix='layoutReview').Set(matrix)


def refresh_stool_extents(stage):
    """Repair stale cylinder extent metadata in the current disposable layer.

    Radius, height and transforms remain the geometry authority. The source
    scene has four old leg extents left over from its earlier stool height.
    """
    from pxr import Gf, UsdGeom
    repairs = []
    for prim in stage.GetPrimAtPath('/World/Stool').GetChildren():
        if not prim.IsA(UsdGeom.Cylinder):
            continue
        cylinder = UsdGeom.Cylinder(prim)
        radius, height = float(cylinder.GetRadiusAttr().Get()), float(cylinder.GetHeightAttr().Get())
        axis = str(cylinder.GetAxisAttr().Get())
        if radius <= 0 or height <= 0 or axis not in ('X', 'Y', 'Z'):
            raise ValueError(f'Invalid stool cylinder geometry: {prim.GetPath()}')
        half = np.full(3, radius)
        half[('X', 'Y', 'Z').index(axis)] = height / 2
        expected = np.asarray([-half, half])
        previous = cylinder.GetExtentAttr().Get()
        old = None if previous is None else np.asarray(previous, dtype=float)
        if old is None or old.shape != (2, 3) or not np.allclose(old, expected, atol=1e-7, rtol=0):
            repairs.append({'path': str(prim.GetPath()), 'previous_extent': None if old is None else old.tolist(),
                            'recomputed_extent': expected.tolist(), 'geometry_parameters_unchanged': True})
        cylinder.CreateExtentAttr().Set([Gf.Vec3f(*map(float, expected[0])), Gf.Vec3f(*map(float, expected[1]))])
    return repairs


def apply_layout(stage, farther, lower, seat_top):
    from pxr import UsdGeom
    move_world(stage, '/World/Patient', [0., farther, lower])
    move_world(stage, '/World/Stool', [0., farther, 0.])
    move_local_z(stage.GetPrimAtPath('/World/Stool/seat'), lower)
    for child in stage.GetPrimAtPath('/World/Stool').GetChildren():
        if child.GetName().startswith('leg'):
            cylinder = UsdGeom.Cylinder(child)
            assert cylinder and str(cylinder.GetAxisAttr().Get()) == 'Z'
            height = float(cylinder.GetHeightAttr().Get()) + lower
            assert height > 0
            cylinder.GetHeightAttr().Set(height)
            move_local_z(child, lower / 2)
        elif child.GetName().startswith('rung'):
            old_z = float(UsdGeom.Xformable(child).GetLocalTransformation().ExtractTranslation()[2])
            move_local_z(child, old_z * lower / seat_top)
    refresh_stool_extents(stage)


def main(argv=None):
    args = parse_args(argv)
    output = args.output
    selected = [next(v for v in VARIANTS if v[0] == name) for name in args.variants]
    inputs = [Path(__file__).resolve(), SOURCE, EPISODE / 'expert_replay.npz', *[PLAN_DIR / f'planned_paths_{p}.json' for p in POINTS]]
    captured = {str(p.relative_to(PROJECT)): sha(p) for p in inputs}
    output.mkdir(parents=True, exist_ok=False)
    targets = {p: np.asarray(json.loads((PLAN_DIR / f'planned_paths_{p}.json').read_text())['points'][0]['position_world']) for p in POINTS}
    with np.load(EPISODE / 'expert_replay.npz', allow_pickle=False) as record:
        body_paths = list(map(str, record['paths']))
        home_poses = record['poses'][0].copy()
        first_row = int(record['row_indices'][0])
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    # Kit receives its own launcher config, not our --output/--variants options.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 1920, 'height': 1080,
                         'renderer': 'RayTracedLighting', 'multi_gpu': False})
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
        from PIL import Image, ImageDraw, ImageFont
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep

        baseline = Usd.Stage.Open(str(SOURCE))
        assert UsdGeom.GetStageMetersPerUnit(baseline) == 1.
        assert str(UsdGeom.GetStageUpAxis(baseline)) == 'Z'
        baseline.SetEditTarget(baseline.GetSessionLayer())
        stool_extent_repairs = refresh_stool_extents(baseline)

        def bounds(stage, path):
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
            r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            return np.asarray([r.GetMin(), r.GetMax()], dtype=float)

        def camera_state(stage):
            camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/RobotCam'))
            matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim()), dtype=float).T
            optical = matrix @ np.diag([1., -1., -1., 1.])
            f, ha, va = [float(attr.Get()) for attr in (camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(), camera.GetVerticalApertureAttr())]
            ho, vo = float(camera.GetHorizontalApertureOffsetAttr().Get()), float(camera.GetVerticalApertureOffsetAttr().Get())
            return {'world_from_optical': optical.tolist(), 'focal_length': f,
                    'horizontal_aperture': ha, 'vertical_aperture': va,
                    'horizontal_aperture_offset': ho, 'vertical_aperture_offset': vo,
                    'clipping_range': list(camera.GetClippingRangeAttr().Get()),
                    'fx': f/ha*1920, 'fy': f/va*1080,
                    'cx': 960-ho/ha*1920, 'cy': 540+vo/va*1080,
                    'width': 1920, 'height': 1080}

        camera_reference = camera_state(baseline)
        initial_bounds = {name: bounds(baseline, '/World/' + name).tolist() for name in ('Patient', 'Stool', 'Table')}
        initial_seat = bounds(baseline, '/World/Stool/seat')
        seat_top = float(initial_seat[1, 2])
        patient_ground_reference = patient_ground_state(baseline)
        ground_reference = {p.GetName(): bounds(baseline, str(p.GetPath()))[0, 2]
                            for p in baseline.GetPrimAtPath('/World/Stool').GetChildren()
                            if p.GetName().startswith('leg')}
        base_path = '/World/StethoArmMount/OpenArmStetho/openarm_left_link0'
        base_reference = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(baseline.GetPrimAtPath(base_path)))
        manifest = {'schema': 'seated_layout_camera_comparison_v1',
                    'scope': 'Static layout and camera preview only; no new motion/contact validation or detector inference.',
                    'input_sha256': captured, 'camera': camera_reference,
                    'baseline_bounds_world_m': initial_bounds,
                    'baseline_seat_top_m': seat_top,
                    'baseline_patient_ground_state': patient_ground_reference,
                    'stool_extent_metadata_repairs': stool_extent_repairs,
                    'selected_variants': args.variants,
                    'arm_pose': 'Same first recorded neutral pose in every layout; not a newly planned motion.',
                    'recorded_pose_measurement_row': first_row,
                    'variants': {}, 'source_files_unchanged': False}
        font, font_source = annotation_font(ImageFont)
        manifest['annotation_font'] = font_source
        rep.orchestrator.set_capture_on_play(False)

        for name, farther, lower in selected:
            delta = np.array([0., farther, lower])
            scene_path = output / (name + '.usda')
            layer = Sdf.Layer.CreateNew(str(scene_path))
            layer.subLayerPaths = [os.path.relpath(SOURCE, scene_path.parent)]
            layer.Save()
            stage = Usd.Stage.Open(str(scene_path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, 'Z')
            apply_layout(stage, farther, lower, seat_top)
            assert camera_state(stage) == camera_reference
            np.testing.assert_allclose(bounds(stage, '/World/Patient'), np.asarray(initial_bounds['Patient']) + delta, atol=1e-6, rtol=0)
            np.testing.assert_allclose(bounds(stage, '/World/Table'), initial_bounds['Table'], atol=1e-9, rtol=0)
            np.testing.assert_allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(base_path))), base_reference, atol=1e-12, rtol=0)
            np.testing.assert_allclose(bounds(stage, '/World/Stool/seat'), initial_seat + delta, atol=1e-6, rtol=0)
            for leg, old_ground in ground_reference.items():
                assert abs(bounds(stage, '/World/Stool/' + leg)[0, 2] - old_ground) < 1e-6
            patient_ground = patient_ground_state(stage)
            verify_no_new_patient_penetration(patient_ground_reference, patient_ground)
            stage.GetRootLayer().Save()
            candidate_bounds = {n: bounds(stage, '/World/' + n).tolist() for n in ('Patient', 'Stool', 'Table')}
            render_stage = Usd.Stage.Open(stage.Flatten())
            for path in ('/Render', '/Replicator', '/Orchestrator'):
                if render_stage.GetPrimAtPath(path):
                    render_stage.RemovePrim(path)
            for prim in render_stage.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                if prim.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            cache = UsdGeom.XformCache()
            for path, pose in zip(body_paths, home_poses):
                prim = render_stage.GetPrimAtPath(path)
                assert prim, path
                inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
                matrix = Gf.Matrix4d(1.)
                matrix.SetRotate(Gf.Quatd(float(pose[3]), Gf.Vec3d(*map(float, pose[4:]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTransformOp(opSuffix='recordedNeutral').Set(matrix * inv)
            side = UsdGeom.Camera.Define(render_stage, '/World/LayoutSideCamera')
            side.CreateFocalLengthAttr(22.)
            side.CreateHorizontalApertureAttr(20.955)
            side.CreateVerticalApertureAttr(20.955 * 1080/1920)
            side.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
            UsdGeom.Xformable(side).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.6,.6,1.9), Gf.Vec3d(0.,.6,.95), Gf.Vec3d(0.,0.,1.)).GetInverse())
            render_path = output / (name + '_render.usdc')
            render_stage.GetRootLayer().Export(str(render_path))
            omni.usd.get_context().open_stage(str(render_path))
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            for _ in range(40):
                app.update()
            live = omni.usd.get_context().get_stage()
            assert camera_state(live) == camera_reference
            images = {}
            depth = None
            for label, cam_path in [('robotcam', '/World/RobotCam'), ('side', '/World/LayoutSideCamera')]:
                product = rep.create.render_product(cam_path, (1920, 1080))
                rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                rgb.attach([product])
                z_ann = None
                if label == 'robotcam':
                    z_ann = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                    z_ann.attach([product])
                for _ in range(15):
                    app.update()
                rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=True)
                assert not timeline.is_playing()
                pixels = np.asarray(rgb.get_data())[:, :, :3].copy()
                assert pixels.shape == (1080, 1920, 3)
                image_path = output / f'{name}_{label}.png'
                Image.fromarray(pixels).save(image_path)
                images[label] = image_path.name
                if z_ann is not None:
                    depth = np.asarray(z_ann.get_data(), dtype=np.float32).copy()
                    np.save(output / f'{name}_depth_m.npy', depth)
                    z_ann.detach([product.path])
                rgb.detach([product.path])
                product.destroy()
            image_path = output / images['robotcam']
            annotated = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(annotated)
            camera_inv = np.linalg.inv(np.asarray(camera_reference['world_from_optical']))
            projections = []
            for point, before in targets.items():
                world = before + delta
                optical = (camera_inv @ np.r_[world, 1.])[:3]
                u = camera_reference['fx'] * optical[0]/optical[2] + camera_reference['cx']
                v = camera_reference['fy'] * optical[1]/optical[2] + camera_reference['cy']
                inside = bool(optical[2] > 0 and 0 <= u < 1920 and 0 <= v < 1080)
                observed = float(depth[int(v), int(u)]) if inside else None
                if observed is not None and not np.isfinite(observed):
                    observed = None
                projections.append({'name': point, 'world_position_m': world.tolist(),
                                    'optical_xyz_m': optical.tolist(), 'uv_px': [float(u), float(v)],
                                    'inside_frame': inside, 'render_depth_m': observed,
                                    'depth_residual_m': None if observed is None else observed-float(optical[2])})
                if inside:
                    color = (0, 228, 242) if point.startswith('R') else (255, 188, 70)
                    draw.ellipse((u-9,v-9,u+9,v+9), fill=color, outline=(0,0,0), width=2)
                    draw.text((u+13,v-17), point, fill=color, font=font, stroke_width=2, stroke_fill=(0,0,0))
            annotated.save(output / f'{name}_robotcam_targets.png')
            manifest['variants'][name] = {'patient_translation_world_m': delta.tolist(),
                'stool_translation_world_m': [0., farther, 0.], 'seat_height_change_m': lower,
                'seat_top_m': seat_top+lower, 'leg_ground_z_unchanged': True,
                'patient_ground_state': patient_ground, 'no_new_patient_ground_penetration': True,
                'camera_pose_optics_unchanged': True, 'robot_base_unchanged': True,
                'patient_scale_unchanged': True, 'bounds_world_m': candidate_bounds,
                'scene': scene_path.name, 'scene_sha256': sha(scene_path), 'images': images,
                'projected_existing_points': projections,
                'point_source': 'Existing accepted-for-simulation points translated with body; no new detector run.'}
            print('LAYOUT_RENDERED', name, sum(p['inside_frame'] for p in projections), flush=True)
        assert all(sha(PROJECT / key) == value for key, value in captured.items())
        manifest['source_files_unchanged'] = True
        manifest['output_sha256'] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        (output / 'layout_comparison.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('LAYOUT_COMPARISON_COMPLETE', flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()

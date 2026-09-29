"""Render camera-only height and pitch variants of the accepted layout on Linux Isaac.

Preserves the accepted episode and all physical geometry. Physical overlay
layers contain only camera-transform edits; disposable replay stage disables
physics. Raw RGB/depth have no presentation overlays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EPISODE = PROJECT/'sim/auscultation_expert_layout_v1/results/expert_v2/physics_all10'
SOURCE = EPISODE/'training_scene.usdc'
CAMERA = '/World/RobotCam'
BODY = '/World/RobotCamBody'
NAMES = [s+str(i) for i in range(1, 6) for s in ('R', 'L')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def down_pitch(world_from_usd, degrees):
    old = np.asarray(world_from_usd, dtype=float)
    new = old.copy()
    angle = np.deg2rad(degrees)
    forward, up = -old[:3, 2], old[:3, 1]
    new[:3, 1] = np.sin(angle)*forward + np.cos(angle)*up
    new[:3, 2] = -(np.cos(angle)*forward - np.sin(angle)*up)
    np.testing.assert_allclose(new[:3, :3].T@new[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(new[:3, 3], old[:3, 3], atol=1e-12)
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['12:5'], help='additional downward pitch degrees:camera height delta cm')
    args = parser.parse_args()
    variants = [tuple(map(float, v.split(':'))) for v in args.variants]
    assert len(set(variants)) == len(variants) and all(0 <= d <= 25 and 0 <= h <= 15 for d,h in variants)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    outcome = json.loads((EPISODE/'outcome.json').read_text())
    assert outcome['success'] and outcome['training_accepted'] and outcome['requested_points'] == NAMES
    assert sha(SOURCE) == outcome['training_scene_sha256']
    assert json.loads((EPISODE/'runner_summary.json').read_text())['accepted']
    source_files = [Path(__file__), SOURCE, EPISODE/'expert_replay.npz', EPISODE/'outcome.json', EPISODE/'runner_summary.json']
    source_hashes = {str(p.relative_to(PROJECT)): sha(p) for p in source_files}
    targets = {p['target']: np.asarray(p['world_target_m']) for p in outcome['points']}
    with np.load(EPISODE/'expert_replay.npz', allow_pickle=False) as r:
        paths, home = r['paths'].copy(), r['poses'][0].copy()
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
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
        source = Usd.Stage.Open(str(SOURCE))
        cache = UsdGeom.XformCache()
        camera = UsdGeom.Camera(source.GetPrimAtPath(CAMERA))
        original = np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T
        np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(source.GetPrimAtPath(BODY))).T, original, atol=1e-12)
        optics = {a.GetName(): repr(a.Get()) for a in camera.GetPrim().GetAttributes()
                  if not a.GetName().startswith('xform')}
        f, ha, va, ho, vo = [float(a.Get()) for a in (camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(),
            camera.GetVerticalApertureAttr(), camera.GetHorizontalApertureOffsetAttr(), camera.GetVerticalApertureOffsetAttr())]
        intrinsics = {'fx': f/ha*1920, 'fy': f/va*1080, 'cx': 960-ho/ha*1920, 'cy': 540+vo/va*1080,
                      'width': 1920, 'height': 1080}
        stage = Usd.Stage.Open(source.Flatten())
        for p in ('/Render', '/Replicator', '/Orchestrator'):
            if stage.GetPrimAtPath(p): stage.RemovePrim(p)
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint): UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        cache = UsdGeom.XformCache()
        for path, pose in zip(paths, home):
            prim = stage.GetPrimAtPath(str(path)); assert prim
            parent_inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            matrix = Gf.Matrix4d(1.).SetRotate(Gf.Quatd(float(pose[3]), Gf.Vec3d(*map(float, pose[4:]))))
            matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
            xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder(); xf.AddTransformOp(opSuffix='recordedHome').Set(matrix*parent_inv)
        render_path = out/'disposable_render.usdc'; stage.GetRootLayer().Export(str(render_path))
        omni.usd.get_context().open_stage(str(render_path))
        timeline = omni.timeline.get_timeline_interface(); timeline.stop()
        for _ in range(45): app.update()
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(CAMERA)
        parent_inv = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder(); op = xf.AddTransformOp(opSuffix='pitchReview')
        body_prim = stage.GetPrimAtPath(BODY)
        body_parent_inv = UsdGeom.XformCache().GetLocalToWorldTransform(body_prim.GetParent()).GetInverse()
        body_xf = UsdGeom.Xformable(body_prim); body_xf.ClearXformOpOrder(); body_op = body_xf.AddTransformOp(opSuffix='cameraPoseReview')
        product = rep.create.render_product(CAMERA, (1920, 1080))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); depth_ann = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
        rgb.attach([product]); depth_ann.attach([product]); rep.orchestrator.set_capture_on_play(False)
        for _ in range(20): app.update()
        try: font = ImageFont.truetype('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf', 26)
        except OSError: font = ImageFont.load_default()
        manifest = {'scope': 'Camera-only static RGB-D comparison; unchanged accepted patient/stool/arm geometry. No new policy/physics execution.',
                    'source_sha256': source_hashes, 'intrinsics': intrinsics, 'original_world_from_camera_usd': original.tolist(),
                    'raw_images_have_no_overlays': True, 'variants': {}}
        for degrees, height_cm in variants:
            name = 'up_'+format(height_cm, 'g').replace('.', 'p')+'cm_down_'+format(degrees, 'g').replace('.', 'p')+'deg'
            transform = down_pitch(original, degrees)
            transform[2, 3] += height_cm / 100.
            assert -transform[2, 2] <= -original[2, 2]+1e-12
            layer_path = out/(name+'.usda')
            layer = Sdf.Layer.CreateNew(str(layer_path)); layer.subLayerPaths = [os.path.relpath(SOURCE, layer_path.parent)]; layer.Save()
            physical = Usd.Stage.Open(str(layer_path))
            physical_xf = UsdGeom.Xformable(physical.GetPrimAtPath(CAMERA)); physical_xf.ClearXformOpOrder()
            physical_xf.AddTransformOp(opSuffix='downwardPitch').Set(Gf.Matrix4d(transform.T)*parent_inv)
            physical_body = UsdGeom.Xformable(physical.GetPrimAtPath(BODY)); physical_body.ClearXformOpOrder()
            physical_body.AddTransformOp(opSuffix='cameraPoseReview').Set(Gf.Matrix4d(transform.T)*body_parent_inv)
            physical.GetRootLayer().Save()
            assert {a.GetName():repr(a.Get()) for a in physical.GetPrimAtPath(CAMERA).GetAttributes() if not a.GetName().startswith('xform')} == optics
            def check_authored(spec):
                if str(spec.path) not in (CAMERA, BODY):
                    assert not list(spec.properties), str(spec.path)
                for child in spec.nameChildren:
                    check_authored(child)
            for spec in physical.GetRootLayer().rootPrims:
                check_authored(spec)
            op.Set(Gf.Matrix4d(transform.T)*parent_inv)
            body_op.Set(Gf.Matrix4d(transform.T)*body_parent_inv)
            for _ in range(5): app.update()
            rep.orchestrator.step(rt_subframes=6, delta_time=0., pause_timeline=True)
            assert not timeline.is_playing()
            pixels = np.asarray(rgb.get_data())[:, :, :3].copy(); depth = np.asarray(depth_ann.get_data(), dtype=np.float32).copy()
            assert pixels.shape == (1080,1920,3) and pixels.std()>5 and depth.shape == (1080,1920)
            Image.fromarray(pixels).save(out/(name+'_rgb.png')); np.save(out/(name+'_depth_m.npy'),depth)
            annotated = Image.fromarray(pixels); draw = ImageDraw.Draw(annotated)
            optical = transform@np.diag([1., -1., -1., 1.]); inv = np.linalg.inv(optical); projected=[]
            for point, world in targets.items():
                xyz = (inv@np.r_[world,1.])[:3]
                u,v = intrinsics['fx']*xyz[0]/xyz[2]+intrinsics['cx'], intrinsics['fy']*xyz[1]/xyz[2]+intrinsics['cy']
                inside = bool(xyz[2]>0 and 0<=u<1920 and 0<=v<1080)
                z = float(depth[int(v),int(u)]) if inside else None
                if z is not None and not np.isfinite(z): z=None
                projected.append({'name':point,'uv_px':[float(u),float(v)],'inside':inside,'optical_xyz_m':xyz.tolist(),
                    'border_margin_px':float(min(u,1919-u,v,1079-v)), 'depth_m':z,'depth_residual_m':None if z is None else z-float(xyz[2])})
                if inside:
                    color=(0,235,240) if point.startswith('R') else (255,170,40)
                    draw.ellipse((u-8,v-8,u+8,v+8),fill=color,outline='black',width=2)
                    draw.text((u+12,v-14),point,fill=color,font=font,stroke_width=2,stroke_fill='black')
            annotated.save(out/(name+'_targets.png'))
            manifest['variants'][name]={'downward_pitch_delta_deg':degrees,'world_from_camera_usd':transform.tolist(),
                'world_from_optical':optical.tolist(),'look_elevation_deg':float(np.rad2deg(np.arcsin(-transform[2,2]))),
                'projected_targets':projected,'patient_stool_robot_unchanged':True,'intrinsics_unchanged':True,
                'camera_position_unchanged':height_cm == 0.,'camera_height_delta_m':height_cm/100.,'physical_overlay':layer_path.name,'visual_camera_body_follows_optical_pose':True}
            print('CAMERA_RENDERED',name,'points_inside',sum(p['inside'] for p in projected),flush=True)
        rgb.detach([product.path]); depth_ann.detach([product.path]); product.destroy()
        assert all(sha(PROJECT/p)==h for p,h in source_hashes.items())
        manifest['source_files_unchanged']=True
        manifest['output_sha256']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        (out/'comparison.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        print('CAMERA_COMPARISON_COMPLETE',flush=True)
    finally:
        app.close()


if __name__=='__main__': main()

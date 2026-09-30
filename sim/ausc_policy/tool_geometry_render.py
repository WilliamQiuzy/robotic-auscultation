"""Isaac-only static geometry review; no policy, robot motion or force qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .tool_protrusion import MOVING, FACE_Z_M, apply_to_stage, wrist_contact_offset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Owned Linux rendering only')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=1440, height=1080,
        renderer='RayTracedLighting', anti_aliasing=3))
    try:
        import omni.usd
        import omni.replicator.core as rep
        import omni.timeline
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from sim.auscultation_expert_layout_v1.scene import WRIST
        from .isaac_case import approved_assets
        before_file = args.output / 'before.usdc'
        with approved_assets(Path(config['asset_root'])) as scene:
            _, sources = scene.physical_sources()
            source_hashes = {str(k): digest(v) for k, v in sources.items()}
            stage = scene.prepare_stage(before_file)
        def protected(s):
            result = {}
            for prim in s.Traverse():
                path = str(prim.GetPath())
                if path.startswith(MOVING + '/Visuals/') or path.startswith(MOVING + '/Colliders/') or path.startswith(MOVING + '/ContactTCP'):
                    continue
                result[path] = dict(attrs={a.GetName():str(a.Get()) for a in prim.GetAttributes()},
                                    rels={r.GetName():str(r.GetTargets()) for r in prim.GetRelationships()})
            return result
        protected_before = protected(stage)
        offsets_before = {str(prim.GetPath()): [str(prim.GetAttribute(a).Get()) for a in
            ('physxCollision:contactOffset', 'physxCollision:restOffset', 'physics:collisionEnabled')]
            for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)}
        report = apply_to_stage(stage)
        assert protected(stage) == protected_before, 'Unexpected protected scene change'
        offsets_after = {str(prim.GetPath()): [str(prim.GetAttribute(a).Get()) for a in
            ('physxCollision:contactOffset', 'physxCollision:restOffset', 'physics:collisionEnabled')]
            for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)}
        assert offsets_after == offsets_before, 'Collision enablement or margins changed'
        cache = UsdGeom.XformCache()
        body = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MOVING))
        def relative(prim):
            return cache.GetLocalToWorldTransform(prim) * body.GetInverse()
        diaphragm = UsdGeom.Cylinder(stage.GetPrimAtPath(MOVING + '/Colliders/Diaphragm'))
        face = relative(diaphragm.GetPrim()).Transform(Gf.Vec3d(0, 0, -diaphragm.GetHeightAttr().Get()/2))
        lip_vertices = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(MOVING + '/Colliders/RetentionLip')):
            if prim.IsA(UsdGeom.Mesh):
                lip_vertices.extend(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()).tolist())
        lip_vertices = np.asarray(lip_vertices)
        actual_protrusion = float(lip_vertices[:, 2].min() - face[2])
        assert abs(actual_protrusion - .002) < 1e-7
        np.testing.assert_allclose(face, [0, 0, FACE_Z_M], atol=1e-7)
        tcp = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MOVING + '/ContactTCP'))
        wrist = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(WRIST))
        wrist_tcp = (tcp * wrist.GetInverse()).ExtractTranslation()
        np.testing.assert_allclose(wrist_tcp, [0, 0, wrist_contact_offset()], atol=1e-6)
        lower_bore_radius = float(np.linalg.norm(lip_vertices[:, :2], axis=1).min())
        assert lower_bore_radius > .0234
        visual = UsdGeom.Mesh(stage.GetPrimAtPath(MOVING + '/Visuals/thinklabs_black_edge'))
        assert abs(float(np.asarray(visual.GetPointsAttr().Get())[:, 2].min()) - FACE_Z_M) < 1e-7
        candidate = args.output / 'candidate.usdc'
        stage.GetRootLayer().Export(str(candidate))
        report.update(measured_usd_protrusion_m=actual_protrusion,
            measured_wrist_tcp_m=list(wrist_tcp), measured_lower_bore_radius_m=lower_bore_radius,
            camera_patient_table_stool_and_physics_unchanged=True,
            original_collision_count=len(offsets_before), source_hashes=source_hashes,
            candidate_sha256=digest(candidate), before_sha256=digest(before_file),
            rendering='Isaac Sim RayTracedLighting, stopped timeline; additional review cameras only')
        (args.output/'geometry_check.json').write_text(json.dumps(report, indent=2)+'\n')
        stage = None
        for version, filename in [('before', before_file), ('after', candidate)]:
            omni.usd.get_context().open_stage(str(filename))
            for _ in range(35): app.update()
            s = omni.usd.get_context().get_stage()
            omni.timeline.get_timeline_interface().stop()
            product = rep.create.render_product('/World/RobotCam', (1440, 810))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            rep.orchestrator.step(rt_subframes=12, pause_timeline=True)
            image = np.asarray(annotator.get_data())[:, :, :3]
            if image.shape != (810, 1440, 3) or image.std() < 3:
                raise RuntimeError('Missing or blank actual Isaac image')
            Image.fromarray(image).save(args.output/(version+'_scene.png'))
            annotator.detach([product]); product.destroy()
            # Inspection visibility only. The saved candidate retains the whole scene.
            for prim in s.GetPrimAtPath('/World').GetChildren():
                if prim.GetName() != 'Stethoscope' and UsdGeom.Boundable(prim):
                    UsdGeom.Imageable(prim).MakeInvisible()
                elif prim.GetName() != 'Stethoscope' and prim.IsA(UsdGeom.Xform):
                    # Do not hide Xform light rigs.
                    if not any('Light' in str(q.GetTypeName()) for q in Usd.PrimRange(prim)):
                        UsdGeom.Imageable(prim).MakeInvisible()
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(s.GetPrimAtPath(MOVING))
            cam = UsdGeom.Camera.Define(s, '/World/ToolReviewCamera')
            cam.CreateClippingRangeAttr(Gf.Vec2f(.001, 100))
            cam.CreateFocalLengthAttr(55)
            cam.CreateHorizontalApertureAttr(24)
            cam.CreateVerticalApertureAttr(18)
            op = cam.AddTransformOp()
            views = [('side', (.35, 0, 0), (0, 0, 0)),
                     ('oblique', (.24, -.30, -.095), (0, 0, 0)),
                     ('face', (.065, -.065, -.28), (0, 0, -.025))]
            for name, eye, target in views:
                eye = transform.Transform(Gf.Vec3d(*eye))
                target = transform.Transform(Gf.Vec3d(*target))
                up = transform.TransformDir(Gf.Vec3d(0, 0, 1))
                op.Set(Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse())
                product = rep.create.render_product(cam.GetPath(), (1440, 1080))
                annotator.attach([product])
                for _ in range(12): app.update()
                rep.orchestrator.step(rt_subframes=16, pause_timeline=True)
                image = np.asarray(annotator.get_data())[:, :, :3]
                if image.shape != (1080, 1440, 3) or image.std() < 3:
                    raise RuntimeError('Invalid inspection render')
                name = version+'_'+name+'.png'
                Image.fromarray(image).save(args.output/name)
                print('RENDERED', name, flush=True)
                annotator.detach([product]); product.destroy()
        assert source_hashes == {str(k): digest(v) for k, v in sources.items()}
        report.update(completed=True, actual_physics_executed=False, source_layers_unchanged=True,
            images={p.name:digest(p) for p in args.output.glob('*.png')})
        (args.output/'COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
        print('TOOL_GEOMETRY_REVIEW_COMPLETE', flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        (args.output/'FAILED.json').write_text(json.dumps(dict(error=type(error).__name__, reason=str(error)), indent=2)+'\n')
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()

"""Owned Linux static review of the explicitly recessed membrane/ring revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from .retainer_geometry import GEOMETRY_VERSION, MOVING, apply_to_stage

SOURCE_SHA256 = '4637a4e7218da2b91ef2688f45f879102a243082d11d709c61a3a953a46d8081'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def build_candidate(config, output):
    """CPU USD construction/export only; completion requires the owned review."""
    source = Path(config['source'])
    if config.get('source_sha256') != SOURCE_SHA256 or digest(source) != SOURCE_SHA256:
        raise ValueError('Exact hash-bound v1 source required')
    from pxr import Usd
    from .approved_camera import apply_approved_camera
    from .geometry_export import export_geometry
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = Usd.Stage.Open(str(source))
    stage = Usd.Stage.Open(original.Flatten())
    report = apply_to_stage(stage)
    camera = apply_approved_camera(stage)
    write(output/'approved_camera_restoration.json', camera)
    candidate = output/'candidate.usdc'
    stage.GetRootLayer().Export(str(candidate))
    entries = export_geometry(stage, output/'geometry')
    if digest(source) != SOURCE_SHA256:
        raise ValueError('Read-only source changed during review')
    report.update(source_sha256=SOURCE_SHA256, candidate_sha256=digest(candidate),
        collision_geometry_sha256=digest(output/'geometry/collision_geometry.json'),
        collision_arrays_sha256=digest(output/'geometry/collision_geometry.npz'),
        collision_count=len(entries), source_layers_unchanged=True,
        approved_camera_restoration=camera, actual_physics_executed=False,
        review_worker_sha256=digest(__file__),
        geometry_module_sha256=digest(Path(__file__).with_name('retainer_geometry.py')))
    write(output/'geometry_check.json', report)
    return stage, report


def render_candidate(app, candidate, output):
    """Render the saved candidate with the timeline stopped and temporary cameras."""
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom
    omni.usd.get_context().open_stage(str(candidate))
    omni.timeline.get_timeline_interface().stop()
    for _ in range(35):
        app.update()
    stage = omni.usd.get_context().get_stage()
    annotator = rep.AnnotatorRegistry.get_annotator('rgb')
    def capture(camera, name, resolution):
        product = rep.create.render_product(camera, resolution)
        annotator.attach([product])
        try:
            rep.orchestrator.step(rt_subframes=16, pause_timeline=True, delta_time=0.)
            frame = np.asarray(annotator.get_data())[:, :, :3]
            if frame.shape != (resolution[1], resolution[0], 3) or frame.std() < 3:
                raise RuntimeError('Missing or blank actual Isaac image: '+name)
            Image.fromarray(frame).save(output/name)
        finally:
            annotator.detach()
            product.destroy()
    capture('/World/RobotCam', 'after_scene.png', (1440, 810))
    # Inspection-only visibility and camera edits never enter the saved candidate.
    for prim in stage.GetPrimAtPath('/World').GetChildren():
        if prim.GetName() == 'Stethoscope':
            continue
        if UsdGeom.Boundable(prim):
            UsdGeom.Imageable(prim).MakeInvisible()
        elif prim.IsA(UsdGeom.Xform) and not any('Light' in str(q.GetTypeName()) for q in Usd.PrimRange(prim)):
            UsdGeom.Imageable(prim).MakeInvisible()
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(MOVING))
    camera = UsdGeom.Camera.Define(stage, '/World/RetainerReviewCamera')
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 100))
    camera.CreateFocalLengthAttr(55)
    camera.CreateHorizontalApertureAttr(24)
    camera.CreateVerticalApertureAttr(18)
    op = camera.AddTransformOp()
    views = [('after_side.png', (.19, 0, -.050), (0, 0, -.038)),
             ('after_oblique.png', (.13, -.16, -.13), (0, 0, -.035)),
             ('after_face.png', (.035, -.035, -.20), (0, 0, -.040))]
    for name, eye, target in views:
        op.Set(Gf.Matrix4d().SetLookAt(transform.Transform(Gf.Vec3d(*eye)),
            transform.Transform(Gf.Vec3d(*target)), transform.TransformDir(Gf.Vec3d(0, 0, 1))).GetInverse())
        for _ in range(12):
            app.update()
        capture(str(camera.GetPath()), name, (1440, 1080))
        print('RENDERED', name, flush=True)
    return {path.name: digest(path) for path in output.glob('*.png')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1'
            or config.get('experiment') != 'retainer_geometry_review'):
        raise RuntimeError('Owned Linux retainer review required')
    if args.output.exists():
        raise FileExistsError('Review output must be a new directory: '+str(args.output))
    # Reject a stale source before launching Isaac or creating output artifacts.
    if config.get('source_sha256') != SOURCE_SHA256 or digest(config['source']) != SOURCE_SHA256:
        raise ValueError('Exact hash-bound v1 source required')
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=1440, height=1080,
        renderer='RayTracedLighting', anti_aliasing=3))
    try:
        stage, report = build_candidate(config, args.output)
        stage = None
        images = render_candidate(app, args.output/'candidate.usdc', args.output)
        if digest(config['source']) != SOURCE_SHA256:
            raise ValueError('Read-only source changed during rendering')
        if digest(args.output/'candidate.usdc') != report['candidate_sha256']:
            raise ValueError('Saved candidate changed during inspection rendering')
        report.update(completed=True, geometry_id=GEOMETRY_VERSION, images=images,
            rendering='Isaac RayTracedLighting; stopped timeline; temporary inspection cameras only',
            physical_validation=False, training_eligible=False)
        write(args.output/'COMPLETE.json', report)
        print('RETAINER_GEOMETRY_REVIEW_COMPLETE', flush=True)
    except Exception as error:
        traceback.print_exc()
        if args.output.is_dir():
            write(args.output/'FAILED.json', dict(error=type(error).__name__, reason=str(error)))
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()

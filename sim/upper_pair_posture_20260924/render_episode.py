"""Alternative episode: actual saved physics poses, continuous 1x, two synchronized views.

Render only on the authorized Linux Isaac Sim host. Does not rerun the controller.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sim.curobo_expert_v1.render_review import sha, FIELDS, CAMERA_CONFIG, PROJECT
from sim.curobo_expert_v1.quality import validate_episode_provenance, verify_hashes
from sim.route_b_full_video_20260924.timeline import full_schedule

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, FPS = 1920, 720, 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='UPPER POSTURE ALTERNATIVE')
    args = parser.parse_args()
    episode, out = args.episode.resolve(), args.output.resolve()
    if out == episode or episode in out.parents:
        raise ValueError('output must be outside the immutable source episode')
    provenance = validate_episode_provenance(episode)
    if provenance['status'] != 'passed':
        raise ValueError('episode provenance rejected: ' + str(provenance['reasons']))
    source = episode / 'training_scene.usdc'
    outcome = json.loads((episode / 'outcome.json').read_text())
    if sha(source) != outcome.get('training_scene_sha256'):
        raise ValueError('scene provenance mismatch')
    source_files = [source, episode/'outcome.json', episode/'expert_replay.npz',
                    episode/'expert_trace.jsonl', CAMERA_CONFIG, Path(__file__),
                    PROJECT/'sim/route_b_full_video_20260924/timeline.py']
    source_files += [PROJECT/'sim/curobo_expert_v1'/x for x in
                     ('render_review.py', 'quality.py', 'provenance.py')]
    source_files += [PROJECT/'sim/expert_quality_20260924/quality.py']
    inputs = {str(p.relative_to(PROJECT)): sha(p) for p in source_files}
    rows = []
    with (episode/'expert_trace.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.append({k: row.get(k) for k in FIELDS})
    times = np.asarray([row['time_s'] for row in rows])
    with np.load(episode/'expert_replay.npz', allow_pickle=False) as record:
        poses, indices, paths = (record[k] for k in ('poses', 'row_indices', 'paths'))
    if poses.shape != (len(indices), len(paths), 7) or not np.isfinite(poses).all():
        raise ValueError('malformed body poses')
    if not np.allclose(np.linalg.norm(poses[:, :, 3:], axis=2), 1, atol=1e-4):
        raise ValueError('nonunit recorded rotation')
    rr, kk, tt = full_schedule(times, indices, FPS)
    points = outcome['points']
    if ([p['target'] for p in points] != outcome['requested_points'] or
            outcome['requested_points'] not in (['R1','L1','R2'],['R1','L1','R2','L2','R3','L3','R4','L4','R5','L5'])):
        raise ValueError('unexpected point order')
    end = 0
    for point in points:
        start, next_end = point['start_physics_row'], point['end_physics_row']
        if start != end or next_end <= start or any(r['target'] != point['target'] for r in rows[start:next_end]):
            raise ValueError('point trace coverage mismatch')
        end = next_end
    if end != len(rows) or rows[-1]['subphase'] != 'home_settle':
        raise ValueError('episode is not complete through final home settle')
    config = json.loads(CAMERA_CONFIG.read_text())
    assert config['downward_pitch_delta_deg'] == 7 and config['camera_height_delta_m'] == .05
    out.mkdir(parents=True, exist_ok=False)
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 1280, 'height': 720,
                         'renderer': 'RayTracedLighting', 'multi_gpu': False})
    encoder = None
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from PIL import Image
        import cv2
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        import imageio_ffmpeg
        stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
        for path in ('/Render', '/Replicator', '/Orchestrator'):
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        stage.GetRootLayer().Export(str(out/'disposable_render.usdc'))
        omni.usd.get_context().open_stage(str(out/'disposable_render.usdc'))
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(35):
            app.update()
        stage = omni.usd.get_context().get_stage()
        cache, transforms = UsdGeom.XformCache(), []
        path_set = set(map(str, paths))
        for path in paths:
            prim = stage.GetPrimAtPath(str(path))
            assert prim
            parent = prim.GetParent()
            while parent and not parent.IsPseudoRoot():
                if str(parent.GetPath()) in path_set:
                    raise ValueError('dynamic parent in replay')
                parent = parent.GetParent()
            inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            transforms.append((xf.AddTransformOp(opSuffix='reviewReplay'), inv))
        camera_world = np.asarray(config['world_from_camera_usd'])
        for path in ('/World/RobotCam', '/World/RobotCamBody'):
            prim = stage.GetPrimAtPath(path)
            inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='approved7deg').Set(Gf.Matrix4d(camera_world.T)*inv)
        camera = UsdGeom.Camera.Define(stage, '/World/QualityReviewCamera')
        camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 100.))
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateFocalLengthAttr(22.)
        xf = UsdGeom.Xformable(camera)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.45,-1.,1.90), Gf.Vec3d(-.12,.50,1.24), Gf.Vec3d(0,0,1)).GetInverse())
        products = {}
        for view, path in [('posture','/World/QualityReviewCamera'), ('robotcam','/World/RobotCam')]:
            product = rep.create.render_product(path, (1280,720))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            products[view] = (annotator, product)
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(15):
            app.update()
        video = out/'alternative_full_1x.mp4'
        encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
        manifest = {'schema': 'continuous_recorded_route_v1', 'source_sha256': inputs,
            'episode': str(episode.relative_to(PROJECT)), 'video': video.name,
            'resolution': [WIDTH, HEIGHT], 'fps': FPS, 'playback_speed': 1.,
            'frame_count': len(rr), 'source_physics_rows': len(rows),
            'source_duration_s': len(rows)/240, 'duration_s': len(rr)/FPS,
            'terminal_video_padding_s': len(rr)/FPS-len(rows)/240,
            'sampling': 'Continuous end-of-frame sampling; latest preceding saved pose; no interpolation. Final fractional interval retained with <1 frame terminal padding.',
            'source_pose_dt_s': 1/30, 'source_physics_dt_s': 1/240,
            'first_saved_pose_s': float(times[indices[0]]), 'last_saved_pose_s': float(times[indices[-1]]),
            'source_outcome_accepted': outcome.get('training_accepted') is True,
            'scope': 'Recorded Isaac physics replay; no controller rerun. RGB review, not RGB-D training export. No audio or clinical acceptance claim.',
            'camera_config': str(CAMERA_CONFIG.relative_to(PROJECT)),
            'layout': {'posture': [0,0,1280,720], 'robotcam': [1280,0,640,360], 'metadata': [1280,360,640,360]},
            'point_intervals': [{k:p[k] for k in ('target','start_physics_row','end_physics_row')} for p in points],
            'frames': []}
        sampled = set()
        for f, (r,k,t) in enumerate(zip(rr,kk,tt)):
            for (op, inv), pose in zip(transforms, poses[k]):
                mat = Gf.Matrix4d(1.).SetRotate(Gf.Quatd(float(pose[3]),Gf.Vec3d(*map(float,pose[4:]))))
                mat.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
                op.Set(mat*inv)
            rep.orchestrator.step(rt_subframes=3, delta_time=0., pause_timeline=True)
            if timeline.is_playing() or abs(timeline.get_current_time()) > 1e-12:
                raise RuntimeError('physics advanced during recorded replay')
            images = {}
            for view, (annotator, _) in products.items():
                pixels = np.asarray(annotator.get_data())[:,:,:3].copy()
                if pixels.shape != (720,1280,3) or pixels.std() < 5:
                    raise RuntimeError('invalid RGB frame')
                images[view] = pixels
            row = rows[int(r)]
            canvas = np.full((HEIGHT, WIDTH, 3), (22,28,39), dtype=np.uint8)
            canvas[:,:1280] = images['posture']
            canvas[:360,1280:] = cv2.resize(images['robotcam'], (640,360), interpolation=cv2.INTER_AREA)
            labels = [
                (args.label+' | 1x', .78),
                (f"Time  {t:07.2f} / {len(rows)/240:.2f} s", .8),
                (f"Point  {row['target']}    Phase  {row['phase']}", .72),
                (f"{row['subphase']}", .65),
                (' '.join(outcome['requested_points']), .65),
                ('LEFT: whole arm   TOP: approved camera', .58),
                ('Saved physics poses | continuous original time', .58),
                ('No audio | engineering review', .58)]
            for line, ((label, scale)) in enumerate(labels):
                cv2.putText(canvas, label, (1300,396+40*line), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (230,236,245), 1, cv2.LINE_AA)
            encoder.stdin.write(canvas.tobytes())
            key = (row['target'], row['phase'], row['subphase'])
            if key not in sampled or f == len(rr)-1:
                sampled.add(key)
                Image.fromarray(canvas).save(out/f'frame_{f:05d}.jpg', quality=92)
            manifest['frames'].append({'frame_index': f, 'video_time_s': f/FPS,
                'simulation_interval_end_s': float(t), 'measurement_row': int(r),
                'measurement_time_s': float(times[r]), 'pose_index': int(k),
                'pose_row': int(indices[k]), 'pose_time_s': float(times[indices[k]]),
                'target': row['target'], 'phase': row['phase'], 'subphase': row['subphase']})
            if f % 300 == 0 or f == len(rr)-1:
                print('FULL_VIDEO_PROGRESS', f+1, '/', len(rr), row['target'], row['subphase'], flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('video encoder failed')
        encoder = None
        if any(sha(PROJECT/rel) != digest for rel,digest in inputs.items()):
            raise RuntimeError('source changed during render')
        if verify_hashes(episode, provenance['artifact_sha256']):
            raise RuntimeError('bound episode changed during render')
        manifest.update(video_sha256=sha(video), episode_provenance=provenance, source_files_unchanged=True)
        (out/'review_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        print('FULL_VIDEO_RENDER_COMPLETE', flush=True)
    finally:
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()
        # Kit owns shared SyntheticData graph teardown for the two render products.
        app.close()


if __name__ == '__main__':
    main()

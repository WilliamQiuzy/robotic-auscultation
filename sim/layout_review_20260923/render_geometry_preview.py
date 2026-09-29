"""CPU raycast of exported scene geometry; explicitly not an Isaac render."""
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / 'geometry_export'
OUTPUT = ROOT / 'geometry_preview'
NAMES = ('baseline', 'farther5_lower2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Helvetica.ttc']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def side_camera():
    eye = np.array([2.6, .6, 1.9])
    forward = np.array([0., .6, .95]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    world = np.eye(4)
    world[:3, :3] = np.stack([right, down, forward], axis=1)
    world[:3, 3] = eye
    return dict(world_from_optical=world.tolist(), fx=22/20.955*1920,
                fy=22/20.955*1920, cx=960., cy=540., width=1920, height=1080)


def render(scene, colors, camera):
    w, h = camera['width'], camera['height']
    vv, uu = np.mgrid[:h, :w].astype(np.float32)
    local = np.stack([(uu+.5-camera['cx'])/camera['fx'],
                      (vv+.5-camera['cy'])/camera['fy'], np.ones_like(uu)], axis=-1)
    world = np.array(camera['world_from_optical'], dtype=np.float32)
    directions = local @ world[:3, :3].T
    origins = np.broadcast_to(world[:3, 3], directions.shape)
    rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)
    answer = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
    depth = answer['t_hit'].numpy()
    ids = answer['primitive_ids'].numpy()
    normals = answer['primitive_normals'].numpy()
    valid = np.isfinite(depth)
    normals[np.sum(normals*directions, axis=-1) > 0] *= -1
    light = np.array([-.25, -.45, 1.], np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(np.sum(normals*light, axis=-1), 0, 1)
    camera_light = np.clip(-np.sum(normals*directions, axis=-1)/np.linalg.norm(directions, axis=-1), 0, 1)
    illumination = .42 + .40*diffuse + .18*camera_light
    rgb = np.full((h, w, 3), .91, np.float32)
    rgb[valid] = colors[ids[valid]] * illumination[valid, None]
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*np.power(rgb, 1/2.4)-.055)
    # Simple neutral matte shader, deliberately no synthetic texture inference.
    return Image.fromarray((rgb*255).astype(np.uint8)), depth


def annotate(raw, depth, camera, targets):
    image = raw.copy()
    draw = ImageDraw.Draw(image)
    inv = np.linalg.inv(np.asarray(camera['world_from_optical']))
    records = []
    for target in targets:
        name = target['name']
        optical = (inv @ np.r_[target['position_world'], 1.])[:3]
        u = camera['fx']*optical[0]/optical[2]+camera['cx']
        v = camera['fy']*optical[1]/optical[2]+camera['cy']
        inside = bool(optical[2] > 0 and 0 <= u < camera['width'] and 0 <= v < camera['height'])
        measured = float(depth[int(v), int(u)]) if inside else None
        if measured is not None and not np.isfinite(measured):
            measured = None
        residual = None if measured is None else measured-float(optical[2])
        occluded = None if residual is None else residual < -.02
        records.append(dict(name=name, uv_px=[float(u),float(v)], inside_frame=inside,
                            target_optical_depth_m=float(optical[2]), raycast_depth_m=measured,
                            depth_residual_m=residual, nearer_geometry_by_over_2cm=occluded))
        if inside:
            color = (0, 235, 245) if name.startswith('R') else (255, 190, 75)
            draw.ellipse((u-9,v-9,u+9,v+9),fill=color,outline=(10,10,10),width=2)
            draw.text((u+13,v-18),name,fill=color,font=font(28),stroke_width=2,stroke_fill=(0,0,0))
    return image, records


def comparison(paths, labels, destination):
    sheet = Image.new('RGB', (1920, 650), '#f2f4f6')
    draw = ImageDraw.Draw(sheet)
    for i, (path, title) in enumerate(zip(paths, labels)):
        draw.text((i*960+20, 12), title, font=font(24), fill='#152637')
        source = Image.open(path).convert('RGB').resize((960,540), Image.Resampling.LANCZOS)
        sheet.paste(source, (i*960,55))
    draw.text((20,608), 'Same USD geometry and camera calibration | Mac geometry preview, not Isaac lighting', font=font(22),fill='#3c4b58')
    sheet.save(destination)


def main():
    OUTPUT.mkdir(exist_ok=False)
    manifest = dict(schema='robotcam_geometry_preview_v1',
                    scope='CPU raycast of actual exported visual control meshes with simple matte shading; not an Isaac RGB render or a new physics rollout.',
                    renderer='Open3D RaycastingScene '+o3d.__version__, variants={})
    for name in NAMES:
        metadata_path = EXPORT / (name+'.json')
        mesh_path = EXPORT / (name+'.npz')
        meta = json.loads(metadata_path.read_text())
        with np.load(mesh_path, allow_pickle=False) as data:
            vertices = data['vertices'].astype(np.float32)
            triangles = data['triangles'].astype(np.uint32)
            colors = data['triangle_colors'].astype(np.float32)
        assert len(colors) == len(triangles)
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(triangles))
        raw, depth = render(scene, colors, meta['camera'])
        raw.save(OUTPUT/(name+'_robotcam.png'))
        np.save(OUTPUT/(name+'_depth_m.npy'), depth)
        annotated, targets = annotate(raw, depth, meta['camera'], meta['targets'])
        annotated.save(OUTPUT/(name+'_robotcam_targets.png'))
        side, _ = render(scene, colors, side_camera())
        side.save(OUTPUT/(name+'_side.png'))
        manifest['variants'][name] = dict(input_npz_sha256=sha(mesh_path), input_metadata_sha256=sha(metadata_path),
            camera=meta['camera'], projected_existing_targets=targets,
            within_frame_count=sum(x['inside_frame'] for x in targets),
            nearer_geometry_count=sum(x['nearer_geometry_by_over_2cm'] is True for x in targets),
            vertex_count=len(vertices), triangle_count=len(triangles))
        print('GEOMETRY_PREVIEW_RENDERED', name, manifest['variants'][name]['within_frame_count'], flush=True)
    assert manifest['variants'][NAMES[0]]['camera'] == manifest['variants'][NAMES[1]]['camera']
    for view in ['robotcam', 'robotcam_targets', 'side']:
        comparison([OUTPUT/(n+'_'+view+'.png') for n in NAMES],
                   ['Original layout', 'Patient / stool: 5 cm farther, seat 2 cm lower'],
                   OUTPUT/('comparison_'+view+'.png'))
    manifest['output_sha256'] = {p.name:sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}
    (OUTPUT/'preview_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('GEOMETRY_PREVIEW_COMPLETE',flush=True)


if __name__ == '__main__':
    main()

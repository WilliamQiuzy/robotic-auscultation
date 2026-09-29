"""Offline inference with the deployed v9 weights. Never connects to hardware."""
import os
import sys
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/robot/Desktop/stethoscope/.wam-releases/code_release-f0b5c832731d8d29eb94/auscultation')
RELEASE = Path('/home/robot/wam_data/perception_runtime/releases/perception_runtime-7c450b029584cf07584e')
CHECKPOINT = RELEASE / 'model/perc_lmh_v9_occ/best.pt'
BASE = RELEASE / 'torch_home/hub/checkpoints/dinov2_vitb14_pretrain.pth'
EXPECTED = '68037e6932fd7315af33722a69060f0d9838afbc1a60d6f0349da909917f6a14'
assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == EXPECTED
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
os.environ['AUSC_OFFLINE_RUNTIME'] = '1'
os.environ['TORCH_HOME'] = str(RELEASE / 'torch_home')
os.environ['AUSC_DINOV2_REPO'] = str(RELEASE / 'torch_home/hub/facebookresearch_dinov2_main')
os.environ['AUSC_DINOV2_CHECKPOINT'] = str(BASE)
sys.path.insert(0, str(CODE))
import numpy as np
import torch
import cv2
torch.set_num_threads(2)
cv2.setNumThreads(2)
from ausc.perception.tracking.detector import NormalsDetector
from ausc.perception.tracking.geometry import Intrinsics, sample_depth_m

meta = json.loads((ROOT / 'asset_manifest.json').read_text())
intr = Intrinsics(**meta['intrinsics_rendered'])
T = np.asarray(meta['world_from_camera_optical'])
detector = NormalsDetector(str(CHECKPOINT), device='cuda', half=True,
                          fill_holes=True, plausibility_gate=True)
outputs = {'checkpoint_sha256': EXPECTED, 'model': 'perc_lmh_v9_occ',
           'code_release': str(CODE.parent.name),
           'input': 'depth-derived surface normals from rendered RGB-D',
           'scope': 'single-frame detector only, not full tracking/person-gate/robot authorization pipeline',
           'point_semantics': 'model-proposed R1/L1 through R5/L5, not user-confirmed anatomical ground truth',
           'confidence_semantics': 'flip-consistency heuristic, not calibrated correctness probability',
           'frames': {}}

for stem in ['before', 'raised']:
    bgr = cv2.imread(str(ROOT / (stem + '_rgb.png')))
    rgb = bgr[:, :, ::-1].copy()
    depth = cv2.imread(str(ROOT / (stem + '_depth_mm.png')), cv2.IMREAD_UNCHANGED)
    result = detector.detect(rgb, depth, intr)
    assert result is not None
    normal_rgb, _, _ = detector._normal_image(rgb, depth, intr)
    raw_xy, _, _ = detector._points_from_im(normal_rgb)
    uv = result.crop.det_to_full(result.xy_det)
    raw_uv = result.crop.det_to_full(raw_xy[:10])
    z = sample_depth_m(depth, uv, intr.depth_scale)
    camera_points = intr.unproject(uv, z)
    world_points = camera_points @ T[:3, :3].T + T[:3, 3]
    rows = []
    for i, name in enumerate(result.order):
        u, v = uv[i]
        inside = bool(0 <= u < intr.width and 0 <= v < intr.height)
        depth_valid = bool(inside and np.isfinite(z[i]) and z[i] > 0)
        row = {'name': name, 'uv_px': uv[i].tolist(), 'raw_uv_px': raw_uv[i].tolist(),
               'inside_frame': inside, 'depth_valid': depth_valid,
               'camera_xyz_m': camera_points[i].tolist() if depth_valid else None,
               'world_xyz_m': world_points[i].tolist() if depth_valid else None,
               'accepted_target': False}
        if depth_valid:
            ui, vi = int(round(u)), int(round(v))
            ya, yb = max(0, vi-5), min(depth.shape[0], vi+6)
            xa, xb = max(0, ui-5), min(depth.shape[1], ui+6)
            yy, xx = np.mgrid[ya:yb, xa:xb]
            zz = depth[ya:yb, xa:xb].astype(float) / 1000.
            valid = (zz > 0) & (np.abs(zz-z[i]) < .025)
            p = intr.unproject(np.c_[xx[valid], yy[valid]], zz[valid])
            if len(p) >= 12:
                center = p.mean(0)
                _, _, vh = np.linalg.svd(p-center, full_matrices=False)
                n = vh[-1]
                if n @ camera_points[i] > 0: n = -n
                row['normal_camera_from_depth'] = n.tolist()
                row['normal_world_from_depth'] = (T[:3, :3] @ n).tolist()
                row['local_plane_residual_m'] = float(np.sqrt(np.mean(((p-center) @ n)**2)))
        rows.append(row)
        if inside:
            x, y = int(round(u)), int(round(v))
            color = (35, 95, 240) if name.startswith('R') else (235, 170, 30)
            cv2.circle(bgr, (x, y), 11, (255,255,255), -1)
            cv2.circle(bgr, (x, y), 8, color, -1)
            tx = x - 70 if name.startswith('R') else x + 16
            cv2.putText(bgr, name, (tx, y+8), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(bgr, name, (tx, y+8), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)
    status = 'MODEL CANDIDATES - HUMAN REVIEW REQUIRED'
    cv2.rectangle(bgr, (0,0), (bgr.shape[1], 92), (25,25,25), -1)
    cv2.putText(bgr, status, (28,36), cv2.FONT_HERSHEY_SIMPLEX, .9, (230,230,255), 2, cv2.LINE_AA)
    line = f'{stem} | v9 depth normals | back gate={result.back_present} | flip score={result.confidence:.3f}'
    cv2.putText(bgr, line, (28,73), cv2.FONT_HERSHEY_SIMPLEX, .75, (230,230,230), 2, cv2.LINE_AA)
    cv2.imwrite(str(ROOT / (stem + '_points.png')), bgr)
    cv2.imwrite(str(ROOT / (stem + '_model_normals.png')), normal_rgb[:, :, ::-1])
    frame = {'back_present_gate': bool(result.back_present), 'flip_consistency_score': float(result.confidence),
             'crop': vars(result.crop), 'points': rows}
    outputs['frames'][stem] = frame
    print(stem, 'back gate', result.back_present, 'flip score', result.confidence,
          'in frame', sum(r['inside_frame'] for r in rows), 'depth valid', sum(r['depth_valid'] for r in rows), flush=True)
(ROOT / 'model_predictions.json').write_text(json.dumps(outputs, indent=2, allow_nan=False))
print('DETECTOR_COMPLETE', flush=True)

"""Measure neutral-pose posterior cloth visibility in actual aligned depth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    args = parser.parse_args()
    folder = args.comparison.resolve()
    data = json.loads((folder/'comparison.json').read_text())
    project = ROOT.parents[1]
    for path, expected in data['source_sha256'].items():
        assert digest(project/path) == expected, path
    for path, expected in data['output_sha256'].items():
        assert digest(folder/path) == expected, path
    samples = np.load(ROOT/'back_roi_samples.npz', allow_pickle=False)
    intr = data['intrinsics']
    report = {
        'scope': 'Neutral recorded arm pose only. Posterior cloth control-mesh proxy sampled uniformly in world X/Z. Not anatomical full-back ground truth, surface area, or dynamic visibility.',
        'method': 'Projection into actual Isaac RGB-D; sample matches rendered optical-Z depth within 5/10/20 mm. Denominator includes out-of-frame and occluded samples.',
        'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [folder/'comparison.json', ROOT/'back_roi_samples.npz']},
        'source_and_render_hashes_verified': True,
        'results': {}}
    for name, variant in data['variants'].items():
        depth = np.load(folder/(name+'_depth_m.npy'), allow_pickle=False)
        inv = np.linalg.inv(variant['world_from_optical'])
        metrics = {}
        for key in ('center', 'roi'):
            pts = samples[key]
            xyz = (np.c_[pts, np.ones(len(pts))]@inv.T)[:, :3]
            uv = xyz[:, :2]/xyz[:, 2, None]*[intr['fx'], intr['fy']]+[intr['cx'], intr['cy']]
            inside = (xyz[:, 2] > 0)&(uv[:, 0] >= 0)&(uv[:, 0] < intr['width'])&(uv[:, 1] >= 0)&(uv[:, 1] < intr['height'])
            pixels = uv[inside].astype(int)
            residual = depth[pixels[:, 1], pixels[:, 0]] - xyz[inside, 2]
            matching = {str(t): float(np.count_nonzero(np.isfinite(residual)&(np.abs(residual) <= t))/len(pts)) for t in (.005, .01, .02)}
            metrics[key] = {'count': len(pts), 'in_frame_fraction': float(inside.mean()), 'depth_matching_fraction': matching,
                'occluded_by_10mm_fraction': float(np.count_nonzero(np.isfinite(residual)&(residual < -.01))/len(pts))}
        targets = variant['projected_targets']
        metrics['target_count_in_frame'] = sum(p['inside'] for p in targets)
        metrics['target_count_depth_matching_within_5mm'] = sum(p['depth_residual_m'] is not None and abs(p['depth_residual_m']) <= .005 for p in targets)
        metrics['minimum_target_border_margin_px'] = min(p['border_margin_px'] for p in targets)
        report['results'][name] = metrics
        print(name, 'center', round(metrics['center']['depth_matching_fraction']['0.01']*100, 2),
              'ROI', round(metrics['roi']['depth_matching_fraction']['0.01']*100, 2),
              'targets', metrics['target_count_depth_matching_within_5mm'], 'margin', round(metrics['minimum_target_border_margin_px'], 1))
    (folder.parent/(folder.name+'_depth_coverage.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

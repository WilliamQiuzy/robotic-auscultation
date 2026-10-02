"""Two observed SMPL sources x bare/parallel-surface proxy x ten patches.

Linux CPU geometry study, not a new clothing asset or native simulation. Source
postures, scene objects and camera are never changed. Raw source and generated
arrays stay Linux. No IK/path/physical or training qualification is issued.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from .contact_surface_gap import ray_offsets, sha
from .surface_patch_plane import offset_surface_proxy, probe_surface_patch

NAMES = [side+str(i) for i in range(1, 6) for side in 'LR']
CASES = ['yaw_p000474_35', 'yaw_p000476_35']


def bound_cell_sources(cell):
    result = {}
    for name in ('collision_geometry.json', 'collision_geometry.npz', 'v6a_anchors.json'):
        record = cell['artifacts'][name]
        data = Path(record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Source hash mismatch: '+name)
        result[name] = data
    return result


def select_cells(summary, names):
    selected = []
    if len(set(names)) != len(names):
        raise ValueError('Repeated case')
    for name in names:
        matches = [c for c in summary['cells'] if c['cell_id'] == name]
        if len(matches) != 1:
            raise ValueError('Missing or repeated case: '+name)
        selected.append(matches[0])
    return selected


def map_anchor(mesh, position, normal):
    p = np.asarray(position, float); n = np.asarray(normal, float)
    offset = ray_offsets(p[None, :], -n, mesh)[0]
    if not np.isfinite(offset):
        raise ValueError('Intended surface anchor is missing in bounded ray')
    mapped = p-offset*n
    tangential = (mapped-p)-((mapped-p)@n)*n
    return dict(position_world_m=mapped.tolist(), reference_normal_world=n.tolist(),
        signed_ray_offset_m=float(offset), tangential_displacement_m=float(np.linalg.norm(tangential)),
        method='first intersection along unchanged anatomical reference normal; no tangential target optimization')


def evaluate_patch(mesh, position, normal):
    try:
        mapped = map_anchor(mesh, position, normal)
    except ValueError as error:
        if str(error) != 'Intended surface anchor is missing in bounded ray':
            raise
        return dict(status='unresolved_missing_central_hit', anchor_map=None,
            sampled_geometry=probe_surface_patch(mesh, position, normal))
    result = probe_surface_patch(mesh, mapped['position_world_m'], normal)
    return dict(status='sampled_patch_complete' if result['missing_samples'] == 0
        else 'unresolved_missing_patch_hits', anchor_map=mapped, sampled_geometry=result)


def run(source, expected_sha, output):
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Linux CPU only; explicitly set CUDA_VISIBLE_DEVICES empty')
    start = time.monotonic(); source = Path(source); output = Path(output)
    if output.exists():
        raise FileExistsError('Retain prior study')
    if sha(source) != expected_sha:
        raise ValueError('Source summary hash mismatch')
    summary = json.loads(source.read_text()); cells = select_cells(summary, CASES)
    sources = [bound_cell_sources(c) for c in cells]
    output.mkdir(parents=True, exist_ok=False)
    details = []; records = []; artifacts = []; mappings = []
    for cell, bound in zip(cells, sources):
        if cell['split'] != 'train':
            raise ValueError('Training source cases only for design exploration')
        entries = json.loads(bound['collision_geometry.json'])['colliders']
        patient = [e for e in entries if e['path'].startswith('/World/Patient/')]
        if (len(patient) != 1 or patient[0]['path'] != '/World/Patient/SMPLSurface'
                or patient[0]['body'] != 'world' or patient[0]['approximation'] != 'none'):
            raise ValueError('Explicit single static SMPL surface required')
        e = patient[0]; transform = np.asarray(e['world_from_body'])
        if not np.array_equal(transform, np.eye(4)):
            raise ValueError('Expected materialized world coordinates')
        with np.load(io.BytesIO(bound['collision_geometry.npz']), allow_pickle=False) as arrays:
            vertices = np.array(arrays[e['key']+'_v'], float); faces = np.array(arrays[e['key']+'_f'])
        if vertices.shape != (6890, 3) or faces.shape != (13776, 3):
            raise ValueError('Unexpected source SMPL topology')
        anchors = json.loads(bound['v6a_anchors.json'])['anchors']
        if set(anchors) != set(NAMES):
            raise ValueError('Complete ten anchors required')
        proxy, receipt = offset_surface_proxy(vertices, faces, .0005)
        for condition, current in [('bare_smpl', vertices), ('parallel_surface_proxy_0p5mm', proxy)]:
            artifact = output/(cell['cell_id']+'_'+condition+'.npz')
            np.savez_compressed(artifact, vertices=current, faces=faces)
            artifacts.append(dict(path=artifact.name, sha256=sha(artifact), bytes=artifact.stat().st_size,
                condition=condition, cell=cell['cell_id']))
            triangles = current[faces]
            for name in NAMES:
                anchor = anchors[name]
                evaluation = evaluate_patch(triangles, anchor['position_m'], anchor['normal'])
                mapped = evaluation['anchor_map']; result = evaluation['sampled_geometry']
                row = dict(cell=cell['cell_id'], identity=cell['identity'], source_sequence=cell['sequence'],
                    source_frame=cell['frame'], target=name, condition=condition,
                    status=evaluation['status'], sampled_support_complete=evaluation['status'] == 'sampled_patch_complete',
                    missing_samples=result['missing_samples'], requested_samples=result['requested_samples'],
                    ring_height_envelope=result['ring_plane'], disk_height_envelope=result['plane'],
                    anchor_map=mapped, training_eligible=False)
                records.append(row); details.append(dict(**row, sampled_geometry=result))
                print(json.dumps(dict(cell=row['cell'], condition=condition, target=name,
                    missing=result['missing_samples'], ring_span_mm=(1000*result['ring_plane']['residual_span_m']
                        if result['ring_plane'] else None))), flush=True)
        mappings.append(dict(cell=cell['cell_id'], proxy_recipe=receipt))
    # Rebind every input after the run, including source identity/pose metadata.
    if sha(source) != expected_sha:
        raise ValueError('Source summary changed during diagnostic')
    for cell in cells:
        bound_cell_sources(cell)
    (output/'details.json').write_text(json.dumps(details, indent=2, allow_nan=False)+'\n')
    report = dict(schema='paired_back_surface_geometry_probe_v1', requested_patches=40,
        completed_patches=len(records), records=records, artifacts=artifacts, proxy_recipes=mappings,
        source_summary=dict(path=str(source), sha256=expected_sha), source_cells=cells,
        source_sha256={f:sha(Path(__file__).with_name(f)) for f in
            ('surface_pair_probe.py','surface_patch_plane.py','contact_surface_gap.py')},
        details_sha256=sha(output/'details.json'), wall_s=time.monotonic()-start,
        scope='Two fixed observed source/shape cases and yaw35; bare SMPL plus 0.5mm mathematical offset. Fixed samples and affine height residuals only. Not a garment pattern, material model, continuous rim bound, native installation or expert fit.',
        physical_material_changed=False, scene_installed=False, ik_executed=False,
        collision_qualified=False, local_self_intersections_checked=False,
        flatness_qualified=False, calibrated_cloth=False, physics_qualified=False, training_eligible=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('source_sha256'); parser.add_argument('output')
    args = parser.parse_args(); run(args.source, args.source_sha256, args.output)

"""Read-only source geometry preflight and minimal remote transfer manifest.

Requires pxr and NumPy, never imports Isaac Sim or creates a renderer.
"""
import copy
import json
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom

from sim.layout_review_20260923 import render_layouts as task


def bounds(stage, path):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.asarray([r.GetMin(), r.GetMax()], float)


def camera_state(stage):
    c = UsdGeom.Camera(stage.GetPrimAtPath('/World/RobotCam'))
    return {'matrix': np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(c.GetPrim())).tolist(),
            'optics': {name: str(c.GetPrim().GetAttribute(name).Get()) for name in (
                'focalLength', 'horizontalAperture', 'verticalAperture',
                'horizontalApertureOffset', 'verticalApertureOffset', 'clippingRange', 'projection')}}


def main():
    inputs = [Path(task.__file__).resolve(), task.SOURCE, task.EPISODE/'expert_replay.npz',
              *[task.PLAN_DIR/f'planned_paths_{name}.json' for name in task.POINTS]]
    rows = [{'path': str(p.relative_to(task.PROJECT)), 'size_bytes': p.stat().st_size, 'sha256': task.sha(p)} for p in inputs]
    baseline = Usd.Stage.Open(str(task.SOURCE))
    assert baseline and UsdGeom.GetStageMetersPerUnit(baseline) == 1.
    assert str(UsdGeom.GetStageUpAxis(baseline)) == 'Z'
    baseline.SetEditTarget(baseline.GetSessionLayer())
    extent_repairs = task.refresh_stool_extents(baseline)
    camera = camera_state(baseline)
    patient = task.patient_ground_state(baseline)
    orig = {n: bounds(baseline, '/World/'+n) for n in ('Patient', 'Stool', 'Table')}
    seat = bounds(baseline, '/World/Stool/seat')
    legs = {p.GetName(): bounds(baseline, str(p.GetPath()))[0, 2]
            for p in baseline.GetPrimAtPath('/World/Stool').GetChildren() if p.GetName().startswith('leg')}
    base_path = '/World/StethoArmMount/OpenArmStetho/openarm_left_link0'
    base_matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(baseline.GetPrimAtPath(base_path)))
    variants = []
    for name, farther, lower in task.VARIANTS:
        # Shared source is read-only; only this anonymous session receives edits.
        stage = Usd.Stage.Open(baseline.GetRootLayer(), Sdf.Layer.CreateAnonymous('layout-preflight-session.usda'))
        stage.SetEditTarget(stage.GetSessionLayer())
        task.apply_layout(stage, farther, lower, float(seat[1, 2]))
        delta = np.asarray([0., farther, lower])
        changed = task.patient_ground_state(stage)
        assert task.verify_no_new_patient_penetration(patient, changed)
        assert camera_state(stage) == camera
        np.testing.assert_allclose(bounds(stage, '/World/Patient'), orig['Patient']+delta, rtol=0, atol=1e-6)
        np.testing.assert_allclose(bounds(stage, '/World/Table'), orig['Table'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(bounds(stage, '/World/Stool/seat'), seat+delta, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(base_path))), base_matrix, rtol=0, atol=1e-12)
        for leg, old_z in legs.items():
            assert abs(bounds(stage, '/World/Stool/'+leg)[0, 2]-old_z) < 1e-6
        variants.append({'name': name, 'patient_translation_world_m': delta.tolist(),
                         'seat_top_m': float(seat[1, 2])+lower, 'patient_ground': changed,
                         'patient_and_seat_translation_verified': True, 'stool_leg_foot_z_unchanged': True,
                         'camera_and_robot_base_and_table_unchanged': True})
    assets = {}
    for prim in baseline.Traverse():
        for attr in prim.GetAttributes():
            typ = str(attr.GetTypeName())
            if typ not in ('asset', 'asset[]'):
                continue
            value = attr.Get()
            values = list(value) if typ == 'asset[]' and value is not None else [value]
            for asset in values:
                if isinstance(asset, Sdf.AssetPath) and asset.path:
                    assets.setdefault(asset.path, []).append(str(attr.GetPath()))
    external_assets = [{'asset_path': p, 'attribute_count': len(attrs), 'example_attribute': attrs[0],
                        'classification': 'https_texture_requires_network_or_existing_cache' if p.startswith('https://') else 'Isaac_runtime_MDL_module' if p == 'OmniPBR.mdl' else 'other_dependency_needs_review'} for p,attrs in sorted(assets.items())]
    assert all(row['classification'] != 'other_dependency_needs_review' for row in external_assets)
    assert len([row for row in external_assets if row['asset_path'].startswith('https://')]) == 3
    assert len([layer for layer in baseline.GetUsedLayers() if not layer.anonymous]) == 1

    # Exercise the requested no-font fallback and penetration rejection paths.
    class MissingFonts:
        @staticmethod
        def truetype(*args, **kwargs): raise OSError('no installed fonts')
        @staticmethod
        def load_default(size=0): return ('bundled', size)
    assert task.annotation_font(MissingFonts)[0] == ('bundled', 28)
    assert len(task.parse_args([]).variants) == 5
    assert task.parse_args(['--output', '/tmp/layout-render-review', '--variants', 'baseline', 'farther_5cm_lower_2cm']).variants == ['baseline', 'farther_5cm_lower_2cm']
    fault = copy.deepcopy(patient)
    fault['patient_meshes'][0]['penetration_m'] += .02
    try:
        task.verify_no_new_patient_penetration(patient, fault)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected
    assert all(task.sha(task.PROJECT/row['path']) == row['sha256'] for row in rows)
    manifest = {'schema': 'layout_render_minimal_inputs_v1', 'required_files': rows,
                'total_bytes': sum(row['size_bytes'] for row in rows), 'required_file_count': len(rows),
                'relative_paths_must_be_preserved': True, 'external_assets': external_assets,
                'external_usd_layers_required': [], 'network_asset_reachability_tested': False,
                'baseline_patient_ground': patient, 'baseline_camera': camera,
                'stool_extent_metadata_repairs': extent_repairs,
                'geometry_preflight': {'accepted': True, 'variants': variants,
                    'fault_injected_patient_ground_penetration_rejected': rejected,
                    'missing_font_fallback_checked': True, 'cli_default_and_subset_checked': True,
                    'source_files_unchanged': True,
                    'scope': 'USD geometry and CLI/font checks only; no render, collision planner, physics or detector run.'},
                'known_existing_limitation': 'Patient shoes are already suspended above the floor; lowering 2cm does not introduce penetration but does not establish foot support.',
                'generator_sha256': task.sha(Path(__file__))}
    (task.ROOT/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (task.ROOT/'transfer_inputs.txt').write_text('\n'.join(row['path'] for row in rows)+'\n')
    print(json.dumps({'required_file_count':len(rows),'total_bytes':manifest['total_bytes'],
        'shoe_minimum_z_before_m':patient['minimum_shoe_clearance_m'],
        'shoe_minimum_z_after_lower2cm_m':variants[2]['patient_ground']['minimum_shoe_clearance_m'],
        'external_assets':[row['asset_path'] for row in external_assets], 'accepted':True},indent=2))


if __name__ == '__main__':
    main()

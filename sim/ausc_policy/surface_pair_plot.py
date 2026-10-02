"""Linux CPU charts and bounded summaries; no mesh/trace export to Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(layer_file, pair_file, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if __package__ in (None, ''):
        from storage_policy import checked_mac_exports
    else:
        from .storage_policy import checked_mac_exports
    layer_file, pair_file, output = Path(layer_file), Path(pair_file), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous evidence')
    layers = json.loads(layer_file.read_text()); pairs = json.loads(pair_file.read_text())
    if len(layers['points']) != 10 or pairs['completed_patches'] != 40:
        raise ValueError('Expected fixed ten and forty patch denominators')
    output.mkdir(parents=True)
    names = [p['target'] for p in layers['points']]
    layer_rows = []
    for p in layers['points']:
        selected = {}
        for key in ['skin', 'shirt', 'all_patient']:
            x = p['anchor_patch']['layers'][key]
            selected[key] = {k:x[k] for k in ['hit_count', 'missing_count', 'signed_offset_stats_m']}
        row = dict(target=p['target'], layers=selected)
        layer_rows.append(row)
    pair_rows = []
    for p in pairs['records']:
        fit = p['ring_height_envelope']
        pair_rows.append(dict(cell=p['cell'], target=p['target'], condition=p['condition'],
            status=p['status'], missing_samples=p['missing_samples'],
            ring_reference_height_span_mm=fit['original_height_span_m']*1000 if fit else None,
            ring_affine_residual_span_mm=fit['residual_span_m']*1000 if fit else None,
            affine_normal_angle_deg=fit['normal_angle_deg'] if fit else None))
    review = dict(schema='contact_surface_review_v2', native_layer_points=layer_rows,
        paired_patches=pair_rows, requested_pairs=40, completed_pairs=40,
        total_requested_ray_samples=40*97,
        paired_missing_ray_samples=sum(r['missing_samples'] for r in pair_rows),
        proxy_recipes=pairs['proxy_recipes'],
        runtime_seconds=dict(layers=layers['wall_s'], pairs=pairs['wall_s']),
        bindings={str(layer_file):sha(layer_file), str(pair_file):sha(pair_file)},
        replay_binding=layers.get('expected_replay_sha256'),
        source_sha256=sha(__file__), storage_policy_sha256=sha(Path(__file__).with_name('storage_policy.py')),
        training_eligible=False, physics_executed=False,
        scope='Actual original mesh ray diagnosis and uncalibrated 0.5mm mathematical offset sensitivity. No new native simulation, clothing material, complete surface or flatness qualification.')
    (output/'summary.json').write_text(json.dumps(review, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    x = np.arange(10)
    axes[0].bar(x-.18, [r['layers']['skin']['hit_count'] for r in layer_rows], .36,
                label='Skin mesh', color='#086f83')
    axes[0].bar(x+.18, [r['layers']['shirt']['hit_count'] for r in layer_rows], .36,
                label='Scrubs mesh', color='#e7a139')
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 112), ylabel='Hit rays / 97',
        title='Original medical asset: skin coverage is incomplete near the ten targets')
    axes[0].axhline(97, color='#777777', linestyle=':', linewidth=1)
    axes[0].legend(ncol=2, loc='upper right')
    for cell, color in [('yaw_p000474_35', '#086f83'), ('yaw_p000476_35', '#a54731')]:
        for cond, style, marker in [('bare_smpl', '-', 'o'), ('parallel_surface_proxy_0p5mm', '--', 'x')]:
            values = {r['target']:r['ring_affine_residual_span_mm'] for r in pair_rows
                if r['cell'] == cell and r['condition'] == cond}
            label = cell.split('_')[1]+' / '+('bare SMPL' if cond == 'bare_smpl' else '0.5 mm offset proxy')
            axes[1].plot(x, [values[n] for n in names], style, marker=marker, color=color, label=label)
    axes[1].set(xticks=x, xticklabels=names, ylim=(0, 6.5), ylabel='Residual height span (mm)',
        title='40 sampled patches: removing affine tilt still leaves local nonplanarity')
    axes[1].legend(ncol=2, loc='upper left', fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Geometry diagnosis only - no learned-policy or acoustic qualification', fontsize=13)
    fig.savefig(output/'surface_comparison.png', dpi=150); plt.close(fig)
    if sha(layer_file) != review['bindings'][str(layer_file)] or sha(pair_file) != review['bindings'][str(pair_file)]:
        raise ValueError('Source changed during figure/export preparation')
    entries = [dict(path=name, kind=kind, sha256=sha(output/name)) for name,kind in
               [('summary.json','summary'), ('surface_comparison.png','chart')]]
    exports = checked_mac_exports(output, entries)
    (output/'export_receipt.json').write_text(json.dumps(dict(files=exports, raw_stays_linux=True), indent=2)+'\n')
    print(json.dumps(dict(files=len(exports), bytes=sum(x['bytes'] for x in exports), output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('layers'); p.add_argument('pairs'); p.add_argument('output')
    a = p.parse_args(); prepare(a.layers, a.pairs, a.output)

"""M0 CPU diagnostics over bound TRAIN32 evidence; no model or native imports.

Run: python -m sim.ausc_policy.diagnostics --config configs/ausc_policy/m0_diagnostics.json
Only the small prepared cache and explicit evaluation shards are loaded. Source
code is hashed/read, and the existing adapter's pure NumPy codecs are reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def finite(value, name):
    result = np.asarray(value, dtype=np.float64)
    need(result.size > 0 and np.isfinite(result).all(), name + ' must be nonempty and finite')
    return result


def constant_velocity(q, dq, horizon, period_s):
    q, dq = finite(q, 'q'), finite(dq, 'dq')
    need(q.ndim == 2 and q.shape == dq.shape and horizon > 0 and period_s > 0,
         'Invalid constant velocity dimensions/period')
    return q[:, None, :] + dq[:, None, :] * np.arange(1, horizon + 1)[None, :, None] * period_s


def _error_summary(error):
    absolute = np.abs(error)
    return dict(mae_rad=float(absolute.mean()), rmse_rad=float(np.sqrt(np.mean(error**2))),
                median_absolute_rad=float(np.median(absolute)),
                median_sample_mae_rad=float(np.median(absolute.reshape(len(error), -1).mean(axis=1))),
                per_joint_mae_rad=absolute.mean(axis=tuple(range(error.ndim-1))).tolist(),
                per_joint_median_absolute_rad=np.median(absolute.reshape(-1, error.shape[-1]), axis=0).tolist())


def _error_groups(error, phases, period_s):
    result = dict(full_chunk=_error_summary(error), h0=_error_summary(error[:, :1]),
                  per_horizon=[], by_phase={})
    for h in range(error.shape[1]):
        result['per_horizon'].append(dict(horizon_index=h, offset_s=round((h+1)*period_s, 10),
                                          **_error_summary(error[:, h:h+1])))
    for phase in sorted(set(phases)):
        selected = error[np.asarray(phases) == phase]
        result['by_phase'][str(phase)] = dict(samples=len(selected), full_chunk=_error_summary(selected),
            h0=_error_summary(selected[:, :1]), per_horizon=[dict(horizon_index=h,
                offset_s=round((h+1)*period_s, 10), **_error_summary(selected[:, h:h+1]))
                for h in range(error.shape[1])])
    return result


def chunk_metrics(prediction, reference, measured_q, phases, period_s):
    prediction, reference = finite(prediction, 'prediction'), finite(reference, 'reference')
    q = finite(measured_q, 'measured_q')
    need(prediction.ndim == 3 and prediction.shape == reference.shape and
         q.shape == (len(reference), reference.shape[2]) and len(phases) == len(reference),
         'Prediction/reference/anchor/phase shapes differ')
    absolute = _error_groups(prediction-reference, phases, period_s)
    # Subtracting the same measured anchor leaves the error unchanged. Report
    # both explicitly; step-to-step delta below is a distinct diagnostic.
    step_pred = np.diff(np.concatenate([q[:, None], prediction], axis=1), axis=1)
    step_ref = np.diff(np.concatenate([q[:, None], reference], axis=1), axis=1)
    return dict(absolute=absolute, delta_from_observation=absolute,
                step_delta=_error_groups(step_pred-step_ref, phases, period_s),
                delta_definition='prediction[h]-q_observed versus reference[h]-q_observed; errors equal absolute errors',
                step_delta_definition='h0 relative to observed q; later horizons relative to preceding predicted/reference command')


def dimension_audit(raw, encoded, *, clamp_applied, buckets=256, near_constant_span=1e-8):
    """Quantiles of physical/latent values, and the *actual* encoded pipeline.

    Quantile clipping is explicitly counterfactual: statistics here are fitted
    to this diagnostic subset, never used to assert the training normalizer.
    Action histograms are hypothetical because continuous actions are not
    tokenized; the caller marks state histograms actual after packing checks.
    """
    raw, encoded = finite(raw, 'raw audit values'), finite(encoded, 'encoded audit values')
    need(raw.shape == encoded.shape and raw.ndim == 2 and buckets == 256, 'Invalid audit shape/buckets')
    rows = []
    for j in range(raw.shape[1]):
        x, z = raw[:, j], encoded[:, j]
        q01, q99 = np.quantile(x, [.01, .99])
        near = bool(q99-q01 <= near_constant_span)
        # Native processor operates on float32 before rounding (including ties).
        bins = np.rint((np.clip(z.astype(np.float32), -1, 1)+1)/2*255).astype(np.int64)
        outside = float(np.mean(np.abs(z) > 1))
        rows.append(dict(dimension=j, count=len(x), min=float(x.min()), max=float(x.max()),
            q01=float(q01), q99=float(q99), quantile_span=float(q99-q01), range=float(np.ptp(x)),
            near_constant=near, encoded_min=float(z.min()), encoded_max=float(z.max()),
            encoded_outside_unit_fraction=outside,
            actual_clipped_fraction=outside if clamp_applied else 0.0,
            actual_clamp_applied=bool(clamp_applied),
            counterfactual_quantile_clipped_fraction=None if near else float(np.mean((x < q01) | (x > q99))),
            counterfactual_quantiles_fitted_on='this diagnostic subset only; not the loaded training normalizer',
            occupied_buckets_256=int(len(np.unique(bins))), bucket_histogram_256=np.bincount(bins, minlength=256).tolist(),
            bucket_histogram_is_actual=False))
    return rows


def audit_temporal(rows, times, period_s):
    times = finite(times, 'times')
    need(times.ndim == 2 and len(rows) == len(times) and np.all(times >= 0), 'Invalid timestamp shape')
    need(np.allclose(np.diff(times, axis=1), period_s, atol=1e-7, rtol=0), 'Nonuniform cached observation timestamps')
    horizon = times.shape[1]
    seen = {}
    for index, row in enumerate(rows):
        need(row['sample_index'] == index and row['split'] == 'train', 'Row order/split changed')
        need(row['label_indices'] == list(range(row['start'], row['start']+horizon)), 'Original label indices changed')
        pairs = row['temporal_label_pairs']
        need(len(pairs) == horizon, 'Incomplete temporal label pairs')
        for h, pair in enumerate(pairs):
            lo, hi = pair['physics_interval_inclusive']
            need(pair['following_frame_index'] == pair['frame_index']+1 and hi-lo == 12
                 and pair['command_physics_row'] == hi, 'Invalid following-frame command mapping')
            if h:
                need(pair['frame_index'] == pairs[h-1]['following_frame_index'] and
                     lo == pairs[h-1]['command_physics_row'], 'Discontinuous original temporal mapping')
            key = (row['episode_id'], row['label_indices'][h])
            if key in seen:
                need(seen[key] == pair, 'Overlapping cached labels disagree')
            seen[key] = pair
    return dict(status='cached_original_pair_mapping_consistent', observations=len(rows),
        label_pairs=len(rows)*horizon, period_s=period_s, physics_steps_per_command=12,
        command_offsets_s=[round((h+1)*period_s, 10) for h in range(horizon)],
        cache_times_semantics='observation frame times; command time = times[h] + 0.05 s',
        action_formula='expert_trace[label_pair.command_physics_row].q_command',
        historical_qualification_evidence=True, independent_direct_trace_validation=False,
        limitation='本轮核对已绑定缓存中的原始索引和时间映射，未重新读取 frames/expert_trace 做独立逐值复验。')


def seed_variability(predictions, reference, measured_q):
    predictions, reference = finite(predictions, 'seed predictions'), finite(reference, 'reference')
    q = finite(measured_q, 'measured_q')
    need(predictions.ndim == 4 and predictions.shape[:2] == (20, 16), 'Expected 20 observations and 16 seeds')
    need(reference.shape == (20, predictions.shape[2], predictions.shape[3]) and
         q.shape == (20, predictions.shape[3]), 'Seed reference/anchor shapes differ')
    std = predictions[:, :, 0].std(axis=1, ddof=0)
    increment = np.abs(reference[:, 0]-q)
    ratio = np.divide(std, increment, out=np.zeros_like(std), where=increment > 0)
    return dict(status='available', observations=20, seeds=16, std_ddof=0, units='rad',
        per_observation_h0_std_rad=std.tolist(), per_observation_true_increment_rad=increment.tolist(),
        per_observation_std_to_true_increment=[[float(ratio[i,j]) if increment[i,j] > 0 else None
            for j in range(std.shape[1])] for i in range(20)],
        per_joint_mean_std_rad=std.mean(axis=0).tolist(), per_joint_median_std_rad=np.median(std, axis=0).tolist(),
        zero_true_increment_count=(increment == 0).sum(axis=0).tolist(),
        per_joint_std_ge_true_increment_fraction=(std >= increment).mean(axis=0).tolist(),
        per_joint_median_ratio_nonzero_increment=[float(np.median(ratio[increment[:,j] > 0,j]))
            if np.any(increment[:,j] > 0) else None for j in range(std.shape[1])])


def load_predictions(path, cached):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    need(metadata['array_sha256'] == sha256(path), 'Prediction shard hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in ('raw','normalized','reference','normalized_reference','hold','measured')}
    for key, array in arrays.items():
        need(fingerprint(array) == metadata['array_fingerprints'][key], 'Prediction fingerprint mismatch: '+key)
        finite(array, key)
    for key in ('reference','normalized_reference','hold'):
        need(np.array_equal(arrays[key], cached[key]), key+' differs from bound cache')
    need(np.array_equal(arrays['measured'], cached['state'][:, :7]), 'Measured anchor differs from bound cache')
    need(metadata['ode_steps'] == 10, 'Expected 10 ODE inference steps')
    need(arrays['raw'].shape == cached['reference'].shape and arrays['normalized'].shape == arrays['raw'].shape,
         'Predicted chunk shape differs')
    return arrays, metadata


def _load_adapter(path):
    spec = importlib.util.spec_from_file_location('_ausc_diagnostic_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # This module's top level imports NumPy only.
    return module


def _pipeline_audit(data, adapter, config, packing, processor_path):
    cfg = config['config']
    need(cfg['normalization_mapping'] == {k:'IDENTITY' for k in ('ACTION','STATE','VISUAL')}, 'Unsupported actual normalization mapping')
    need(cfg['num_state_tokens'] == 256 and cfg['action_mode'] == 'continuous', 'Actual native token/action mode differs')
    need(config['normalization'] == data.preparation['normalization'], 'Loaded normalization differs from prepared cache')
    freeze = json.loads((Path(adapter.__file__).parent/'SOURCE_FREEZE.json').read_text())
    expected = freeze['public_lerobot']['src/lerobot/policies/molmoact2/processor_molmoact2.py']
    need(sha256(processor_path) == expected, 'Native processor source hash differs')
    need(config['dependency_sha256']['adapter.py'] == sha256(adapter.__file__), 'Run used a different adapter')
    for key in ('state','reference','normalized_reference','hold','times','schedule'):
        need(config['input_arrays'][key] == fingerprint(data.arrays[key]), 'Run input array changed: '+key)
    encoded = np.stack([data.row(i)['state'] for i in range(len(data.rows))])
    need(len(packing) == len(data.rows), 'Packing observation count differs')
    offsets = set()
    for i, record in enumerate(packing):
        bins = np.rint((np.clip(encoded[i], -1, 1)+1)/2*255).astype(np.int64)
        for mode in ('prediction','training'):
            own = record[mode]
            need(own['sample_index'] == i and own['native_processor'] and not own['truncation'], 'Native packing identity/truncation differs')
            need(own['tensors']['observation.state'] == fingerprint(encoded[i:i+1]), 'Actual packed state differs')
            need(len(own['state_token_ids']) == 43, 'Expected 41 state buckets and two delimiters')
            offsets.update((np.asarray(own['state_token_ids'][1:-1])-bins).tolist())
        padded = np.zeros((1, 10, 32), np.float32)
        padded[0, :, :7] = data.arrays['normalized_reference'][i]
        need(record['training']['tensors']['action'] == fingerprint(padded), 'Actual action packing modified label values')
    need(len(offsets) == 1, 'Native 256-bucket IDs disagree with float32 round-to-nearest recipe')
    state_rows = dimension_audit(data.arrays['state'], encoded, clamp_applied=True)
    for row in state_rows:
        row['bucket_histogram_is_actual'] = True
    residual = np.stack([data.action_codec.denormalize_chunk(data.arrays['normalized_reference'][i],
        row['phase'], row['target'], data.stats) for i,row in enumerate(data.rows)])
    action_rows = dimension_audit(residual.reshape(-1, 7), data.arrays['normalized_reference'].reshape(-1, 7), clamp_applied=False)
    action_by_phase_horizon = []
    phases = np.asarray([r['phase'] for r in data.rows])
    for phase in sorted(set(phases)):
        for h in range(10):
            for row in dimension_audit(residual[phases == phase, h], data.arrays['normalized_reference'][phases == phase, h], clamp_applied=False):
                row.update(phase=str(phase), horizon_index=h)
                action_by_phase_horizon.append(row)
    return dict(actual_pipeline='frozen physical/asinh StateCodec -> native packer with IDENTITY; V11 bounded latent residual action codec -> unchanged continuous labels',
        generic_normalizer_and_clamp_bypassed=True,
        state_tokenizer_clamp_applied=True, state_codec_outside_domain='reject',
        state_codec_bins=4096, actual_native_state_buckets=256,
        state_tokenizer_base_id=next(iter(offsets)),
        action_clamp_applied=False, action_hard_projection_applied=False,
        action_bound_mapping='V11 softplus interval map; smooth saturation is not hard clipping',
        actual_training_normalization_source='configuration.json normalization, exact equality with preparation.json checked',
        counterfactual_note='子集 q01/q99 仅作反事实量程审计；动作桶占用是假设离散化，不是连续动作训练所用token。',
        state=state_rows, action_latent=action_rows, action_latent_by_phase_horizon=action_by_phase_horizon)


def _report_zh(report):
    lines = ['# M0 离线诊断', '', '范围：已绑定 TRAIN32 训练子集；不代表闭环、泛化或临床有效性。', '',
             '| 方法 | 整块 MAE (rad) | 整块逐样本 MAE 中位数 (rad) | 首步 MAE (rad) |',
             '|---|---:|---:|---:|']
    for name, result in report['metrics'].items():
        full, h0 = result['absolute']['full_chunk'], result['absolute']['h0']
        lines.append(f"| {name} | {full['mae_rad']:.8g} | {full['median_sample_mae_rad']:.8g} | {h0['mae_rad']:.8g} |")
    lines += ['', '绝对动作与相对观测 q 的 delta 误差数学上相同；另外导出逐步指令增量误差。分阶段、每个视界与每关节指标见 JSON/CSV。',
        '', '时间对齐：已核对缓存保存的原始 label index、相邻帧与 12 个物理步映射。第 0 项是 +50 ms 指令，最后一项是 +500 ms；缓存 times 是观测帧时间。历史 qualification 证据保留，本轮未独立重读原始 trace。',
        '', '实际归一化：冻结 physical/asinh StateCodec + native 256 桶；native adapter 配置为 IDENTITY，绕过 generic normalizer/clamp。状态 token 函数含 clamp，实际改变比例由表中数值报告。动作使用 V11 分阶段有界 latent 编码，连续标签未硬裁剪。子集分位数裁剪比例属于反事实。']
    comparisons = report['baseline_comparisons']
    for name, values in comparisons.items():
        lines.append(f"\n{name}：整块逐样本 MAE 中位数 / hold = {values['median_ratio_to_hold']:.5g}，/ 匀速 = {values['median_ratio_to_constant_velocity']:.5g}。这只是点估计，未定义“明显优于”的统计门。")
    lines += ['', '20 观测 × 16 种子：' + ('已读取真实预测并导出首步逐关节标准差/真实运动比。' if report['seed_variability']['status'] == 'available' else '尚无输入预测，不填造结果。'),
        '', '**M0 尚未完成。** 交接区尚未测量，不能以关节误差伪造 TCP 位置容差通过；状态输入 ACT/DP 的训练场景闭环复现仍是 M0 退出条件。',
        '', '复现信息：配置、当前诊断代码、输入 shard 与实际依赖源码哈希见 JSON；没有全量逐文件审计。']
    return '\n'.join(lines)+'\n'


def _write_exports(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    metric_rows = []
    for model, values in report['metrics'].items():
        for representation in ('absolute','delta_from_observation','step_delta'):
            result = values[representation]
            groups = [('ALL', result)] + list(result['by_phase'].items())
            for phase, group in groups:
                slices = [('full_chunk', group['full_chunk']), ('h0', group['h0'])]
                slices += [(f"h{h['horizon_index']}", h) for h in group['per_horizon']]
                for horizon, metrics in slices:
                    metric_rows.append(dict(model=model, representation=representation, phase=phase, horizon=horizon,
                        mae_rad=metrics['mae_rad'], rmse_rad=metrics['rmse_rad'], median_absolute_rad=metrics['median_absolute_rad'],
                        median_sample_mae_rad=metrics['median_sample_mae_rad']))
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    audit_rows = []
    for group in ('state','action_latent','action_latent_by_phase_horizon'):
        for row in report['normalization_audit'][group]:
            audit_rows.append(dict(group=group, phase=row.get('phase','ALL'), horizon_index=row.get('horizon_index','ALL'),
                                  **{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items() if k not in ('phase','horizon_index')}))
    with (output/'normalization.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0])); writer.writeheader(); writer.writerows(audit_rows)
    (output/'REPORT_zh.md').write_text(_report_zh(report))


def run(config_path, *, root=None, output=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    resolve = lambda value: root/Path(value)
    adapter_path = resolve(config['adapter_source'])
    adapter = _load_adapter(adapter_path)
    data = adapter.load_prepared(resolve(config['prepared_directory']), resolve(config['frozen_runtime']))
    arrays, rows = data.arrays, data.rows
    temporal = audit_temporal(rows, arrays['times'], config['period_s'])
    # The export source is bound by preparation; inspect just this relevant file.
    exporter = resolve(config['label_exporter_source'])
    binding = [v for k,v in data.preparation['source_bindings'].items() if k.endswith('/sim/auscultation_pi05_v6/export_joint_dataset.py')]
    need(len(binding) == 1 and binding[0] == sha256(exporter), 'Original temporal exporter source changed')
    code_files = [Path(__file__), adapter_path, adapter_path.parent/'SOURCE_FREEZE.json', exporter,
                  resolve(config['processor_source']), resolve(config['preparation_source'])]
    need(sha256(resolve(config['preparation_source'])) == data.preparation['implementation_sha256']['prepare_cpu.py'], 'Preparation source changed')
    freeze = json.loads((adapter_path.parent/'SOURCE_FREEZE.json').read_text())
    code_files += [resolve(config['frozen_runtime'])/relative for relative in freeze['runtime']]
    used_inputs = [config_path, resolve(config['prepared_directory'])/'cache.npz', resolve(config['prepared_directory'])/'preparation.json']
    q = arrays['state'][:, :7]
    phases = [r['phase'] for r in rows]
    predictions = dict(hold=arrays['hold'], constant_velocity=constant_velocity(q, arrays['state'][:, 7:14], 10, config['period_s']))
    native_metadata, missing = {}, []
    normalization = None
    for update in config['prediction_updates']:
        path = next((resolve(directory)/f'eval_{update:04d}.npz' for directory in config['prediction_directories']
                     if (resolve(directory)/f'eval_{update:04d}.npz').is_file()), None)
        if path is None:
            missing.append(update)
            continue
        predicted, metadata = load_predictions(path, arrays)
        actual_config_path, packing_path = path.parent/'configuration.json', path.parent/'packing.json'
        actual_config = json.loads(actual_config_path.read_text())
        need(metadata['prediction_seed_per_example'] == actual_config['seeds']['prediction_per_example'], 'Prediction seed differs from run configuration')
        audit = _pipeline_audit(data, adapter, actual_config, json.loads(packing_path.read_text()), resolve(config['processor_source']))
        if normalization is not None:
            need(audit == normalization, 'Prediction runs used different packing/normalization')
        normalization = audit
        for i in range(len(rows)):
            decoded = data.decode(predicted['normalized'][i], data.row(i))
            need(np.array_equal(decoded, predicted['raw'][i]), 'Raw prediction differs from frozen V11 decode')
        name = f'native_{update:04d}'
        predictions[name] = predicted['raw']
        native_metadata[name] = dict(path=str(path.relative_to(root)), update=metadata['update'],
            ode_steps=metadata['ode_steps'], prediction_seed_per_example=metadata['prediction_seed_per_example'])
        used_inputs += [path, path.with_suffix('.json'), actual_config_path, packing_path]
    need(normalization is not None, 'No actual native evidence available to establish processor behavior')
    metrics = {name:chunk_metrics(value, arrays['reference'], q, phases, config['period_s']) for name,value in predictions.items()}
    comparisons = {}
    for name in native_metadata:
        median = metrics[name]['absolute']['full_chunk']['median_sample_mae_rad']
        comparisons[name] = {f'median_ratio_to_{base}':median/metrics[base]['absolute']['full_chunk']['median_sample_mae_rad']
                             for base in ('hold','constant_velocity')}
    variability = dict(status='missing', required_shape=[20,16,10,7], reason='No multi-seed GPU prediction shard configured')
    if config.get('seed_predictions'):
        path = resolve(config['seed_predictions'])
        with np.load(path, allow_pickle=False) as archive:
            multi = {k:archive[k] for k in ('raw','reference','measured','sample_indices','seeds')}
        indices = multi['sample_indices']
        need(indices.shape == (20,) and indices.dtype.kind in 'iu' and len(set(indices.tolist())) == 20
             and np.all((indices >= 0)&(indices < len(rows))), 'Invalid seed observation IDs')
        need(multi['seeds'].shape == (16,) and multi['seeds'].dtype.kind in 'iu' and len(set(multi['seeds'].tolist())) == 16, 'Invalid noise seed IDs')
        need(np.array_equal(multi['reference'], arrays['reference'][indices]) and np.array_equal(multi['measured'], q[indices]), 'Seed observation references differ from cache')
        variability = seed_variability(multi['raw'], multi['reference'], multi['measured'])
        variability.update(sample_indices=indices.tolist(), seed_ids=multi['seeds'].tolist())
        used_inputs.append(path)
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=root, text=True, capture_output=True, check=False)
    report = dict(schema='ausc_policy_m0_offline_diagnostics_v1', scope='TRAIN32 offline pipeline evidence only',
        config_sha256=sha256(config_path), code_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        code_sha256={str(p.relative_to(root)):sha256(p) for p in code_files},
        input_shards_sha256={str(p.relative_to(root)) if p.is_relative_to(root) else str(p):sha256(p) for p in set(used_inputs)},
        original_training_shards=[dict(path=path, sha256=digest, verification='historical preparation binding; original shard not reopened')
            for path,digest in sorted({r['source_npz_path']:r['source_npz_sha256'] for r in rows}.items())],
        temporal_alignment=temporal, normalization_audit=normalization, native_predictions=native_metadata,
        missing_prediction_updates=missing, metrics=metrics, baseline_comparisons=comparisons,
        seed_variability=variability, acceptance=dict(status='not_assessed', m0_complete=False,
            capture_region='not_measured', position_error='not_computed_from_joint_error',
            required_remaining=['20 observations x 16 seeds if unavailable', 'measured capture region and TCP error', 'state-only ACT/DP training-scene closed-loop reproduction']))
    output = Path(output) if output else resolve(config['output_directory'])
    _write_exports(report, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ausc_policy/m0_diagnostics.json'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = run(args.config, root=args.root, output=args.output)
    print(json.dumps(dict(status='offline_diagnostics_exported', m0_complete=False,
        native_updates=[v['update'] for v in report['native_predictions'].values()],
        missing_updates=report['missing_prediction_updates']), ensure_ascii=False))


if __name__ == '__main__':
    main()

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
                reference_command_measured_gap=_error_groups(reference-q[:, None], phases, period_s),
                predicted_command_measured_gap=_error_groups(prediction-q[:, None], phases, period_s),
                command_measured_gap_definition='future command minus current measured q; includes servo tracking error and is not measured single-step motion',
                delta_definition='prediction[h]-q_observed versus reference[h]-q_observed; errors equal absolute errors',
                step_delta_definition='mixed differences: h0 command minus current measured q; h>0 command minus preceding command; neither establishes measured trajectory motion')


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


def _std_signal_comparison(std, signal):
    """Compare a sampling std with a named nonnegative signal, not a motion label."""
    ratio = np.divide(std, signal, out=np.zeros_like(std), where=signal > 0)
    return dict(per_observation_signal_rad=signal.tolist(),
        per_observation_std_to_signal_ratio=[[float(ratio[i,j]) if signal[i,j] > 0 else None
            for j in range(std.shape[1])] for i in range(len(std))],
        per_joint_zero_signal_count=(signal == 0).sum(axis=0).tolist(),
        per_joint_std_ge_signal_fraction=(std >= signal).mean(axis=0).tolist(),
        std_ge_fraction_includes_zero_signal=True,
        per_joint_median_ratio_nonzero_signal=[float(np.median(ratio[signal[:,j] > 0,j]))
            if np.any(signal[:,j] > 0) else None for j in range(std.shape[1])])


def seed_variability(predictions, reference, measured_q, *, measured_dq=None, period_s=.05):
    """Inputs are decoded radians; dq, when present, is measured rad/s.

    reference[:, 0] - measured_q mixes a future command with a current measured
    state. It includes present tracking lag and cannot substitute for either
    command(t+dt)-command(t) or measured_q(t+dt)-measured_q(t). dq*dt is only a
    local constant-velocity estimate; those two observed trace differences
    require independent trace evidence not contained in this cache.
    """
    predictions, reference = finite(predictions, 'seed predictions'), finite(reference, 'reference')
    q = finite(measured_q, 'measured_q')
    need(np.isfinite(period_s) and period_s > 0, 'Invalid seed diagnostic period')
    need(predictions.ndim == 4 and predictions.shape[:2] == (20, 16), 'Expected 20 observations and 16 seeds')
    need(reference.shape == (20, predictions.shape[2], predictions.shape[3]) and
         q.shape == (20, predictions.shape[3]), 'Seed reference/anchor shapes differ')
    std = predictions[:, :, 0].std(axis=1, ddof=0)
    gap = _std_signal_comparison(std, np.abs(reference[:, 0]-q))
    gap.update(signal_definition='abs(q_command(t+period_s)-q_measured(t))',
        contains_servo_tracking_gap=True, is_observed_trace_increment=False)
    result = dict(status='available', observations=20, seeds=16, std_ddof=0, units='rad', period_s=period_s,
        per_observation_h0_std_rad=std.tolist(), command_measured_gap_comparison=gap,
        command_measured_gap_is_observed_single_step_motion=False,
        per_joint_mean_std_rad=std.mean(axis=0).tolist(), per_joint_median_std_rad=np.median(std, axis=0).tolist(),
        direct_trace_increment=dict(status='unavailable',
            required=['q_command(t+period_s)-q_command(t)', 'q_measured(t+period_s)-q_measured(t)'],
            reason='The bound observation cache has no independently validated current command or next measured q.'),
        local_velocity_estimate_comparison=dict(status='unavailable', reason='No measured dq supplied'),
        interpretation='Ratios to the command/measured gap or dq*dt are descriptive comparisons, not proof that seed noise exceeds observed motion or causes closed-loop failure.')
    if measured_dq is not None:
        dq = finite(measured_dq, 'measured_dq')
        need(dq.shape == q.shape, 'Seed measured dq shape differs from measured_q')
        estimate = _std_signal_comparison(std, np.abs(dq*period_s))
        estimate.update(status='available', signal_definition='abs(qdot_measured(t) * period_s)',
            is_estimate=True, is_observed_trace_increment=False,
            limitation='Local constant-velocity estimate; acceleration and tracking dynamics are not observed over the next interval.')
        result['local_velocity_estimate_comparison'] = estimate
    return result


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
    need(cfg['num_state_tokens'] == 256 and cfg['action_mode'] == 'continuous'
         and cfg['num_inference_steps'] == 10 and cfg['chunk_size'] == 10, 'Actual native token/action/inference mode differs')
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
        j = row['dimension']
        row.update(loaded_normalization_low=float(data.codec.low[j]), loaded_normalization_high=float(data.codec.high[j]),
                   loaded_asinh_scale=float(data.codec.scale[j]), loaded_asinh_applied=bool(data.codec.nonlinear[j]))
    residual = np.stack([data.action_codec.denormalize_chunk(data.arrays['normalized_reference'][i],
        row['phase'], row['target'], data.stats) for i,row in enumerate(data.rows)])
    action_rows = dimension_audit(residual.reshape(-1, 7), data.arrays['normalized_reference'].reshape(-1, 7), clamp_applied=False)
    physical_rows = dimension_audit(data.arrays['reference'].reshape(-1, 7),
        data.arrays['normalized_reference'].reshape(-1, 7), clamp_applied=False)
    action_by_phase_horizon = []
    phases = np.asarray([r['phase'] for r in data.rows])
    for phase in sorted(set(phases)):
        for h in range(10):
            for row in dimension_audit(residual[phases == phase, h], data.arrays['normalized_reference'][phases == phase, h], clamp_applied=False):
                row.update(phase=str(phase), horizon_index=h)
                j = row['dimension']
                own = data.stats['actions_by_phase'][phase]
                for key in ('low','high','quantile_low','quantile_high'):
                    values = np.asarray(own[key])
                    row['loaded_'+key] = float(values[h,j] if values.ndim == 2 else values[j])
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
        state=state_rows, action_physical=physical_rows, action_latent=action_rows, action_latent_by_phase_horizon=action_by_phase_horizon)


def _report_zh(report):
    lines = ['# M0 离线诊断', '', '范围：已绑定 TRAIN32 训练子集；不代表闭环、泛化或临床有效性。', '',
             '| 方法 | 整块 MAE (rad) | 元素绝对误差中位数 (rad) | 逐样本整块 MAE 中位数 (rad) | 首步 MAE (rad) |',
             '|---|---:|---:|---:|---:|']
    for name, result in report['metrics'].items():
        full, h0 = result['absolute']['full_chunk'], result['absolute']['h0']
        lines.append(f"| {name} | {full['mae_rad']:.8g} | {full['median_absolute_rad']:.8g} | {full['median_sample_mae_rad']:.8g} | {h0['mae_rad']:.8g} |")
    lines += ['', '![整块拟合与基线曲线](fit_vs_baselines.svg)',
        '', '绝对动作与相对观测 q 的 delta 误差数学上相同；另外导出参考/预测的“未来指令－当前实测 q”差值，以及混合差分误差（h0 是指令－实测，后续是相邻指令－指令）。这些量均不等于实测轨迹运动。分阶段、每个视界与每关节指标见 JSON/CSV。',
        '', '时间对齐：已核对缓存保存的原始 label index、相邻帧与 12 个物理步映射。第 0 项是 +50 ms 指令，最后一项是 +500 ms；缓存 times 是观测帧时间。历史 qualification 证据保留，本轮未独立重读原始 trace。',
        '', '实际归一化：冻结 physical/asinh StateCodec + native 256 桶；native adapter 配置为 IDENTITY，绕过 generic normalizer/clamp。状态 token 函数含 clamp，实际改变比例由表中数值报告。动作使用 V11 分阶段有界 latent 编码，连续标签未硬裁剪。子集分位数裁剪比例属于反事实。']
    audit = report['normalization_audit']
    lines.append('\n本批状态实际被 clamp 改变的比例：' + f"{np.mean([r['actual_clipped_fraction'] for r in audit['state']]):.3%}" +
                 '。近常数状态维（0-based）：' + str([r['dimension'] for r in audit['state'] if r['near_constant']]) + '。')
    comparisons = report['baseline_comparisons']
    if comparisons:
        name = next(reversed(comparisons))
        values = comparisons[name]
        lines.append(f"\n{name}：整块逐样本 MAE 中位数 / hold = {values['median_ratio_to_hold']:.5g}，/ 匀速 = {values['median_ratio_to_constant_velocity']:.5g}。元素绝对误差中位数 / hold = {values['element_median_ratio_to_hold']:.5g}，/ 匀速 = {values['element_median_ratio_to_constant_velocity']:.5g}。比值小于 1 表示点估计较低；未定义“明显优于”的统计门。")
        lines.append('FREE 阶段整块逐样本 MAE 中位数 (rad)：' + '，'.join(f"{key} = {report['metrics'][key]['absolute']['by_phase']['FREE']['full_chunk']['median_sample_mae_rad']:.8g}" for key in ('hold','constant_velocity',name))+'。')
    lines += ['', '20 观测 × 16 种子：' + ('已读取真实预测，导出首步逐关节标准差及其相对“未来指令－当前实测 q”差值的比值。' if report['seed_variability']['status'] == 'available' else '尚无输入预测，不填造结果。'),
        '', '种子诊断的分母口径：reference[:, 0] − measured 是 +50 ms 指令与当前实测关节之差，包含伺服跟踪滞后，不能称为真实单步运动。缓存 dq × 0.05 s 另列为局部匀速估计，也不是原始轨迹增量。原始 command→command 和 measured→measured 的单步增量尚需 trace 独立核验；不能据此宣称种子噪声导致闭环失败。',
        '', '**M0 尚未完成。** 交接区尚未测量，不能以关节误差伪造 TCP 位置容差通过；状态输入 ACT/DP 的训练场景闭环复现仍是 M0 退出条件。',
        '', '复现信息：配置、当前诊断代码、输入 shard 与实际依赖源码哈希见 JSON；没有全量逐文件审计。']
    return '\n'.join(lines)+'\n'


def _write_curve(report, output):
    """Dependency-free SVG: the two median/mean estimands are kept separate."""
    native = [(own['update'], name) for name,own in report['native_predictions'].items()]
    native.sort()
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="430" viewBox="0 0 1000 430">',
           '<rect width="1000" height="430" fill="#fbfcfe"/>',
           '<g font-family="Arial,sans-serif" fill="#172c45">',
           '<text x="38" y="30" font-size="18" font-weight="bold">TRAIN32 full-chunk error vs. baselines</text>',
           '<text x="38" y="51" font-size="12">Physical joint radians; 10 horizons, +50 to +500 ms. Log y axis. Lower is better.</text>']
    max_update = max(u for u,_ in native) or 1
    colors = {'native':'#1166b2', 'hold':'#8c4a13', 'constant_velocity':'#188474'}
    for panel,(metric,title) in enumerate([('mae_rad','Mean absolute error'), ('median_sample_mae_rad','Median of per-observation chunk MAE')]):
        left, top, width, height = 75+panel*490, 103, 395, 237
        getter = lambda name: report['metrics'][name]['absolute']['full_chunk'][metric]
        values = [getter(name) for _,name in native] + [getter(k) for k in ('hold','constant_velocity')]
        low = np.floor(np.log10(min(values)))-.15
        high = np.ceil(np.log10(max(values)))+.15
        x = lambda update: left+width*update/max_update
        y = lambda value: top+height*(high-np.log10(value))/(high-low)
        svg.append(f'<text x="{left}" y="84" font-size="14" font-weight="bold">{title}</text>')
        for exponent in range(int(np.ceil(low)), int(np.floor(high))+1):
            yy = y(10.**exponent)
            svg += [f'<line x1="{left}" y1="{yy}" x2="{left+width}" y2="{yy}" stroke="#dae1e9"/>',
                    f'<text x="{left-9}" y="{yy+4}" text-anchor="end" font-size="11">1e{exponent}</text>']
        svg.append(f'<path d="M {left} {top} V {top+height} H {left+width}" fill="none" stroke="#8696a8"/>')
        for update,_ in native:
            svg.append(f'<text x="{x(update)}" y="{top+height+18}" text-anchor="middle" font-size="11">{update}</text>')
        svg.append(f'<text x="{left+width/2}" y="{top+height+37}" text-anchor="middle" font-size="11">Training update</text>')
        for baseline in ('hold','constant_velocity'):
            yy = y(getter(baseline))
            svg.append(f'<line x1="{left}" y1="{yy}" x2="{left+width}" y2="{yy}" stroke="{colors[baseline]}" stroke-width="2" stroke-dasharray="6 4"/>')
        points = ' '.join(f'{x(update)},{y(getter(name))}' for update,name in native)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colors["native"]}" stroke-width="3"/>')
        for update,name in native:
            svg.append(f'<circle cx="{x(update)}" cy="{y(getter(name))}" r="4" fill="{colors["native"]}"/>')
    for i,(key,label) in enumerate([('native','Native MolmoAct2'),('hold','Hold'),('constant_velocity','q + dq * (h+1) * 0.05')]):
        left = 100+i*295
        svg += [f'<line x1="{left}" y1="409" x2="{left+25}" y2="409" stroke="{colors[key]}" stroke-width="3"/>',
                f'<text x="{left+33}" y="413" font-size="12">{label}</text>']
    svg.append('</g></svg>')
    (output/'fit_vs_baselines.svg').write_text('\n'.join(svg)+'\n')


def _write_exports(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    metric_rows = []
    for model, values in report['metrics'].items():
        for representation in ('absolute','delta_from_observation','step_delta','reference_command_measured_gap','predicted_command_measured_gap'):
            result = values[representation]
            groups = [('ALL', result)] + list(result['by_phase'].items())
            for phase, group in groups:
                slices = [('full_chunk', group['full_chunk'])]
                slices += [(f"h{h['horizon_index']}", h) for h in group['per_horizon']]
                for horizon, metrics in slices:
                    metric_rows.append(dict(model=model, representation=representation, phase=phase, horizon=horizon,
                        mae_rad=metrics['mae_rad'], rmse_rad=metrics['rmse_rad'], median_absolute_rad=metrics['median_absolute_rad'],
                        median_sample_mae_rad=metrics['median_sample_mae_rad'],
                        **{f'joint_{j}_mae_rad':value for j,value in enumerate(metrics['per_joint_mae_rad'])}))
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    audit_rows = []
    for group in ('state','action_physical','action_latent','action_latent_by_phase_horizon'):
        for row in report['normalization_audit'][group]:
            audit_rows.append(dict(group=group, phase=row.get('phase','ALL'), horizon_index=row.get('horizon_index','ALL'),
                                  **{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items() if k not in ('phase','horizon_index')}))
    with (output/'normalization.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in audit_rows for k in row)))
        writer.writeheader(); writer.writerows(audit_rows)
    variability = report['seed_variability']
    if variability['status'] == 'available':
        seed_rows = []
        gap = variability['command_measured_gap_comparison']
        estimate = variability['local_velocity_estimate_comparison']
        for i, sample in enumerate(variability['sample_indices']):
            for j, std in enumerate(variability['per_observation_h0_std_rad'][i]):
                seed_rows.append(dict(sample_index=sample, joint_index=j, h0_std_rad=std,
                    h0_command_measured_gap_rad=gap['per_observation_signal_rad'][i][j],
                    std_to_command_measured_gap=gap['per_observation_std_to_signal_ratio'][i][j],
                    local_velocity_estimate_abs_rad=estimate['per_observation_signal_rad'][i][j],
                    std_to_local_velocity_estimate=estimate['per_observation_std_to_signal_ratio'][i][j],
                    observed_trace_increment_available=False))
        with (output/'seed_variability.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0])); writer.writeheader(); writer.writerows(seed_rows)
    (output/'REPORT.md').write_text(_report_zh(report))
    _write_curve(report, output)


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
        need(metadata['update'] == update, 'Evaluation metadata update differs from requested checkpoint')
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
        comparisons[name].update({f'element_median_ratio_to_{base}':metrics[name]['absolute']['full_chunk']['median_absolute_rad']/metrics[base]['absolute']['full_chunk']['median_absolute_rad']
                                  for base in ('hold','constant_velocity')})
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
        variability = seed_variability(multi['raw'], multi['reference'], multi['measured'],
                                       measured_dq=arrays['state'][indices, 7:14], period_s=config['period_s'])
        variability.update(sample_indices=indices.tolist(), seed_ids=multi['seeds'].tolist())
        used_inputs.append(path)
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=root, text=True, capture_output=True, check=False)
    report = dict(schema='ausc_policy_m0_offline_diagnostics_v2', scope='TRAIN32 offline pipeline evidence only',
        config_sha256=sha256(config_path), code_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        code_sha256={str(p.relative_to(root)):sha256(p) for p in code_files},
        input_shards_sha256={str(p.relative_to(root)) if p.is_relative_to(root) else str(p):sha256(p) for p in set(used_inputs)},
        original_training_shards=[dict(path=path, sha256=digest, verification='historical preparation binding; original shard not reopened')
            for path,digest in sorted({r['source_npz_path']:r['source_npz_sha256'] for r in rows}.items())],
        temporal_alignment=temporal, normalization_audit=normalization, native_predictions=native_metadata,
        missing_prediction_updates=missing, metrics=metrics, baseline_comparisons=comparisons,
        seed_variability=variability, acceptance=dict(status='not_assessed', m0_complete=False,
            capture_region='not_measured', position_error='not_computed_from_joint_error',
            required_remaining=['20 observations x 16 seeds if unavailable', 'independent command/measured trace increments', 'measured capture region and TCP error', 'state-only ACT/DP training-scene closed-loop reproduction']))
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

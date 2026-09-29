"""Summarize mirrored D0 evidence without changing or excusing the original audits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = args.evidence.resolve()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('Invalid run ID')
    manifest = json.loads((root/'MIRROR_SHA256.json').read_text())
    for name, expected in manifest.items():
        path = root/name
        if root not in path.resolve().parents or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Mirror binding failed: '+name)
    run = root/args.run_id
    case = run/'output/case'
    outcome = json.loads((case/'outcome.json').read_text())
    result = json.loads((case/'result.json').read_text())
    trace = [json.loads(line) for line in (case/'expert_trace.jsonl').read_text().splitlines()]
    commands = json.loads((case/'control_commands.json').read_text())
    times = np.array([row['time_s'] for row in trace])
    if not np.allclose(np.diff(times), 1/240, rtol=0, atol=1e-8):
        raise ValueError('Nonconsecutive physics clock')
    # A rejected, never-executed command is retained in the original evidence.
    # Check positive-length coverage separately, never modify its audit verdict.
    cursor = 0
    zero = []
    for i, row in enumerate(commands):
        start, end = row['physics_row'], row['physics_row_end_exclusive']
        if start != cursor or end < start or end-start != row['executed_physics_steps']:
            raise ValueError('Actual command coverage mismatch')
        if start == end:
            zero.append(i)
        cursor = end
    if cursor != len(trace):
        raise ValueError('Uncovered physics rows')
    queries, actions = outcome['l2_queries'], outcome['l2_actions']
    errors, labels, phases = [], [], []
    for action in actions:
        if action['completed']:
            end = action['physics_row_end_exclusive']-1
            if abs(times[end]-action['label_time_s']) > 1e-8:
                raise ValueError('Action clock differs from actual execution')
            errors.append(np.asarray(trace[end]['q_command'])-action['proposal'])
            labels.append(action['label_time_s'])
            phases.append(action['phase'])
    errors = np.asarray(errors)
    phase_events = [e for e in outcome['prefix_events'] if e['reason']=='phase_changed']
    last = trace[-1]
    stop = ('next_command_precheck' if zero == [len(commands)-1]
            and outcome['command_collision_samples']==outcome['observed_collision_samples']==len(trace)
            and 'command/observation mesh collision rejected' in outcome['failure_reason']
            else 'not_disambiguated')
    dynamics = json.loads((case/'measured_dynamics.json').read_text())
    summary = dict(run_id=args.run_id, source_commit=json.loads((run/'run.json').read_text())['code_commit'],
        source_manifest_sha256=digest(root/'MIRROR_SHA256.json'), verified_mirror_files=len(manifest),
        source_sha256={name:digest(case/name) for name in ('expert_trace.jsonl','control_commands.json','outcome.json','result.json')},
        execution_completed=result['execution_completed'], case_passed=result['case_passed'], m0_passed=False,
        physics_steps=len(trace), simulation_end_s=float(times[-1]), policy_queries=len(queries),
        completed_actions=len(errors), attempted_actions=len(actions),
        prefix_completed_counts=dict(Counter(a['prefix_index'] for a in actions if a['completed'])),
        phase_changes=phase_events, handoffs=outcome['d0_handoffs'], stop_location=stop,
        zero_execution_command_indices=zero, executed_command_coverage_contiguous=True,
        audit_verdict_unchanged=True,
        final=dict(gap_mm=1000*last['gap_m'], tangent_mm=1000*last['tangent_m'], axis_angle_deg=last['angle_deg'],
            hall_N=last['hall_N'], hold_s=last['hold_elapsed_s']),
        max_hall_N=max(r['hall_N'] for r in trace), max_forbidden_N=max(r['forbidden_N'] for r in trace),
        maximum_elbow_z_m=max(r['elbow_z_m'] for r in trace),
        observed_centered_jerk_rad_s3=dynamics['centered']['jerk_rad_s3']['max'],
        command_target_error_scope='Executed shaped command minus current ACT target at its 50 ms label time; not expert error or prediction error',
        command_target_mean_abs_rad=float(np.abs(errors).mean()),
        command_target_p95_abs_rad=float(np.quantile(np.abs(errors),.95)),
        command_target_max_abs_rad=float(np.abs(errors).max()),
        failure_reason=outcome['failure_reason'],
        scope='Single approved body, R4, state ACT L2 diagnostic; no patient contact/hold success, no generalization claim')
    (root/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    start = queries[0]['time_s']; phase_time = phase_events[0]['time_s']
    mask = times >= start
    values = [('gap_m',1000,'Normal gap (mm)',25,(23,27)),
              ('tangent_m',1000,'Tangential error (mm)',3,None),
              ('angle_deg',1,'Tool-axis error (deg)',5,None)]
    for ax,(name,scale,label,threshold,band) in zip(axes,values):
        y = np.array([row[name] for row in trace])*scale
        ax.plot(times[mask][::12],y[mask][::12],color='#166b78',lw=1.6)
        ax.axhline(threshold,color='#b55d31',ls='--',lw=1,label='Original handoff gate')
        if band:ax.axhspan(*band,color='#b55d31',alpha=.10)
        ax.axvspan(phase_time,times[-1],color='#d0dde1',alpha=.3)
        ax.annotate(f'{y[-1]:.2f}',(times[-1],y[-1]),xytext=(-4,10),textcoords='offset points',ha='right')
        ax.set_ylabel(label);ax.grid(alpha=.16)
    axes[0].legend(loc='upper right');axes[-1].set_xlabel('Simulator time (s); shaded region = ACT APPROACH')
    fig.suptitle('ACT D0 / R4: no L3 handoff, stopped by 3 mm command-clearance gate',fontsize=13)
    fig.savefig(root/'trajectory.svg');fig.savefig(root/'trajectory.png',dpi=160);plt.close(fig)
    report = f'''# ACT D0：R4 真实闭环未通过

单身体诊断；{len(queries)} 次 ACT 查询，{len(trace):,} 个实际物理步，仿真时长 {times[-1]:.3f} s。执行进程正常收尾不等于模型通过。没有进入 L3、CONTACT 或 HOLD，保持 0 s。

| 最后实测状态（既定目标坐标系） | 数值 | 原接管门 |
|---|---:|---:|
| 法向间距 | {summary['final']['gap_mm']:.3f} mm | 25 ± 2 mm |
| 切向偏差 | {summary['final']['tangent_mm']:.3f} mm | ≤3 mm |
| 工具轴夹角 | {summary['final']['axis_angle_deg']:.3f}° | ≤5° |

APPROACH 在 {phase_time:.2f} s 开始。下一条命令的 RetentionLip 与衣服距离为 2.992 mm，低于原 3 mm 自由段门；该命令执行前已被拒绝。最后一条 control row 执行 0 步，两个成功碰撞检查计数均为 {len(trace):,}。这是间隙保护触发，不是已发生患者接触的证据。Hall、前端物理力和禁触力记录均为 0。

H10/E5 的已完成索引分布为 {summary['prefix_completed_counts']}，共有 {len(errors)} 个完整 50 ms 动作，另有阶段变化打断与末尾失败。执行端在标签时刻距 ACT 目标的关节平均绝对差为 {summary['command_target_mean_abs_rad']:.6f} rad、p95 为 {summary['command_target_p95_abs_rad']:.6f} rad。这衡量整形后的指令跟踪差，不能直接当成模型相对专家误差。

原独立审计仍判失败。末尾保留的零执行拒绝行使严格训练绑定器报 coverage 错误；另行逐行核查，已执行的物理步连续且无遗漏，但这不使回合合格。实测关节 centered jerk 最大 {summary['observed_centered_jerk_rad_s3']:.3f} rad/s³，也超过原 20 门；未放宽阈值。

下一步：把正确专家目标送入同一 Ruckig 执行路径，对比终端速度为零与有限差分速度提示，分离执行器影响与策略误差。结果出来之前，不改模型或把低离线误差称为成功。ACT 无 RGB 输入；此实验也不证明视觉、Hall 或深度被 VLA 有效使用。

录像 `review.mp4` 来自本回合实际 RGB-D 相机的 RGB 帧，69.35 s / 20 fps / 640×360；推理时物理暂停，所以录像不是实时推理性能证明。保留批准的场景、相机下转 7° 与所有安全门。

来源：代码 `{summary['source_commit']}`；`MIRROR_SHA256.json` 校验 {len(manifest)} 个文件；`summary.json` 保存精确输入哈希与数值。第一次资产路径启动失败独立保留，未计入策略结果。
'''
    (root/'REPORT.md').write_text(report)
    print(json.dumps(summary, ensure_ascii=False))


if __name__=='__main__':main()

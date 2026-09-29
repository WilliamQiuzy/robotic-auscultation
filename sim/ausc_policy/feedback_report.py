"""Plot completed CPU response probes; never labels these as physical recovery."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verified(folder):
    owner = json.loads((folder/'result.json').read_text())
    marker = json.loads((folder/'output/COMPLETE.json').read_text())
    path = folder/'output/metrics.json'; raw = path.read_bytes()
    if not (owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
            and owner['stage']['exit_code'] == 0 and marker['completed']
            and marker['metrics_sha256'] == hashlib.sha256(raw).hexdigest()):
        raise ValueError('Probe must be completed and its metrics bound to the completion marker')
    value = json.loads(raw)
    if any(value[k] for k in ('m0_passed','physics_executed','training_updates','contact_evaluated')):
        raise ValueError('Unexpected probe scope')
    return value, dict(run=owner['run_id'], code_commit=owner['code_commit'],
                      metrics_sha256=marker['metrics_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-probe', type=Path, required=True)
    parser.add_argument('--robot-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target, target_binding = verified(args.target_probe)
    robot, robot_binding = verified(args.robot_probe)
    if (target['receipt'] != robot['receipt'] or
            [x['source_start'] for x in target['anchors']] != [x['source_start'] for x in robot['anchors']]
            or robot['intervention'] != 'robot_state_response'):
        raise ValueError('Paired probe checkpoint or anchor identity differs')
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    rows = robot['responses']; color = '#397177'
    values = [np.array([r['changed_joint_response_gain'] for r in rows
                        if r['joint'] == j and abs(r['joint_shift_rad']) == .02]) for j in range(7)]
    quantiles = np.array([np.percentile(x,[10,50,90]) for x in values])
    axes[0].errorbar(range(1,8), quantiles[:,1],
        yerr=[quantiles[:,1]-quantiles[:,0],quantiles[:,2]-quantiles[:,1]],
        fmt='o', capsize=4, color=color, label='Median; 10th-90th percentile')
    axes[0].set(xticks=range(1,8), xlabel='Perturbed joint (±0.02 rad)',
        ylabel='Change in proposed joint / input joint change', title='h5 output largely follows the changed joint')
    per_anchor = []
    for magnitude in (.005,.02):
        per_anchor.append([np.median([r['tcp_perturbation_ratio'] for r in rows
            if r['source_start'] == a['source_start'] and abs(r['joint_shift_rad']) == magnitude
            and r['tcp_perturbation_ratio'] is not None]) for a in robot['anchors']])
    for i, values_for_anchor in enumerate(per_anchor):
        axes[1].scatter(np.full(len(values_for_anchor),i+1)+np.linspace(-.11,.11,len(values_for_anchor)),
                        values_for_anchor, s=24, color=color, alpha=.75)
        axes[1].plot([i+.8,i+1.2],[np.median(values_for_anchor)]*2,color='#263a40',linewidth=2)
    axes[1].set(xticks=[1,2], xticklabels=['±0.005 rad','±0.02 rad'], xlim=(.5,2.5),
        xlabel='Joint intervention; each dot = one anchor median',
        ylabel='Norm of h5 TCP change / norm of initial TCP change',
        title='Most initial TCP displacement remains in the proposal')
    for ax in axes:
        ax.axhline(1,color='#ab6655',linestyle='--',linewidth=1,label='Copy-current-q reference = 1')
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',alpha=.18); ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Fixed ACT checkpoint · 16 R4 APPROACH TRAIN anchors · 480 CPU inferences',fontsize=12)
    fig.text(.04,.025,'h5 = +250 ms raw proposal. Fixed target; q and TCP changed consistently. No PhysX or recovery oracle.\n'
             'Local response ratios do not establish closed-loop stability, successful recovery, safety, or generalization.',fontsize=9)
    fig.tight_layout(rect=(0,.115,1,.96))
    for suffix in ('svg','png'):
        fig.savefig(args.output/('response.'+suffix),dpi=150)
    plt.close(fig)
    summary = dict(target_probe=target_binding,robot_probe=robot_binding,groups=robot['groups'],
        checkpoint_receipt=robot['receipt'],per_joint_002rad_gain_p10_p50_p90=quantiles.tolist(),
        zero_clean_repeat=robot['clean_repeat_max_abs_rad']==0,
        m0_passed=False, physics_executed=False, generalization_established=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

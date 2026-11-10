"""Plot checked public scalars; no simulator, data arrays or model execution."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=root / 'docs/drafts/evidence/act14-source59-near-20261110.public.json')
    p.add_argument('--output', type=Path, default=root / 'docs/drafts/assets/act14-source59-near-fit-20261110.png')
    a = p.parse_args()
    e = json.loads(a.evidence.read_bytes())
    assert e['physical']['passed'] == 3 and e['physical']['denominator'] == 4
    assert e['scope']['fresh_cases_tested'] == 0 and not e['physical']['model_qualified']
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.8), layout='constrained', facecolor='#f5f4ee')
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    green, rust, gray = '#315d45', '#a64732', '#879084'
    for ax in axes:
        ax.set_facecolor('#f5f4ee')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#d8dbd3', linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)
    cases = e['physical']['cases']
    holds = [c['physical']['continuous_force_and_plane_hold_s'] for c in cases]
    axes[0].bar(range(4), holds, color=[green, green, green, rust], width=.58)
    axes[0].scatter([3], [0], s=95, marker='x', linewidths=2.5, color=rust, zorder=4)
    axes[0].axhline(3, linestyle='--', color=gray, linewidth=1)
    axes[0].set_xticks(range(4), ['TRAIN\nR3', 'TRAIN\nL3', 'DEV VAL\nR3', 'DEV VAL\nL3'])
    axes[0].set_ylim(-.15, 3.85)
    axes[0].set_ylabel('Qualified force + local-plane hold (s)', fontsize=11)
    axes[0].set_title('Actual Isaac closure: 3 / 4', loc='left', fontsize=15, fontweight='bold', pad=18)
    for i, c in enumerate(cases):
        text = '3.000 s\n' + f"{c['physical']['contact_force_peak_N']:.3f} N peak" if c['qualified_case_pass'] else 'No contact\nL2 timeout'
        axes[0].text(i, holds[i]+.13, text, ha='center', va='bottom', fontsize=10, color=green if c['qualified_case_pass'] else rust)
    rows = e['offline']['rows']
    x = np.arange(3)
    acts = [r['e10_mae_mrad'] for r in rows]
    holds = [r['e10_hold_mae_mrad'] for r in rows]
    axes[1].bar(x-.18, acts, .34, color=rust, label='Frozen ACT')
    axes[1].bar(x+.18, holds, .34, color=gray, label='Zero-action HOLD')
    axes[1].set_xticks(x, ['All near\n3,000 windows', 'R3\n1,500 windows', 'L3\n1,500 windows'])
    axes[1].set_ylim(0, 1.65)
    axes[1].set_ylabel('Masked E10 joint target MAE (mrad)', fontsize=11)
    axes[1].set_title('Near TRAIN fit: L3 trails HOLD', loc='left', fontsize=15, fontweight='bold', pad=18)
    axes[1].legend(frameon=False, loc='upper left', fontsize=10)
    for i, (act, hold) in enumerate(zip(acts, holds)):
        axes[1].text(i-.18, act+.035, f'{act:.3f}', ha='center', fontsize=10)
        axes[1].text(i+.18, hold+.035, f'{hold:.3f}', ha='center', fontsize=10)
    fig.suptitle('Source59 · frozen final1,000 ACT updates · model not accepted', fontsize=18, fontweight='bold')
    fig.supxlabel('Left: reused development cases, original physical gates.  Right: overlapping TRAIN teacher windows.\n'
                  'Fresh identities remain untested.  State17/H2 neural approach + analytic Hall contact.', fontsize=11)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160, facecolor=fig.get_facecolor())
    vector = a.output.with_suffix('.svg')
    fig.savefig(vector, facecolor=fig.get_facecolor())
    vector.write_bytes(b'\n'.join(line.rstrip() for line in vector.read_bytes().splitlines()) + b'\n')
    plt.close(fig)
    print(a.output)


if __name__ == '__main__':
    main()

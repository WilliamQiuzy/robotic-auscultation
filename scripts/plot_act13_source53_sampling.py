"""Plot public derived counts only; no model, patient arrays, or simulator."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/drafts'
e = json.loads((DRAFT / 'evidence/act13-source53-balanced-20261109.public.json').read_text())
rows = e['pool']['strata']
labels = [f"{x['body']} / {x['target']}" for x in rows]
counts = {(x['body'], x['target'], x['band']): x['count'] for x in e['sampling']['actual_strata']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True, layout='constrained')
fig.set_facecolor('#f5f4ee')
for ax in axes:
    ax.set_facecolor('#f5f4ee')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#d8d9d0', linewidth=.7)
y = np.arange(len(rows))
colors = ['#227269', '#a77c43']
for ax, mode in zip(axes, ['pool', 'draws']):
    fine = [x['fine'] if mode == 'pool' else counts[x['body'], x['target'], 'fine'] for x in rows]
    recovery = [x['recovery'] if mode == 'pool' else counts[x['body'], x['target'], 'recovery'] for x in rows]
    ax.barh(y, fine, color=colors[0], height=.65, label='Fine correction')
    ax.barh(y, recovery, left=fine, color=colors[1], height=.65, label='Recovery')
    for i, (f, r) in enumerate(zip(fine, recovery)):
        ax.text(f + r + (2 if mode == 'pool' else 10), i, f'{f} + {r}', va='center', fontsize=10)
    ax.set_xlim(0, 160 if mode == 'pool' else 850)
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_title('Existing real TRAIN pool: 1,324 windows', loc='left', pad=18, weight='bold')
axes[1].set_title('Actual replacement sampling: 8,000 draws', loc='left', pad=18, weight='bold')
axes[0].set_xlabel('Eligible existing windows (fine 186 + recovery 1,138)')
axes[1].set_xlabel('Draws, including repeated windows (fine 4,000 + recovery 4,000)')
fig.suptitle('V11 / Source53: change exposure, reuse the same real data', x=.01, ha='left', weight='bold', fontsize=20)
fig.supxlabel('Teal: fine correction · Ochre: recovery · Labels show fine + recovery counts\nSix real TRAIN bodies × R3/L3 · one known pose2 · zero new demonstrations\nCounts describe training exposure; they do not establish contact success.', fontsize=12)
out = DRAFT / 'assets/act13-source53-existing-pool-and-draws.actual.png'
fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
print(out)

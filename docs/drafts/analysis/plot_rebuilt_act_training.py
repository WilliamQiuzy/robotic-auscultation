"""Render aggregate offline metrics only; never open arrays, weights or Isaac.

From the repository root:
  python docs/drafts/analysis/plot_rebuilt_act_training.py
Requires an existing Matplotlib installation. The public scalar extract is the
only input; its source-summary hash identifies the separately retained receipt.
"""
from pathlib import Path
import hashlib
import json
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[3]
DRAFT = ROOT / 'docs/drafts'
DATA = DRAFT / 'evidence/rebuilt-act-offline-20261106.public.json'
FONT = ROOT / 'docs/assets/fonts/Manrope-Variable.ttf'


def main():
    data = json.loads(DATA.read_text())
    rows = data['evaluations']
    assert len(rows) == 40
    assert [r['step'] for r in rows] == list(range(500, 20001, 500))
    selected = min(rows, key=lambda r: (r['validation_act_h20_mae_rad'], r['step']))
    assert selected['step'] == data['selected_step'] == 4500
    assert data['metric']['horizon'] == 20
    font_cache = TemporaryDirectory(prefix='ausc-plot-font-')
    if FONT.exists():
        # Matplotlib does not select variable-font axes as a browser does.
        # Make temporary static weights so the site's font stays readable.
        for weight in (400, 600):
            target = Path(font_cache.name) / f'Manrope-{weight}.ttf'
            with TTFont(FONT) as variable:
                static = instantiateVariableFont(variable, {'wght': weight})
                static.save(target)
            font_manager.fontManager.addfont(str(target))
        plt.rcParams['font.family'] = ['Manrope', 'DejaVu Sans', 'sans-serif']
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 6.25), dpi=160)
    fig.subplots_adjust(left=.10, right=.975, bottom=.20, top=.79)
    fig.suptitle('Rebuilt ACT: an offline prediction milestone', x=.10, y=.955,
        ha='left', fontsize=17, fontweight='semibold')
    fig.text(.10, .89, '4 TRAIN identities / 7,558 windows · 1 selection-VAL identity / 2,193 windows',
        fontsize=12, color='#525b56')
    steps = [r['step'] for r in rows]
    for key, label, color in [('train', 'TRAIN (full cohort)', '#195548'),
                               ('validation', 'Selection VAL (full cohort)', '#ad4e20')]:
        values = [r[key + '_act_h20_mae_rad'] * 1000 for r in rows]
        line, = ax.plot(steps, values, color=color, lw=1.8, marker='o', ms=3.8,
            label=label, zorder=3)
        line.set_gid(key + '-evaluations')
        ax.scatter([selected['step']], [selected[key + '_act_h20_mae_rad'] * 1000],
            s=90, facecolors='white', edgecolors=color, linewidths=2, zorder=5)
    ax.axvline(selected['step'], color='#5a645f', lw=1.2, ls=(0, (4, 3)), zorder=2)
    annotation = ('Selected step 4,500\nEarliest full selection-VAL minimum\n'
        'VAL 26.523 mrad · TRAIN 5.664 mrad')
    ax.annotate(annotation,
        xy=(selected['step'], selected['validation_act_h20_mae_rad'] * 1000),
        xytext=(9500, 38.5), ha='left', va='top', fontsize=12,
        arrowprops={'arrowstyle': '->', 'color': '#5a645f', 'lw': 1.1},
        bbox={'boxstyle': 'round,pad=.5', 'fc': 'white', 'ec': '#cbd0cc', 'lw': .8})
    ax.set(xlim=(0, 20500), ylim=(0, 45), xlabel='Optimizer update',
        ylabel='H20 raw target-space MAE (mrad)')
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis='y', color='#dfe3df', lw=.8)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=2,
        frameon=False, fontsize=12, borderaxespad=0, handlelength=2)
    fig.text(.10, .075, 'All 40 full-cohort evaluations; 500-update intervals; no smoothing. Lower is better.',
        fontsize=12, color='#525b56')
    fig.text(.10, .025, 'Offline target prediction is not measured tracking, patient contact or native qualification.',
        fontsize=12, color='#525b56')
    assets = DRAFT / 'assets'
    for suffix in ('png', 'svg'):
        target = assets / ('rebuilt-act-training-20261106.' + suffix)
        fig.savefig(target, dpi=160,
            facecolor='white', metadata={'Creator': 'Matplotlib; public aggregate metric extract'})
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)
    font_cache.cleanup()
    print('Rendered exact 40-point offline curve; selected step 4,500.')


if __name__ == '__main__':
    main()

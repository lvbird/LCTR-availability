# -*- coding: utf-8 -*-
"""Regenerate Figure 8 – Refrigerant policy scenario composition, 2025-2050."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10

COLORS = {
    'R410A': '#000000',
    'R32':   '#00A4FE',
    'R290':  '#00DD00',
}

LABELS = {
    'R410A': 'R-410A (GWP≈2088)',
    'R32':   'HFC-32 (GWP=675)',
    'R290':  'HC-290 (GWP=3)',
}

SCENARIO_TITLES = {
    'BAU': 'BAU',
    'MTP': 'MTP',
    'APD': 'APD',
}

years = list(range(2025, 2051))


def compute_mix():
    mix = {}
    for sc in ['BAU', 'MTP', 'APD']:
        rows = {}
        for y in years:
            r410a = max(0.0, 0.15 * (2029 - y) / (2029 - 2025)) if y <= 2029 else 0.0
            if sc == 'BAU':
                r290 = 0.0
            elif sc == 'MTP':
                r290 = min(0.90, 0.90 * (y - 2035) / (2050 - 2035)) if y > 2035 else 0.0
            else:  # APD
                r290 = min(1.0, (y - 2025) / (2035 - 2025))
            r32 = max(0.0, 1.0 - r410a - r290)
            rows[y] = {'R410A': r410a, 'R32': r32, 'R290': r290}
        mix[sc] = pd.DataFrame.from_dict(rows, orient='index')
    return mix


def main():
    mix = compute_mix()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.22, left=0.07, right=0.97, top=0.88, bottom=0.22)

    for ax, sc in zip(axes, ['BAU', 'MTP', 'APD']):
        df = mix[sc]
        yr = df.index.tolist()
        r410a = df['R410A'].values * 100
        r32   = df['R32'].values   * 100
        r290  = df['R290'].values  * 100

        ax.stackplot(
            yr,
            r410a, r32, r290,
            colors=[COLORS['R410A'], COLORS['R32'], COLORS['R290']],
            linewidth=0,
        )
        ax.set_xlim(2025, 2050)
        ax.set_ylim(0, 100)
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        ax.set_xticklabels(['2025', '2030', '2035', '2040', '2045', '2050'], fontsize=9)
        ax.tick_params(axis='both', which='both', length=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(SCENARIO_TITLES[sc], fontsize=11, pad=6)
        ax.set_xlabel('Year', fontsize=10)

    axes[0].set_ylabel('Refrigerant share (%)', fontsize=10)
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    # Shared legend below panels
    patches = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in ['R410A', 'R32', 'R290']]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.52, 0.01))

    root = Path(__file__).resolve().parent.parent
    out_dir = root / f"Figures_BCC-CSM2-MR_3hr"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / 'FigureS13_Refrigerant_Scenarios.png'
    out_tif = out_dir / 'FigureS13_Refrigerant_Scenarios.tif'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out_tif, dpi=600, bbox_inches='tight')
    print(f'Saved: {out} / {out_tif}')
    plt.close(fig)


if __name__ == '__main__':
    main()

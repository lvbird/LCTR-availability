"""
Generate Supplementary Figure S9: Annual CDH23 for Beijing and Guangzhou,
comparing BCC-CSM2-MR (solid) and MRI-ESM2-0 (dashed) under three SSP scenarios.

Thin faint lines = raw annual values; thick lines = 5-year rolling mean.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if ROOT.name == "figures_scripts":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from figure_io import read_excel_cached

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS = ['BCC-CSM2-MR', 'MRI-ESM2-0']
SSPS = ['ssp126', 'ssp245', 'ssp585']
SSP_LABELS = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
SSP_COLORS = {'ssp126': '#1a9850', 'ssp245': '#fd8d3c', 'ssp585': '#d73027'}
MODEL_STYLE = {'BCC-CSM2-MR': '-', 'MRI-ESM2-0': '--'}

CITIES = ['北京', '广州']
CITY_LABELS = {'北京': 'Beijing', '广州': 'Guangzhou'}
CDH_BASE = 23
SMOOTH_WINDOW = 5  # years

RESULT_DIR = ROOT / 'CMIP6' / 'results' / '3hr'
OUT_DIR = ROOT / 'Figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Load data and compute CDH23 ───────────────────────────────────────────────
# data[model][ssp][city] = pd.Series(year → CDH23)
data = {}

for model in MODELS:
    data[model] = {}
    for ssp in SSPS:
        fname = RESULT_DIR / f'{model}_{ssp}_TempDist_3hr.xlsx'
        print(f'Loading {fname.name} …')
        df = read_excel_cached(str(fname))

        # Temperature bin columns: integer degrees ≥ 24 contribute to CDH23
        temp_cols = sorted(
            [(c, int(c)) for c in df.columns
             if str(c).lstrip('-').isdigit() and int(c) >= 24],
            key=lambda x: x[1]
        )

        data[model][ssp] = {}
        for city in CITIES:
            sub = df[df['City'] == city].set_index('Year').sort_index()
            cdh = sum(sub[col] * (temp - CDH_BASE) for col, temp in temp_cols)
            data[model][ssp][city] = cdh

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
fig.subplots_adjust(wspace=0.30)

for ax, city in zip(axes, CITIES):
    for ssp in SSPS:
        color = SSP_COLORS[ssp]
        for model in MODELS:
            s = data[model][ssp][city]
            ls = MODEL_STYLE[model]

            # Thin faint raw annual line
            ax.plot(s.index, s.values,
                    color=color, linestyle=ls,
                    linewidth=0.8, alpha=0.25)

            # Thick smoothed line (5-year rolling mean, centre-aligned)
            smoothed = s.rolling(window=SMOOTH_WINDOW, center=True,
                                 min_periods=3).mean()
            ax.plot(smoothed.index, smoothed.values,
                    color=color, linestyle=ls,
                    linewidth=2.0, alpha=1.0)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Cooling Degree Hours (base 23 °C, K·h)', fontsize=11)
    ax.set_title(CITY_LABELS[city], fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Legend (inside Guangzhou panel) ──────────────────────────────────────────
ax_gz = axes[1]
legend_handles = [
    mlines.Line2D([], [], color=SSP_COLORS[ssp], linewidth=2,
                  label=SSP_LABELS[ssp])
    for ssp in SSPS
] + [
    mlines.Line2D([], [], color='#888888', linewidth=2,
                  linestyle='-', label='BCC-CSM2-MR (solid)'),
    mlines.Line2D([], [], color='#888888', linewidth=2,
                  linestyle='--', label='MRI-ESM2-0 (dashed)'),
]
ax_gz.legend(handles=legend_handles, fontsize=9,
             loc='upper left', frameon=True, framealpha=0.85,
             edgecolor='#cccccc')

# ── Save ──────────────────────────────────────────────────────────────────────
out_png = OUT_DIR / 'FigS07_CDH23_Model_Comparison.png'
out_pdf = OUT_DIR / 'FigS07_CDH23_Model_Comparison.pdf'
out_tif = OUT_DIR / 'FigS07_CDH23_Model_Comparison.tif'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_tif, dpi=600, bbox_inches='tight')
print(f'\nSaved: {out_png}')
print(f'Saved: {out_pdf}')
print(f'Saved: {out_tif}')
print(f'Saved: {out_tif}')
plt.show()

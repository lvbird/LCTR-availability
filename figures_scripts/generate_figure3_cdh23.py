"""
Generate Figure 3 (updated): CDH23 annual trends for Beijing and Guangzhou
under SSP1-2.6, SSP2-4.5, SSP5-8.5, years 2025-2059.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "figures_scripts":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from figure_io import read_excel_cached

CLIMATE_MODEL = os.environ.get("LCTR_FIGURE_MODEL", "BCC-CSM2-MR")
OUT_DIR = ROOT / f"Figures_{CLIMATE_MODEL}_3hr"
OUT_DIR.mkdir(exist_ok=True)

CITIES = ['北京', '广州']
CITY_LABELS = {'北京': 'Beijing', '广州': 'Guangzhou'}
YEAR_START = 2025
YEAR_END = 2059
SSP_FILES = {
    'SSP1-2.6': f'CMIP6/results/3hr/{CLIMATE_MODEL}_ssp126_TempDist_3hr.xlsx',
    'SSP2-4.5': f'CMIP6/results/3hr/{CLIMATE_MODEL}_ssp245_TempDist_3hr.xlsx',
    'SSP5-8.5': f'CMIP6/results/3hr/{CLIMATE_MODEL}_ssp585_TempDist_3hr.xlsx',
}
SSP_COLORS = {
    'SSP1-2.6': '#2196F3',   # blue
    'SSP2-4.5': '#757575',   # grey
    'SSP5-8.5': '#F44336',   # red
}

# ── Load and compute CDH23 ────────────────────────────────────────────────────
results = {}  # results[ssp][city] = pd.Series(year -> CDH23)

for ssp, fname in SSP_FILES.items():
    print(f'Loading {ssp} ...')
    df = read_excel_cached(fname)
    temp_cols = sorted(
        [(c, int(c)) for c in df.columns if str(c).lstrip("-").isdigit() and int(c) >= 24],
        key=lambda x: x[1]
    )

    results[ssp] = {}
    for city in CITIES:
        sub = df[df['City'] == city].copy()
        sub = sub[(sub['Year'] >= YEAR_START) & (sub['Year'] <= YEAR_END)]
        sub = sub.set_index('Year').sort_index()
        cdh23 = sum(sub[col] * (temp - 23) for col, temp in temp_cols)
        results[ssp][city] = cdh23
        print(f'  {city}: {cdh23.mean():.0f} mean CDH23')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
fig.subplots_adjust(wspace=0.32)

for ax, city in zip(axes, CITIES):
    for ssp, color in SSP_COLORS.items():
        s = results[ssp][city]
        ax.plot(s.index, s.values, color=color, linewidth=1.4, label=ssp)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Cooling Degree Hours (CDH23)', fontsize=11)
    ax.set_title(CITY_LABELS[city], fontsize=12, fontweight='bold')
    ax.set_xlim(YEAR_START, YEAR_END)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

# Shared legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 1.01))

# Panel labels
for ax, lbl in zip(axes, ['(a)', '(b)']):
    ax.text(-0.12, 1.04, lbl, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

out_png = OUT_DIR / 'Figure3_CDH23_Beijing_Guangzhou.png'
out_pdf = OUT_DIR / 'Figure3_CDH23_Beijing_Guangzhou.pdf'
out_tif = OUT_DIR / 'Figure3_CDH23_Beijing_Guangzhou.tif'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_tif, dpi=600, bbox_inches='tight')
print(f'\nFigure saved to {out_png} / {out_pdf} / {out_tif}')
plt.show()

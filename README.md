# Code & Data Availability — LCTR Study

**Manuscript:** *Lifetime Cooling Temperature Rise (LCTR) of residential air conditioners in China under climate change*
**Last updated:** 2026-05-20
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20303579.svg)](https://doi.org/10.5281/zenodo.20303579)

---

## Repository Structure

```
availability/
├── README.md                         ← this file
├── CODE_AVAILABILITY.md              ← draft code availability statement
├── DATA_AVAILABILITY.md              ← draft data availability statement + data manifest
├── RUN_CMIP6_MODELS.md               ← step-by-step guide for rerunning CMIP6 calculations
├── run_cmim6_models.ps1              ← PowerShell helper: runs LCTR_computation.py for BCC + MRI
├── LCTR_computation.py               ← main LCTR calculation script (standalone)
├── EM_raw.xlsx                       ← provincial grid emission factor inputs by SSP scenario
│
├── CMIP6/
│   ├── data/                         ← raw CMIP6 downloads (large; excluded from git)
│   └── results/3hr/                  ← processed 3-hourly temperature distribution inputs
│       ├── BCC-CSM2-MR_ssp{126,245,585}_TempDist_3hr.xlsx   (6 files)
│       └── MRI-ESM2-0_ssp{126,245,585}_TempDist_3hr.xlsx    (6 files)
│
├── RAC_inputs/
│   └── RAC_EER_simulation_result.xlsx           ← thermodynamic simulation outputs (EER/SEER tables)
│
├── RAC_market/                       ← RAC stock & sales projection scripts and data
│   ├── Stock_prediction.py
│   ├── National_sales_estimation_ssp245_weibull.py
│   ├── Analyze_weibull_sales.py
│   ├── Draw_smoothcal.py             ← generates Supp. Fig. S11 (provincial AC penetration)
│   ├── Draw_national_stock_sales_ssp245.py  ← generates Supp. Fig. S12 (national stock & sales)
│   ├── AC_stock_forecast_by_province_2025_2050_smoothcal.xlsx
│   ├── household_size.xlsx
│   ├── population.xlsx
│   ├── population_proportion.xlsx
│   ├── AC_stock_urban_per_100_households.xlsx
│   ├── AC_stock_rural_per_100_households.xlsx
│   ├── Logistic_Fitting_Results_dynamic_CDD_hh_income.xlsx
│   └── National_AC_sales_estimation_AllSSP_weibull.xlsx
│
├── results/                          ← canonical LCTR result tables (all models, all outputs)
│   ├── CMIP6_3hr_model_comparison_summary.xlsx
│   ├── LCTR_Result_National_Total_{BCC-CSM2-MR,MRI-ESM2-0}_3hr.xlsx
│   ├── LCTR_Result_Provincial_Total_{BCC-CSM2-MR,MRI-ESM2-0}_3hr.xlsx
│   ├── LCTR_Breakdown_2025_SSP245_{BCC-CSM2-MR,MRI-ESM2-0}_3hr.xlsx
│   ├── LCTR_Target_Time_Analysis_{BCC-CSM2-MR,MRI-ESM2-0}_3hr.xlsx
│   └── LCTR_Trend_Beijing_Kunming_{BCC-CSM2-MR,MRI-ESM2-0}_3hr.xlsx
│
├── figure_csv_cache/                 ← CSV exports of key Excel sheets (openpyxl-free fallback)
│   ├── BCC-CSM2-MR_ssp{126,245,585}_TempDist_3hr__Sheet1.csv
│   ├── EM_raw__ssp245.csv
│   ├── LCTR_Result_National_Total_BCC-CSM2-MR_3hr__Unit_LCTR_Detail.csv
│   ├── LCTR_Result_National_Total_BCC-CSM2-MR_3hr__National_Total_Impact.csv
│   └── LCTR_Result_Provincial_Total_BCC-CSM2-MR_3hr__All_Data_Detail.csv
│
├── figures_scripts/                  ← Python scripts to regenerate all manuscript figures
│   ├── figure_io.py                  ← shared Excel/CSV reader helper
│   ├── prepare_figure_csv_cache.py   ← regenerates figure_csv_cache/ from Excel inputs
│   ├── generate_figure1_LCTR_concept.py        ← Fig. 1: analytical framework / LCTR concept
│   ├── generate_figure3_cdh23.py               ← Fig. 3: CDH23 trend (Beijing, Guangzhou)
│   ├── generate_figure4_region_scatter.py      ← Fig. 4: four-quadrant provincial scatter
│   ├── generate_figure6_combined.py            ← Fig. 6: per-cohort LCTR (a) + cumulative (b)
│   ├── generate_figureS1_sensitivity.py        ← Supp. Fig. S1: parameter sensitivity
│   ├── generate_figureS6_provincial_LCTR.py    ← Supp. Fig. S6: provincial LCTR bar chart
│   ├── generate_figureS7_CDH_comparison.py     ← Supp. Fig. S7: CDH23 BCC vs MRI comparison
│   └── generate_figureS13_refrigerant_scenarios.py  ← Supp. Fig. S13: refrigerant scenario trajectories
│
└── Figures/                          ← pre-generated manuscript figures (PDF + PNG + TIFF 600 dpi)
    ├── Fig01_LCTR_Concept.{pdf,png,tif}
    ├── Fig03_CDH23_Beijing_Guangzhou.{pdf,png,tif}
    ├── Fig04_Region_Scatter.{pdf,png,tif}
    ├── Fig04_region_scatter_data.csv
    ├── Fig06_National_LCTR_Combined.{pdf,png,tif}
    ├── FigS01_Sensitivity.{pdf,png,tif}
    ├── FigS06_Provincial_LCTR.{pdf,png,tif}
    ├── FigS07_CDH23_Model_Comparison.{pdf,png,tif}
    ├── FigS11_Provincial_AC_Penetration.{png,tif}
    ├── FigS12_National_Stock_Sales.{png,tif}
    └── FigS13_Refrigerant_Scenarios.{png,tif}
```

---

## Regenerating Figures

All figures can be regenerated from this repository. Run from inside `availability/`:

```powershell
$env:MPLBACKEND = "Agg"

# Step 0 (optional): build CSV cache if openpyxl is unavailable
python figures_scripts\prepare_figure_csv_cache.py

# Main manuscript figures
python figures_scripts\generate_figure1_LCTR_concept.py      # Fig. 1
python figures_scripts\generate_figure3_cdh23.py             # Fig. 3
python figures_scripts\generate_figure4_region_scatter.py    # Fig. 4
python figures_scripts\generate_figure6_combined.py          # Fig. 6 (two-panel)

# Supplementary figures
python figures_scripts\generate_figureS1_sensitivity.py      # Supp. Fig. S1
python figures_scripts\generate_figureS6_provincial_LCTR.py  # Supp. Fig. S6
python figures_scripts\generate_figureS7_CDH_comparison.py   # Supp. Fig. S7
python RAC_market\Draw_smoothcal.py                          # Supp. Fig. S11
python RAC_market\Draw_national_stock_sales_ssp245.py        # Supp. Fig. S12
python figures_scripts\generate_figureS13_refrigerant_scenarios.py  # Supp. Fig. S13
```

Scripts save outputs to `Figures_{MODEL}_3hr/` (BCC-CSM2-MR by default).
To switch to MRI-ESM2-0: set `$env:LCTR_FIGURE_MODEL = "MRI-ESM2-0"` before running.

If `openpyxl` is not installed, each script falls back automatically to `figure_csv_cache/`.

---

## Rerunning the LCTR Calculation

See `RUN_CMIP6_MODELS.md` for a step-by-step guide. The helper script `run_cmim6_models.ps1` sets
the `LCTR_CLIMATE_MODEL` environment variable and calls `LCTR_computation.py` for both
BCC-CSM2-MR and MRI-ESM2-0.

---

## Dependencies

```
python >= 3.10
pandas
numpy
matplotlib
openpyxl      # for reading .xlsx (optional; CSV fallback provided)
scipy         # for Weibull fitting in RAC_market scripts
```

Install with:

```bash
pip install pandas numpy matplotlib openpyxl scipy
```

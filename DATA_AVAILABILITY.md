# Data Availability

## Draft Statement For Manuscript

The processed datasets required to reproduce the calculations are provided with the article/code archive. The CMIP6 3-hourly temperature-derived inputs used in the revised analysis are stored as processed city-year temperature-bin and CDH tables for BCC-CSM2-MR and MRI-ESM2-0 under SSP1-2.6, SSP2-4.5, and SSP5-8.5. Raw CMIP6 outputs are publicly available from the CMIP6 archive; the download and processing notebooks used in this study are provided to document model selection, variables, scenarios, and preprocessing steps. Provincial RAC sales projections, grid emission factor trajectories, thermodynamic model outputs, processed LCTR result tables, and plotting-ready figure data are included in the accompanying data archive. Data derived from third-party public sources remain subject to the terms of the original providers.

## Primary Data Manifest

Climate inputs:

- `CMIP6/results/3hr/BCC-CSM2-MR_ssp126_TempDist_3hr.xlsx`
- `CMIP6/results/3hr/BCC-CSM2-MR_ssp245_TempDist_3hr.xlsx`
- `CMIP6/results/3hr/BCC-CSM2-MR_ssp585_TempDist_3hr.xlsx`
- `CMIP6/results/3hr/MRI-ESM2-0_ssp126_TempDist_3hr.xlsx`
- `CMIP6/results/3hr/MRI-ESM2-0_ssp245_TempDist_3hr.xlsx`
- `CMIP6/results/3hr/MRI-ESM2-0_ssp585_TempDist_3hr.xlsx`
- `CMIP6/results/3hr/*_CDH_3hr.xlsx`

Core LCTR inputs:

- `EM_raw.xlsx`: provincial/city grid emission factor input by SSP.
- `RAC_market/National_AC_sales_estimation_AllSSP_weibull.xlsx`: provincial annual RAC sales flow by SSP.
- `RAC_market/AC_stock_forecast_by_province_2025_2050_smoothcal.xlsx`: provincial RAC stock forecast to 2050 under all SSPs.
- `RAC_market/Logistic_Fitting_Results_dynamic_CDD_hh_income.xlsx`: logistic regression fit parameters for provincial penetration rates.
- `RAC/Result_SEER_*.xlsx`, `RAC/Result_fixed_charge_*.xlsx`, and `RAC_EER模拟结果.xlsx`: thermodynamic simulation outputs supporting refrigerant performance assumptions.
- `AC_stock_forecast_by_province_2025_2045_smoothcal.xlsx` and `Regression/*.xlsx`: RAC stock, sales, income, population, and penetration projection inputs/intermediates.

LCTR result tables — all files below are in `results/`:

- `CMIP6_3hr_model_comparison_summary.xlsx`
- `LCTR_Result_National_Total_BCC-CSM2-MR_3hr.xlsx`
- `LCTR_Result_National_Total_MRI-ESM2-0_3hr.xlsx`
- `LCTR_Result_Provincial_Total_BCC-CSM2-MR_3hr.xlsx`
- `LCTR_Result_Provincial_Total_MRI-ESM2-0_3hr.xlsx`
- `LCTR_Trend_Beijing_Kunming_BCC-CSM2-MR_3hr.xlsx`
- `LCTR_Trend_Beijing_Kunming_MRI-ESM2-0_3hr.xlsx`
- `LCTR_Breakdown_2025_SSP245_BCC-CSM2-MR_3hr.xlsx`
- `LCTR_Breakdown_2025_SSP245_MRI-ESM2-0_3hr.xlsx`
- `LCTR_Target_Time_Analysis_BCC-CSM2-MR_3hr.xlsx`
- `LCTR_Target_Time_Analysis_MRI-ESM2-0_3hr.xlsx`

BCC-based regenerated figure outputs:

- `Figures_BCC-CSM2-MR_3hr/Figure3_CDH23_Beijing_Guangzhou.pdf`
- `Figures_BCC-CSM2-MR_3hr/Figure8_Refrigerant_Scenarios.png`
- `Figures_BCC-CSM2-MR_3hr/Figure16_Region_Scatter.pdf`
- `Figures_BCC-CSM2-MR_3hr/Figure16_region_scatter_data.csv`
- `Figures_BCC-CSM2-MR_3hr/Figure18_Cohort_LCTR.pdf`
- `Figures_BCC-CSM2-MR_3hr/Figure19_Cumulative_LCTR.pdf`
- `Figures_BCC-CSM2-MR_3hr/Figure20_Provincial_LCTR.pdf`
- `Figures_BCC-CSM2-MR_3hr/Figure_Sensitivity_LCTR.pdf`

RAC stock and sales projection figures:

- `RAC_market/Figure6_Provincial_AC_Penetration.png`
- `RAC_market/Figure7_China_AC_Stock_Sales.png`

Figure assets:

- `Figures/`: manuscript figure image files.
- `generate_figure*.py`: scripts for regenerating Python-generated figures.
- `绘图/`: Origin project files and exported figure assets used during drafting.

Archiving note:

- For journal submission, archive the processed Excel inputs and outputs above, the scripts listed in `CODE_AVAILABILITY.md`, and a copy of this manifest. Very large raw CMIP6 NetCDF files can be excluded if the public CMIP6 source and the download notebooks are provided.

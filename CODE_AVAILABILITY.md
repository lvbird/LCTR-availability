# Code Availability

## Draft Statement For Manuscript

The code used to download and process CMIP6 climate inputs, estimate RAC stock and sales, simulate refrigerant performance, compute LCTR, and generate figures is provided with the accompanying code archive. The main LCTR calculation is implemented in `LCTR_computation.py`; setting the environment variable `LCTR_CLIMATE_MODEL` to `BCC-CSM2-MR` or `MRI-ESM2-0` reproduces the two CMIP6 3-hourly model runs reported in the revised analysis. The code archive also includes CMIP6 preprocessing notebooks, RAC thermodynamic model scripts, RAC sales projection scripts, and plotting scripts. External proprietary software dependencies, if any, are documented in the archive and are not redistributed.

## Primary Code Manifest

Main LCTR calculation:

- `LCTR_computation.py`
- `availability/run_cmip6_models.ps1`

CMIP6 climate processing:

- `CMIP6/01_download_cmip6.ipynb`
- `CMIP6/02_process_cdh.ipynb`

RAC stock and sales projection:

- `RAC_market/Stock_prediction.py`
- `RAC_market/National_sales_estimation_ssp245_weibull.py`
- `RAC_market/Analyze_weibull_sales.py`
- `RAC_market/Draw_smoothcal.py`
- `RAC_market/Draw_national_stock_sales_ssp245.py`

RAC thermodynamic and refrigerant-performance model:

- `RAC/Main*.m`
- `RAC/Operation_3IHXs.m`
- `RAC/Decoupled_Solver.m`
- `RAC/Compressor*.m`
- `RAC/Condenser.m`
- `RAC/EvaporatorL.m`
- `RAC/CondensingHTF*.m`
- `RAC/BoilingHTF*.m`
- `RAC/Generate_Ref_LUT.m`
- `RAC/Verify_*.m`

Figure generation and analysis:

- `figures_scripts/figure_io.py`
- `figures_scripts/prepare_figure_csv_cache.py`
- `figures_scripts/generate_figure3_cdh23.py`
- `figures_scripts/generate_figure8.py`
- `figures_scripts/generate_figure9_LCTR_concept.py`
- `figures_scripts/generate_figure16_region_scatter.py`
- `figures_scripts/generate_figure18_cohort_LCTR.py`
- `figures_scripts/generate_figure19_cumulative_LCTR.py`
- `figures_scripts/generate_figure20_provincial_LCTR.py`
- `figures_scripts/generate_figure_sensitivity.py`
- `_compute_breakdown_2040.py`
- `_extract_scatter_data.py`
- `_verify_scatter_data.py`

The updated Figure 16 notebook cell in `figures_scripts/LCTR_computation_for_RAC.ipynb` now calls `generate_figure16_region_scatter.py` instead of using hard-coded CDH/EM/LCTR arrays.

Known cleanup before public archiving:

- Remove temporary scratch scripts beginning with `_tmp_` unless a generated figure depends on them.
- Replace local absolute paths in plotting scripts, such as `os.chdir(...)`, with paths relative to the repository root.
- Do not redistribute proprietary REFPROP binaries unless the license allows it; document that REFPROP is required for the MATLAB RAC simulation workflow.

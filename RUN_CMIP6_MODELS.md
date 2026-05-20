# Reproducing The BCC/MRI LCTR Runs

From the workspace root, run:

```powershell
.\availability\run_cmip6_models.ps1
```

From the standalone `availability` package, run:

```powershell
.\run_cmip6_models.ps1
```

The helper executes:

```powershell
$env:LCTR_CLIMATE_MODEL='BCC-CSM2-MR'
python LCTR_computation.py

$env:LCTR_CLIMATE_MODEL='MRI-ESM2-0'
python LCTR_computation.py
```

Input files expected by `LCTR_computation.py`:

- `CMIP6/results/3hr/{model}_{scenario}_TempDist_3hr.xlsx`
- `EM_raw.xlsx`
- `Regression/National_AC_sales_estimation_AllSSP_weibull.xlsx`

Output files are written to the workspace root with model-specific suffixes, for example:

- `LCTR_Result_National_Total_BCC-CSM2-MR_3hr.xlsx`
- `LCTR_Result_National_Total_MRI-ESM2-0_3hr.xlsx`

If `LCTR_CLIMATE_MODEL` is not set, `LCTR_computation.py` preserves the previous behavior and reads `Processed_Hourly_Stats_{scenario}_Ordered.xlsx`.

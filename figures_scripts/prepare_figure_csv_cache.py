"""
Prepare CSV fallbacks for figure scripts.

This is useful on Python environments that have matplotlib but do not have
openpyxl. The figure scripts call read_excel_cached(), which reads Excel first
and falls back to these CSV files only when needed.
"""
from pathlib import Path

import pandas as pd


CACHE_DIR = Path("figure_csv_cache")
CACHE_DIR.mkdir(exist_ok=True)


def safe_sheet(sheet_name):
    sheet = "Sheet1" if sheet_name in (0, None) else str(sheet_name)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sheet)


def cache_excel(path, sheet_name=0, *, header=0):
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    out = CACHE_DIR / f"{Path(path).stem}__{safe_sheet(sheet_name)}.csv"
    df.to_csv(out, index=False, header=(header is not None), encoding="utf-8-sig")
    print(f"Wrote {out}")


FILES = [
    ("CMIP6/results/3hr/BCC-CSM2-MR_ssp126_TempDist_3hr.xlsx", 0, 0),
    ("CMIP6/results/3hr/BCC-CSM2-MR_ssp245_TempDist_3hr.xlsx", 0, 0),
    ("CMIP6/results/3hr/BCC-CSM2-MR_ssp585_TempDist_3hr.xlsx", 0, 0),
    ("EM_raw.xlsx", "ssp245", 0),
    ("LCTR_Result_National_Total_BCC-CSM2-MR_3hr.xlsx", "Unit_LCTR_Detail", 0),
    ("LCTR_Result_National_Total_BCC-CSM2-MR_3hr.xlsx", "National_Total_Impact", None),
    ("LCTR_Result_Provincial_Total_BCC-CSM2-MR_3hr.xlsx", "All_Data_Detail", 0),
]


if __name__ == "__main__":
    for path, sheet, header in FILES:
        cache_excel(path, sheet_name=sheet, header=header)


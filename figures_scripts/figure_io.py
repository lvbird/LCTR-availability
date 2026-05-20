from pathlib import Path

import pandas as pd


def _cache_name(path, sheet_name):
    sheet = "Sheet1" if sheet_name in (0, None) else str(sheet_name)
    safe_sheet = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sheet)
    return Path("figure_csv_cache") / f"{Path(path).stem}__{safe_sheet}.csv"


def read_excel_cached(path, *args, sheet_name=0, **kwargs):
    """Read Excel normally, falling back to a prepared CSV cache if openpyxl is unavailable."""
    try:
        return pd.read_excel(path, *args, sheet_name=sheet_name, **kwargs)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        cache = _cache_name(path, sheet_name)
        if not cache.exists():
            raise FileNotFoundError(
                f"Missing CSV cache {cache}. Run prepare_figure_csv_cache.py first."
            ) from exc
        return pd.read_csv(cache, **kwargs)


from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = 'Arial'

BASE_DIR = Path(__file__).resolve().parent

FILE_FORECAST = BASE_DIR / "AC_stock_forecast_by_province_2025_2050_smoothcal.xlsx"
FILE_POP = BASE_DIR / "population.xlsx"
FILE_URBAN_RATE = BASE_DIR / "population_proportion.xlsx"
FILE_HH_SIZE = BASE_DIR / "household_size.xlsx"
FILE_URBAN_PR = BASE_DIR / "AC_stock_urban_per_100_households.xlsx"
FILE_RURAL_PR = BASE_DIR / "AC_stock_rural_per_100_households.xlsx"


PROVINCE_EN = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "上海": "Shanghai",
    "重庆": "Chongqing",
    "河北": "Hebei",
    "山西": "Shanxi",
    "内蒙古": "Inner Mongolia",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "山东": "Shandong",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "海南": "Hainan",
    "四川": "Sichuan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "西藏": "Tibet",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
}


def wide_to_long(path: Path, value_name: str, drop_value_na: bool = True) -> pd.DataFrame:
    df = pd.read_excel(path)
    province_col = df.columns[0]
    df = df.rename(columns={province_col: "Province"})
    df["Province"] = df["Province"].astype(str).str.strip()

    year_cols = [col for col in df.columns if col != "Province"]
    long_df = df.melt(
        id_vars="Province",
        value_vars=year_cols,
        var_name="Year",
        value_name=value_name,
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")

    long_df = long_df.dropna(subset=["Year"])
    if drop_value_na:
        long_df = long_df.dropna(subset=[value_name])
    return long_df


def build_historical_panel() -> pd.DataFrame:
    pop = wide_to_long(FILE_POP, "Total_Pop_10k")
    urb = wide_to_long(FILE_URBAN_RATE, "Urban_Rate_pct", drop_value_na=False)
    hh = wide_to_long(FILE_HH_SIZE, "HH_Size")
    upr = wide_to_long(FILE_URBAN_PR, "Urban_PR_per_hh")
    rpr = wide_to_long(FILE_RURAL_PR, "Rural_PR_per_hh")

    hist = pop.merge(urb, on=["Province", "Year"], how="inner")
    hist = hist.merge(hh, on=["Province", "Year"], how="inner")
    hist = hist.merge(upr, on=["Province", "Year"], how="inner")
    hist = hist.merge(rpr, on=["Province", "Year"], how="inner")

    hist = hist[(hist["Year"] >= 2000) & (hist["Year"] <= 2024)]
    hist = hist[hist["Province"] != "全国"].copy()
    hist = hist.sort_values(["Province", "Year"])
    hist["Urban_Rate_pct"] = hist.groupby("Province")["Urban_Rate_pct"].transform(lambda s: s.ffill().bfill())
    hist = hist.dropna(subset=["Urban_Rate_pct", "HH_Size", "Urban_PR_per_hh", "Rural_PR_per_hh", "Total_Pop_10k"])

    hist["Urban_Rate"] = hist["Urban_Rate_pct"] / 100.0
    hist["Urban_Pop_10k"] = hist["Total_Pop_10k"] * hist["Urban_Rate"]
    hist["Rural_Pop_10k"] = hist["Total_Pop_10k"] - hist["Urban_Pop_10k"]

    hist["Urban_HH_thousand"] = (hist["Urban_Pop_10k"] / hist["HH_Size"]) * 10.0
    hist["Rural_HH_thousand"] = (hist["Rural_Pop_10k"] / hist["HH_Size"]) * 10.0
    total_hh = hist["Urban_HH_thousand"] + hist["Rural_HH_thousand"]

    hist["PR_per_hh"] = (
        hist["Urban_PR_per_hh"] * hist["Urban_HH_thousand"]
        + hist["Rural_PR_per_hh"] * hist["Rural_HH_thousand"]
    ) / total_hh

    hist["Stock_thousand"] = (
        hist["Urban_HH_thousand"] * hist["Urban_PR_per_hh"] / 100.0
        + hist["Rural_HH_thousand"] * hist["Rural_PR_per_hh"] / 100.0
    )

    return hist[["Province", "Year", "PR_per_hh", "Stock_thousand"]]


def build_forecast_panel() -> pd.DataFrame:
    fc = pd.read_excel(FILE_FORECAST, sheet_name="SSP245_province")
    fc["Province"] = fc["Province"].astype(str).str.strip()
    fc = fc[(fc["Year"] >= 2025) & (fc["Year"] <= 2050)].copy()

    total_hh = fc["Urban_Households"] + fc["Rural_Households"]
    fc["PR_per_hh"] = (
        fc["Urban_PR_per_hh"] * fc["Urban_Households"]
        + fc["Rural_PR_per_hh"] * fc["Rural_Households"]
    ) / total_hh
    fc["PR_per_hh"] = fc["PR_per_hh"] * 100.0
    fc["Stock_thousand"] = fc["Total_Stock"]

    return fc[["Province", "Year", "PR_per_hh", "Stock_thousand"]]


def _spread_right_labels(y_last: pd.Series, min_gap_frac: float = 0.022) -> pd.Series:
    ordered = y_last.sort_values()
    if ordered.empty:
        return ordered
    ymin, ymax = ordered.min(), ordered.max()
    span = max(ymax - ymin, 1.0)
    min_gap = min_gap_frac * span
    adjusted = ordered.copy().astype(float)
    for i in range(1, len(adjusted)):
        if adjusted.iloc[i] - adjusted.iloc[i - 1] < min_gap:
            adjusted.iloc[i] = adjusted.iloc[i - 1] + min_gap
    return adjusted.reindex(y_last.index)


FONT_TICK  = 13
FONT_LABEL = 14
FONT_ANNO  = 11


def plot_province_panel(panel: pd.DataFrame, value_col: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.subplots_adjust(right=0.78)

    provinces = sorted([str(p) for p in panel["Province"].unique()],
                       key=lambda p: panel[panel["Province"] == p][value_col].iloc[-1]
                                     if not panel[panel["Province"] == p].empty else 0)
    n = len(provinces)
    cmap = matplotlib.colormaps.get_cmap("tab20")

    lines = {}
    for idx, prov in enumerate(provinces):
        sub = panel[panel["Province"] == prov].sort_values("Year")
        color = cmap(idx)
        ln, = ax.plot(sub["Year"], sub[value_col],
                      linewidth=1.4, alpha=0.88, color=color)
        ax.scatter(sub["Year"], sub[value_col], s=10, alpha=0.8, color=color, zorder=3)
        lines[prov] = (ln, color)

    year_last = panel["Year"].max()
    end_data  = panel[panel["Year"] == year_last].copy()
    y_last    = end_data.set_index("Province")[value_col]
    y_adj     = _spread_right_labels(y_last)

    x_end  = year_last
    x_tick = year_last + 0.9
    x_text = year_last + 1.3
    for prov, y_raw in y_last.items():
        color = lines[prov][1]
        ax.annotate(
            "",
            xy=(x_tick, y_adj.loc[prov]),
            xytext=(x_end, y_raw),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.75),
        )
        ax.text(x_text, y_adj.loc[prov],
                str(PROVINCE_EN.get(prov, prov)),
                fontsize=FONT_ANNO, va="center", ha="left", color=color)

    ax.set_xlabel("Year", fontsize=FONT_LABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, labelpad=8)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_xlim(2000, year_last + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)
    ax.grid(axis="x", alpha=0.12, linestyle=":", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tif"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_national_total(panel: pd.DataFrame, output_path: Path) -> None:
    national = panel.groupby("Year", as_index=False)["Stock_thousand"].sum()
    national = national.sort_values("Year")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.plot(national["Year"], national["Stock_thousand"],
            color="#1a6faf", linewidth=2.5, zorder=3)
    ax.scatter(national["Year"], national["Stock_thousand"],
               color="#1a6faf", s=20, zorder=4)

    ax.set_xlabel("Year", fontsize=FONT_LABEL, labelpad=6)
    ax.set_ylabel("AC stock (thousand units)", fontsize=FONT_LABEL, labelpad=8)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(6, integer=False))
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.grid(axis="x", alpha=0.12, linestyle=":", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1999.5, national["Year"].max() + 0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tif"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    hist = build_historical_panel()
    fc = build_forecast_panel()

    common_provinces = sorted(set(hist["Province"]).intersection(set(fc["Province"])))
    hist = hist[hist["Province"].isin(common_provinces)].copy()
    fc = fc[fc["Province"].isin(common_provinces)].copy()

    panel = pd.concat([hist, fc], ignore_index=True)
    panel = panel.sort_values(["Province", "Year"]).reset_index(drop=True)

    plot_province_panel(
        panel=panel,
        value_col="PR_per_hh",
        ylabel="AC penetration (units per 100 households)",
        output_path=BASE_DIR / "Figure6_Provincial_AC_Penetration.png",
    )

    plot_national_total(
        panel=panel,
        output_path=BASE_DIR / "Figure6_China_AC_Stock.png",
    )

    plot_national_total(
        panel=panel,
        output_path=BASE_DIR / "chart_3_china_total_stock_2000_2045_ssp245_smoothcal.png",
    )

    print("Saved:")
    print(BASE_DIR / "Figure6_Provincial_AC_Penetration.png")
    print(BASE_DIR / "Figure6_China_AC_Stock.png")


if __name__ == "__main__":
    main()

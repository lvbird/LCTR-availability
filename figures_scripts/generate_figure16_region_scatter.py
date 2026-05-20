"""
Generate Figure 16 regional heterogeneity scatter plot.

Data source defaults to BCC-CSM2-MR 3-hourly CMIP6 processed outputs:
- CDH23: cumulative 2025-2034 from CMIP6/results/3hr/{model}_ssp245_TempDist_3hr.xlsx
- EM: 2025-2034 average, PCHIP-interpolated from EM_raw.xlsx, sheet ssp245
- LCTR: R32 unit LCTR for 2025 production cohort, SSP2-4.5, TY=2100
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import PchipInterpolator
ROOT = Path(__file__).resolve().parent
if ROOT.name == "figures_scripts":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from figure_io import read_excel_cached

CLIMATE_MODEL = os.environ.get("LCTR_FIGURE_MODEL", "BCC-CSM2-MR")
OUT_DIR = ROOT / f"Figures_{CLIMATE_MODEL}_3hr"
OUT_DIR.mkdir(exist_ok=True)

CDH_THRESH = 70000
EM_THRESH = 0.35

PROVINCE_TO_CITY = {
    "Liaoning": "沈阳",
    "Jilin": "长春",
    "Heilongjiang": "哈尔滨",
    "Beijing": "北京",
    "Tianjin": "天津",
    "Hebei": "石家庄",
    "Shanxi": "太原",
    "Inner Mongolia": "呼和浩特",
    "Shandong": "济南",
    "Shanghai": "上海",
    "Jiangsu": "南京",
    "Zhejiang": "杭州",
    "Anhui": "合肥",
    "Fujian": "福州",
    "Jiangxi": "南昌",
    "Henan": "郑州",
    "Hubei": "武汉",
    "Hunan": "长沙",
    "Chongqing": "重庆",
    "Sichuan": "成都",
    "Guangdong": "广州",
    "Guangxi": "南宁",
    "Hainan": "海口",
    "Guizhou": "贵阳",
    "Yunnan": "昆明",
    "Shaanxi": "西安",
    "Gansu": "兰州",
    "Qinghai": "西宁",
    "Ningxia": "银川",
    "Xinjiang": "乌鲁木齐",
    "Tibet": "拉萨",
}


def interpolate_em_2025_2034():
    raw = read_excel_cached("EM_raw.xlsx", sheet_name="ssp245")
    raw = raw.set_index(raw.columns[0]).astype(float)
    years = np.arange(2025, 2035)
    out = {}
    for city in raw.columns:
        ser = raw[city].dropna()
        interp = PchipInterpolator(ser.index.astype(int), ser.values, extrapolate=True)
        out[city] = float(np.mean(np.clip(interp(years), 0.0, 1.0)))
    return out


def build_dataframe():
    temp = read_excel_cached(f"CMIP6/results/3hr/{CLIMATE_MODEL}_ssp245_TempDist_3hr.xlsx")
    temp_cols = [(c, int(c)) for c in temp.columns if str(c).lstrip("-").isdigit() and int(c) >= 24]
    temp = temp[(temp["Year"] >= 2025) & (temp["Year"] <= 2034)].copy()
    temp["CDH23"] = sum(temp[col] * (temp_bin - 23) for col, temp_bin in temp_cols)
    cdh = temp.groupby("City")["CDH23"].sum()

    em = interpolate_em_2025_2034()

    lctr = read_excel_cached(
        f"results/LCTR_Result_National_Total_{CLIMATE_MODEL}_3hr.xlsx",
        sheet_name="Unit_LCTR_Detail"
    )
    lctr = lctr[
        (lctr["情景"] == "ssp245")
        & (lctr["年份"] == 2025)
        & (lctr["制冷剂"] == "R32")
    ]
    lctr_by_city = dict(zip(lctr["地点"], lctr["Unit_LCTR_pK"]))

    rows = []
    for province, city in PROVINCE_TO_CITY.items():
        rows.append({
            "Province": province,
            "City": city,
            "CDH23": float(cdh.get(city, 0.0)),
            "EM": float(em[city]),
            "LCTR": float(lctr_by_city[city]),
        })
    return pd.DataFrame(rows)


def zone(row):
    if row["CDH23"] >= CDH_THRESH and row["EM"] >= EM_THRESH:
        return "High Load / High Carbon"
    if row["CDH23"] >= CDH_THRESH and row["EM"] < EM_THRESH:
        return "High Load / Low Carbon"
    if row["CDH23"] < CDH_THRESH and row["EM"] >= EM_THRESH:
        return "Low Load / High Carbon"
    return "Low Load / Low Carbon"


def main():
    rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    df = build_dataframe()
    df["Zone"] = df.apply(zone, axis=1)
    df.to_csv(OUT_DIR / "Figure16_region_scatter_data.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=300)
    scatter = ax.scatter(
        df["CDH23"], df["EM"], c=df["LCTR"], s=165,
        cmap="RdYlGn_r", alpha=0.88, edgecolors="black", linewidth=0.7,
        vmin=max(0.0, df["LCTR"].min() * 0.85), vmax=df["LCTR"].max() * 1.05,
        zorder=10
    )

    ax.axvline(CDH_THRESH, color="gray", linestyle="--", alpha=0.65, linewidth=1.2)
    ax.axhline(EM_THRESH, color="gray", linestyle="--", alpha=0.65, linewidth=1.2)

    x_min, x_max = df["CDH23"].min(), df["CDH23"].max()
    y_min, y_max = df["EM"].min(), df["EM"].max()
    ax.set_xlim(max(0, x_min - 10000), x_max + 20000)
    ax.set_ylim(max(0, y_min - 0.04), y_max + 0.12)

    # Quadrant corner labels placed in axes-fraction coordinates
    _zone_kw = dict(fontsize=10, fontweight="bold", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              alpha=0.75, edgecolor="none"))
    ax.text(0.98, 0.98, "High Load / High Carbon",
            color="#a50026", ha="right", va="top", **_zone_kw)
    ax.text(0.02, 0.98, "Low Load / High Carbon",
            color="#d73027", ha="left", va="top", **_zone_kw)
    ax.text(0.02, 0.02, "Low Load / Low Carbon",
            color="#1a9850", ha="left", va="bottom", **_zone_kw)
    ax.text(0.98, 0.02, "High Load / Low Carbon",
            color="#4575b4", ha="right", va="bottom", **_zone_kw)

    # Province labels — fully manual annotate, no adjust_text.
    # Each entry: (delta_CDH, delta_EM, ha, va)
    # Coordinates confirmed against actual data (see _tmp_scatter_data.py)
    LABEL_OFFSETS = {
        # Low CDH / isolated
        "Tibet":           (  4000,  0.028, "left", "center"),
        "Qinghai":         (  4000,  0.030, "left",  "top"),
        "Gansu":           (  8000,  0.022, "left", "center"),
        "Yunnan":          (  5000,  0.030, "left",  "top"),
        # NE/NW high-EM stacked column (x ≈ 27k–32k, y ≈ 0.60–0.78)
        "Inner Mongolia":  (  9000,  0.038, "right", "bottom"),
        "Xinjiang":        ( -7000,  0.022, "right", "center"),
        "Heilongjiang":    ( -7000, -0.018, "right", "top"),
        # Mid-left column (x ≈ 34k–41k)
        "Liaoning":        ( -4000,  0.028, "right", "bottom"),
        "Jilin":           ( -4000, -0.030, "right", "top"),
        # Ningxia (42k, 0.64) above Shaanxi/Shanxi
        "Ningxia":         (  4000,  0.020, "left",  "bottom"),
        # Sichuan isolated low-y
        "Sichuan":         (  4000, -0.032, "left",  "top"),
        # Shaanxi / Shanxi (44k–47k, 0.59–0.69)
        "Shaanxi":         ( -9000,  0.030, "right", "bottom"),
        "Shanxi":          (  5000,  0.026, "left",  "bottom"),
        # Beijing / Shanghai / Guizhou (57k–61k, 0.30–0.50)
        "Beijing":         ( -5000, -0.028, "right", "top"),
        "Shanghai":        ( -5000, -0.030, "right", "top"),
        "Guizhou":         ( -5000, -0.028, "right", "top"),
        # HIGH CDH ZONE ──────────────────────────────────────────────────
        # Tianjin (73674, 0.555) / Hebei (73830, 0.671) — nearly same x!
        "Tianjin":         (-5000,  0.032, "right", "bottom"),
        "Hebei":           ( 4000,  0.034, "left",  "bottom"),
        # Henan (76803, 0.500) / Zhejiang (77547, 0.380) — similar x, diff y
        "Henan":           (-5000, -0.032, "right", "top"),
        "Zhejiang":        (-5000, 0.044, "right", "center"),
        # Shandong (78747, 0.489) — just right of Henan, push up
        "Shandong":        ( 5000, -0.038, "left",  "top"),
        # Jiangsu (82455, 0.518) — right and down
        "Jiangsu":         ( 5000, 0.024, "left",  "bottom"),
        # Fujian (86499, 0.328)
        "Fujian":          (  5000, -0.024, "left",  "top"),
        # Anhui (94908, 0.696) — high EM, push up-right
        "Anhui":           (  4000,  0.025, "left",  "bottom"),
        # Hubei (95895, 0.261) / Chongqing (107724, 0.270) — similar y, stagger
        "Hubei":           ( -5000, -0.028, "right", "top"),
        "Chongqing":       (  5000, -0.028, "left",  "top"),
        # Hunan (101613, 0.403)
        "Hunan":           (  5000, -0.024, "left",  "top"),
        # Jiangxi (112263, 0.431)
        "Jiangxi":         (  5000,  0.022, "left",  "bottom"),
        # Guangxi (133098, 0.326) / Guangdong (135012, 0.332) — nearly same!
        "Guangxi":         (-6000, -0.034, "right", "top"),
        "Guangdong":       ( 5000, -0.030, "left",  "top"),
        # Hainan far right (198132, 0.229)
        "Hainan":          (-4000,  0.032, "right", "bottom"),
    }

    _arrow = dict(arrowstyle="-", color="#555555", lw=0.55, alpha=0.75,
                  shrinkA=2, shrinkB=7)   # shrinkB≈marker radius in pts
    for _, row in df.iterrows():
        dx, dy, ha, va = LABEL_OFFSETS.get(row["Province"], (0, 0.028, "center", "bottom"))
        ax.annotate(
            row["Province"],
            xy=(row["CDH23"], row["EM"]),
            xytext=(row["CDH23"] + dx, row["EM"] + dy),
            fontsize=7.4,
            ha=ha, va=va,
            alpha=0.92,
            zorder=20,
            arrowprops=_arrow,
        )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("HFC-32 unit LCTR (pK), 2025 cohort, TY=2100", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Cumulative CDH$_{23}$ (2025-2034, K h)", fontsize=12)
    ax.set_ylabel("Average grid emission factor (2025-2034, kg CO$_2$/kWh)", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35, zorder=0)

    fig.tight_layout()
    out_pdf = OUT_DIR / "Figure16_Region_Scatter.pdf"
    out_png = OUT_DIR / "Figure16_Region_Scatter.png"
    out_tif = OUT_DIR / "Figure16_Region_Scatter.tif"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_tif, dpi=600, bbox_inches="tight")
    print(f"Saved: {out_pdf} / {out_png} / {out_tif}")
    print(df[["Province", "City", "CDH23", "EM", "LCTR", "Zone"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
generate_figure20_provincial_LCTR.py
──────────────────────────────────────────────────────────────────────────────
Section 3.4: Provincial LCTR distribution
Horizontal bar chart showing top-N province cumulative LCTR contribution
(MTP-weighted, SSP2-4.5, prod_year 2025-2050, T=2100), coloured by climate zone.
──────────────────────────────────────────────────────────────────────────────
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.patches import Patch
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

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.major.width"] = 0.8
rcParams["ytick.major.width"] = 0.8
rcParams["xtick.major.size"] = 3.5
rcParams["ytick.major.size"] = 3.5

# ── Load provincial data ──────────────────────────────────────────────────────
df_prov = read_excel_cached(
    f"results/LCTR_Result_Provincial_Total_{CLIMATE_MODEL}_3hr.xlsx",
    sheet_name="All_Data_Detail"
)

# Filter SSP245, 2025-2050 production cohorts
df = df_prov[(df_prov["情景"] == "ssp245") & (df_prov["年份"] <= 2050)].copy()

# Apply MTP weights per year �?use correct weights from LCTR_computation.py
# (NOT Scenario setting.xlsx which has old weights)
def make_mtp_weights():
    rows = {}
    for y in range(2025, 2051):
        r410a = max(0.0, 0.15 * (2029 - y) / (2029 - 2025)) if y <= 2029 else 0.0
        r290  = min(0.90, 0.90 * (y - 2035) / (2050 - 2035)) if y > 2035 else 0.0
        r32   = max(0.0, 1.0 - r410a - r290)
        rows[y] = {"年份": y, "w_R410A": r410a, "w_R32": r32, "w_R290": r290}
    return pd.DataFrame.from_dict(rows, orient="index")

mtp_weights = make_mtp_weights()

# Merge weights into provincial data
df = df.merge(mtp_weights, on="年份", how="left")

# Compute weighted contribution per row
def get_weight(row):
    ref = row["制冷剂"]
    if ref == "R410A": return row["w_R410A"]
    elif ref == "R32":  return row["w_R32"]
    elif ref == "R290": return row["w_R290"]
    return 0.0

df["weight"] = df.apply(get_weight, axis=1)
df["weighted_pK"] = df["Provincial_Total_LCTR_pK"] * df["weight"]

# Aggregate by province
prov_mtp = df.groupby("地点")["weighted_pK"].sum().sort_values(ascending=False)

# Convert to mK: pK * 1e-12 K * 1e3 mK/K = pK / 1e9 mK
prov_mK = prov_mtp / 1e9

print("MTP-weighted provincial totals (mK, SSP245, 2025-2050, T=2100):")
print(prov_mK.to_string())
print(f"\nNational total (sum of provinces): {prov_mK.sum():.4f} mK")

# ── Climate zone classification ───────────────────────────────────────────────
# 4 zones: HH (high load, high carbon), HL, LH, LL.
# Recomputed from the same BCC/MRI temperature-bin and EM inputs used in LCTR.
CDH_THRESH = 70000
EM_THRESH = 0.35

def load_zone_map():
    temp = read_excel_cached(f"CMIP6/results/3hr/{CLIMATE_MODEL}_ssp245_TempDist_3hr.xlsx")
    temp_cols = [(c, int(c)) for c in temp.columns if str(c).lstrip("-").isdigit() and int(c) >= 24]
    temp = temp[(temp["Year"] >= 2025) & (temp["Year"] <= 2034)].copy()
    temp["CDH23"] = sum(temp[col] * (temp_bin - 23) for col, temp_bin in temp_cols)
    cdh = temp.groupby("City")["CDH23"].sum()

    em_raw = read_excel_cached("EM_raw.xlsx", sheet_name="ssp245")
    em_raw = em_raw.set_index(em_raw.columns[0]).astype(float)
    em = em_raw.loc[[y for y in range(2025, 2035) if y in em_raw.index]].mean()

    out = {}
    for city in cdh.index:
        high_load = float(cdh[city]) >= CDH_THRESH
        high_carbon = float(em.get(city, np.nan)) >= EM_THRESH
        if high_load and high_carbon:
            out[city] = "HH"
        elif high_load and not high_carbon:
            out[city] = "HL"
        elif not high_load and high_carbon:
            out[city] = "LH"
        else:
            out[city] = "LL"
    return out

zone_map = load_zone_map()
zone_colors = {
    "HH": "#d62728",   # red
    "HL": "#ff7f0e",   # orange
    "LH": "#1f77b4",   # blue
    "LL": "#2ca02c",   # green
}
zone_labels = {
    "HH": "High Load / High Carbon",
    "HL": "High Load / Low Carbon",
    "LH": "Low Load / High Carbon",
    "LL": "Low Load / Low Carbon",
}

# ── English province names mapping ────────────────────────────────────────────
name_en = {
    "广州":"Guangdong", "郑州":"Henan", "南京":"Jiangsu",
    "长沙":"Hunan", "合肥":"Anhui", "石家庄":"Hebei",
    "杭州":"Zhejiang", "济南":"Shandong", "南昌":"Jiangxi",
    "南宁":"Guangxi", "武汉":"Hubei", "西安":"Shaanxi",
    "福州":"Fujian", "重庆":"Chongqing", "上海":"Shanghai",
    "沈阳":"Liaoning", "北京":"Beijing", "天津":"Tianjin",
    "太原":"Shanxi", "呼和浩特":"Inner Mongolia", "海口":"Hainan",
    "长春":"Jilin", "哈尔滨":"Heilongjiang", "乌鲁木齐":"Xinjiang",
    "兰州":"Gansu", "银川":"Ningxia", "成都":"Sichuan",
    "贵阳":"Guizhou", "昆明":"Yunnan", "西宁":"Qinghai",
    "拉萨":"Tibet",
}

# Take top N provinces
N = 20
top_prov = prov_mK.head(N)
prov_names = [name_en.get(p, p) for p in top_prov.index]
prov_zones  = [zone_map.get(p, "LH") for p in top_prov.index]
prov_colors = [zone_colors[z] for z in prov_zones]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 6.2))

y_pos = np.arange(len(top_prov))
bars = ax.barh(y_pos, top_prov.values, height=0.72,
               color=prov_colors, edgecolor="white", linewidth=0.4, alpha=0.88)

ax.set_yticks(y_pos)
ax.set_yticklabels(prov_names, fontsize=9.5)
ax.invert_yaxis()

ax.set_xlabel("Cumulative MTP provincial LCTR contribution (mK)", fontsize=10.5)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.02))
ax.tick_params(axis="both", labelsize=10)

# Value labels
for i, v in enumerate(top_prov.values):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5)

ax.set_xlim(0, top_prov.values[0] * 1.22)

# Legend
handles = [Patch(facecolor=zone_colors[z], alpha=0.85, label=zone_labels[z])
           for z in ["HH","HL","LH","LL"]]
ax.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.85,
          title="Climate Zone\n(CDH23 / Grid EM)", title_fontsize=8.5)

plt.tight_layout()
out_pdf = OUT_DIR / "Figure20_Provincial_LCTR.pdf"
out_png = OUT_DIR / "Figure20_Provincial_LCTR.png"
out_tif = OUT_DIR / "Figure20_Provincial_LCTR.tif"
plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_tif, dpi=600, bbox_inches="tight")
plt.show()
print(f"Saved: {out_pdf} / {out_png} / {out_tif}")

"""
generate_figure6_combined.py
──────────────────────────────────────────────────────────────────────────────
Figure 6 (combined two-panel) for main manuscript:
  (a) Per-cohort lifetime LCTR contribution (previously Supplementary Fig. S6)
  (b) Cumulative national LCTR at TY=2060 and TY=2100 (previously Fig. 6)

Combines generate_figure18_cohort_LCTR.py and generate_figure19_cumulative_LCTR.py
into a single publication figure with shared legend context.
──────────────────────────────────────────────────────────────────────────────
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
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

# ── Colour scheme (shared across both panels) ────────────────────────────────
C_BAU = "#d62728"
C_MTP = "#ff7f0e"
C_APD = "#2ca02c"

# ── Refrigerant market-share helper ──────────────────────────────────────────
def get_shares(policy, year):
    r410a = max(0.0, 0.15 * (2029 - year) / (2029 - 2025)) if year <= 2029 else 0.0
    if policy == "BAU":
        r290 = 0.0
    elif policy == "MTP":
        r290 = min(0.90, 0.90 * (year - 2035) / (2050 - 2035)) if year > 2035 else 0.0
    else:  # APD
        r290 = min(1.0, (year - 2025) / (2035 - 2025))
    r32 = max(0.0, 1.0 - r410a - r290)
    return r410a, r32, r290


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (a) DATA  — per-cohort lifetime LCTR
# ═══════════════════════════════════════════════════════════════════════════════
raw_nat = read_excel_cached(
    f"results/LCTR_Result_National_Total_{CLIMATE_MODEL}_3hr.xlsx",
    sheet_name="National_Total_Impact", header=None
)
nat = raw_nat.iloc[4:, :].copy()
nat.columns = [
    "prod_year",
    "s126_BAU", "s126_MTP", "s126_APD",
    "s245_BAU", "s245_MTP", "s245_APD",
    "s585_BAU", "s585_MTP", "s585_APD"
]
nat = nat.dropna(subset=["prod_year"]).copy()
nat["prod_year"] = nat["prod_year"].astype(int)
nat = nat[nat["prod_year"] <= 2050].reset_index(drop=True)
for c in nat.columns[1:]:
    nat[c] = pd.to_numeric(nat[c])
nat_s = nat.sort_values("prod_year")
years = nat_s["prod_year"].values


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (b) DATA  — cumulative LCTR at TY=2060 and TY=2100
# ═══════════════════════════════════════════════════════════════════════════════
# TY=2060/2100 ratio from Macro_National_Total
df_macro_raw = read_excel_cached(
    f"results/LCTR_Target_Time_Analysis_{CLIMATE_MODEL}_3hr.xlsx",
    sheet_name="Macro_National_Total",
    header=None,
)
try:
    val_2060 = df_macro_raw.iloc[3, [4, 5, 6]].values.astype(float)
    val_2100 = df_macro_raw.iloc[11, [4, 5, 6]].values.astype(float)
    RATIO = {"R410A": val_2060[0] / val_2100[0],
             "R32":   val_2060[1] / val_2100[1],
             "R290":  val_2060[2] / val_2100[2]}
    print(f"TY ratios (2060/2100): R410A={RATIO['R410A']:.4f}  "
          f"R32={RATIO['R32']:.4f}  R290={RATIO['R290']:.4f}")
except Exception as e:
    print(f"Warning: macro ratio lookup failed ({e}), using hard-coded values")
    RATIO = {"R410A": 2.0246, "R32": 2.1131, "R290": 1.1822}

# Unit LCTR (TY=2100, SSP2-4.5)
unit = read_excel_cached(
    f"results/LCTR_Result_National_Total_{CLIMATE_MODEL}_3hr.xlsx",
    sheet_name="Unit_LCTR_Detail",
)
unit_245 = unit[unit["情景"] == "ssp245"].copy()

# Provincial sales
sales = read_excel_cached(
    "RAC_market/National_AC_sales_estimation_AllSSP_weibull.xlsx",
    sheet_name="provincial_sales_flow",
)
sales = sales[sales["SSP_Scenario"] == "SSP245"][
    ["Province", "Year", "New_Sales_Units"]].copy()
sales["Year"] = sales["Year"].astype(int)
sales = sales[(sales["Year"] >= 2025) & (sales["Year"] <= 2050)]
sales_piv = sales.pivot(
    index="Year", columns="Province", values="New_Sales_Units").fillna(0)

CITY_TO_PROVINCE = {
    "广州": "广东", "郑州": "河南", "南京": "江苏", "长沙": "湖南", "合肥": "安徽",
    "石家庄": "河北", "太原": "山西", "呼和浩特": "内蒙古", "济南": "山东", "上海": "上海",
    "北京": "北京", "天津": "天津", "杭州": "浙江", "南宁": "广西", "武汉": "湖北",
    "郑州": "河南", "长沙": "湖南", "合肥": "安徽", "沈阳": "辽宁", "长春": "吉林",
    "哈尔滨": "黑龙江", "福州": "福建", "重庆": "重庆", "西安": "陕西", "南昌": "江西",
    "成都": "四川", "贵阳": "贵州", "昆明": "云南", "乌鲁木齐": "新疆",
    "兰州": "甘肃", "银川": "宁夏", "西宁": "青海", "拉萨": "西藏", "海口": "海南",
}
city_sales = {
    c: sales_piv[p].to_dict() if p in sales_piv.columns else {}
    for c, p in CITY_TO_PROVINCE.items()
}

# Q_r(y): sum over cities of unit_lctr * sales
qrows = []
for year in range(2025, 2051):
    for ref in ["R410A", "R32", "R290"]:
        sub = unit_245[(unit_245["年份"] == year) & (unit_245["制冷剂"] == ref)]
        Q = sum(
            row["Unit_LCTR_pK"] * city_sales.get(row["地点"], {}).get(year, 0)
            for _, row in sub.iterrows()
        )
        qrows.append({"year": year, "ref": ref, "Q": Q})
Q_df = pd.DataFrame(qrows).pivot(index="year", columns="ref", values="Q")

# Cumulative LCTR
cumul_2060 = {"BAU": 0.0, "MTP": 0.0, "APD": 0.0}
cumul_2100 = {"BAU": 0.0, "MTP": 0.0, "APD": 0.0}
for year in range(2025, 2051):
    for policy in ["BAU", "MTP", "APD"]:
        r410a, r32, r290 = get_shares(policy, year)
        Qr = Q_df.loc[year]
        cumul_2100[policy] += r410a * Qr["R410A"] + r32 * Qr["R32"] + r290 * Qr["R290"]
        cumul_2060[policy] += (
            r410a * Qr["R410A"] * RATIO["R410A"]
            + r32  * Qr["R32"]   * RATIO["R32"]
            + r290 * Qr["R290"]  * RATIO["R290"]
        )
vals_2060 = {k: v / 1e9 for k, v in cumul_2060.items()}
vals_2100 = {k: v / 1e9 for k, v in cumul_2100.items()}

print("\n=== Cumulative national LCTR (2025-2050 cohorts, SSP2-4.5) ===")
for p in ["BAU", "MTP", "APD"]:
    s60  = (vals_2060["BAU"] - vals_2060[p]) / vals_2060["BAU"] * 100
    s100 = (vals_2100["BAU"] - vals_2100[p]) / vals_2100["BAU"] * 100
    print(f"  {p}: TY=2060={vals_2060[p]:.4f} mK  TY=2100={vals_2100[p]:.4f} mK"
          f"  (saves vs BAU: {s60:.1f}% / {s100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT  — 1 row × 2 columns, width ratio 1:1
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(13.5, 5.0),
    gridspec_kw={"width_ratios": [1.1, 1.0]}
)

# ── Panel (a): Per-cohort LCTR ───────────────────────────────────────────────
ax_a.fill_between(
    years, nat_s["s126_BAU"], nat_s["s585_BAU"],
    color="#aec7e8", alpha=0.38, linewidth=0,
    label="SSP range (BAU)"
)
ax_a.plot(years, nat_s["s585_BAU"], color="#6b4226", lw=0.8, alpha=0.6, ls="-")
ax_a.plot(years, nat_s["s126_BAU"], color="#4a90d9", lw=0.8, alpha=0.6, ls="-")
ax_a.plot(years, nat_s["s245_BAU"], color=C_BAU, lw=1.9, label="BAU")
ax_a.plot(years, nat_s["s245_MTP"], color=C_MTP, lw=1.9, ls="--", label="MTP")
ax_a.plot(years, nat_s["s245_APD"], color=C_APD, lw=1.9, ls=":",  label="APD")

y585_end = nat_s.loc[nat_s["prod_year"] == 2050, "s585_BAU"].values[0]
y126_end = nat_s.loc[nat_s["prod_year"] == 2050, "s126_BAU"].values[0]
ax_a.text(2050.4, y585_end + 0.001, "SSP5-8.5", fontsize=8, color="#6b4226", va="bottom")
ax_a.text(2050.4, y126_end - 0.001, "SSP1-2.6", fontsize=8, color="#4a90d9", va="top")

ax_a.set_xlabel("Production year", fontsize=12)
ax_a.set_ylabel("Per-cohort LCTR contribution (mK)", fontsize=12)
ax_a.set_xlim(2024, 2054)
ax_a.set_ylim(0.04, 0.105)
ax_a.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax_a.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax_a.tick_params(axis="both", labelsize=11)
ax_a.legend(fontsize=10, loc="upper right", framealpha=0.88)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.text(0.03, 0.97, "a", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", va="top")

# ── Panel (b): Cumulative LCTR bar chart ─────────────────────────────────────
COLORS = {
    "BAU": ("#8b0000", "#f4a0a0"),
    "MTP": ("#7b4000", "#f5c48a"),
    "APD": ("#1a5200", "#a8d88a"),
}
policies = ["BAU", "MTP", "APD"]
x = np.arange(len(policies))
bar_w = 0.35

for i, policy in enumerate(policies):
    c_dark, c_light = COLORS[policy]
    v60  = vals_2060[policy]
    v100 = vals_2100[policy]
    ax_b.bar(x[i] - bar_w / 2, v60, bar_w,
             color=c_dark, label="TY = 2060" if i == 0 else None,
             edgecolor="white", linewidth=0.6)
    ax_b.bar(x[i] + bar_w / 2, v100, bar_w,
             color=c_light, label="TY = 2100" if i == 0 else None,
             edgecolor="white", linewidth=0.6)
    ax_b.text(x[i] - bar_w / 2, v60 + 0.04,
              f"{v60:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_b.text(x[i] + bar_w / 2, v100 + 0.04,
              f"{v100:.2f}", ha="center", va="bottom", fontsize=9.5)

# Annotation arrows
save_2060 = (vals_2060["BAU"] - vals_2060["APD"]) / vals_2060["BAU"] * 100
save_2100 = (vals_2100["BAU"] - vals_2100["APD"]) / vals_2100["BAU"] * 100
x_bau_60  = x[0] - bar_w / 2
x_apd_60  = x[2] - bar_w / 2
x_bau_100 = x[0] + bar_w / 2
x_apd_100 = x[2] + bar_w / 2
y_arrow_60  = vals_2060["BAU"] + 0.48
y_arrow_100 = vals_2060["BAU"] + 0.95
label_gap = 0.16

for x_pos, v_bar, y_arrow, c in [
    (x_bau_60,  vals_2060["BAU"], y_arrow_60,  "#333333"),
    (x_apd_60,  vals_2060["APD"], y_arrow_60,  "#333333"),
    (x_bau_100, vals_2100["BAU"], y_arrow_100, "#888888"),
    (x_apd_100, vals_2100["APD"], y_arrow_100, "#888888"),
]:
    ax_b.plot([x_pos, x_pos], [v_bar + label_gap, y_arrow],
              ls=":", color=c, lw=0.9, zorder=3)

ax_b.annotate("",
              xy=(x_apd_60, y_arrow_60), xytext=(x_bau_60, y_arrow_60),
              arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6))
ax_b.text((x_bau_60 + x_apd_60) / 2, y_arrow_60 + 0.03,
          f"APD saves {save_2060:.1f}%  (TY = 2060)",
          ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#333333")

ax_b.annotate("",
              xy=(x_apd_100, y_arrow_100), xytext=(x_bau_100, y_arrow_100),
              arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.3,
                              linestyle="dashed"))
ax_b.text((x_bau_100 + x_apd_100) / 2, y_arrow_100 + 0.03,
          f"APD saves {save_2100:.1f}%  (TY = 2100)",
          ha="center", va="bottom", fontsize=9.5, color="#888888", style="italic")

ax_b.set_xticks(x)
ax_b.set_xticklabels(policies, fontsize=13)
ax_b.set_ylabel("Cumulative national LCTR,\n2025–2050 cohorts (mK)", fontsize=12)
ymax = y_arrow_100 + 0.55
ax_b.set_ylim(0, ymax)
ax_b.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax_b.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
ax_b.tick_params(axis="both", labelsize=11)
ax_b.legend(fontsize=11, loc="upper right", framealpha=0.85)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.text(0.03, 0.97, "b", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", va="top")

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(w_pad=2.5)

out_pdf = OUT_DIR / "Figure6_National_Aggregate_Combined.pdf"
out_png = OUT_DIR / "Figure6_National_Aggregate_Combined.png"
out_tif = OUT_DIR / "Figure6_National_Aggregate_Combined.tif"
plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_tif, dpi=600, bbox_inches="tight")
plt.show()
print(f"\nSaved:\n  {out_pdf}\n  {out_png}\n  {out_tif}")

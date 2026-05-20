"""
Analyze_weibull_sales.py
对 National_AC_sales_estimation_SSP245_weibull.xlsx 进行可视化诊断，
检查 Weibull(k=5, λ=10) 下销量/报废量/保有量数据的合理性。
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "National_AC_sales_estimation_SSP245_weibull.xlsx"
OUT_DIR = BASE_DIR
YEAR_RANGE = (1990, 2045)

# ─────────────────────────────────────────────
# 读数据
# ─────────────────────────────────────────────
xls = pd.ExcelFile(FILE)
nat = pd.read_excel(FILE, sheet_name="national_sales_flow")
prov = pd.read_excel(FILE, sheet_name="provincial_sales_flow")
summary = pd.read_excel(FILE, sheet_name="summary")

nat["Year"] = nat["Year"].astype(int)
prov["Year"] = prov["Year"].astype(int)

# 聚焦 1990-2045
nat = nat[(nat["Year"] >= YEAR_RANGE[0]) & (nat["Year"] <= YEAR_RANGE[1])].copy()
prov = prov[(prov["Year"] >= YEAR_RANGE[0]) & (prov["Year"] <= YEAR_RANGE[1])].copy()

print("=== 汇总参数 ===")
print(summary.to_string(index=False))
print()

# ─────────────────────────────────────────────
# 辅助：格式化纵轴
# ─────────────────────────────────────────────
def fmt_million(x, _):
    if x >= 1e8:
        return f"{x/1e8:.1f}亿"
    elif x >= 1e4:
        return f"{x/1e4:.0f}万"
    else:
        return f"{x:.0f}"

def thousands_to_wan(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/100:.0f}万" if x < 100000 else f"{x/100000:.1f}亿")
    )

# ─────────────────────────────────────────────
# 图1：全国总量 — 保有量 / 新销售 / 报废量（单位：千台 → 万台）
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("全国空调总量：保有量 / 新销量 / 报废量（SSP245, Weibull k=5 λ=10）", fontsize=13)

for ax, col, title, color in zip(
    axes,
    ["Stock_Thousand_Units", "New_Sales_Thousand_Units", "Retirements_Thousand_Units"],
    ["总保有量（千台）", "年度新销量（千台）", "年度报废量（千台）"],
    ["steelblue", "darkorange", "firebrick"],
):
    ax.plot(nat["Year"], nat[col], color=color, lw=2)
    ax.axvline(2025, color="gray", ls="--", lw=1, label="预测起始")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("年份")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    thousands_to_wan(ax)
    ax.grid(True, alpha=0.3)
    # 标注负值（不合理）
    neg = nat[nat[col] < 0]
    if not neg.empty:
        ax.scatter(neg["Year"], neg[col], color="red", zorder=5, label="负值！")
        ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "diag_1_national_totals.png", dpi=150)
plt.close()
print("图1 已保存：diag_1_national_totals.png")

# ─────────────────────────────────────────────
# 图2：全国总量 — 三条线叠放，检查 Stock balance
# stock(t) = stock(t-1) + sales(t) - retirements(t)
# ─────────────────────────────────────────────
nat_sorted = nat.sort_values("Year").copy()
nat_sorted["Implied_Stock"] = (
    nat_sorted["Stock_Thousand_Units"].shift(1)
    + nat_sorted["New_Sales_Thousand_Units"]
    - nat_sorted["Retirements_Thousand_Units"]
)
nat_sorted["Balance_Error"] = nat_sorted["Stock_Thousand_Units"] - nat_sorted["Implied_Stock"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("全国保有量 balance 检验", fontsize=13)

ax = axes[0]
ax.plot(nat_sorted["Year"], nat_sorted["Stock_Thousand_Units"], lw=2, label="实际保有量")
ax.plot(nat_sorted["Year"], nat_sorted["Implied_Stock"], ls="--", lw=1.5, label="推算保有量\n(stock_{t-1}+sales-retire)")
ax.axvline(2025, color="gray", ls=":", lw=1)
ax.set_title("保有量：实际 vs 推算")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
thousands_to_wan(ax)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(nat_sorted["Year"], nat_sorted["Balance_Error"], color="steelblue", width=0.8)
ax.axhline(0, color="black", lw=1)
ax.set_title("Balance Error（应≈0）")
ax.set_xlabel("年份")
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
thousands_to_wan(ax)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "diag_2_balance_check.png", dpi=150)
plt.close()
print("图2 已保存：diag_2_balance_check.png")

# ─────────────────────────────────────────────
# 图3：全国新销量年增长率（检查异常跳变）
# ─────────────────────────────────────────────
nat_sorted["Sales_YoY"] = nat_sorted["New_Sales_Thousand_Units"].pct_change() * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(nat_sorted["Year"], nat_sorted["Sales_YoY"], color=[
    "firebrick" if v < -30 or v > 60 else "steelblue"
    for v in nat_sorted["Sales_YoY"].fillna(0)
], width=0.8)
ax.axhline(0, color="black", lw=1)
ax.axvline(2025, color="gray", ls="--", lw=1, label="预测起始")
ax.set_title("全国年度新销量 YoY 增长率（%）\n（红色 = 增长率 < -30% 或 > +60%，需关注）", fontsize=12)
ax.set_xlabel("年份")
ax.set_ylabel("增长率 (%)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "diag_3_sales_yoy.png", dpi=150)
plt.close()
print("图3 已保存：diag_3_sales_yoy.png")

# ─────────────────────────────────────────────
# 图4：省级销量热图（2025-2045）
# ─────────────────────────────────────────────
prov_2025 = prov[prov["Year"] >= 2025].copy()
pivot_sales = prov_2025.pivot_table(
    index="Province", columns="Year", values="New_Sales_Thousand_Units", aggfunc="first"
)
# 按2025年销量降序排列
pivot_sales = pivot_sales.sort_values(pivot_sales.columns[0], ascending=False)

fig, ax = plt.subplots(figsize=(18, 10))
im = ax.imshow(pivot_sales.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(pivot_sales.columns)))
ax.set_xticklabels(pivot_sales.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot_sales.index)))
ax.set_yticklabels(pivot_sales.index, fontsize=8)
plt.colorbar(im, ax=ax, label="新销量（千台）")
ax.set_title("各省年度新销量热图（2025-2045，千台）", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "diag_4_provincial_sales_heatmap.png", dpi=150)
plt.close()
print("图4 已保存：diag_4_provincial_sales_heatmap.png")

# ─────────────────────────────────────────────
# 图5：省级负值销量检查（最重要！）
# ─────────────────────────────────────────────
neg_prov = prov[prov["New_Sales_Thousand_Units"] < 0].copy()
neg_prov_hist = neg_prov[neg_prov["Year"] < 2025]
neg_prov_fc = neg_prov[neg_prov["Year"] >= 2025]

print(f"\n省级负值销量：总计 {len(neg_prov)} 条，其中历史期 {len(neg_prov_hist)} 条，预测期 {len(neg_prov_fc)} 条")
if not neg_prov.empty:
    print(neg_prov[["Province", "Year", "New_Sales_Thousand_Units"]].to_string(index=False))

if not neg_prov.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    for pname, gdf in neg_prov.groupby("Province"):
        ax.scatter(gdf["Year"], gdf["New_Sales_Thousand_Units"], label=pname, zorder=5)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("省级负值新销量分布（各点为一条负值记录）", fontsize=12)
    ax.set_xlabel("年份")
    ax.set_ylabel("新销量（千台）")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "diag_5_negative_sales.png", dpi=150)
    plt.close()
    print("图5 已保存：diag_5_negative_sales.png")
else:
    print("无负值省级销量，跳过图5。")

# ─────────────────────────────────────────────
# 图6：重点省份 全周期 保有量/销量/报废量 折线
# ─────────────────────────────────────────────
HIGHLIGHT_PROVS = ["广东", "山东", "江苏", "河南", "浙江", "湖南", "黑龙江", "内蒙古", "云南", "青海"]
highlight_provs = [p for p in HIGHLIGHT_PROVS if p in prov["Province"].unique()]

fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=False)
fig.suptitle("重点省份：保有量 / 新销量 / 报废量（1990-2045）", fontsize=13)

for ax, pname in zip(axes.flat, highlight_provs):
    sub = prov[prov["Province"] == pname].sort_values("Year")
    ax.plot(sub["Year"], sub["Stock_Thousand_Units"], lw=1.5, label="保有量", color="steelblue")
    ax.plot(sub["Year"], sub["New_Sales_Thousand_Units"], lw=1.5, label="销量", color="darkorange")
    ax.plot(sub["Year"], sub["Retirements_Thousand_Units"], lw=1.5, label="报废量", color="firebrick")
    ax.axvline(2025, color="gray", ls="--", lw=0.8)
    ax.set_title(pname, fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="x", labelsize=7)
    thousands_to_wan(ax)
    ax.grid(True, alpha=0.25)
    # 高亮负值
    neg_s = sub[sub["New_Sales_Thousand_Units"] < 0]
    if not neg_s.empty:
        ax.scatter(neg_s["Year"], neg_s["New_Sales_Thousand_Units"], color="red", zorder=6, s=40)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "diag_6_key_provinces.png", dpi=150)
plt.close()
print("图6 已保存：diag_6_key_provinces.png")

# ─────────────────────────────────────────────
# 图7：Weibull 退役概率密度 vs 年龄（可视化参数合理性）
# ─────────────────────────────────────────────
k, lam = 5.0, 10.0
ages = np.arange(1, 26)
cdf = 1 - np.exp(-((ages / lam) ** k))
pdf = np.diff(np.concatenate([[0], cdf]))
median_life = lam * (np.log(2) ** (1 / k))
mean_life = lam * (1 + 1 / k)  # Γ(1+1/k), 用简化近似

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Weibull 退役分布（k={k}, λ={lam}）\n中值寿命≈{median_life:.1f}年，均值寿命≈{mean_life:.1f}年（近似）", fontsize=11)

ax = axes[0]
ax.bar(ages, pdf * 100, color="steelblue", width=0.8)
ax.set_title("每年退役概率 (PDF，%)")
ax.set_xlabel("使用年龄")
ax.set_ylabel("%")
ax.axvline(median_life, color="firebrick", ls="--", lw=1.5, label=f"中值 {median_life:.1f}年")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ages, cdf * 100, color="darkorange", lw=2)
ax.set_title("累积退役率 (CDF，%)")
ax.set_xlabel("使用年龄")
ax.set_ylabel("%")
ax.axhline(50, color="gray", ls=":", lw=1)
ax.axvline(median_life, color="firebrick", ls="--", lw=1.5, label=f"中值 {median_life:.1f}年")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "diag_7_weibull_distribution.png", dpi=150)
plt.close()
print("图7 已保存：diag_7_weibull_distribution.png")

# ─────────────────────────────────────────────
# 图8：省级加总 vs 全国总量（一致性检验）
# ─────────────────────────────────────────────
prov_agg = (
    prov.groupby("Year")[["Stock_Thousand_Units", "New_Sales_Thousand_Units", "Retirements_Thousand_Units"]]
    .sum()
    .reset_index()
)
merged = nat.merge(prov_agg, on="Year", suffixes=("_nat", "_prov"))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("省级加总 vs 全国口径（一致性检验）", fontsize=13)

for ax, col, title in zip(
    axes,
    ["Stock_Thousand_Units", "New_Sales_Thousand_Units", "Retirements_Thousand_Units"],
    ["保有量", "新销量", "报废量"],
):
    ax.plot(merged["Year"], merged[f"{col}_nat"], lw=2, label="全国口径", color="steelblue")
    ax.plot(merged["Year"], merged[f"{col}_prov"], lw=1.5, ls="--", label="省级加总", color="darkorange")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    thousands_to_wan(ax)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "diag_8_national_vs_provincial.png", dpi=150)
plt.close()
print("图8 已保存：diag_8_national_vs_provincial.png")

# ─────────────────────────────────────────────
# 控制台诊断摘要
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("诊断摘要")
print("=" * 55)

nat_neg = nat[nat["New_Sales_Thousand_Units"] < 0]
print(f"[全国] 负值新销量年份数: {len(nat_neg)}  {list(nat_neg['Year']) if not nat_neg.empty else ''}")

print(f"[全国] 销量峰值: {nat['New_Sales_Thousand_Units'].max():.0f}千台 "
      f"({nat.loc[nat['New_Sales_Thousand_Units'].idxmax(), 'Year']}年)")
print(f"[全国] 保有量峰值: {nat['Stock_Thousand_Units'].max():.0f}千台 "
      f"({nat.loc[nat['Stock_Thousand_Units'].idxmax(), 'Year']}年)")
print(f"[全国] 2025年新销量: {nat.loc[nat['Year']==2025, 'New_Sales_Thousand_Units'].values[0]:.0f}千台")
print(f"[全国] 2025年保有量: {nat.loc[nat['Year']==2025, 'Stock_Thousand_Units'].values[0]:.0f}千台")

# 找保有量下降的年份（预测期）
fc_nat = nat[nat["Year"] >= 2025].sort_values("Year")
stock_drop = fc_nat[fc_nat["Stock_Thousand_Units"].diff() < -fc_nat["Stock_Thousand_Units"].max() * 0.02]
if not stock_drop.empty:
    print(f"[警告] 预测期保有量出现显著下降年份: {list(stock_drop['Year'])}")

# 找销量年增超过80%或负值的年份
yoy = nat_sorted[nat_sorted["Year"] >= 2001].copy()
extreme_yoy = yoy[(yoy["Sales_YoY"] > 80) | (yoy["Sales_YoY"] < -40)]
if not extreme_yoy.empty:
    print(f"[警告] 极端销量增长年份:")
    print(extreme_yoy[["Year", "New_Sales_Thousand_Units", "Sales_YoY"]].to_string(index=False))

print("=" * 55)
print("所有图表已保存至:", OUT_DIR)

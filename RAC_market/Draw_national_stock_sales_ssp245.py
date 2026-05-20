from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd

matplotlib.rcParams['font.family'] = 'Arial'

BASE_DIR = Path(__file__).resolve().parent

FILE_SALES_FLOW = BASE_DIR / "National_AC_sales_estimation_AllSSP_weibull.xlsx"
OUT_PNG = BASE_DIR / "Figure7_China_AC_Stock_Sales.png"

FONT_TICK  = 13
FONT_LABEL = 14

SSP_COLORS = {
    'SSP126': '#2ca02c',   # green
    'SSP245': '#1a6faf',   # blue
    'SSP585': '#d62728',   # red
}
SSP_LABELS = {
    'SSP126': 'Total stock – SSP1-2.6',
    'SSP245': 'Total stock – SSP2-4.5',
    'SSP585': 'Total stock – SSP5-8.5',
}


def build_flow_panel() -> pd.DataFrame:
    """Load national sales flow for all SSPs, 2000-2050."""
    df = pd.read_excel(FILE_SALES_FLOW, sheet_name="national_sales_flow")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Stock_Thousand_Units"] = pd.to_numeric(df["Stock_Thousand_Units"], errors="coerce")
    df["New_Sales_Thousand_Units"] = pd.to_numeric(df["New_Sales_Thousand_Units"], errors="coerce")
    df = df.dropna(subset=["Year", "Stock_Thousand_Units"])
    df["Year"] = df["Year"].astype(int)
    return df[(df["Year"] >= 2000) & (df["Year"] <= 2050)].copy()


def plot_stock_and_sales(flow: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax2 = ax1.twinx()

    # --- Bars: SSP245 new sales (right axis) ---
    ssp245_sales = (
        flow[flow["SSP_Scenario"] == "SSP245"]
        .sort_values("Year")
        .drop_duplicates(subset="Year", keep="last")
    )
    bars = ax2.bar(
        ssp245_sales["Year"],
        ssp245_sales["New_Sales_Thousand_Units"].clip(lower=0),
        width=0.72,
        color="#f4a261",
        alpha=0.45,
        label="New sales (SSP2-4.5)",
        zorder=1,
    )

    # --- Lines: stock for all three SSPs (left axis) ---
    line_handles = []
    for ssp in ["SSP126", "SSP245", "SSP585"]:
        sub = (
            flow[flow["SSP_Scenario"] == ssp]
            .sort_values("Year")
            .drop_duplicates(subset="Year", keep="last")
        )
        ln, = ax1.plot(
            sub["Year"], sub["Stock_Thousand_Units"],
            color=SSP_COLORS[ssp], linewidth=2.2,
            label=SSP_LABELS[ssp], zorder=3,
        )
        ax1.scatter(sub["Year"], sub["Stock_Thousand_Units"],
                    color=SSP_COLORS[ssp], s=12, zorder=4)
        line_handles.append(ln)

    # --- Axes styling ---
    ax1.set_xlabel("Year", fontsize=FONT_LABEL, labelpad=6)
    ax1.set_ylabel("AC stock (thousand units)", fontsize=FONT_LABEL,
                   color="#333333", labelpad=8)
    ax2.set_ylabel("AC new sales (thousand units)", fontsize=FONT_LABEL,
                   color="#c46e1d", labelpad=8)

    ax1.tick_params(axis="both", labelsize=FONT_TICK)
    ax2.tick_params(axis="y",   labelsize=FONT_TICK, labelcolor="#c46e1d")

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlim(1999.2, 2050.8)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax1.grid(axis="x", alpha=0.10, linestyle=":", linewidth=0.6)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    all_handles = line_handles + [bars]
    all_labels  = [SSP_LABELS[s] for s in ["SSP126", "SSP245", "SSP585"]] + ["New sales (SSP2-4.5)"]
    ax1.legend(all_handles, all_labels, loc="upper left",
               frameon=True, framealpha=0.9, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tif"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    flow = build_flow_panel()
    plot_stock_and_sales(flow, OUT_PNG)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()

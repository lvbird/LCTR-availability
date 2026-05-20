import numpy as np
import pandas as pd
from pathlib import Path
import re

# ----------------- 参数设置 -----------------
HIST_START = 1990
HIST_ANCHOR_YEAR = 2000
FORECAST_END = 2050

# 威布尔分布参数（shape=k, scale=lambda）
# scale=10 对应约9.2年中值寿命，与房间空调10年设计寿命相符
WEIBULL_SHAPE = 5.0
WEIBULL_SCALE = 10.0

# 1990 年初始保有量假设：每百户 0.01 台
INITIAL_STOCK_PER_100_HH_1990 = 0.01

# 输入文件
FILE_AC_URBAN = 'AC_stock_urban_per_100_households.xlsx'
FILE_AC_RURAL = 'AC_stock_rural_per_100_households.xlsx'
FILE_POP = 'population.xlsx'
FILE_URBAN_RATE = 'population_proportion.xlsx'
FILE_HH_SIZE = 'household_size.xlsx'
FILE_FORECAST = 'AC_stock_forecast_by_province_2025_2050_smoothcal.xlsx'

# 输出文件
OUT_FILE = 'National_AC_sales_estimation_AllSSP_weibull.xlsx'
# 处理的气候情景列表（对应 Stock_prediction 输出的 sheet 前缀）
SSP_SCENARIOS = ['SSP126', 'SSP245', 'SSP585']
BASE_DIR = Path(__file__).resolve().parent
TARGET_PROVINCES = ['黑龙江', '内蒙古', '云南', '青海']


def p(name: str) -> Path:
    return BASE_DIR / name


def standardize_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.map(_normalize_province_name)
    if not out.index.is_unique:
        out = out.groupby(level=0).first()

    new_cols = []
    for c in out.columns:
        s = str(c).strip()
        try:
            v = float(s)
            if v.is_integer():
                new_cols.append(int(v))
            else:
                new_cols.append(s)
        except Exception:
            new_cols.append(s)
    out.columns = new_cols
    return out


def best_province_match(prov_name: str, candidate_index) -> str | None:
    cands = [str(x).strip() for x in candidate_index]
    if prov_name in cands:
        return prov_name
    matches = [x for x in cands if prov_name in x or x in prov_name]
    if matches:
        return matches[0]
    return None


def _normalize_province_name(v: str) -> str:
    s = str(v).strip()
    s = re.sub(r'\s+', '', s)

    if s in {'全国', '中国', 'China', 'CHINA'}:
        return s

    for suffix in ['维吾尔自治区', '壮族自治区', '回族自治区', '特别行政区', '自治区', '省', '市', '地区']:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break

    return s


def to_numeric_safe(v):
    return pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0]


def weibull_cdf(age: int, k: float, lam: float) -> float:
    age = max(float(age), 0.0)
    return 1.0 - np.exp(-((age / lam) ** k))


def weibull_discard_prob(age: int, k: float, lam: float) -> float:
    if age <= 0:
        return 0.0
    return weibull_cdf(age, k, lam) - weibull_cdf(age - 1, k, lam)


def build_historical_national_stock() -> tuple[pd.Series, float]:
    df_ac_u = standardize_panel(pd.read_excel(p(FILE_AC_URBAN), index_col=0))
    df_ac_r = standardize_panel(pd.read_excel(p(FILE_AC_RURAL), index_col=0))
    df_pop = standardize_panel(pd.read_excel(p(FILE_POP), index_col=0))
    df_urban = standardize_panel(pd.read_excel(p(FILE_URBAN_RATE), index_col=0))
    df_hh = standardize_panel(pd.read_excel(p(FILE_HH_SIZE), index_col=0))

    # 各历史面板首行通常为“全国”汇总，这里剔除，仅保留省级行。
    df_ac_u = df_ac_u.iloc[1:, :]
    df_ac_r = df_ac_r.iloc[1:, :]
    df_pop = df_pop.iloc[1:, :]
    df_urban = df_urban.iloc[1:, :]
    df_hh = df_hh.iloc[1:, :]

    # population_proportion 前几年存在缺失，这里按“省内就近年份”补齐（先向后填，再向前填）。
    urb_years = sorted([c for c in df_urban.columns if isinstance(c, int)])
    df_urban[urb_years] = (
        df_urban[urb_years]
        .apply(pd.to_numeric, errors='coerce')
        .bfill(axis=1)
        .ffill(axis=1)
    )

    province_list = sorted(
        set(df_ac_u.index)
        & set(df_ac_r.index)
        & set(df_pop.index)
        & set(df_urban.index)
        & set(df_hh.index)
    )

    year_candidates = [
        {c for c in df_ac_u.columns if isinstance(c, int)},
        {c for c in df_ac_r.columns if isinstance(c, int)},
        {c for c in df_pop.columns if isinstance(c, int)},
        {c for c in df_urban.columns if isinstance(c, int)},
        {c for c in df_hh.columns if isinstance(c, int)},
    ]
    years = sorted(set.intersection(*year_candidates))
    years = [y for y in years if y <= 2024]

    national_stock = {}

    for y in years:
        total_stock_wan = 0.0

        for prov in province_list:
            pr_u = to_numeric_safe(df_ac_u.at[prov, y]) / 100.0
            pr_r = to_numeric_safe(df_ac_r.at[prov, y]) / 100.0
            pop_tot = to_numeric_safe(df_pop.at[prov, y])
            urban_rate = to_numeric_safe(df_urban.at[prov, y])
            hh_size = to_numeric_safe(df_hh.at[prov, y])

            if not np.isfinite(pop_tot) or not np.isfinite(urban_rate) or not np.isfinite(hh_size) or hh_size <= 0:
                continue

            if np.isnan(pr_u):
                pr_u = 0.0
            if np.isnan(pr_r):
                pr_r = 0.0

            # population_proportion.xlsx 为百分比口径（如 56.3）
            if urban_rate > 1.5:
                urban_rate = urban_rate / 100.0
            urban_rate = float(np.clip(urban_rate, 0.0, 1.0))

            hh_u = pop_tot * urban_rate / hh_size
            hh_r = pop_tot * (1.0 - urban_rate) / hh_size

            stock_wan = pr_u * hh_u + pr_r * hh_r
            if np.isfinite(stock_wan):
                total_stock_wan += stock_wan

        # 历史由万人口口径推得 stock 为“万台”，统一转成“千台”以衔接 SSP245 预测文件。
        national_stock[y] = total_stock_wan * 10.0

    s_hist = pd.Series(national_stock, dtype=float).sort_index()

    # 2000 年全国总户数（万户）用于构造 1990 初值
    pop_2000 = 0.0
    hh_2000 = 0.0
    for prov in province_list:
        pop_val = to_numeric_safe(df_pop.at[prov, HIST_ANCHOR_YEAR])
        hh_val = to_numeric_safe(df_hh.at[prov, HIST_ANCHOR_YEAR])
        if np.isfinite(pop_val) and np.isfinite(hh_val) and hh_val > 0:
            pop_2000 += pop_val
            hh_2000 += pop_val / hh_val

    init_stock_1990_qiantai = (INITIAL_STOCK_PER_100_HH_1990 / 100.0) * hh_2000 * 10.0
    return s_hist, init_stock_1990_qiantai


def build_historical_provincial_stock_and_init() -> tuple[pd.DataFrame, pd.Series]:
    df_ac_u = standardize_panel(pd.read_excel(p(FILE_AC_URBAN), index_col=0))
    df_ac_r = standardize_panel(pd.read_excel(p(FILE_AC_RURAL), index_col=0))
    df_pop = standardize_panel(pd.read_excel(p(FILE_POP), index_col=0))
    df_urban = standardize_panel(pd.read_excel(p(FILE_URBAN_RATE), index_col=0))
    df_hh = standardize_panel(pd.read_excel(p(FILE_HH_SIZE), index_col=0))

    df_ac_u = df_ac_u.iloc[1:, :]
    df_ac_r = df_ac_r.iloc[1:, :]
    df_pop = df_pop.iloc[1:, :]
    df_urban = df_urban.iloc[1:, :]
    df_hh = df_hh.iloc[1:, :]

    urb_years = sorted([c for c in df_urban.columns if isinstance(c, int)])
    df_urban[urb_years] = (
        df_urban[urb_years]
        .apply(pd.to_numeric, errors='coerce')
        .bfill(axis=1)
        .ffill(axis=1)
    )

    province_list = sorted(
        set(df_ac_u.index)
        & set(df_ac_r.index)
        & set(df_pop.index)
        & set(df_urban.index)
        & set(df_hh.index)
    )

    year_candidates = [
        {c for c in df_ac_u.columns if isinstance(c, int)},
        {c for c in df_ac_r.columns if isinstance(c, int)},
        {c for c in df_pop.columns if isinstance(c, int)},
        {c for c in df_urban.columns if isinstance(c, int)},
        {c for c in df_hh.columns if isinstance(c, int)},
    ]
    years = sorted(set.intersection(*year_candidates))
    years = [y for y in years if y <= 2024]

    stock_panel = pd.DataFrame(index=years, columns=province_list, dtype=float)
    for y in years:
        for prov in province_list:
            pr_u = to_numeric_safe(df_ac_u.at[prov, y]) / 100.0
            pr_r = to_numeric_safe(df_ac_r.at[prov, y]) / 100.0
            pop_tot = to_numeric_safe(df_pop.at[prov, y])
            urban_rate = to_numeric_safe(df_urban.at[prov, y])
            hh_size = to_numeric_safe(df_hh.at[prov, y])

            if not np.isfinite(pop_tot) or not np.isfinite(urban_rate) or not np.isfinite(hh_size) or hh_size <= 0:
                stock_panel.at[y, prov] = np.nan
                continue

            if np.isnan(pr_u):
                pr_u = 0.0
            if np.isnan(pr_r):
                pr_r = 0.0

            if urban_rate > 1.5:
                urban_rate = urban_rate / 100.0
            urban_rate = float(np.clip(urban_rate, 0.0, 1.0))

            hh_u = pop_tot * urban_rate / hh_size
            hh_r = pop_tot * (1.0 - urban_rate) / hh_size
            stock_wan = pr_u * hh_u + pr_r * hh_r

            stock_panel.at[y, prov] = stock_wan * 10.0 if np.isfinite(stock_wan) else np.nan

    hh_2000_by_prov = pd.Series(index=province_list, dtype=float)
    for prov in province_list:
        pop_val = to_numeric_safe(df_pop.at[prov, HIST_ANCHOR_YEAR])
        hh_val = to_numeric_safe(df_hh.at[prov, HIST_ANCHOR_YEAR])
        if np.isfinite(pop_val) and np.isfinite(hh_val) and hh_val > 0:
            hh_2000_by_prov.at[prov] = pop_val / hh_val
        else:
            hh_2000_by_prov.at[prov] = np.nan

    init_stock_1990_by_prov = (INITIAL_STOCK_PER_100_HH_1990 / 100.0) * hh_2000_by_prov * 10.0
    return stock_panel.sort_index(), init_stock_1990_by_prov


def _choose_provincial_anchor(stock_series: pd.Series) -> tuple[int | None, float | None]:
    valid = pd.to_numeric(stock_series, errors='coerce')
    valid = valid[(valid > 0) & np.isfinite(valid)]
    if valid.empty:
        return None, None

    anchor_year = int(valid.index[0])
    anchor_value = float(valid.iloc[0])
    return anchor_year, anchor_value


def _parse_year_value(v) -> int | None:
    if isinstance(v, (int, np.integer)):
        y = int(v)
        if 1900 <= y <= 2100:
            return y
        return None
    if isinstance(v, float) and np.isfinite(v):
        if float(v).is_integer():
            y = int(v)
            if 1900 <= y <= 2100:
                return y
        return None

    s = str(v).strip()
    m = re.search(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', s)
    if m:
        return int(m.group(1))
    return None


def _parse_provincial_forecast_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_cols = {c.lower(): c for c in df.columns}
    prov_col = None
    year_col = None
    stock_col = None

    prov_candidates = ['province', 'prov', 'region', '省份', '省', '地区', '行政区']
    year_candidates = ['year', '年份', '年']
    stock_candidates = ['总保有量', '保有量', '总量']

    # 优先精确识别总保有量字段，避免误选 Urban_Stock / Rural_Stock。
    for key, raw in lower_cols.items():
        if key in {'total_stock', 'total stock', 'totalstock'}:
            stock_col = raw
            break

    for key, raw in lower_cols.items():
        if prov_col is None and any(k in key for k in prov_candidates):
            prov_col = raw
        if year_col is None and any(k == key or k in key for k in year_candidates):
            year_col = raw
        if stock_col is None and any(k in key for k in stock_candidates):
            stock_col = raw

    if prov_col is not None and year_col is not None and stock_col is not None:
        out = pd.DataFrame({
            'Province': df[prov_col].map(_normalize_province_name),
            'Year': df[year_col].map(_parse_year_value),
            'Total_Stock': pd.to_numeric(df[stock_col], errors='coerce'),
        })
        out = out.dropna(subset=['Province', 'Year', 'Total_Stock'])
        out['Year'] = out['Year'].astype(int)
        return out

    year_cols = []
    year_map = {}
    for c in df.columns:
        y = _parse_year_value(c)
        if y is not None:
            year_cols.append(c)
            year_map[c] = y

    if len(year_cols) < 3:
        return pd.DataFrame(columns=['Province', 'Year', 'Total_Stock'])

    candidate_prov_cols = [c for c in df.columns if c not in year_cols]
    if candidate_prov_cols:
        prov_col_wide = candidate_prov_cols[0]
        wide = df[[prov_col_wide] + year_cols].copy()
        wide = wide.rename(columns={prov_col_wide: 'Province'})
    else:
        wide = df[year_cols].copy().reset_index()
        wide = wide.rename(columns={wide.columns[0]: 'Province'})

    long_df = wide.melt(id_vars='Province', value_vars=year_cols, var_name='YearRaw', value_name='Total_Stock')
    long_df['Year'] = long_df['YearRaw'].map(year_map)
    long_df['Province'] = long_df['Province'].map(_normalize_province_name)
    long_df['Total_Stock'] = pd.to_numeric(long_df['Total_Stock'], errors='coerce')
    long_df = long_df.drop(columns=['YearRaw'])
    long_df = long_df.dropna(subset=['Province', 'Year', 'Total_Stock'])
    long_df['Year'] = long_df['Year'].astype(int)
    return long_df


def build_ssp245_forecast_national_stock(ssp_scenario: str = 'SSP245') -> pd.Series:
    sheet = f'{ssp_scenario}_national'
    nat = pd.read_excel(p(FILE_FORECAST), sheet_name=sheet)
    nat['Year'] = pd.to_numeric(nat['Year'], errors='coerce')
    nat['Total_Stock'] = pd.to_numeric(nat['Total_Stock'], errors='coerce')
    nat = nat.dropna(subset=['Year', 'Total_Stock']).copy()
    nat['Year'] = nat['Year'].astype(int)
    return nat.set_index('Year')['Total_Stock'].sort_index()


def build_ssp245_forecast_provincial_stock(ssp_scenario: str = 'SSP245') -> pd.DataFrame:
    xls = pd.ExcelFile(p(FILE_FORECAST))
    records = []

    preferred_sheets = [
        s for s in xls.sheet_names
        if (ssp_scenario.lower() in s.lower()) and ('province' in s.lower() or '省' in s)
    ]
    candidate_sheets = preferred_sheets if preferred_sheets else xls.sheet_names

    for sheet in candidate_sheets:
        sheet_l = sheet.lower()
        if 'national' in sheet_l or '全国' in sheet_l:
            continue

        raw = pd.read_excel(p(FILE_FORECAST), sheet_name=sheet)
        parsed = _parse_provincial_forecast_sheet(raw)
        if parsed.empty:
            continue

        parsed['__sheet__'] = sheet
        records.append(parsed)

    if not records:
        raise ValueError(f'未在预测文件中解析出省级 {ssp_scenario} 保有量。')

    all_long = pd.concat(records, ignore_index=True)
    all_long = all_long[(all_long['Year'] >= 2025) & (all_long['Year'] <= FORECAST_END)].copy()
    all_long = all_long[~all_long['Province'].isin(['全国', 'China', 'CHINA'])]

    if all_long.empty:
        raise ValueError(f'省级预测数据存在，但 2025-{FORECAST_END} 区间无可用值（情景：{ssp_scenario}）。')

    all_long = all_long.sort_values(['Province', 'Year', '__sheet__'])
    all_long = all_long.drop_duplicates(subset=['Province', 'Year'], keep='last')

    panel = all_long.pivot(index='Year', columns='Province', values='Total_Stock').sort_index()
    panel.columns = [str(c).strip() for c in panel.columns]
    return panel


def _extrapolate_to_year(s: pd.Series, target_year: int) -> pd.Series:
    """将 stock 序列线性外推至 target_year（基于最后两个已知点的斜率）。"""
    valid = s.dropna().sort_index()
    if valid.empty:
        return s
    last_yr = int(valid.index.max())
    if last_yr >= target_year:
        return s
    slope = (float(valid.iloc[-1]) - float(valid.iloc[-2])) if len(valid) >= 2 else 0.0
    base_val = float(valid.iloc[-1])
    ext = pd.Series(
        {y: max(0.0, base_val + slope * (y - last_yr)) for y in range(last_yr + 1, target_year + 1)},
        dtype=float,
    )
    return pd.concat([s, ext]).sort_index()


def build_complete_stock_series(ssp_scenario: str = 'SSP245') -> pd.Series:
    hist_stock, init_stock_1990 = build_historical_national_stock()
    forecast_stock = build_ssp245_forecast_national_stock(ssp_scenario)

    if HIST_ANCHOR_YEAR not in hist_stock.index:
        raise ValueError('历史 stock 序列缺少 2000 年，无法构造 1990-2000 指数增长边界。')

    stock_2000 = float(hist_stock.loc[HIST_ANCHOR_YEAR])
    if stock_2000 <= 0:
        raise ValueError('2000 年保有量非正，无法构造指数增长边界。')

    years_early = list(range(HIST_START, HIST_ANCHOR_YEAR + 1))
    r = np.log(stock_2000 / init_stock_1990) / (HIST_ANCHOR_YEAR - HIST_START)

    early_values = [init_stock_1990 * np.exp(r * (y - HIST_START)) for y in years_early]
    early_stock = pd.Series(early_values, index=years_early, dtype=float)

    full = pd.concat([
        early_stock,
        hist_stock[hist_stock.index > HIST_ANCHOR_YEAR],
        forecast_stock[forecast_stock.index >= 2025],
    ])

    full = full[~full.index.duplicated(keep='first')].sort_index()
    full = full[(full.index >= HIST_START) & (full.index <= FORECAST_END)]
    # 若预测文件不覆盖 FORECAST_END，则线性外推
    full = _extrapolate_to_year(full, FORECAST_END)
    return full


def build_complete_provincial_stock_panel(ssp_scenario: str = 'SSP245') -> pd.DataFrame:
    hist_panel, init_stock_1990_by_prov = build_historical_provincial_stock_and_init()

    try:
        forecast_panel = build_ssp245_forecast_provincial_stock(ssp_scenario)
    except Exception as ex:
        print(f'警告：省级预测解析失败，将按 2024 年省份占比拆分全国预测。原因: {ex}')
        forecast_nat = build_ssp245_forecast_national_stock(ssp_scenario)

        if 2024 not in hist_panel.index:
            raise ValueError('缺少 2024 年省级历史保有量，无法拆分全国预测。')

        shares = hist_panel.loc[2024].copy()
        shares = shares / shares.sum()

        forecast_panel = pd.DataFrame(index=forecast_nat.index, columns=hist_panel.columns, dtype=float)
        for y in forecast_nat.index:
            forecast_panel.loc[y, :] = shares.values * float(forecast_nat.loc[y])

    prov_list = sorted(set(hist_panel.columns) | set(forecast_panel.columns))
    all_years = list(range(HIST_START, FORECAST_END + 1))
    out_panel = pd.DataFrame(index=all_years, columns=prov_list, dtype=float)

    for prov in prov_list:
        hist_s = hist_panel[prov] if prov in hist_panel.columns else pd.Series(dtype=float)
        fc_s = forecast_panel[prov] if prov in forecast_panel.columns else pd.Series(dtype=float)

        anchor_year, anchor_value = _choose_provincial_anchor(hist_s)
        if anchor_year is None or anchor_value is None:
            continue

        init_1990 = float(init_stock_1990_by_prov.get(prov, np.nan))
        if not np.isfinite(init_1990) or init_1990 <= 0:
            init_1990 = max(anchor_value * 1e-6, 1e-9)

        years_early = list(range(HIST_START, anchor_year + 1))
        if anchor_year == HIST_START:
            early_stock = pd.Series([anchor_value], index=years_early, dtype=float)
        else:
            r = np.log(anchor_value / init_1990) / (anchor_year - HIST_START)
            early_values = [init_1990 * np.exp(r * (y - HIST_START)) for y in years_early]
            early_values[-1] = anchor_value
            early_stock = pd.Series(early_values, index=years_early, dtype=float)

        full_s = pd.concat([
            early_stock,
            hist_s[hist_s.index > anchor_year],
            fc_s[fc_s.index >= 2025],
        ])

        full_s = full_s[~full_s.index.duplicated(keep='first')].sort_index()
        full_s = full_s[(full_s.index >= HIST_START) & (full_s.index <= FORECAST_END)]
        # 若预测文件不覆盖 FORECAST_END，则线性外推
        full_s = _extrapolate_to_year(full_s, FORECAST_END)
        out_panel.loc[full_s.index, prov] = full_s.values

    return out_panel


def estimate_sales_from_stock(stock_series: pd.Series) -> pd.DataFrame:
    years = sorted(stock_series.index.tolist())
    sales = {}
    retire = {}

    for y in years:
        stock_y = float(stock_series.loc[y])

        if y == years[0]:
            retire[y] = 0.0
            sales[y] = stock_y
            continue

        stock_prev = float(stock_series.loc[y - 1]) if (y - 1) in stock_series.index else np.nan
        if not np.isfinite(stock_prev):
            raise ValueError(f'缺少 {y - 1} 年保有量，无法递推到 {y} 年。')

        if y <= HIST_ANCHOR_YEAR:
            retire_y = 0.0
        else:
            retire_y = 0.0
            for y0, sale_y0 in sales.items():
                if y0 >= y:
                    continue
                age = y - y0
                retire_y += sale_y0 * weibull_discard_prob(age, WEIBULL_SHAPE, WEIBULL_SCALE)

        sales_y = stock_y - stock_prev + retire_y

        retire[y] = retire_y
        sales[y] = sales_y

    out = pd.DataFrame({
        'Year': years,
        'Stock_Thousand_Units': [stock_series.loc[y] for y in years],
        'Retirements_Thousand_Units': [retire[y] for y in years],
        'New_Sales_Thousand_Units': [sales[y] for y in years],
    })

    out['Stock_Units'] = out['Stock_Thousand_Units'] * 1000.0
    out['Retirements_Units'] = out['Retirements_Thousand_Units'] * 1000.0
    out['New_Sales_Units'] = out['New_Sales_Thousand_Units'] * 1000.0

    out['Implied_Stock_Balance_Error'] = (
        out['Stock_Thousand_Units']
        - out['Stock_Thousand_Units'].shift(1)
        - out['New_Sales_Thousand_Units']
        + out['Retirements_Thousand_Units']
    )
    out.loc[out['Year'] == HIST_START, 'Implied_Stock_Balance_Error'] = 0.0

    return out


def estimate_sales_panel_from_stock(stock_panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for prov in stock_panel.columns:
        s = pd.to_numeric(stock_panel[prov], errors='coerce').dropna()
        if s.empty:
            continue

        est = estimate_sales_from_stock(s)
        est.insert(0, 'Province', prov)
        records.append(est)

    if not records:
        return pd.DataFrame(columns=[
            'Province', 'Year',
            'Stock_Thousand_Units', 'Retirements_Thousand_Units', 'New_Sales_Thousand_Units',
            'Stock_Units', 'Retirements_Units', 'New_Sales_Units',
            'Implied_Stock_Balance_Error',
        ])

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(['Province', 'Year']).reset_index(drop=True)
    return out



def _run_single_ssp(ssp_scenario: str):
    """对单个 SSP 情景运行威布尔产品流反推，返回 (national_df, provincial_df)。"""
    print(f'  [{ssp_scenario}] 正在构建全国保有量序列...')
    stock_series = build_complete_stock_series(ssp_scenario)
    result = estimate_sales_from_stock(stock_series)
    result.insert(0, 'SSP_Scenario', ssp_scenario)

    print(f'  [{ssp_scenario}] 正在构建省级保有量序列并反推省级销量...')
    prov_stock_panel = build_complete_provincial_stock_panel(ssp_scenario)
    prov_result = estimate_sales_panel_from_stock(prov_stock_panel)
    prov_result.insert(0, 'SSP_Scenario', ssp_scenario)

    return result, prov_result


def main():
    all_nat = []
    all_prov = []

    for ssp in SSP_SCENARIOS:
        nat_df, prov_df = _run_single_ssp(ssp)
        all_nat.append(nat_df)
        all_prov.append(prov_df)

    nat_combined = pd.concat(all_nat, ignore_index=True)
    prov_combined = pd.concat(all_prov, ignore_index=True)

    # 省级加总校验（以 SSP245 为参照）
    ref_nat = nat_combined[nat_combined['SSP_Scenario'] == 'SSP245'].copy()
    ref_prov = prov_combined[prov_combined['SSP_Scenario'] == 'SSP245'].copy()
    neg_cnt = int((ref_nat['New_Sales_Thousand_Units'] < 0).sum())
    neg_cnt_prov = int((ref_prov['New_Sales_Thousand_Units'] < 0).sum())

    prov_agg = (
        ref_prov
        .groupby('Year', as_index=False)[['Stock_Thousand_Units', 'Retirements_Thousand_Units', 'New_Sales_Thousand_Units']]
        .sum()
        .rename(columns={
            'Stock_Thousand_Units': 'Prov_Sum_Stock_Thousand_Units',
            'Retirements_Thousand_Units': 'Prov_Sum_Retirements_Thousand_Units',
            'New_Sales_Thousand_Units': 'Prov_Sum_New_Sales_Thousand_Units',
        })
    )
    nat_cmp = ref_nat[['Year', 'Stock_Thousand_Units', 'Retirements_Thousand_Units', 'New_Sales_Thousand_Units']].copy()
    nat_cmp = nat_cmp.rename(columns={
        'Stock_Thousand_Units': 'National_Stock_Thousand_Units',
        'Retirements_Thousand_Units': 'National_Retirements_Thousand_Units',
        'New_Sales_Thousand_Units': 'National_New_Sales_Thousand_Units',
    })
    prov_nat_check = nat_cmp.merge(prov_agg, on='Year', how='left')
    prov_nat_check['Stock_Diff_Thousand_Units'] = (
        prov_nat_check['Prov_Sum_Stock_Thousand_Units'] - prov_nat_check['National_Stock_Thousand_Units']
    )
    prov_nat_check['New_Sales_Diff_Thousand_Units'] = (
        prov_nat_check['Prov_Sum_New_Sales_Thousand_Units'] - prov_nat_check['National_New_Sales_Thousand_Units']
    )

    with pd.ExcelWriter(p(OUT_FILE)) as writer:
        nat_combined.to_excel(writer, sheet_name='national_sales_flow', index=False)
        prov_combined.to_excel(writer, sheet_name='provincial_sales_flow', index=False)
        prov_nat_check.to_excel(writer, sheet_name='province_national_check_ssp245', index=False)

        summary = pd.DataFrame({
            'Metric': [
                'Weibull_shape_k',
                'Weibull_scale_lambda_years',
                'Initial_stock_1990_per100hh',
                'SSP_scenarios',
                'Negative_sales_year_count_SSP245',
                'Negative_sales_province_year_count_SSP245',
            ],
            'Value': [
                WEIBULL_SHAPE,
                WEIBULL_SCALE,
                INITIAL_STOCK_PER_100_HH_1990,
                str(SSP_SCENARIOS),
                neg_cnt,
                neg_cnt_prov,
            ],
        })
        summary.to_excel(writer, sheet_name='summary', index=False)

    print(f'\n完成！所有 SSP 情景结果已输出到: {p(OUT_FILE)}')
    print('说明：SSP_Scenario 列区分 SSP126/SSP245/SSP585，单位主口径为"千台"，并同时给出"台"。')


if __name__ == '__main__':
    main()

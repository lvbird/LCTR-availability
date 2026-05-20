import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 确保工作目录为脚本所在目录（避免从父目录运行时相对路径失效）
os.chdir(Path(__file__).resolve().parent)

# ----------------- 1. 内置参数与预定义 -----------------
CDD_FILE = 'CDD_2000_2045_yearly_by_province_and_ssp.xlsx'
HOUSEHOLD_SIZE_FILE = 'household_size.xlsx'
PROVINCE_ALPHA_CAP = {
    '云南': 1.0, '吉林': 0.5, '四川': 2.5, '天津': 2.5, '山东': 1.8, '山西': 1.8,
    '广东': 2.5, '广西': 2.8, '新疆': 1.6, '河南': 2.5, '甘肃': 1.0, '西藏': 0.5,
    '辽宁': 1.5, '重庆': 2.5, '陕西': 1.8, '青海': 0.5
}

def calc_cms(cdd):
    # 公式2：计算天花板 CMS，支持标量或数组输入
    cdd = np.asarray(cdd, dtype=float)
    return 1 - 0.949 * np.exp(-0.00187 * cdd)

def logistic_func(hhi, alpha, beta, gamma):
    # 公式3：Logistic 回归函数 (拟合目标)
    return alpha / (1 + gamma * np.exp(-beta * hhi))

def logistic_func_fixed_alpha(hhi, beta, gamma, alpha_fixed):
    # 固定 alpha 时，仅对 beta、gamma 进行二次拟合
    return alpha_fixed / (1 + gamma * np.exp(-beta * hhi))

def standardize_panel(df):
    # 清洗索引和列名，并尽可能把年份列转为 int
    out = df.copy()
    out.index = out.index.map(lambda x: str(x).strip())

    new_cols = []
    for c in out.columns:
        c_str = str(c).strip()
        try:
            c_num = float(c_str)
            if c_num.is_integer():
                new_cols.append(int(c_num))
            else:
                new_cols.append(c_str)
        except Exception:
            new_cols.append(c_str)
    out.columns = new_cols
    return out

def best_province_match(prov_name, candidate_index):
    # 省名对齐：先精确匹配，再做包含匹配
    candidate_index = [str(x).strip() for x in candidate_index]
    if prov_name in candidate_index:
        return prov_name

    matches = [x for x in candidate_index if prov_name in x or x in prov_name]
    if matches:
        return matches[0]
    return None

def build_cdd_panel(cdd_excel_path):
    # 读取动态 CDD，输出：index=Province, columns=Year, values=CDD
    sheets = pd.read_excel(cdd_excel_path, sheet_name=None)

    # 1) 优先用 summary_long（结构最稳）
    if 'summary_long' in sheets:
        s = sheets['summary_long'].copy()
        s.columns = [str(c).strip() for c in s.columns]

        province_col = 'province' if 'province' in s.columns else None
        year_col = 'year' if 'year' in s.columns else None

        cdd_candidates = [c for c in s.columns if c.lower().startswith('cdd_')]
        preferred_order = ['CDD_ssp245', 'CDD_ssp126', 'CDD_ssp585']
        cdd_col = None
        for p in preferred_order:
            if p in s.columns:
                cdd_col = p
                break
        if cdd_col is None and cdd_candidates:
            cdd_col = cdd_candidates[0]

        if province_col is not None and year_col is not None and cdd_col is not None:
            tmp = s[[province_col, year_col, cdd_col]].copy()
            tmp.columns = ['Province', 'Year', 'CDD']
            tmp['Province'] = tmp['Province'].astype(str).str.strip()
            tmp['Year'] = pd.to_numeric(tmp['Year'], errors='coerce')
            tmp['CDD'] = pd.to_numeric(tmp['CDD'], errors='coerce')
            tmp = tmp.dropna(subset=['Province', 'Year', 'CDD'])
            tmp['Year'] = tmp['Year'].astype(int)
            panel = tmp.pivot_table(index='Province', columns='Year', values='CDD', aggfunc='mean')
            return standardize_panel(panel)

    # 2) 回退到 yearly 宽表（year 在行，province 在列）
    for sheet_name in ['ssp245_yearly', 'ssp126_yearly', 'ssp585_yearly']:
        if sheet_name not in sheets:
            continue

        w = sheets[sheet_name].copy()
        w.columns = [str(c).strip() for c in w.columns]

        year_col = None
        for c in w.columns:
            c_low = str(c).lower()
            if c_low == 'year' or c_low.startswith('unnamed'):
                year_col = c
                break
        if year_col is None:
            year_col = w.columns[0]

        w = w.rename(columns={year_col: 'Year'})
        w['Year'] = pd.to_numeric(w['Year'], errors='coerce')
        w = w.dropna(subset=['Year'])
        w['Year'] = w['Year'].astype(int)

        province_cols = [c for c in w.columns if c != 'Year']
        if not province_cols:
            continue

        # 转成省 x 年
        panel = w.set_index('Year')[province_cols].T
        panel.index.name = 'Province'
        panel.columns.name = None
        panel = standardize_panel(panel)
        return panel

    raise ValueError(f'无法从 {cdd_excel_path} 解析出动态 CDD 数据，请检查文件格式。')

def build_household_size_panels(hh_excel_path):
    # household_size.xlsx 当前为单表（地区 x 年），城镇/农村共用
    sheets = pd.read_excel(hh_excel_path, sheet_name=None)
    first_name = list(sheets.keys())[0]
    df = sheets[first_name].copy()
    df.columns = [str(c).strip() for c in df.columns]

    if '地区' in df.columns:
        df = df.set_index('地区')
    else:
        df = df.set_index(df.columns[0])

    panel = standardize_panel(df)
    year_cols = [c for c in panel.columns if isinstance(c, int)]
    panel = panel[year_cols]

    # 如果后续有分城乡文件，可在这里替换为分别读取
    return panel.copy(), panel.copy()

# ----------------- 2. 数据读取与预处理 -----------------
print('正在读取数据...')
df_ac_urban = pd.read_excel('AC_stock_urban_per_100_households.xlsx', index_col=0)
df_ac_rural = pd.read_excel('AC_stock_rural_per_100_households.xlsx', index_col=0)
df_income_urban = pd.read_excel('Income_urban.xlsx', index_col=0)
df_income_rural = pd.read_excel('Income_rural.xlsx', index_col=0)
df_cpi = pd.read_excel('CPI.xlsx', index_col=0)

df_ac_urban = standardize_panel(df_ac_urban)
df_ac_rural = standardize_panel(df_ac_rural)
df_income_urban = standardize_panel(df_income_urban)
df_income_rural = standardize_panel(df_income_rural)
df_cpi = standardize_panel(df_cpi)

print('正在读取动态 CDD 与 household size...')
df_cdd_dynamic = build_cdd_panel(CDD_FILE)
df_hh_urban, df_hh_rural = build_household_size_panels(HOUSEHOLD_SIZE_FILE)

# --- CPI 处理：转不变价（基期设为 2000 年）---
print('正在剔除通货膨胀，计算 household real income...')
df_cpi_fixed = pd.DataFrame(index=df_cpi.index, columns=df_cpi.columns, dtype=float)
if 2000 in df_cpi.columns:
    df_cpi_fixed[2000] = 1.0
else:
    raise ValueError('CPI 数据中缺少 2000 年，无法按当前设定构造定基 CPI。')

for year in sorted([c for c in df_cpi.columns if isinstance(c, int) and c > 2000]):
    prev = year - 1
    if prev in df_cpi_fixed.columns:
        df_cpi_fixed[year] = df_cpi_fixed[prev] * (df_cpi[year] / 100.0)

# 1) 先把人均收入换成不变价
# 2) 再乘 household size 得到 household income（拟合使用）
df_income_urban_real_pc = df_income_urban / df_cpi_fixed
df_income_rural_real_pc = df_income_rural / df_cpi_fixed

df_income_urban_real_hh = df_income_urban_real_pc * df_hh_urban
df_income_rural_real_hh = df_income_rural_real_pc * df_hh_rural

# ----------------- 3. 核心拟合循环 -----------------
print('开始执行 Logistic 曲线拟合（动态 CDD + household income）...')
results = []

data_maps = {
    'Urban': {'ac': df_ac_urban, 'income_hh': df_income_urban_real_hh},
    'Rural': {'ac': df_ac_rural, 'income_hh': df_income_rural_real_hh}
}

province_base = sorted(set(df_ac_urban.index).union(set(df_ac_rural.index)))

for prov in province_base:
    prov_in_cdd = best_province_match(prov, df_cdd_dynamic.index)
    if prov_in_cdd is None:
        print(f'警告：动态 CDD 中未找到省份 {prov}，跳过该省。')
        continue

    for region_type, dfs in data_maps.items():
        prov_in_ac = best_province_match(prov, dfs['ac'].index)
        prov_in_income = best_province_match(prov, dfs['income_hh'].index)

        if prov_in_ac is None or prov_in_income is None:
            print(f'警告：{prov}-{region_type} 在 AC 或收入表中缺失，跳过。')
            continue

        ac_row = dfs['ac'].loc[prov_in_ac]
        income_row = dfs['income_hh'].loc[prov_in_income]
        cdd_row = df_cdd_dynamic.loc[prov_in_cdd]

        ac_years = {c for c in ac_row.index if isinstance(c, int)}
        income_years = {c for c in income_row.index if isinstance(c, int)}
        cdd_years = {c for c in cdd_row.index if isinstance(c, int)}
        common_years = sorted(ac_years & income_years & cdd_years)

        if len(common_years) < 5:
            print(f'{prov}-{region_type} 有效共同年份不足，跳过。')
            continue

        # PR (每户几台)
        pr_actual = ac_row[common_years].astype(float).values / 100.0
        # HHI: household income（不变价）
        hhi_actual = income_row[common_years].astype(float).values
        # 动态 CDD + 动态 CMS（按年份）
        cdd_actual = cdd_row[common_years].astype(float).values
        cms_actual = calc_cms(cdd_actual)

        # AF（按年份）
        af_actual = pr_actual / cms_actual

        valid_mask = (
            np.isfinite(af_actual)
            & np.isfinite(hhi_actual)
            & np.isfinite(cdd_actual)
            & np.isfinite(cms_actual)
            & (cms_actual > 0)
        )

        x_data = hhi_actual[valid_mask]
        y_data = af_actual[valid_mask]
        cdd_used = cdd_actual[valid_mask]
        cms_used = cms_actual[valid_mask]

        if len(x_data) < 5:
            print(f'{prov}-{region_type} 清洗后数据点不足，跳过。')
            continue

        initial_guess = [2.0, 0.0001, 10.0]
        alpha_upper_bound = 4.0
        alpha_prov_cap = PROVINCE_ALPHA_CAP.get(prov, alpha_upper_bound)
        param_bounds = ([0.1, 1e-8, 0.1], [alpha_upper_bound, 0.01, 10000])

        try:
            popt, pcov = curve_fit(
                logistic_func,
                x_data,
                y_data,
                p0=initial_guess,
                bounds=param_bounds,
                maxfev=100000
            )

            alpha_fit, beta_fit, gamma_fit = popt
            beta_raw, gamma_raw = beta_fit, gamma_fit
            alpha_capped = bool(alpha_fit > alpha_prov_cap + 1e-12)

            if alpha_capped:
                alpha_used = alpha_prov_cap
                p0_bg = [max(beta_fit, 1e-8), max(gamma_fit, 0.1)]
                popt_bg, _ = curve_fit(
                    lambda hhi, beta, gamma: logistic_func_fixed_alpha(hhi, beta, gamma, alpha_used),
                    x_data,
                    y_data,
                    p0=p0_bg,
                    bounds=([1e-8, 0.1], [0.01, 10000]),
                    maxfev=100000
                )
                beta_fit, gamma_fit = popt_bg
            else:
                alpha_used = alpha_fit
            y_pred = logistic_func(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            alpha_bound_hit = bool(np.isclose(alpha_fit, alpha_upper_bound, rtol=0, atol=1e-6))
            y_pred_used = logistic_func(x_data, alpha_used, beta_fit, gamma_fit)
            ss_res_used = np.sum((y_data - y_pred_used) ** 2)
            r_squared_used = 1 - (ss_res_used / ss_tot) if ss_tot > 0 else np.nan

        except Exception as e:
            alpha_fit, alpha_used, beta_fit, gamma_fit, beta_raw, gamma_raw, r_squared, r_squared_used = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            alpha_bound_hit = False
            alpha_capped = False
            alpha_prov_cap = PROVINCE_ALPHA_CAP.get(prov, np.nan)
            print(f'拟合失败: {prov}-{region_type} ({e})')

        results.append({
            'Province': prov,
            'Region_Type': region_type,
            'N_Obs': int(len(x_data)),
            'CDD_Mean_Used': float(np.nanmean(cdd_used)),
            'CMS_Mean_Used': float(np.nanmean(cms_used)),
            'Alpha': alpha_used,
            'Alpha_Raw': alpha_fit,
            'Alpha_Province_Cap': alpha_prov_cap,
            'Alpha_Capped': alpha_capped,
            'Beta': beta_fit,
            'Beta_Raw': beta_raw,
            'Gamma': gamma_fit,
            'Gamma_Raw': gamma_raw,
            'R_Square': r_squared,
            'R_Square_Used': r_squared_used,
            'Alpha_Bound_Hit': alpha_bound_hit
        })

# ----------------- 4. 导出结果 -----------------
df_results = pd.DataFrame(results)
output_file = 'Logistic_Fitting_Results_dynamic_CDD_hh_income.xlsx'
try:
    df_results.to_excel(output_file, index=False)
except PermissionError:
    output_file = 'Logistic_Fitting_Results_dynamic_CDD_hh_income_v2.xlsx'
    df_results.to_excel(output_file, index=False)
print(f'拟合完成！结果已保存至: {output_file}')
print('该版本已使用动态 CDD（逐年）与 household real income（人均不变价收入 × household size）。')



# ----------------- 预测口径修正（新增） -----------------


# 目标年份
BASE_YEAR = 2024
END_YEAR = 2050
HH_FLOOR_2050 = 2.3

# 省份英文列名到中文省名映射（与文献预测CSV对齐）
EN2CN = {
    'Beijing': '北京', 'Tianjin': '天津', 'Hebei': '河北', 'Shanxi': '山西', 'NeiMongol': '内蒙古',
    'Liaoning': '辽宁', 'Jilin': '吉林', 'Heilongjiang': '黑龙江', 'Shanghai': '上海', 'Jiangsu': '江苏',
    'Zhejiang': '浙江', 'Anhui': '安徽', 'Fujian': '福建', 'Jiangxi': '江西', 'Shandong': '山东',
    'Henan': '河南', 'Hubei': '湖北', 'Hunan': '湖南', 'Guangdong': '广东', 'Guangxi': '广西',
    'Hainan': '海南', 'Chongqing': '重庆', 'Sichuan': '四川', 'Guizhou': '贵州', 'Yunnan': '云南',
    'Xizang': '西藏', 'Shaanxi': '陕西', 'Gansu': '甘肃', 'Qinghai': '青海', 'Ningxia': '宁夏',
    'Xinjiang': '新疆'
}

if 'df_income_urban_real_pc' not in globals() or 'df_income_rural_real_pc' not in globals():
    raise RuntimeError('请先运行第1个单元，再运行本单元。')

# ---------- 5.1 household size 外推到2050 ----------
def project_household_size_to_2050(hh_panel, base_year=2024, end_year=2050, floor_2050=2.3):
    """
    用“指数趋近下限”外推户规模：
    hs_t = floor + (hs_base - floor) * exp(-k*(t-base_year))

    说明：
    - 相比线性外推，指数方式更平滑，且不会在后期出现不合理负值。
    - k 由各省近10年变化自动估计；若数据不足则使用温和值。
    """
    panel = hh_panel.copy()
    panel.columns = [int(c) if str(c).isdigit() else c for c in panel.columns]

    hist_years = sorted([c for c in panel.columns if isinstance(c, int)])
    if base_year not in hist_years:
        raise ValueError(f'household size 数据缺少基准年 {base_year}。')

    future_years = list(range(base_year + 1, end_year + 1))
    out = panel.copy()

    for prov in out.index:
        y0 = pd.to_numeric(out.at[prov, base_year], errors='coerce')
        if not np.isfinite(y0):
            continue

        # 若已低于下限，不再继续下降
        floor = min(floor_2050, y0)

        # 用近10年估计衰减速度 k
        back_year = max(hist_years[0], base_year - 10)
        yb = pd.to_numeric(out.at[prov, back_year], errors='coerce') if back_year in out.columns else np.nan

        if np.isfinite(yb) and (yb - floor) > 1e-6 and (y0 - floor) > 1e-6:
            span = base_year - back_year
            k = -np.log((y0 - floor) / (yb - floor)) / max(span, 1)
            k = float(np.clip(k, 0.0, 0.30))
        else:
            k = 0.03

        for y in future_years:
            out.at[prov, y] = floor + (y0 - floor) * np.exp(-k * (y - base_year))

    # 强制不低于 floor_2050
    for y in future_years:
        out[y] = np.maximum(pd.to_numeric(out[y], errors='coerce'), floor_2050)

    return out

# 当前 household_size.xlsx 为单表，这里城乡共用；若后续有分城乡表可替换
hh_base_panel = df_hh_urban.copy()
df_hh_projected = project_household_size_to_2050(
    hh_base_panel,
    base_year=BASE_YEAR,
    end_year=END_YEAR,
    floor_2050=HH_FLOOR_2050
)

# ---------- 5.2 文献预测收入口径转换（基于增长率） ----------
def read_lit_income_csv(path):
    raw = pd.read_csv(path)
    # 删除无意义首列（通常是行号）
    if raw.columns[0] not in ['Year', 'year']:
        raw = raw.drop(columns=[raw.columns[0]])

    year_col = 'Year' if 'Year' in raw.columns else 'year'
    raw[year_col] = pd.to_numeric(raw[year_col], errors='coerce').astype('Int64')
    raw = raw.dropna(subset=[year_col]).copy()
    raw[year_col] = raw[year_col].astype(int)

    # 英文省名转中文
    cols = [c for c in raw.columns if c != year_col]
    rename_map = {c: EN2CN[c] for c in cols if c in EN2CN}
    raw = raw.rename(columns=rename_map)

    keep_cols = [year_col] + [c for c in raw.columns if c in EN2CN.values()]
    raw = raw[keep_cols]

    panel = raw.set_index(year_col).T
    panel.index.name = 'Province'
    panel.columns.name = None

    # 自动识别是否为对数口径：若中位数 < 30，默认是 ln(value)
    vals = pd.to_numeric(panel.stack(), errors='coerce')
    is_log_scale = np.nanmedian(vals) < 30

    if is_log_scale:
        panel_level = np.exp(panel.astype(float))
    else:
        panel_level = panel.astype(float)

    return panel_level

def build_equivalent_real_income(hist_real_pc, lit_level_panel, base_year=2024, end_year=2050):
    """
    规则：
    real_equiv[y] = real_hist[base_year] * lit_level[y] / lit_level[base_year]
    仅计算到 end_year。
    """
    years = [y for y in range(base_year, end_year + 1) if y in lit_level_panel.columns]
    out = pd.DataFrame(index=hist_real_pc.index, columns=years, dtype=float)

    for prov in out.index:
        prov_lit = prov if prov in lit_level_panel.index else None
        if prov_lit is None:
            continue

        base_real = pd.to_numeric(hist_real_pc.at[prov, base_year], errors='coerce') if base_year in hist_real_pc.columns else np.nan
        base_lit = pd.to_numeric(lit_level_panel.at[prov_lit, base_year], errors='coerce') if base_year in lit_level_panel.columns else np.nan

        if (not np.isfinite(base_real)) or (not np.isfinite(base_lit)) or abs(base_lit) < 1e-12:
            continue

        for y in years:
            lit_y = pd.to_numeric(lit_level_panel.at[prov_lit, y], errors='coerce')
            if np.isfinite(lit_y):
                out.at[prov, y] = base_real * (lit_y / base_lit)

    return out

urban_lit_level = read_lit_income_csv('Urban_income_prediction_raw.csv')
rural_lit_level = read_lit_income_csv('Rural_income_prediction_raw.csv')

df_income_urban_real_pc_proj = build_equivalent_real_income(
    hist_real_pc=df_income_urban_real_pc,
    lit_level_panel=urban_lit_level,
    base_year=BASE_YEAR,
    end_year=END_YEAR
)

df_income_rural_real_pc_proj = build_equivalent_real_income(
    hist_real_pc=df_income_rural_real_pc,
    lit_level_panel=rural_lit_level,
    base_year=BASE_YEAR,
    end_year=END_YEAR
)

# household income（预测）= real per-capita income（预测）* household size（预测）
common_years_urban = sorted(set(df_income_urban_real_pc_proj.columns) & set(df_hh_projected.columns))
common_years_rural = sorted(set(df_income_rural_real_pc_proj.columns) & set(df_hh_projected.columns))

df_income_urban_real_hh_proj = df_income_urban_real_pc_proj[common_years_urban] * df_hh_projected[common_years_urban]
df_income_rural_real_hh_proj = df_income_rural_real_pc_proj[common_years_rural] * df_hh_projected[common_years_rural]

# ---------- 5.3 导出中间结果 ----------
df_hh_projected.to_excel('household_size_projected_to_2050.xlsx')
df_income_urban_real_pc_proj.to_excel('Urban_income_real_pc_equivalent_2024_2050.xlsx')
df_income_rural_real_pc_proj.to_excel('Rural_income_real_pc_equivalent_2024_2050.xlsx')
df_income_urban_real_hh_proj.to_excel('Urban_income_real_hh_equivalent_2024_2050.xlsx')
df_income_rural_real_hh_proj.to_excel('Rural_income_real_hh_equivalent_2024_2050.xlsx')

print('口径修正完成（至2050年）：')
print('- household size 外推：household_size_projected_to_2050.xlsx')
print('- 城镇 real pc 等效预测：Urban_income_real_pc_equivalent_2024_2050.xlsx')
print('- 农村 real pc 等效预测：Rural_income_real_pc_equivalent_2024_2050.xlsx')
print('- 城镇 real household income：Urban_income_real_hh_equivalent_2024_2050.xlsx')
print('- 农村 real household income：Rural_income_real_hh_equivalent_2024_2050.xlsx')


# -----------------  未来保有量预测（到2045，三种CDD情景） -----------------

FORECAST_START = 2025
FORECAST_END = 2050
YEARS = list(range(FORECAST_START, FORECAST_END + 1))

# 平滑校准参数（用于缓解 2024->2025 拼接断点）
CALIBRATION_RATIO_CAP = 3.0
CALIBRATION_POWER = 1.0

# 与第2单元保持一致
EN2CN_EXT = {
    'Beijing': '北京', 'Tianjin': '天津', 'Hebei': '河北', 'Shanxi': '山西', 'NeiMongol': '内蒙古',
    'Inner Mongolia': '内蒙古',
    'Liaoning': '辽宁', 'Jilin': '吉林', 'Heilongjiang': '黑龙江', 'Shanghai': '上海', 'Jiangsu': '江苏',
    'Zhejiang': '浙江', 'Anhui': '安徽', 'Fujian': '福建', 'Jiangxi': '江西', 'Shandong': '山东',
    'Henan': '河南', 'Hubei': '湖北', 'Hunan': '湖南', 'Guangdong': '广东', 'Guangxi': '广西',
    'Hainan': '海南', 'Chongqing': '重庆', 'Sichuan': '四川', 'Guizhou': '贵州', 'Yunnan': '云南',
    'Xizang': '西藏', 'Shaanxi': '陕西', 'Gansu': '甘肃', 'Qinghai': '青海', 'Ningxia': '宁夏',
    'Xinjiang': '新疆'
}

PROVINCE_ALIASES = {
    'beijing': '北京', 'tianjin': '天津', 'hebei': '河北', 'shanxi': '山西',
    'neimongol': '内蒙古', 'neimonggol': '内蒙古', 'innermongolia': '内蒙古',
    'liaoning': '辽宁', 'jilin': '吉林', 'heilongjiang': '黑龙江',
    'shanghai': '上海', 'jiangsu': '江苏', 'jiangxi': '江西', 'zhejiang': '浙江',
    'anhui': '安徽', 'fujian': '福建', 'shandong': '山东', 'henan': '河南',
    'hubei': '湖北', 'hunan': '湖南', 'guangdong': '广东', 'guangxi': '广西',
    'hainan': '海南', 'chongqing': '重庆', 'sichuan': '四川', 'guizhou': '贵州',
    'yunnan': '云南', 'xizang': '西藏', 'tibet': '西藏',
    'shaanxi': '陕西', 'gansu': '甘肃', 'qinghai': '青海',
    'ningxia': '宁夏', 'xinjiang': '新疆'
}

def normalize_province_name(name):
    s = str(name).strip()
    if not s:
        return s
    if s in EN2CN_EXT.values():
        return s
    if s in EN2CN_EXT:
        return EN2CN_EXT[s]

    s_clean = ''.join(ch for ch in s.lower() if ch.isalnum())
    if s_clean == 'jiangsu1':
        return '江西'
    if s_clean in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[s_clean]
    if s_clean.endswith('1') and s_clean[:-1] in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[s_clean[:-1]]
    return s

def to_year_columns(df):
    out = df.copy()
    new_cols = []
    for c in out.columns:
        s = str(c).strip()
        if s.startswith('X') and s[1:].isdigit():
            new_cols.append(int(s[1:]))
        elif s.isdigit():
            new_cols.append(int(s))
        else:
            new_cols.append(s)
    out.columns = new_cols
    return out

def read_population_total_wpp(path, years, scenario='SSP2'):
    """
    从 Population_WPP.xlsx 读取全国总人口时间序列。
    若有情景列则优先取 scenario；否则取唯一序列。
    """
    sheets = pd.read_excel(path, sheet_name=None)

    for _, raw in sheets.items():
        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # 尝试宽表（年份在列）
        tmp = to_year_columns(df)
        year_cols = [c for c in tmp.columns if isinstance(c, int)]
        if len(year_cols) >= 10:
            non_year_cols = [c for c in tmp.columns if c not in year_cols]
            cand = tmp.copy()

            # 若有场景列，优先筛 scenario
            scen_col = None
            for c in non_year_cols:
                if any(k in str(c).lower() for k in ['ssp', 'scenario', '情景']):
                    scen_col = c
                    break
            if scen_col is not None:
                mask = cand[scen_col].astype(str).str.upper() == str(scenario).upper()
                if mask.any():
                    cand = cand[mask]

            # 若有“TOTAL/全国”列，优先筛
            text_cols = [c for c in non_year_cols if c != scen_col]
            for c in text_cols:
                m = cand[c].astype(str).str.upper().isin(['TOTAL', '全国'])
                if m.any():
                    cand = cand[m]
                    break

            if len(cand) >= 1:
                s = pd.to_numeric(cand.iloc[0][year_cols], errors='coerce')
                s.index = year_cols
                s = s.reindex(years)
                return s

        # 尝试长表（year/value）
        col_map = {str(c).strip().lower(): c for c in df.columns}
        year_col = col_map.get('year', None)
        if year_col is None:
            for k, v in col_map.items():
                if 'year' in k or '年份' in k:
                    year_col = v
                    break
        val_col = None
        for c in df.columns:
            c_low = str(c).lower()
            if any(k in c_low for k in ['population', 'pop', 'value', '数量', '总人口']):
                val_col = c
                break

        if year_col is not None and val_col is not None:
            d = df.copy()
            d['__year__'] = pd.to_numeric(d[year_col], errors='coerce')
            d['__val__'] = pd.to_numeric(d[val_col], errors='coerce')
            d = d.dropna(subset=['__year__', '__val__'])
            d['__year__'] = d['__year__'].astype(int)

            # 如果有情景列，则筛选
            scen_col = None
            for c in df.columns:
                if any(k in str(c).lower() for k in ['ssp', 'scenario', '情景']):
                    scen_col = c
                    break
            if scen_col is not None:
                m = d[scen_col].astype(str).str.upper() == str(scenario).upper()
                if m.any():
                    d = d[m]

            s = d.set_index('__year__')['__val__'].groupby(level=0).mean().reindex(years)
            if s.notna().any():
                return s

        # 回退：文件可能是两列数值且首行被当成表头（如 2025.0 / 1417733.636）
        try:
            nohead = pd.read_excel(path, sheet_name=0, header=None)
            if nohead.shape[1] >= 2:
                col_y = pd.to_numeric(nohead.iloc[:, 0], errors='coerce')
                col_v = pd.to_numeric(nohead.iloc[:, 1], errors='coerce')
                d0 = pd.DataFrame({'Year': col_y, 'Pop': col_v}).dropna()

                # 也把原先被当成列名的两个值拼回序列
                h_y = pd.to_numeric(pd.Series([df.columns[0]]), errors='coerce').iloc[0]
                h_v = pd.to_numeric(pd.Series([df.columns[1] if len(df.columns) > 1 else np.nan]), errors='coerce').iloc[0]
                if np.isfinite(h_y) and np.isfinite(h_v):
                    d0 = pd.concat([pd.DataFrame({'Year': [h_y], 'Pop': [h_v]}), d0], ignore_index=True)

                d0['Year'] = d0['Year'].astype(int)
                s0 = d0.groupby('Year')['Pop'].mean().reindex(years)
                if s0.notna().any():
                    return s0
        except Exception:
            pass

    raise ValueError('无法从 Population_WPP.xlsx 解析全国总人口序列。')

def read_pop_share_from_pop_total(path, years, scenario='SSP2'):
    """
    用 Pop_TOTAL.csv 的 SSP2 计算各省占全国比例（每年）。
    注意仅使用占比，不使用其绝对人口。
    """
    df = pd.read_csv(path)
    df = to_year_columns(df)

    # 识别前两列：地区与情景
    col0, col1 = df.columns[0], df.columns[1]
    year_cols = [c for c in df.columns if isinstance(c, int)]

    d = df.copy()
    d[col0] = d[col0].astype(str).str.strip()
    d[col1] = d[col1].astype(str).str.strip().str.upper()

    d = d[d[col1] == str(scenario).upper()]

    total_row = d[d[col0].str.upper() == 'TOTAL']
    if total_row.empty:
        raise ValueError('Pop_TOTAL.csv 中未找到 TOTAL 行。')

    total = pd.to_numeric(total_row.iloc[0][year_cols], errors='coerce')

    prov_rows = d[d[col0].str.upper() != 'TOTAL'].copy()
    prov_rows['Province'] = prov_rows[col0].map(normalize_province_name)

    share = pd.DataFrame(index=prov_rows['Province'].tolist(), columns=years, dtype=float)
    for _, r in prov_rows.iterrows():
        prov = r['Province']
        for y in years:
            if y in year_cols and y in total.index:
                denom = pd.to_numeric(total[y], errors='coerce')
                numer = pd.to_numeric(r[y], errors='coerce')
                if np.isfinite(denom) and abs(denom) > 1e-12 and np.isfinite(numer):
                    share.at[prov, y] = numer / denom

    # 同名省份（若映射冲突）取均值
    share = share.groupby(share.index).mean()
    return share

def read_urbanization_panel(path, years):
    """
    读取各省城镇化率，兼容：
    - 宽表（省份x年份）
    - 长表（province, year, value, scenario）

    返回 dict: {scenario_name: panel(index=Province, columns=Year)}
    若文件不含情景，则返回 {'all': panel}
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # 宽表尝试
    tmp = to_year_columns(df)
    year_cols = [c for c in tmp.columns if isinstance(c, int)]
    if len(year_cols) >= 5:
        non_year_cols = [c for c in tmp.columns if c not in year_cols]
        idx_col = non_year_cols[0] if non_year_cols else None
        panel = tmp.copy()
        if idx_col is not None:
            panel = panel.set_index(idx_col)
        panel.index = panel.index.map(normalize_province_name)
        panel = panel[year_cols]

        # 若包含场景列，拆分
        scen_col = None
        for c in non_year_cols:
            if any(k in str(c).lower() for k in ['ssp', 'scenario', '情景']):
                scen_col = c
                break
        if scen_col is not None:
            out = {}
            for scen, g in tmp.groupby(scen_col):
                p = g.set_index(idx_col)
                p.index = p.index.map(normalize_province_name)
                p = p[year_cols].groupby(level=0).mean().reindex(columns=years)
                out[str(scen)] = p
            return out

        return {'all': panel.groupby(level=0).mean().reindex(columns=years)}

    # 回退：年份在行、省份在列（当前 urbanization_rate.csv 即此格式）
    first_col = df.columns[0]
    year_try = pd.to_numeric(df[first_col], errors='coerce')
    if year_try.notna().sum() >= max(5, int(0.8 * len(df))):
        w = df.copy()
        w['__Year__'] = pd.to_numeric(w[first_col], errors='coerce').astype('Int64')
        w = w.dropna(subset=['__Year__']).copy()
        w['__Year__'] = w['__Year__'].astype(int)

        prov_cols = [c for c in w.columns if c not in [first_col, '__Year__']]
        p = w.set_index('__Year__')[prov_cols].T
        p.index = p.index.map(normalize_province_name)
        p = p.groupby(level=0).mean()

        # 百分比转比例（若需要）
        med = np.nanmedian(pd.to_numeric(p.stack(), errors='coerce'))
        if np.isfinite(med) and med > 1.5:
            p = p / 100.0

        return {'all': p.reindex(columns=years)}

    # 长表尝试
    col_map = {str(c).strip().lower(): c for c in df.columns}
    prov_col = None
    for k in ['province', 'prov', 'region', '地区', '省份']:
        if k in col_map:
            prov_col = col_map[k]
            break
    year_col = None
    for k in ['year', '年份']:
        if k in col_map:
            year_col = col_map[k]
            break
    val_col = None
    for c in df.columns:
        c_low = str(c).lower()
        if any(k in c_low for k in ['urban', 'rate', '城镇化', 'value']):
            val_col = c
            break

    scen_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ['ssp', 'scenario', '情景']):
            scen_col = c
            break

    if prov_col is None or year_col is None or val_col is None:
        raise ValueError('无法识别 urbanization_rate.csv 的列结构。')

    d = df.copy()
    d['Province'] = d[prov_col].map(normalize_province_name)
    d['Year'] = pd.to_numeric(d[year_col], errors='coerce')
    d['Value'] = pd.to_numeric(d[val_col], errors='coerce')
    d = d.dropna(subset=['Province', 'Year', 'Value'])
    d['Year'] = d['Year'].astype(int)

    # 百分比转比例
    if d['Value'].median() > 1.5:
        d['Value'] = d['Value'] / 100.0

    out = {}
    if scen_col is None:
        p = d.pivot_table(index='Province', columns='Year', values='Value', aggfunc='mean').reindex(columns=years)
        out['all'] = p
    else:
        for scen, g in d.groupby(scen_col):
            p = g.pivot_table(index='Province', columns='Year', values='Value', aggfunc='mean').reindex(columns=years)
            out[str(scen)] = p
    return out

def build_chen_provincial_population(pop_proj_dir, ssp_num, years, wpp_national_series=None):
    """
    从 Chen et al. 省级人口预测文件夹读取分情景省级人口。
    每省每年有一个独立 CSV 文件（101行×14列，包含年龄×教育程度×性别队列）。
    总人口 = 文件中所有数值之和。

    pop_proj_dir        : DATA_Provincial_Population_Projection/ 路径
    ssp_num             : 1, 2 或 5
    years               : 目标年份列表
    wpp_national_series : 可选 WPP 全国总量 Series（用于校准到 WPP 绝对值）
    返回 DataFrame，index=省份（中文），columns=年份
    """
    pop_dir = Path(pop_proj_dir)
    records = {}

    for prov_dir in sorted(pop_dir.iterdir()):
        if not prov_dir.is_dir():
            continue
        prov_en = prov_dir.name
        prov_cn = normalize_province_name(prov_en)

        yr_data = {}
        for year in years:
            csv_file = prov_dir / f'Pop_E_{prov_en}_SSP{ssp_num}_{year}.csv'
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    nums = pd.to_numeric(df.values.flatten(), errors='coerce')
                    yr_data[year] = float(np.nansum(nums))
                except Exception:
                    pass
        if yr_data:
            records[prov_cn] = yr_data

    panel = pd.DataFrame(records).T  # province × year
    panel = panel.groupby(level=0).sum()
    panel = panel.reindex(columns=years, fill_value=np.nan)

    # WPP 校准：将 Chen et al. 省级总量按比例缩放至 WPP 全国总量
    if wpp_national_series is not None:
        chen_national = panel.sum(axis=0)
        for y in years:
            if y not in panel.columns:
                continue
            try:
                wpp_val = float(wpp_national_series.loc[y])
                chen_val = float(chen_national.loc[y])
                if chen_val > 0 and np.isfinite(wpp_val) and np.isfinite(chen_val):
                    panel[y] = panel[y] * (wpp_val / chen_val)
            except (KeyError, TypeError):
                pass

    return panel


def read_chen_urbanization_panel(urb_dir, ssp_num, years):
    """
    读取 Chen et al. SSP 分情景城镇化率数据。
    文件格式：urbanShareSSP{ssp_num}.csv，行=年份，列=省份（英文名），值∈[0,1]。
    返回 DataFrame，index=省份（中文），columns=年份
    """
    csv_file = Path(urb_dir) / f'urbanShareSSP{ssp_num}.csv'
    df = pd.read_csv(csv_file)
    # 首列为年份
    df = df.rename(columns={df.columns[0]: 'Year'})
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df = df.set_index('Year')

    # 省份英文名 → 中文名
    df.columns = [normalize_province_name(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    # 处理重复列（如 Jiangsu.1 → 江西，与 Jiangsu → 江苏 各自独立）
    df = df.T.groupby(level=0).mean().T

    # 确保值在 [0,1]
    if np.nanmedian(df.stack()) > 1.5:
        df = df / 100.0
    df = df.clip(0, 1)

    # 返回 province × year 面板
    available = [y for y in years if y in df.index]
    if not available:
        raise ValueError(f'urbanShareSSP{ssp_num}.csv 中无所请求年份的数据')
    panel = df.loc[available].T  # province × year
    panel = panel.reindex(columns=years)
    return panel


def calibrate_urban_rate_panel(
    urban_rate_panel,
    anchor_urban_rate,
    ratio_cap=3.0,
    power=1.0,
):
    """
    以 2024 年历史城镇化率为锚点，对 2025+ 城镇化率增量做剩余空间比例缩放。
    目的：消除历史-预测拼接断崖，同时保持后续趋势平滑且不超过 1。
    """
    out = urban_rate_panel.copy()
    out = out.apply(pd.to_numeric, errors='coerce')
    out = out.clip(lower=0, upper=1)

    years_sorted = sorted([c for c in out.columns if isinstance(c, int)])
    if not years_sorted:
        return out

    for prov in out.index:
        anchor_val = pd.to_numeric(anchor_urban_rate.get(prov, np.nan), errors='coerce')
        series = pd.to_numeric(out.loc[prov, years_sorted], errors='coerce').values.astype(float)

        if len(series) == 0:
            continue

        if not np.isfinite(anchor_val):
            out.loc[prov, years_sorted] = np.maximum.accumulate(series)
            continue

        anchor_val = float(np.clip(anchor_val, 0.0, 1.0))

        if len(series) >= 2 and np.isfinite(series[0]) and np.isfinite(series[1]):
            first_inc = series[1] - series[0]
            prev_model0 = max(0.0, series[0] - max(first_inc, 0.0))
        else:
            prev_model0 = max(0.0, series[0] - 0.005 if np.isfinite(series[0]) else 0.0)

        prev_model = np.concatenate([[prev_model0], series[:-1]])
        calibrated = np.empty_like(series)
        current = anchor_val

        for i in range(len(series)):
            model_i = series[i]
            prev_i = prev_model[i]

            if not np.isfinite(model_i):
                calibrated[i] = current
                continue

            model_inc = max(model_i - prev_i, 0.0)
            rem_actual = max(1.0 - current, 0.0)
            rem_model = max(1.0 - prev_i, 1e-9)
            ratio = float(np.clip(rem_actual / rem_model, 0.0, ratio_cap))
            adj_inc = model_inc * (ratio ** power)
            current = min(1.0, current + adj_inc)

            # 城镇化率通常单调上升；这里施加不可逆约束避免拼接后回落。
            current = max(current, calibrated[i - 1] if i > 0 else anchor_val)
            calibrated[i] = current

        out.loc[prov, years_sorted] = calibrated

    return out

def get_cdd_by_scenario(cdd_file, years):
    sheets = pd.read_excel(cdd_file, sheet_name=None)
    mapping = {
        'SSP126': 'ssp126_yearly',
        'SSP245': 'ssp245_yearly',
        'SSP585': 'ssp585_yearly'
    }
    out = {}
    for scen, sheet in mapping.items():
        if sheet not in sheets:
            raise ValueError(f'CDD 文件缺少 sheet: {sheet}')
        w = sheets[sheet].copy()
        w.columns = [str(c).strip() for c in w.columns]

        year_col = 'year' if 'year' in [c.lower() for c in w.columns] else w.columns[0]
        if year_col != 'year':
            for c in w.columns:
                c_low = str(c).lower()
                if c_low == 'year' or c_low.startswith('unnamed'):
                    year_col = c
                    break

        w = w.rename(columns={year_col: 'Year'})
        w['Year'] = pd.to_numeric(w['Year'], errors='coerce')
        w = w.dropna(subset=['Year'])
        w['Year'] = w['Year'].astype(int)

        year_slice = w[w['Year'].isin(years)].copy()
        prov_cols = [c for c in year_slice.columns if c != 'Year']

        panel = year_slice.set_index('Year')[prov_cols].T
        panel.index = panel.index.map(lambda x: str(x).strip())
        panel = panel.groupby(level=0).mean().reindex(columns=years)
        out[scen] = panel

    return out

def smooth_cdd_linear_monotonic_panel(panel):
    # 对每个省做线性拟合并施加单调不降约束，降低年际抖动
    out = panel.copy()
    x = np.array(out.columns, dtype=float)
    for prov in out.index:
        y = pd.to_numeric(out.loc[prov], errors='coerce').values.astype(float)
        mask = np.isfinite(y)
        if mask.sum() >= 2:
            coef = np.polyfit(x[mask], y[mask], deg=1)
            y_hat = coef[0] * x + coef[1]
            y_hat = np.maximum.accumulate(y_hat)
            out.loc[prov] = y_hat
    return out


def calibrate_penetration_by_increment(
    grp,
    anchor_u,
    anchor_r,
    alpha_u,
    alpha_r,
    ratio_cap=3.0,
    power=1.0,
):
    """
    以 2024 年真实值为锚点，对模型增量进行“剩余空间比例”缩放：
    - 不直接采用模型绝对值，而是采用逐年增量；
    - 按 (实际剩余空间 / 模型剩余空间) 缩放增量；
    - 再逐年累加，保证与 2024 年平滑接合。
    """
    out = grp.sort_values('Year').copy()

    def _calibrate_one_region(pr_col, cms_col, alpha_value, anchor_value):
        pr_model = pd.to_numeric(out[pr_col], errors='coerce').values.astype(float)
        cms = pd.to_numeric(out[cms_col], errors='coerce').values.astype(float)

        if len(pr_model) == 0:
            return pr_model

        if (not np.isfinite(anchor_value)):
            return np.maximum.accumulate(pr_model)

        alpha_value = float(alpha_value) if np.isfinite(alpha_value) else np.nan
        if not np.isfinite(alpha_value):
            # 无有效 alpha 时，至少保证平滑接合到 2024
            calibrated = np.empty_like(pr_model)
            current = float(anchor_value)
            for i in range(len(pr_model)):
                if i == 0:
                    current = max(current, pr_model[i])
                else:
                    current = max(current, pr_model[i], calibrated[i - 1])
                calibrated[i] = current
            return calibrated

        cap = alpha_value * cms
        cap = np.where(np.isfinite(cap), cap, np.nan)
        # 若锚点高于模型上限，短期放宽上限避免首年强制下跳。
        cap_eff = np.maximum(cap, anchor_value)

        if len(pr_model) >= 2 and np.isfinite(pr_model[0]) and np.isfinite(pr_model[1]):
            first_inc = pr_model[1] - pr_model[0]
            prev_model0 = max(0.0, pr_model[0] - max(first_inc, 0.0))
        else:
            prev_model0 = max(0.0, pr_model[0] * 0.98 if np.isfinite(pr_model[0]) else 0.0)

        prev_model = np.concatenate([[prev_model0], pr_model[:-1]])
        calibrated = np.empty_like(pr_model)

        current = float(anchor_value)
        for i in range(len(pr_model)):
            model_i = pr_model[i]
            prev_i = prev_model[i]
            cap_i = cap_eff[i] if np.isfinite(cap_eff[i]) else np.nan

            if not np.isfinite(model_i):
                calibrated[i] = current
                continue

            model_inc = max(model_i - prev_i, 0.0)
            if np.isfinite(cap_i):
                rem_actual = max(cap_i - current, 0.0)
                rem_model = max(cap_i - prev_i, 1e-9)
                ratio = rem_actual / rem_model
                ratio = float(np.clip(ratio, 0.0, ratio_cap))
                adj_inc = model_inc * (ratio ** power)
                current = min(cap_i, current + adj_inc)
            else:
                current = current + model_inc

            # 不允许下降，且保持非负。
            current = max(current, calibrated[i - 1] if i > 0 else anchor_value, 0.0)
            calibrated[i] = current

        return calibrated

    out['Urban_PR_per_hh_raw'] = out['Urban_PR_per_hh']
    out['Rural_PR_per_hh_raw'] = out['Rural_PR_per_hh']

    out['Urban_PR_per_hh'] = _calibrate_one_region('Urban_PR_per_hh', 'CMS', alpha_u, anchor_u)
    out['Rural_PR_per_hh'] = _calibrate_one_region('Rural_PR_per_hh', 'CMS', alpha_r, anchor_r)

    # AF/Stock 按校准后 PR 回算
    cms_val = pd.to_numeric(out['CMS'], errors='coerce')
    out['Urban_AF'] = np.where(cms_val > 0, out['Urban_PR_per_hh'] / cms_val, np.nan)
    out['Rural_AF'] = np.where(cms_val > 0, out['Rural_PR_per_hh'] / cms_val, np.nan)
    out['Urban_Stock'] = out['Urban_PR_per_hh'] * out['Urban_Households']
    out['Rural_Stock'] = out['Rural_PR_per_hh'] * out['Rural_Households']
    out['Total_Stock'] = out['Urban_Stock'] + out['Rural_Stock']

    return out

# ---------- 6.1 读取输入 ----------
if 'df_results' not in globals():
    raise RuntimeError('请先运行第1个单元，生成拟合参数 df_results。')
if 'df_income_urban_real_hh_proj' not in globals() or 'df_income_rural_real_hh_proj' not in globals():
    raise RuntimeError('请先运行第2个单元，生成预测收入变量。')
if 'df_hh_projected' not in globals():
    raise RuntimeError('请先运行第2个单元，生成预测户规模变量。')

# ---------- SSP 分情景人口与城镇化率（Chen et al. 数据）----------
# SSP情景 → Chen et al. SSP编号映射（仅使用 SSP1/2/5，对应气候情景 126/245/585）
_SSP_TO_CHEN = {'SSP126': 1, 'SSP245': 2, 'SSP585': 5}
_POP_PROJ_DIR = Path('DATA_Provincial_Population_Projection')
_URB_RATE_DIR = Path('DATA_Provincial_Urbanization_Rate/DATA_Provincial_Urbanization_Rate')

# WPP 全国总人口（用于校准 Chen et al. 省级总量）
nat_total_pop_by_ssp = {}
for _ssp_num in [1, 2, 5]:
    try:
        nat_total_pop_by_ssp[_ssp_num] = read_population_total_wpp(
            'Population_WPP.xlsx', years=YEARS, scenario=f'SSP{_ssp_num}')
    except Exception:
        # 若 WPP 文件无对应 SSP 变体，退而使用 SSP2
        nat_total_pop_by_ssp[_ssp_num] = read_population_total_wpp(
            'Population_WPP.xlsx', years=YEARS, scenario='SSP2')

# 构建分情景省级人口与城镇化率原始面板
print('正在构建 Chen et al. SSP 分情景省级人口与城镇化率面板...')
prov_total_pop_by_ssp = {}
_urban_rate_raw_by_ssp = {}
for _scen_key, _ssp_num in _SSP_TO_CHEN.items():
    prov_total_pop_by_ssp[_scen_key] = build_chen_provincial_population(
        _POP_PROJ_DIR, _ssp_num, YEARS, nat_total_pop_by_ssp[_ssp_num])
    _urban_rate_raw_by_ssp[_scen_key] = read_chen_urbanization_panel(
        _URB_RATE_DIR, _ssp_num, YEARS)
    print(f'  {_scen_key} (SSP{_ssp_num}): {len(prov_total_pop_by_ssp[_scen_key])} 个省份已加载')

# 城镇化率平滑校准（锚定 2024 历史值，缓解拼接断崖）
if BASE_YEAR not in df_cpi.columns:
    raise ValueError(f'缺少 {BASE_YEAR} 年，无法执行城镇化率平滑校准。')

df_urban_hist = pd.read_excel('population_proportion.xlsx', index_col=0)
df_urban_hist = standardize_panel(df_urban_hist)
if BASE_YEAR not in df_urban_hist.columns:
    raise ValueError(f'population_proportion.xlsx 缺少 {BASE_YEAR} 年，无法执行城镇化率平滑校准。')

anchor_urban_rate = pd.to_numeric(df_urban_hist[BASE_YEAR], errors='coerce')
if np.nanmedian(anchor_urban_rate) > 1.5:
    anchor_urban_rate = anchor_urban_rate / 100.0
anchor_urban_rate.index = anchor_urban_rate.index.map(normalize_province_name)
anchor_urban_rate = anchor_urban_rate.groupby(level=0).mean().clip(lower=0, upper=1)

urban_rate_dict_calibrated = {}
for _scen_key, _raw_panel in _urban_rate_raw_by_ssp.items():
    panel_cal = _raw_panel.copy()
    panel_cal = panel_cal.clip(lower=0, upper=1)
    panel_cal = calibrate_urban_rate_panel(panel_cal, anchor_urban_rate, ratio_cap=3.0, power=1.0)
    urban_rate_dict_calibrated[_scen_key] = panel_cal

# CDD 三情景（先读取原始，再做线性+单调平滑）
cdd_dict_raw = get_cdd_by_scenario(CDD_FILE, years=YEARS)
cdd_dict = {k: smooth_cdd_linear_monotonic_panel(v) for k, v in cdd_dict_raw.items()}

# 参数表整理
params = df_results[['Province', 'Region_Type', 'Alpha', 'Beta', 'Gamma']].copy()
params['Province'] = params['Province'].astype(str).str.strip()
params['Region_Type'] = params['Region_Type'].astype(str).str.strip()

# ---------- 6.2 逐情景计算保有量 ----------
all_rows = []

for scen_name, cdd_panel in cdd_dict.items():
    # 按气候情景选取匹配的 SSP 人口与城镇化率
    prov_total_pop = prov_total_pop_by_ssp.get(
        scen_name,
        prov_total_pop_by_ssp.get('SSP245', next(iter(prov_total_pop_by_ssp.values())))
    )
    # 选对应城镇化率
    if scen_name in urban_rate_dict_calibrated:
        urban_rate = urban_rate_dict_calibrated[scen_name]
    else:
        urban_rate = urban_rate_dict_calibrated.get('SSP245', next(iter(urban_rate_dict_calibrated.values())))

    # 统一比例口径
    urban_rate = urban_rate.copy()
    if np.nanmedian(pd.to_numeric(urban_rate.stack(), errors='coerce')) > 1.5:
        urban_rate = urban_rate / 100.0
    urban_rate = urban_rate.clip(lower=0, upper=1)

    # 省份交集
    provs = sorted(
        set(prov_total_pop.index)
        & set(urban_rate.index)
        & set(df_hh_projected.index)
        & set(df_income_urban_real_hh_proj.index)
        & set(df_income_rural_real_hh_proj.index)
        & set(cdd_panel.index)
        & set(params['Province'])
    )

    expected_provs = sorted(set(params['Province']))
    missing_for_scenario = sorted(set(expected_provs) - set(provs))
    if missing_for_scenario:
        print(f'警告：{scen_name} 情景仍有省份缺失：{missing_for_scenario}')

    for prov in provs:
        # 取参数
        p_u = params[(params['Province'] == prov) & (params['Region_Type'] == 'Urban')]
        p_r = params[(params['Province'] == prov) & (params['Region_Type'] == 'Rural')]
        if p_u.empty or p_r.empty:
            continue

        alpha_u, beta_u, gamma_u = p_u.iloc[0][['Alpha', 'Beta', 'Gamma']]
        alpha_r, beta_r, gamma_r = p_r.iloc[0][['Alpha', 'Beta', 'Gamma']]

        prov_rows = []

        for y in YEARS:
            # 人口与户数
            pop_tot = pd.to_numeric(prov_total_pop.at[prov, y], errors='coerce') if y in prov_total_pop.columns else np.nan
            u_rate = pd.to_numeric(urban_rate.at[prov, y], errors='coerce') if y in urban_rate.columns else np.nan
            hh_size = pd.to_numeric(df_hh_projected.at[prov, y], errors='coerce') if y in df_hh_projected.columns else np.nan

            if not (np.isfinite(pop_tot) and np.isfinite(u_rate) and np.isfinite(hh_size) and hh_size > 0):
                continue

            pop_u = pop_tot * u_rate
            pop_r = pop_tot * (1 - u_rate)
            hh_u = pop_u / hh_size
            hh_r = pop_r / hh_size

            # 收入（household real income）
            hhi_u = pd.to_numeric(df_income_urban_real_hh_proj.at[prov, y], errors='coerce') if y in df_income_urban_real_hh_proj.columns else np.nan
            hhi_r = pd.to_numeric(df_income_rural_real_hh_proj.at[prov, y], errors='coerce') if y in df_income_rural_real_hh_proj.columns else np.nan

            # CDD -> CMS
            cdd_v = pd.to_numeric(cdd_panel.at[prov, y], errors='coerce') if y in cdd_panel.columns else np.nan
            if not (np.isfinite(hhi_u) and np.isfinite(hhi_r) and np.isfinite(cdd_v)):
                continue

            cms = float(calc_cms(cdd_v))
            if not np.isfinite(cms) or cms <= 0:
                continue

            # AF 与 PR（每户台数）
            af_u = float(logistic_func(hhi_u, alpha_u, beta_u, gamma_u))
            af_r = float(logistic_func(hhi_r, alpha_r, beta_r, gamma_r))
            pr_u = af_u * cms
            pr_r = af_r * cms

            # 总保有量（台）
            stock_u = pr_u * hh_u
            stock_r = pr_r * hh_r
            stock_tot = stock_u + stock_r

            prov_rows.append({
                'Scenario': scen_name,
                'Province': prov,
                'Year': y,
                'Total_Pop': pop_tot,
                'Urbanization_Rate': u_rate,
                'Urban_Pop': pop_u,
                'Rural_Pop': pop_r,
                'HH_Size': hh_size,
                'Urban_Households': hh_u,
                'Rural_Households': hh_r,
                'CDD': cdd_v,
                'CMS': cms,
                'Urban_HHI': hhi_u,
                'Rural_HHI': hhi_r,
                'Urban_AF': af_u,
                'Rural_AF': af_r,
                'Urban_PR_per_hh': pr_u,
                'Rural_PR_per_hh': pr_r,
                'Urban_Stock': stock_u,
                'Rural_Stock': stock_r,
                'Total_Stock': stock_tot
            })

        # 方案E：普及率（每户保有量）不可逆，而非总保有量不可逆
        if prov_rows:
            prov_df = pd.DataFrame(prov_rows).sort_values('Year')
            prov_df['Urban_PR_per_hh'] = np.maximum.accumulate(prov_df['Urban_PR_per_hh'].values)
            prov_df['Rural_PR_per_hh'] = np.maximum.accumulate(prov_df['Rural_PR_per_hh'].values)
            prov_df['Urban_Stock'] = prov_df['Urban_PR_per_hh'] * prov_df['Urban_Households']
            prov_df['Rural_Stock'] = prov_df['Rural_PR_per_hh'] * prov_df['Rural_Households']
            prov_df['Total_Stock'] = prov_df['Urban_Stock'] + prov_df['Rural_Stock']
            all_rows.extend(prov_df.to_dict('records'))

# ---------- 6.3 导出 ----------
df_forecast = pd.DataFrame(all_rows)
if df_forecast.empty:
    raise ValueError('预测结果为空，请检查输入文件口径与省份名称映射。')

# ---------- 6.3 平滑校准（锚定 2024 历史真实值） ----------
if BASE_YEAR not in df_ac_urban.columns or BASE_YEAR not in df_ac_rural.columns:
    raise ValueError(f'历史普及率文件缺少 {BASE_YEAR} 年，无法执行平滑校准。')

anchor_urban = pd.to_numeric(df_ac_urban[BASE_YEAR], errors='coerce') / 100.0
anchor_rural = pd.to_numeric(df_ac_rural[BASE_YEAR], errors='coerce') / 100.0

alpha_u_map = (
    params[params['Region_Type'] == 'Urban']
    .set_index('Province')['Alpha']
    .astype(float)
)
alpha_r_map = (
    params[params['Region_Type'] == 'Rural']
    .set_index('Province')['Alpha']
    .astype(float)
)

calibrated_groups = []
for (scen, prov), g in df_forecast.groupby(['Scenario', 'Province'], sort=False):
    a_u = anchor_urban.get(prov, np.nan)
    a_r = anchor_rural.get(prov, np.nan)
    alpha_u = alpha_u_map.get(prov, np.nan)
    alpha_r = alpha_r_map.get(prov, np.nan)
    calibrated_groups.append(
        calibrate_penetration_by_increment(
            g,
            anchor_u=a_u,
            anchor_r=a_r,
            alpha_u=alpha_u,
            alpha_r=alpha_r,
            ratio_cap=CALIBRATION_RATIO_CAP,
            power=CALIBRATION_POWER,
        )
    )

df_forecast_calibrated = pd.concat(calibrated_groups, ignore_index=True)

# 全国汇总（按情景、年份）
df_national = (
    df_forecast
    .groupby(['Scenario', 'Year'], as_index=False)[['Urban_Stock', 'Rural_Stock', 'Total_Stock']]
    .sum()
)

df_national_calibrated = (
    df_forecast_calibrated
    .groupby(['Scenario', 'Year'], as_index=False)[['Urban_Stock', 'Rural_Stock', 'Total_Stock']]
    .sum()
)

# 分情景分sheet输出
out_file_raw = 'AC_stock_forecast_by_province_2025_2050_raw_new.xlsx'
with pd.ExcelWriter(out_file_raw) as writer:
    for scen in sorted(df_forecast['Scenario'].unique()):
        df_forecast[df_forecast['Scenario'] == scen].to_excel(writer, sheet_name=f'{scen}_province', index=False)
        df_national[df_national['Scenario'] == scen].to_excel(writer, sheet_name=f'{scen}_national', index=False)

out_file_calibrated = 'AC_stock_forecast_by_province_2025_2050_smoothcal.xlsx'
with pd.ExcelWriter(out_file_calibrated) as writer:
    for scen in sorted(df_forecast_calibrated['Scenario'].unique()):
        df_forecast_calibrated[df_forecast_calibrated['Scenario'] == scen].to_excel(writer, sheet_name=f'{scen}_province', index=False)
        df_national_calibrated[df_national_calibrated['Scenario'] == scen].to_excel(writer, sheet_name=f'{scen}_national', index=False)

print('预测完成：')
print(f'- 原始分省+全国结果：{out_file_raw}')
print(f'- 平滑校准分省+全国结果：{out_file_calibrated}')
print('- 口径：WPP全国总量 × Pop_TOTAL(SSP2)省份占比；再按城镇化率拆分城乡人口；套用已拟合参数计算保有量。')
print('- CDD处理：线性拟合 + 单调不降平滑（逐省、逐情景）。')
print('- 方案E约束：普及率（每户保有量）不可逆，再回算总保有量。')
print('- 平滑校准：以2024历史真实普及率为锚点，按“剩余空间比例”缩放模型增量并逐年累加。')
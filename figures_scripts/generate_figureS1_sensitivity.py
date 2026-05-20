# -*- coding: utf-8 -*-
"""
Sensitivity analysis: impact of ±10% adverse change in each parameter on LCTR(2100).
Cities: Beijing (北京), Guangzhou (广州)
Refrigerants: R410A, R32, R290
Production year: 2025, SSP2-4.5
Mirrors LCTR_computation.py logic exactly.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
matplotlib.rcParams['font.family'] = 'Arial'

ROOT = Path(__file__).resolve().parent
if ROOT.name == "figures_scripts":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from figure_io import read_excel_cached

CLIMATE_MODEL = os.environ.get("LCTR_FIGURE_MODEL", "BCC-CSM2-MR")
OUT_DIR = ROOT / f"Figures_{CLIMATE_MODEL}_3hr"
OUT_DIR.mkdir(exist_ok=True)

# ── AGTP tables ──────────────────────────────────────────────────────────
AGTP_R32 = {
    1:1.2468E-12,2:1.9722E-12,3:2.3441E-12,4:2.4812E-12,5:2.4667E-12,
    6:2.3585E-12,7:2.1967E-12,8:2.0082E-12,9:1.8109E-12,10:1.6165E-12,
    11:1.4319E-12,12:1.2611E-12,13:1.1061E-12,14:9.6731E-13,15:8.4444E-13,
    16:7.3660E-13,17:6.4261E-13,18:5.6115E-13,19:4.9086E-13,20:4.3046E-13,
    21:3.7870E-13,22:3.3445E-13,23:2.9671E-13,24:2.6456E-13,25:2.3720E-13,
    26:2.1394E-13,27:1.9416E-13,28:1.7734E-13,29:1.6303E-13,30:1.5085E-13,
    31:1.4047E-13,32:1.3161E-13,33:1.2403E-13,34:1.1753E-13,35:1.1194E-13,
    36:1.0711E-13,37:1.0294E-13,38:9.9312E-14,39:9.6144E-14,40:9.3366E-14,
    41:9.0917E-14,42:8.8747E-14,43:8.6812E-14,44:8.5079E-14,45:8.3517E-14,
    46:8.2101E-14,47:8.0809E-14,48:7.9626E-14,49:7.8534E-14,50:7.7523E-14,
    51:7.6582E-14,52:7.5701E-14,53:7.4874E-14,54:7.4094E-14,55:7.3355E-14,
    56:7.2653E-14,57:7.1985E-14,58:7.1346E-14,59:7.0734E-14,60:7.0147E-14,
    61:6.9582E-14,62:6.9038E-14,63:6.8513E-14,64:6.8005E-14,65:6.7514E-14,
    66:6.7038E-14,67:6.6576E-14,68:6.6128E-14,69:6.5692E-14,70:6.5268E-14,
    71:6.4856E-14,72:6.4454E-14,73:6.4062E-14,74:6.3680E-14,75:6.3307E-14,
    76:6.2943E-14,77:6.2587E-14,78:6.2240E-14,79:6.1900E-14,80:6.1567E-14,
    81:6.1241E-14,82:6.0922E-14,83:6.0609E-14,84:6.0303E-14,85:6.0002E-14,
    86:5.9707E-14,87:5.9418E-14,88:5.9134E-14,89:5.8854E-14,90:5.8580E-14,
    91:5.8310E-14,92:5.8045E-14,93:5.7784E-14,94:5.7527E-14,95:5.7274E-14,
    96:5.7025E-14,97:5.6780E-14,98:5.6538E-14,99:5.6299E-14,100:5.6064E-14,
}
AGTP_125 = {
    1:1.2259E-12,2:2.1063E-12,3:2.7307E-12,4:3.1656E-12,5:3.4599E-12,
    6:3.6500E-12,7:3.7632E-12,8:3.8195E-12,9:3.8341E-12,10:3.8182E-12,
    11:3.7801E-12,12:3.7262E-12,13:3.6612E-12,14:3.5885E-12,15:3.5108E-12,
    16:3.4299E-12,17:3.3475E-12,18:3.2644E-12,19:3.1815E-12,20:3.0993E-12,
    21:3.0183E-12,22:2.9387E-12,23:2.8608E-12,24:2.7847E-12,25:2.7104E-12,
    26:2.6380E-12,27:2.5676E-12,28:2.4991E-12,29:2.4325E-12,30:2.3679E-12,
    51:1.3755E-12,52:1.3422E-12,53:1.3099E-12,54:1.2786E-12,55:1.2483E-12,
    56:1.2188E-12,57:1.1902E-12,58:1.1625E-12,59:1.1356E-12,60:1.1095E-12,
    61:1.0842E-12,62:1.0597E-12,63:1.0358E-12,64:1.0127E-12,65:9.9031E-13,
    70:8.8785E-13,75:7.9973E-13,80:7.2389E-13,85:6.5857E-13,90:6.0224E-13,95:5.5360E-13,100:5.1153E-13,
}
AGTP_290 = {
    1:1.6677E-16,2:1.2542E-16,3:9.4644E-17,4:7.1691E-17,5:5.4538E-17,
    6:4.1697E-17,7:3.2069E-17,8:2.4838E-17,9:1.9401E-17,10:1.5307E-17,
    11:1.2220E-17,12:9.8879E-18,13:8.1239E-18,14:6.7867E-18,15:5.7707E-18,
    16:4.9966E-18,17:4.4046E-18,18:3.9500E-18,19:3.5990E-18,20:3.3262E-18,
    25:2.6097E-18,30:2.3278E-18,35:2.1660E-18,40:2.0464E-18,45:1.9484E-18,
    50:1.8651E-18,55:1.7932E-18,60:1.7305E-18,65:1.6755E-18,70:1.6267E-18,
    75:1.5831E-18,80:1.5436E-18,85:1.5077E-18,90:1.4746E-18,95:1.4438E-18,100:1.4150E-18,
}
AGTP_CO2 = {
    1:1.8706E-16,2:3.1578E-16,3:4.0297E-16,4:4.6080E-16,5:4.9801E-16,
    6:5.2084E-16,7:5.3378E-16,8:5.3998E-16,9:5.4170E-16,10:5.4048E-16,
    11:5.3741E-16,12:5.3324E-16,13:5.2847E-16,14:5.2344E-16,15:5.1836E-16,
    16:5.1337E-16,17:5.0854E-16,18:5.0392E-16,19:4.9952E-16,20:4.9536E-16,
    25:4.7763E-16,30:4.6376E-16,35:4.5241E-16,40:4.4286E-16,45:4.3470E-16,
    50:4.2770E-16,55:4.2169E-16,60:4.1652E-16,65:4.1208E-16,70:4.0827E-16,
    75:4.0501E-16,80:4.0222E-16,85:3.9984E-16,90:3.9780E-16,95:3.9607E-16,100:3.9460E-16,
}

def _fill_gaps(d):
    known = sorted(d.keys())
    for i in range(len(known)-1):
        h0,h1 = known[i],known[i+1]
        v0,v1 = d[h0],d[h1]
        for h in range(h0+1,h1):
            d[h] = v0+(v1-v0)*(h-h0)/(h1-h0)
    return d

for d in [AGTP_R32,AGTP_125,AGTP_290,AGTP_CO2]:
    _fill_gaps(d)

def agtp_ref_fn(ref,h):
    if ref=='R410A': return (AGTP_R32.get(h,0)+AGTP_125.get(h,0))/2
    elif ref=='R32': return AGTP_R32.get(h,0)
    else:            return AGTP_290.get(h,0)

# ── Parameters ───────────────────────────────────────────────────────────
charge_amount = {'R410A':0.762,'R32':0.696,'R290':0.367}
rfm_factors   = {'R410A':10.7,'R32':7.2,'R290':0.05}
RFD = 2.1
MM_sum_base = (15+28)*(0.46*1.8+0.12*12.6+0.19*3.8+0.23*2.8)
RM_sum_base = 0.07*(15+28)*(0.46+0.12+0.19)+0.15*(15+28)*0.23

EER_data = {
    'R32':  {24:11.59,25:10.93,26:10.52,27:10.12,28:9.80,29:8.76,30:7.37,
             31:6.14,32:5.57,33:4.92,34:4.47,35:4.03,36:3.79,37:3.43,38:3.19,39:3.00,40:2.76,41:2.61},
    'R410A':{24:10.47,25:10.00,26:9.71,27:9.43,28:9.21,29:7.94,30:6.72,
             31:5.94,32:5.15,33:4.64,34:4.25,35:3.78,36:3.51,37:3.27,38:3.08,39:2.91,40:2.64,41:2.42},
    'R290': {24:16.18,25:13.12,26:12.85,27:11.95,28:10.69,29:8.65,30:7.30,
             31:6.17,32:5.24,33:4.76,34:4.32,35:4.18,36:3.88,37:3.63,38:3.08,39:2.64,40:1.82,41:1.49}
}
APF_improved = {2025:0,2026:0.054,2027:0.107,2028:0.157,2029:0.204,2030:0.25,
                2031:0.294,2032:0.336,2033:0.376,2034:0.415,2035:0.453,2036:0.489,
                2037:0.524,2038:0.559,2039:0.592,2040:0.624,2041:0.655,2042:0.685,
                2043:0.715,2044:0.743,2045:0.771}

def get_decay_factor(life):
    if 3<=life<=5: return 0.941
    elif 6<=life<=7: return 0.913
    elif life>=8: return 0.874
    return 1.0

# ── Load data ─────────────────────────────────────────────────────────────
df_em_raw = read_excel_cached('EM_raw.xlsx', sheet_name='ssp245')
df_em_raw = df_em_raw.set_index(df_em_raw.columns[0]).astype(float)

def make_em_interp(city):
    ser = df_em_raw[city].dropna()
    interp = PchipInterpolator(ser.index.astype(int), ser.values, extrapolate=True)
    return lambda yr: float(np.clip(interp(yr), 0.05, 1.0))

print("Loading temperature distribution...")
df_temp = read_excel_cached(f'CMIP6/results/3hr/{CLIMATE_MODEL}_ssp245_TempDist_3hr.xlsx')
df_temp = df_temp[(df_temp['Model'] == CLIMATE_MODEL) & (df_temp['Scenario'] == 'ssp245')].copy()
df_temp = df_temp.drop(columns=[c for c in ['Model', 'Scenario'] if c in df_temp.columns])
df_temp.set_index(['Year','City'], inplace=True)

def get_temp_dist(year, city):
    try:
        row = df_temp.loc[(year, city)]
        return {int(k):float(v) for k,v in row.items() if pd.notna(v) and float(v)>0}
    except KeyError:
        avail = sorted(set(y for (y,c) in df_temp.index if c==city))
        nr = min(avail, key=lambda y: abs(y-year))
        row = df_temp.loc[(nr, city)]
        return {int(k):float(v) for k,v in row.items() if pd.notna(v) and float(v)>0}

def calc_aec(ref, year_prod, year_use, temp_dist, eer_scale=1.0):
    life = year_use - year_prod
    if life < 0: return 0.0
    improvement = APF_improved.get(year_prod, 0)
    energy_mult = 1.0 / get_decay_factor(life)
    total_wh = 0.0
    for t in range(24,42):
        hours = temp_dist.get(t, 0)
        if hours==0: continue
        load_w = (t-23)*3500/(35-23)
        base_eer = EER_data[ref].get(t, 3.0)
        eer = (base_eer + improvement) * eer_scale
        power_w = load_w / eer
        total_wh += power_w * hours
    return (total_wh/1000.0)*energy_mult

def compute_lctr(city, ref, year_prod, target_year, *,
                 alr_scale=1.0, em_scale=1.0, aec_scale=1.0,
                 mm_scale=1.0, eer_scale=1.0):
    """Compute LCTR(target_year) for one city/ref/year_prod under scaled parameters."""
    C   = charge_amount[ref]
    RFM = rfm_factors[ref]
    em_fn = make_em_interp(city)
    lifespan = 10
    years_use = range(year_prod, year_prod+lifespan)

    # Scaled parameters
    ALR    = max(0.02, 0.05 - 0.002*(year_prod-2025)) * alr_scale
    MM_sum = MM_sum_base * mm_scale
    RM_sum = RM_sum_base * mm_scale  # same scale for RM

    td = te = temb = 0.0
    for life in range(lifespan):
        yr   = year_prod + life
        em_yr = em_fn(yr) * em_scale
        temp_dist = get_temp_dist(yr, city)
        aec_yr = calc_aec(ref, year_prod, yr, temp_dist, eer_scale=eer_scale) * aec_scale
        eol  = max(0.0, 0.97-0.01*(yr-2025))
        h = target_year - yr
        if h < 1: continue
        if h > 100: h = 100
        ag_ref = agtp_ref_fn(ref, h)
        ag_co2 = AGTP_CO2.get(h, 0)

        if life == 0:
            m_dir  = C * ALR
            m_ene  = C*(1+ALR)*RFM + aec_yr*em_yr
            m_emb  = MM_sum
        elif life == lifespan-1:
            m_dir  = C*(ALR+eol)
            m_ene  = C*ALR*RFM + aec_yr*em_yr + C*(1-eol)*RFD
            m_emb  = RM_sum
        else:
            m_dir  = C * ALR
            m_ene  = C*ALR*RFM + aec_yr*em_yr
            m_emb  = 0.0

        td   += m_dir  * ag_ref
        te   += m_ene  * ag_co2
        temb += m_emb  * ag_co2

    return (td+te+temb)*1e12

# ── Sensitivity analysis ──────────────────────────────────────────────────
YEAR_PROD  = 2025
TARGET_YR  = 2100
DELTA      = 0.10   # 10% adverse

CITIES = ['北京', '广州']
CITY_LABELS = {'北京': 'Beijing', '广州': 'Guangzhou'}
REFS   = ['R410A', 'R32', 'R290']
REF_LABELS = {'R410A':'R410A','R32':'HFC-32','R290':'HC-290'}

PARAMS = [
    ('Embodied Emission',   dict(mm_scale=1+DELTA),   dict()),
    ('Leakage Rate',        dict(alr_scale=1+DELTA),  dict()),
    ('Operating Hours',     dict(aec_scale=1+DELTA),  dict()),
    ('Grid Emission Factor',dict(em_scale=1+DELTA),   dict()),
    ('Energy Efficiency',   dict(eer_scale=1-DELTA),  dict()),   # adverse = lower EER
]

print("\nComputing base case and perturbations...")
results = {}  # results[city][ref] = {'base': val, param: pct_change, ...}

for city in CITIES:
    results[city] = {}
    for ref in REFS:
        base = compute_lctr(city, ref, YEAR_PROD, TARGET_YR)
        row = {'base': base}
        for pname, kw_adv, _ in PARAMS:
            perturbed = compute_lctr(city, ref, YEAR_PROD, TARGET_YR, **kw_adv)
            row[pname] = (perturbed - base) / base * 100.0
        results[city][ref] = row
        print(f"  {CITY_LABELS[city]:12s} {REF_LABELS[ref]:8s}  base={base:.4f} pK")

# ── Cross-check base against file ────────────────────────────────────────
df_ck = read_excel_cached(
    f'results/LCTR_Result_National_Total_{CLIMATE_MODEL}_3hr.xlsx',
    sheet_name='Unit_LCTR_Detail'
)
print("\nCross-check (2025, SSP245, T=2100):")
for city in CITIES:
    for ref in REFS:
        row = df_ck[(df_ck['地点']==city)&(df_ck['情景']=='ssp245')&
                    (df_ck['年份']==2025)&(df_ck['制冷剂']==ref)]
        if not row.empty:
            file_val = float(row['Unit_LCTR_pK'].iloc[0])
            my_val   = results[city][ref]['base']
            print(f"  {city} {ref}: file={file_val:.4f}  computed={my_val:.4f}  diff={abs(file_val-my_val)/file_val*100:.1f}%")

# ── Print sensitivity table ───────────────────────────────────────────────
print("\n── Sensitivity: % change in LCTR(2100) per +10% adverse parameter ──")
param_names = [p[0] for p in PARAMS]
for city in CITIES:
    print(f"\n  [{CITY_LABELS[city]}]")
    header = f"  {'Parameter':<25}" + "".join(f"  {REF_LABELS[r]:>8}" for r in REFS)
    print(header)
    for pname in param_names:
        row_str = f"  {pname:<25}"
        for ref in REFS:
            row_str += f"  {results[city][ref][pname]:>7.2f}%"
        print(row_str)

# ── Plot ──────────────────────────────────────────────────────────────────
COLORS = {'R410A':'#222222','R32':'#2196F3','R290':'#4CAF50'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.24, right=0.97, top=0.96, bottom=0.08)

for ax, city in zip(axes, CITIES):
    pnames = [p[0] for p in reversed(PARAMS)]  # bottom-to-top display
    n_p = len(pnames)
    y_pos = np.arange(n_p)
    bar_h = 0.22
    offsets = {'R410A': -bar_h, 'R32': 0, 'R290': bar_h}

    for ref in REFS:
        vals = [results[city][ref][p] for p in pnames]
        ax.barh(y_pos + offsets[ref], vals, height=bar_h,
                color=COLORS[ref], label=REF_LABELS[ref], alpha=0.88)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pnames, fontsize=11)
    ax.set_xlabel('Change in LCTR (%)', fontsize=12)
    ax.set_title(f'({chr(97+CITIES.index(city))}) {CITY_LABELS[city]}', fontsize=13)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

axes[0].legend(loc='upper right', fontsize=11, frameon=True)

out_png = OUT_DIR / 'FigureS1_Sensitivity_LCTR.png'
out_pdf = OUT_DIR / 'FigureS1_Sensitivity_LCTR.pdf'
out_tif = OUT_DIR / 'FigureS1_Sensitivity_LCTR.tif'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
fig.savefig(out_tif, dpi=600, bbox_inches='tight')
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_tif}")
plt.show()

# -*- coding: utf-8 -*-
"""
Figure: Life-cycle emission and climate response of a single HFC-32 RAC unit
        Beijing, SSP2-4.5, produced 2040, operated 2040�?049 (10-year lifespan)

Three stacked panels sharing the same x-axis (2040�?100):
  (a) Annual physical emissions �?CO�?indirect (left axis) + kg R32 direct (right axis)
  (b) LCCP accumulation �?running cumulative CO₂e (kgCO₂e)
  (c) LCTR evolution    �?running temperature response (pK) at each evaluation year T

Key insight conveyed by Panel (c) vs (b):
  - LCCP plateaus after retirement (2049): past is permanent
  - LCTR decays after some point: warming effect fades, faster for short-lived R32
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import PchipInterpolator
from pathlib import Path

matplotlib.rcParams['font.family'] = 'Arial'

OUT_PDF = Path(__file__).parent.parent / 'Figures' / 'Figure1_LCTR_Concept.pdf'
OUT_PNG = Path(__file__).parent.parent / 'Figures' / 'Figure1_LCTR_Concept.png'
OUT_TIF = Path(__file__).parent.parent / 'Figures' / 'Figure1_LCTR_Concept.tif'

# ===========================================================
# 1. Physical parameters  (mirrors LCTR_computation.py)
# ===========================================================
C       = 0.696   # kg charge �?HFC-32
GWP_R32 = 675     # AR6 GWP100
RFM     = 7.2     # kgCO₂e/kg �?refrigerant manufacturing emission factor
RFD     = 2.1     # kgCO₂e/kg �?refrigerant disposal emission factor

YEAR_PROD = 2040
LIFESPAN  = 10                                   # 10-year lifespan
YEARS_USE = list(range(YEAR_PROD, YEAR_PROD + LIFESPAN))  # 2040�?049

ALR = max(0.02, 0.05 - 0.002 * (YEAR_PROD - 2025))   # = 0.02 (annual leakage rate)

# Structural embodied CO�? (kgCO�?
MM_sum = (15 + 28) * (0.46 * 1.8 + 0.12 * 12.6 + 0.19 * 3.8 + 0.23 * 2.8)
RM_sum = 0.07 * (15 + 28) * (0.46 + 0.12 + 0.19) + 0.15 * (15 + 28) * 0.23

def get_decay_factor(life):
    """Energy-efficiency degradation (same logic as LCTR_computation.py)."""
    if 3 <= life <= 5: return 0.941
    elif 6 <= life <= 7: return 0.913
    elif life >= 8:      return 0.874
    return 1.0

# ===========================================================
# 2. Grid emission factor �?Beijing SSP2-4.5  (PCHIP)
# ===========================================================
_EM_YRS = np.array([2020,  2025,  2030,  2035,   2040,    2050,    2060])
_EM_VAL = np.array([0.615, 0.595, 0.519, 0.289, 0.2601, 0.18114, 0.13934])
_em_interp = PchipInterpolator(_EM_YRS, _EM_VAL, extrapolate=True)

def em(year):
    return float(np.clip(_em_interp(year), 0.05, 1.0))

# ===========================================================
# 3. Annual energy consumption (AEC) �?calibrated to electricity-only CO�?= 871.06
#    (Beijing HFC-32 unit produced 2040, SSP2-4.5; from BCC-CSM2-MR 3-hourly data)
# ===========================================================
LCCP_ENERGY_KNOWN = 871.06   # kgCO�?electricity-only contribution

_denom = sum(em(YEAR_PROD + life) / get_decay_factor(life) for life in range(LIFESPAN))
AEC_BASE = LCCP_ENERGY_KNOWN / _denom   # kWh/yr at life=0 (~535 kWh/yr)

def aec(life):
    """kWh/yr for a unit aged 'life' years."""
    return AEC_BASE / get_decay_factor(life)

# ===========================================================
# 4. AGTP tables  (AR6 values from LCTR_computation.py)
# ===========================================================
AGTP_R32 = {
    1: 1.2468E-12, 2: 1.9722E-12, 3: 2.3441E-12, 4: 2.4812E-12, 5: 2.4667E-12,
    6: 2.3585E-12, 7: 2.1967E-12, 8: 2.0082E-12, 9: 1.8109E-12, 10: 1.6165E-12,
    11: 1.4319E-12, 12: 1.2611E-12, 13: 1.1061E-12, 14: 9.6731E-13, 15: 8.4444E-13,
    16: 7.3660E-13, 17: 6.4261E-13, 18: 5.6115E-13, 19: 4.9086E-13, 20: 4.3046E-13,
    21: 3.7870E-13, 22: 3.3445E-13, 23: 2.9671E-13, 24: 2.6456E-13, 25: 2.3720E-13,
    26: 2.1394E-13, 27: 1.9416E-13, 28: 1.7734E-13, 29: 1.6303E-13, 30: 1.5085E-13,
    31: 1.4047E-13, 32: 1.3161E-13, 33: 1.2403E-13, 34: 1.1753E-13, 35: 1.1194E-13,
    36: 1.0711E-13, 37: 1.0294E-13, 38: 9.9312E-14, 39: 9.6144E-14, 40: 9.3366E-14,
    41: 9.0917E-14, 42: 8.8747E-14, 43: 8.6812E-14, 44: 8.5079E-14, 45: 8.3517E-14,
    46: 8.2101E-14, 47: 8.0809E-14, 48: 7.9626E-14, 49: 7.8534E-14, 50: 7.7523E-14,
    51: 7.6582E-14, 52: 7.5701E-14, 53: 7.4874E-14, 54: 7.4094E-14, 55: 7.3355E-14,
    56: 7.2653E-14, 57: 7.1985E-14, 58: 7.1346E-14, 59: 7.0734E-14, 60: 7.0147E-14,
    61: 6.9582E-14, 62: 6.9038E-14, 63: 6.8513E-14, 64: 6.8005E-14, 65: 6.7514E-14,
    66: 6.7038E-14, 67: 6.6576E-14, 68: 6.6128E-14, 69: 6.5692E-14, 70: 6.5268E-14,
    71: 6.4856E-14, 72: 6.4454E-14, 73: 6.4062E-14, 74: 6.3680E-14, 75: 6.3307E-14,
    76: 6.2943E-14, 77: 6.2587E-14, 78: 6.2240E-14, 79: 6.1900E-14, 80: 6.1567E-14,
    81: 6.1241E-14, 82: 6.0922E-14, 83: 6.0609E-14, 84: 6.0303E-14, 85: 6.0002E-14,
    86: 5.9707E-14, 87: 5.9418E-14, 88: 5.9134E-14, 89: 5.8854E-14, 90: 5.8580E-14,
    91: 5.8310E-14, 92: 5.8045E-14, 93: 5.7784E-14, 94: 5.7527E-14, 95: 5.7274E-14,
    96: 5.7025E-14, 97: 5.6780E-14, 98: 5.6538E-14, 99: 5.6299E-14, 100: 5.6064E-14,
}
AGTP_CO2 = {
    1: 1.8706E-16, 2: 3.1578E-16, 3: 4.0297E-16, 4: 4.6080E-16, 5: 4.9801E-16,
    6: 5.2084E-16, 7: 5.3378E-16, 8: 5.3998E-16, 9: 5.4170E-16, 10: 5.4048E-16,
    11: 5.3741E-16, 12: 5.3324E-16, 13: 5.2847E-16, 14: 5.2344E-16, 15: 5.1836E-16,
    16: 5.1337E-16, 17: 5.0854E-16, 18: 5.0392E-16, 19: 4.9952E-16, 20: 4.9536E-16,
    21: 4.9142E-16, 22: 4.8769E-16, 23: 4.8416E-16, 24: 4.8081E-16, 25: 4.7763E-16,
    26: 4.7460E-16, 27: 4.7171E-16, 28: 4.6894E-16, 29: 4.6630E-16, 30: 4.6376E-16,
    31: 4.6132E-16, 32: 4.5897E-16, 33: 4.5670E-16, 34: 4.5452E-16, 35: 4.5241E-16,
    36: 4.5037E-16, 37: 4.4840E-16, 38: 4.4650E-16, 39: 4.4465E-16, 40: 4.4286E-16,
    41: 4.4112E-16, 42: 4.3944E-16, 43: 4.3782E-16, 44: 4.3624E-16, 45: 4.3470E-16,
    46: 4.3322E-16, 47: 4.3178E-16, 48: 4.3038E-16, 49: 4.2902E-16, 50: 4.2770E-16,
    51: 4.2643E-16, 52: 4.2519E-16, 53: 4.2399E-16, 54: 4.2282E-16, 55: 4.2169E-16,
    56: 4.2059E-16, 57: 4.1953E-16, 58: 4.1849E-16, 59: 4.1749E-16, 60: 4.1652E-16,
    61: 4.1557E-16, 62: 4.1466E-16, 63: 4.1377E-16, 64: 4.1291E-16, 65: 4.1208E-16,
    66: 4.1127E-16, 67: 4.1048E-16, 68: 4.0972E-16, 69: 4.0899E-16, 70: 4.0827E-16,
    71: 4.0758E-16, 72: 4.0691E-16, 73: 4.0625E-16, 74: 4.0562E-16, 75: 4.0501E-16,
    76: 4.0442E-16, 77: 4.0384E-16, 78: 4.0328E-16, 79: 4.0274E-16, 80: 4.0222E-16,
    81: 4.0171E-16, 82: 4.0122E-16, 83: 4.0074E-16, 84: 4.0028E-16, 85: 3.9984E-16,
    86: 3.9940E-16, 87: 3.9898E-16, 88: 3.9858E-16, 89: 3.9818E-16, 90: 3.9780E-16,
    91: 3.9743E-16, 92: 3.9708E-16, 93: 3.9673E-16, 94: 3.9640E-16, 95: 3.9607E-16,
    96: 3.9576E-16, 97: 3.9545E-16, 98: 3.9516E-16, 99: 3.9487E-16, 100: 3.9460E-16,
}

# ===========================================================
# 5. Per-year physical emission breakdown
# ===========================================================
# direct_kg[year_u]         : kg HFC-32 leaked
# indirect_energy[year_u]   : kgCO�?from energy use + refrigerant manufacture handling
# indirect_embodied[year_u] : kgCO�?from unit manufacture/disposal materials
direct_kg        = {}
indirect_energy  = {}
indirect_embodied = {}

for life in range(LIFESPAN):
    yr = YEAR_PROD + life
    em_yr = em(yr)
    aec_yr = aec(life)
    eol = max(0.0, 0.97 - 0.01 * (yr - 2025))   # end-of-life recovery fraction

    if life == 0:                           # manufacture + first operational year
        direct_kg[yr]         = C * ALR
        indirect_energy[yr]   = C * (1 + ALR) * RFM + aec_yr * em_yr
        indirect_embodied[yr] = MM_sum

    elif life == LIFESPAN - 1:              # last operational year + disposal
        direct_kg[yr]         = C * (ALR + eol)
        indirect_energy[yr]   = C * ALR * RFM + aec_yr * em_yr + C * (1 - eol) * RFD
        indirect_embodied[yr] = RM_sum

    else:                                   # normal operation
        direct_kg[yr]         = C * ALR
        indirect_energy[yr]   = C * ALR * RFM + aec_yr * em_yr
        indirect_embodied[yr] = 0.0

print(f"AEC_BASE = {AEC_BASE:.1f} kWh/yr")
print(f"Verified LCCP energy = {sum(indirect_energy[yr] for yr in YEARS_USE):.1f} kgCO2 (target ~878)")
_lccp_total = (sum(indirect_energy.values()) + sum(indirect_embodied.values())
               + sum(v * GWP_R32 for v in direct_kg.values()))
print(f"Verified LCCP total  = {_lccp_total:.1f} kgCO2e")

# ===========================================================
# 6. Compute cumulative series over 2040�?100
# ===========================================================
YEARS_ALL = list(range(2040, 2101))

# --- Panel (b): LCCP cumulative (running total kgCO₂e) ---
lccp_direct_cum   = []
lccp_energy_cum   = []
lccp_embodied_cum = []
_cd = _ce = _cemb = 0.0
for T in YEARS_ALL:
    if T in direct_kg:
        _cd   += direct_kg[T] * GWP_R32
        _ce   += indirect_energy[T]
        _cemb += indirect_embodied[T]
    lccp_direct_cum.append(_cd)
    lccp_energy_cum.append(_ce)
    lccp_embodied_cum.append(_cemb)

# --- Panel (c): LCTR running temperature response (pK) ---
lctr_direct_cum   = []
lctr_energy_cum   = []
lctr_embodied_cum = []
for T in YEARS_ALL:
    td = te = temb = 0.0
    for yr_u in YEARS_USE:
        h = T - yr_u
        if 1 <= h <= 100:
            td   += direct_kg[yr_u]         * AGTP_R32[h]
            te   += indirect_energy[yr_u]   * AGTP_CO2[h]
            temb += indirect_embodied[yr_u] * AGTP_CO2[h]
    lctr_direct_cum.append(td   * 1e12)
    lctr_energy_cum.append(te   * 1e12)
    lctr_embodied_cum.append(temb * 1e12)

# ===========================================================
# 7. Plot
# ===========================================================
YEARS_ARR = np.array(YEARS_ALL)
C_EMBOD  = '#7f7f7f'    # gray   �?embodied
C_ENERGY = '#ff7f0e'    # orange �?energy
C_DIRECT = '#d62728'    # red    �?direct refrigerant
ALPHA    = 0.88

FONT_LABEL = 13
FONT_TICK  = 12
FONT_ANNO  = 11

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.subplots_adjust(hspace=0.07, left=0.10, right=0.94, top=0.97, bottom=0.07)

# ── Panel (a): Annual physical emissions ─────────────────
ax1 = axes[0]
ax1r = ax1.twinx()

# Indirect CO�?(left axis) �?stacked: energy + embodied
emb_vals = [indirect_embodied.get(yr, 0) for yr in YEARS_ALL]
ene_vals = [indirect_energy.get(yr, 0)   for yr in YEARS_ALL]
dir_vals = [direct_kg.get(yr, 0)         for yr in YEARS_ALL]

BAR_W = 0.48
OFFSET = 0.30
# Left axis: stacked indirect bars shifted left
ax1.bar(YEARS_ARR - OFFSET, emb_vals,
        bottom=0, width=BAR_W, color=C_EMBOD,  alpha=ALPHA, label='Embodied CO₂', zorder=3)
ax1.bar(YEARS_ARR - OFFSET, ene_vals,
        bottom=emb_vals, width=BAR_W, color=C_ENERGY, alpha=ALPHA, label='Energy CO₂', zorder=3)
# Right axis: direct leakage bars shifted right
ax1r.bar(YEARS_ARR + OFFSET, dir_vals, width=BAR_W, color=C_DIRECT, alpha=0.75,
         label='Direct leakage (R32)', zorder=4)

ax1.set_ylabel('Annual indirect emission\n(kg CO$_2$)', fontsize=FONT_LABEL, labelpad=6)
ax1r.set_ylabel('Annual direct leakage\n(kg HFC-32)', fontsize=FONT_LABEL, color=C_DIRECT, labelpad=6)
ax1r.tick_params(axis='y', labelcolor=C_DIRECT, labelsize=FONT_TICK)
ax1.tick_params(axis='y', labelsize=FONT_TICK)
ax1.set_ylim(bottom=0)
ax1r.set_ylim(bottom=0)
ax1.spines['top'].set_visible(False)
ax1r.spines['top'].set_visible(False)
ax1.text(0.017, 0.93, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold')

# Legend for panel (a)
handles_a = [
    mpatches.Patch(color=C_EMBOD,  alpha=ALPHA, label='Embodied CO$_2$ (indirect)'),
    mpatches.Patch(color=C_ENERGY, alpha=ALPHA, label='Energy CO$_2$ (indirect)'),
    mpatches.Patch(color=C_DIRECT, alpha=0.75,  label='Direct leakage (HFC-32)'),
]
ax1.legend(handles=handles_a, loc='upper right', fontsize=FONT_ANNO,
           frameon=True, framealpha=0.9)

# Retirement annotation
ax1.axvline(x=2049.5, color='#444', linestyle=':', linewidth=1.2, zorder=5)
ax1.text(2050.0, ax1.get_ylim()[1] * 0.92, 'Retirement', fontsize=FONT_ANNO,
         color='#444', ha='left', va='top')

# ── Panel (b): LCCP cumulative ────────────────────────────
ax2 = axes[1]
emb_b = np.array(lccp_embodied_cum)
ene_b = np.array(lccp_energy_cum)
dir_b = np.array(lccp_direct_cum)

ax2.bar(YEARS_ARR, emb_b,
        width=0.85, color=C_EMBOD,  alpha=ALPHA, label='Embodied', zorder=3)
ax2.bar(YEARS_ARR, ene_b,
        bottom=emb_b, width=0.85, color=C_ENERGY, alpha=ALPHA, label='Energy', zorder=3)
ax2.bar(YEARS_ARR, dir_b,
        bottom=emb_b + ene_b, width=0.85, color=C_DIRECT, alpha=ALPHA, label='Direct × GWP', zorder=3)

final_lccp = emb_b[-1] + ene_b[-1] + dir_b[-1]
ax2.axhline(y=final_lccp, color='#333', linestyle='--', linewidth=1.1, zorder=6)
ax2.text(2052, final_lccp * 1.015,
         f'LCCP = {final_lccp:.0f} kg CO$_2$e',
         fontsize=FONT_ANNO, color='#333', ha='left', va='bottom')
ax2.axvline(x=2049.5, color='#444', linestyle=':', linewidth=1.2, zorder=5)

ax2.set_ylabel('Cumulative CO$_2$ equivalent\n(kg CO$_2$e)', fontsize=FONT_LABEL, labelpad=6)
ax2.tick_params(axis='y', labelsize=FONT_TICK)
ax2.set_ylim(bottom=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(0.017, 0.93, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold')
ax2.legend(loc='center right', fontsize=FONT_ANNO, frameon=True, framealpha=0.9)

# ── Panel (c): LCTR evolution ─────────────────────────────
ax3 = axes[2]
emb_c = np.array(lctr_embodied_cum)
ene_c = np.array(lctr_energy_cum)
dir_c = np.array(lctr_direct_cum)
total_c = emb_c + ene_c + dir_c

ax3.bar(YEARS_ARR, emb_c,
        width=0.85, color=C_EMBOD,  alpha=ALPHA, label='Embodied', zorder=3)
ax3.bar(YEARS_ARR, ene_c,
        bottom=emb_c, width=0.85, color=C_ENERGY, alpha=ALPHA, label='Energy', zorder=3)
ax3.bar(YEARS_ARR, dir_c,
        bottom=emb_c + ene_c, width=0.85, color=C_DIRECT, alpha=ALPHA, label='Direct', zorder=3)

# Peak annotation
peak_idx = int(np.argmax(total_c))
peak_yr  = YEARS_ALL[peak_idx]
peak_val = total_c[peak_idx]
ax3.annotate(f'Peak LCTR = {peak_val:.3f} pK\n(Year {peak_yr})',
             xy=(peak_yr, peak_val),
             xytext=(peak_yr + 10, peak_val * 0.72),
             fontsize=FONT_ANNO,
             arrowprops=dict(arrowstyle='->', color='#333', lw=0.9),
             color='#333')

# Final value (T=2100)
val_2100 = total_c[-1]
ax3.annotate(f'LCTR(2100) = {val_2100:.3f} pK',
             xy=(2100, val_2100),
             xytext=(2085, val_2100 * 1.25),
             fontsize=FONT_ANNO,
             arrowprops=dict(arrowstyle='->', color='#555', lw=0.9),
             color='#555')

ax3.axvline(x=2049.5, color='#444', linestyle=':', linewidth=1.2, zorder=5)

ax3.set_xlabel('Year', fontsize=FONT_LABEL, labelpad=6)
ax3.set_ylabel('Cumulative temperature response\n(pK)', fontsize=FONT_LABEL, labelpad=6)
ax3.tick_params(axis='both', labelsize=FONT_TICK)
ax3.set_ylim(bottom=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.text(0.017, 0.93, '(c)', transform=ax3.transAxes, fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=FONT_ANNO, frameon=True, framealpha=0.9)

# ── Shared x-axis formatting ─────────────────────────────
ax3.set_xlim(2039.3, 2100.7)
ax3.xaxis.set_major_locator(MultipleLocator(5))
ax3.xaxis.set_minor_locator(MultipleLocator(1))
for ax in axes:
    ax.tick_params(axis='x', which='minor', length=2)
    ax.grid(axis='y', alpha=0.18, linestyle='--', linewidth=0.7, zorder=0)

# ── Save ─────────────────────────────────────────────────
fig.savefig(OUT_PDF, format='pdf', bbox_inches='tight')
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
fig.savefig(OUT_TIF, dpi=600, bbox_inches='tight')
print(f'Saved: {OUT_PDF}')
print(f'Saved: {OUT_PNG}')
print(f'Saved: {OUT_TIF}')
plt.show()



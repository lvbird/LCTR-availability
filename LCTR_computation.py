import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator
import os

CLIMATE_MODEL = os.environ.get("LCTR_CLIMATE_MODEL", "").strip()
CLIMATE_DATA_DIR = os.environ.get("LCTR_CLIMATE_DATA_DIR", os.path.join("CMIP6", "results", "3hr"))


def with_climate_tag(filename):
    """Append the active CMIP6 model tag to output files when model mode is enabled."""
    if not CLIMATE_MODEL:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{CLIMATE_MODEL}_3hr{ext}"

# ==========================================
# 1. 基础参数定义 & 数据录入
# ==========================================

# 地点
locations = ['沈阳', '长春', '哈尔滨', '北京', '天津', '石家庄', '太原', '呼和浩特', '济南', '上海', '南京', '杭州', '合肥', '福州', '南昌', '郑州', '武汉', '长沙', '重庆', '成都', '广州', '南宁', '海口', '贵阳', '昆明', '西安', '兰州', '西宁', '银川', '乌鲁木齐', '拉萨']

# 各省空调年销售量/消费量 (台/年)
Sales_Volume = {
    '沈阳': 1123209,    # 辽宁
    '长春': 219387,     # 吉林
    '哈尔滨': 303046,   # 黑龙江
    '北京': 1906949,    # 北京
    '天津': 855340,     # 天津
    '石家庄': 6028792,  # 河北
    '太原': 1027412,    # 山西
    '呼和浩特': 241388, # 内蒙古
    '济南': 6503021,    # 山东
    '上海': 2561701,    # 上海
    '南京': 11280649,   # 江苏
    '杭州': 7719249,    # 浙江
    '合肥': 6155209,    # 安徽
    '福州': 4375775,    # 福建
    '南昌': 4139982,    # 江西
    '郑州': 8461495,    # 河南
    '武汉': 3497416,    # 湖北
    '长沙': 9069984,    # 湖南
    '重庆': 1853622,    # 重庆
    '成都': 7092025,    # 四川
    '广州': 11543028,   # 广东
    '南宁': 3156549,    # 广西
    '海口': 505451,     # 海南
    '贵阳': 754722,     # 贵州
    '昆明': 618374,     # 云南
    '西安': 2788127,    # 陕西
    '兰州': 235416,     # 甘肃
    '西宁': 19209,      # 青海
    '银川': 111144,     # 宁夏
    '乌鲁木齐': 274685, # 新疆
    '拉萨': 27646       # 西藏
}

#不同制冷剂不同工况下基本EER
EER_data = {
    'R32': {
        24: 11.59, 25: 10.93, 26: 10.52, 27: 10.12, 28: 9.80, 29: 8.76, 30: 7.37,
        31: 6.14, 32: 5.57, 33: 4.92, 34: 4.47, 35: 4.03, 36: 3.79, 37: 3.43,
        38: 3.19, 39: 3.00, 40: 2.76, 41: 2.61
    },
    'R410A': {
        24: 10.47, 25: 10.00, 26: 9.71, 27: 9.43, 28: 9.21, 29: 7.94, 30: 6.72,
        31: 5.94, 32: 5.15, 33: 4.64, 34: 4.25, 35: 3.78, 36: 3.51, 37: 3.27,
        38: 3.08, 39: 2.91, 40: 2.64, 41: 2.42
    },
    'R290': {
        24: 16.18, 25: 13.12, 26: 12.85, 27: 11.95, 28: 10.69, 29: 8.65, 30: 7.30,
        31: 6.17, 32: 5.24, 33: 4.76, 34: 4.32, 35: 4.18, 36: 3.88, 37: 3.63,
        38: 3.08, 39: 2.64, 40: 1.82, 41: 1.49
    }
}

# 生产年份带来的能效提升 (APF_improved)
# 注意：这部分将作为附加值加到基础EER上
APF_improved = {
    2025: 0, 2026: 0.054, 2027: 0.107, 2028: 0.157, 2029: 0.204,
    2030: 0.25, 2031: 0.294, 2032: 0.336, 2033: 0.376, 2034: 0.415,
    2035: 0.453, 2036: 0.489, 2037: 0.524, 2038: 0.559, 2039: 0.592,
    2040: 0.624, 2041: 0.655, 2042: 0.685, 2043: 0.715, 2044: 0.743,
    2045: 0.771, 2046: 0.799, 2047: 0.826, 2048: 0.852, 2049: 0.877,
    2050: 0.901
}

# 使用寿命内的能效衰减系数 (原逻辑：APF * factor，这里转换为能耗乘数 1/factor)
# 0-2年: 1.0, 3-5年: 0.941, 6-7年: 0.913, 8-9年: 0.874
# 注：空调寿命设为10年(life 0..9)，10年以上不适用
def get_decay_factor(life):
    if life >= 3 and life <= 5: return 0.941
    elif life >= 6 and life <= 7: return 0.913
    elif life >= 8: return 0.874
    else: return 1.0

#AR6的AGTP数据
AGTP_data = {
    "HFC-32": {
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
        96: 5.7025E-14, 97: 5.6780E-14, 98: 5.6538E-14, 99: 5.6299E-14, 100: 5.6064E-14
    },
    "HFC-125": {
        1: 1.2259E-12, 2: 2.1063E-12, 3: 2.7307E-12, 4: 3.1656E-12, 5: 3.4599E-12,
        6: 3.6500E-12, 7: 3.7632E-12, 8: 3.8195E-12, 9: 3.8341E-12, 10: 3.8182E-12,
        11: 3.7801E-12, 12: 3.7262E-12, 13: 3.6612E-12, 14: 3.5885E-12, 15: 3.5108E-12,
        16: 3.4299E-12, 17: 3.3475E-12, 18: 3.2644E-12, 19: 3.1815E-12, 20: 3.0993E-12,
        21: 3.0183E-12, 22: 2.9387E-12, 23: 2.8608E-12, 24: 2.7847E-12, 25: 2.7104E-12,
        26: 2.6380E-12, 27: 2.5676E-12, 28: 2.4991E-12, 29: 2.4325E-12, 30: 2.3679E-12,
        31: 2.3051E-12, 32: 2.2441E-12, 33: 2.1849E-12, 34: 2.1274E-12, 35: 2.0717E-12,
        36: 2.0176E-12, 37: 1.9651E-12, 38: 1.9142E-12, 39: 1.8647E-12, 40: 1.8168E-12,
        41: 1.7703E-12, 42: 1.7252E-12, 43: 1.6815E-12, 44: 1.6390E-12, 45: 1.5978E-12,
        46: 1.5579E-12, 47: 1.5192E-12, 48: 1.4816E-12, 49: 1.4451E-12, 50: 1.4098E-12,
        51: 1.3755E-12, 52: 1.3422E-12, 53: 1.3099E-12, 54: 1.2786E-12, 55: 1.2483E-12,
        56: 1.2188E-12, 57: 1.1902E-12, 58: 1.1625E-12, 59: 1.1356E-12, 60: 1.1095E-12,
        61: 1.0842E-12, 62: 1.0597E-12, 63: 1.0358E-12, 64: 1.0127E-12, 65: 9.9031E-13,
        66: 9.6856E-13, 67: 9.4746E-13, 68: 9.2698E-13, 69: 9.0712E-13, 70: 8.8785E-13,
        71: 8.6915E-13, 72: 8.5100E-13, 73: 8.3339E-13, 74: 8.1631E-13, 75: 7.9973E-13,
        76: 7.8364E-13, 77: 7.6803E-13, 78: 7.5287E-13, 79: 7.3817E-13, 80: 7.2389E-13,
        81: 7.1004E-13, 82: 6.9659E-13, 83: 6.8354E-13, 84: 6.7087E-13, 85: 6.5857E-13,
        86: 6.4663E-13, 87: 6.3503E-13, 88: 6.2378E-13, 89: 6.1285E-13, 90: 6.0224E-13,
        91: 5.9193E-13, 92: 5.8192E-13, 93: 5.7220E-13, 94: 5.6276E-13, 95: 5.5360E-13,
        96: 5.4469E-13, 97: 5.3604E-13, 98: 5.2763E-13, 99: 5.1946E-13, 100: 5.1153E-13
    },
    "HFO-1234yf": {
        1: 4.1984E-15, 2: 3.1573E-15, 3: 2.3826E-15, 4: 1.8048E-15, 5: 1.3730E-15,
        6: 1.0497E-15, 7: 8.0734E-16, 8: 6.2533E-16, 9: 4.8846E-16, 10: 3.8539E-16,
        11: 3.0766E-16, 12: 2.4896E-16, 13: 2.0456E-16, 14: 1.7089E-16, 15: 1.4532E-16,
        16: 1.2583E-16, 17: 1.1092E-16, 18: 9.9480E-17, 19: 9.0643E-17, 20: 8.3776E-17,
        21: 7.8397E-17, 22: 7.4145E-17, 23: 7.0746E-17, 24: 6.7995E-17, 25: 6.5738E-17,
        26: 6.3857E-17, 27: 6.2266E-17, 28: 6.0897E-17, 29: 5.9700E-17, 30: 5.8639E-17,
        31: 5.7684E-17, 32: 5.6813E-17, 33: 5.6010E-17, 34: 5.5264E-17, 35: 5.4564E-17,
        36: 5.3902E-17, 37: 5.3275E-17, 38: 5.2676E-17, 39: 5.2103E-17, 40: 5.1553E-17,
        41: 5.1024E-17, 42: 5.0513E-17, 43: 5.0021E-17, 44: 4.9545E-17, 45: 4.9084E-17,
        46: 4.8638E-17, 47: 4.8206E-17, 48: 4.7787E-17, 49: 4.7380E-17, 50: 4.6985E-17,
        51: 4.6602E-17, 52: 4.6230E-17, 53: 4.5868E-17, 54: 4.5516E-17, 55: 4.5174E-17,
        56: 4.4842E-17, 57: 4.4518E-17, 58: 4.4203E-17, 59: 4.3896E-17, 60: 4.3597E-17,
        61: 4.3306E-17, 62: 4.3022E-17, 63: 4.2745E-17, 64: 4.2475E-17, 65: 4.2211E-17,
        66: 4.1953E-17, 67: 4.1702E-17, 68: 4.1457E-17, 69: 4.1217E-17, 70: 4.0982E-17,
        71: 4.0753E-17, 72: 4.0528E-17, 73: 4.0308E-17, 74: 4.0093E-17, 75: 3.9882E-17,
        76: 3.9676E-17, 77: 3.9473E-17, 78: 3.9275E-17, 79: 3.9080E-17, 80: 3.8889E-17,
        81: 3.8701E-17, 82: 3.8517E-17, 83: 3.8336E-17, 84: 3.8158E-17, 85: 3.7983E-17,
        86: 3.7811E-17, 87: 3.7641E-17, 88: 3.7475E-17, 89: 3.7311E-17, 90: 3.7149E-17,
        91: 3.6989E-17, 92: 3.6832E-17, 93: 3.6677E-17, 94: 3.6525E-17, 95: 3.6374E-17,
        96: 3.6225E-17, 97: 3.6078E-17, 98: 3.5933E-17, 99: 3.5789E-17, 100: 3.5648E-17
    },
    "CO2": {
        1: 1.8706E-16, 2: 3.1578E-16, 3: 4.0297E-16, 4: 4.6080E-16,
        5: 4.9801E-16, 6: 5.2084E-16, 7: 5.3378E-16, 8: 5.3998E-16,
        9: 5.4170E-16, 10: 5.4048E-16, 11: 5.3741E-16, 12: 5.3324E-16,
        13: 5.2847E-16, 14: 5.2344E-16, 15: 5.1836E-16, 16: 5.1337E-16,
        17: 5.0854E-16, 18: 5.0392E-16, 19: 4.9952E-16, 20: 4.9536E-16,
        21: 4.9142E-16, 22: 4.8769E-16, 23: 4.8416E-16, 24: 4.8081E-16,
        25: 4.7763E-16,  26: 4.7460E-16,
        27: 4.7171E-16,
        28: 4.6894E-16,
        29: 4.6630E-16,
        30: 4.6376E-16,
        31: 4.6132E-16,
        32: 4.5897E-16,
        33: 4.5670E-16,
        34: 4.5452E-16,
        35: 4.5241E-16,
        36: 4.5037E-16,
        37: 4.4840E-16,
        38: 4.4650E-16,
        39: 4.4465E-16,
        40: 4.4286E-16,
        41: 4.4112E-16,
        42: 4.3944E-16,
        43: 4.3782E-16,
        44: 4.3624E-16,
        45: 4.3470E-16,
        46: 4.3322E-16,
        47: 4.3178E-16,
        48: 4.3038E-16,
        49: 4.2902E-16,
        50: 4.2770E-16,
        51: 4.2643E-16,
        52: 4.2519E-16,
        53: 4.2399E-16,
        54: 4.2282E-16,
        55: 4.2169E-16,
        56: 4.2059E-16,
        57: 4.1953E-16,
        58: 4.1849E-16,
        59: 4.1749E-16,
        60: 4.1652E-16,
        61: 4.1557E-16,
        62: 4.1466E-16,
        63: 4.1377E-16,
        64: 4.1291E-16,
        65: 4.1208E-16,
        66: 4.1127E-16,
        67: 4.1048E-16,
        68: 4.0972E-16,
        69: 4.0899E-16,
        70: 4.0827E-16,
        71: 4.0758E-16,
        72: 4.0691E-16,
        73: 4.0625E-16,
        74: 4.0562E-16,
        75: 4.0501E-16,
        76: 4.0442E-16,
        77: 4.0384E-16,
        78: 4.0328E-16,
        79: 4.0274E-16,
        80: 4.0222E-16,
        81: 4.0171E-16,
        82: 4.0122E-16,
        83: 4.0074E-16,
        84: 4.0028E-16,
        85: 3.9984E-16,
        86: 3.9940E-16,
        87: 3.9898E-16,
        88: 3.9858E-16,
        89: 3.9818E-16,
        90: 3.9780E-16,
        91: 3.9743E-16,
        92: 3.9708E-16,
        93: 3.9673E-16,
        94: 3.9640E-16,
        95: 3.9607E-16,
        96: 3.9576E-16,
        97: 3.9545E-16,
        98: 3.9516E-16,
        99: 3.9487E-16,
        100: 3.9460E-16
    },
    "HC-290": {1: 1.6677E-16, 2: 1.2542E-16, 3: 9.4644E-17, 4: 7.1691E-17, 5: 5.4538E-17,
    6: 4.1697E-17, 7: 3.2069E-17, 8: 2.4838E-17, 9: 1.9401E-17, 10: 1.5307E-17,
    11: 1.2220E-17, 12: 9.8879E-18, 13: 8.1239E-18, 14: 6.7867E-18, 15: 5.7707E-18,
    16: 4.9966E-18, 17: 4.4046E-18, 18: 3.9500E-18, 19: 3.5990E-18, 20: 3.3262E-18,
    21: 3.1125E-18, 22: 2.9436E-18, 23: 2.8086E-18, 24: 2.6994E-18, 25: 2.6097E-18,
    26: 2.5350E-18, 27: 2.4718E-18, 28: 2.4175E-18, 29: 2.3700E-18, 30: 2.3278E-18,
    31: 2.2899E-18, 32: 2.2553E-18, 33: 2.2234E-18, 34: 2.1938E-18, 35: 2.1660E-18,
    36: 2.1397E-18, 37: 2.1148E-18, 38: 2.0910E-18, 39: 2.0683E-18, 40: 2.0464E-18,
    41: 2.0254E-18, 42: 2.0051E-18, 43: 1.9856E-18, 44: 1.9667E-18, 45: 1.9484E-18,
    46: 1.9307E-18, 47: 1.9135E-18, 48: 1.8969E-18, 49: 1.8807E-18, 50: 1.8651E-18,
    51: 1.8499E-18, 52: 1.8351E-18, 53: 1.8207E-18, 54: 1.8068E-18, 55: 1.7932E-18,
    56: 1.7800E-18, 57: 1.7671E-18, 58: 1.7546E-18, 59: 1.7424E-18, 60: 1.7305E-18,
    61: 1.7190E-18, 62: 1.7077E-18, 63: 1.6967E-18, 64: 1.6860E-18, 65: 1.6755E-18,
    66: 1.6653E-18, 67: 1.6553E-18, 68: 1.6456E-18, 69: 1.6360E-18, 70: 1.6267E-18,
    71: 1.6176E-18, 72: 1.6087E-18, 73: 1.6000E-18, 74: 1.5914E-18, 75: 1.5831E-18,
    76: 1.5749E-18, 77: 1.5668E-18, 78: 1.5590E-18, 79: 1.5512E-18, 80: 1.5436E-18,
    81: 1.5362E-18, 82: 1.5289E-18, 83: 1.5217E-18, 84: 1.5146E-18, 85: 1.5077E-18,
    86: 1.5008E-18, 87: 1.4941E-18, 88: 1.4875E-18, 89: 1.4810E-18, 90: 1.4746E-18,
    91: 1.4682E-18, 92: 1.4620E-18, 93: 1.4558E-18, 94: 1.4498E-18, 95: 1.4438E-18,
    96: 1.4379E-18, 97: 1.4320E-18, 98: 1.4263E-18, 99: 1.4206E-18, 100: 1.4150E-18}
}

# ---------------------------------------------------------
# 2. 辅助函数：排放因子插值 & 数据读取
# ---------------------------------------------------------

def get_em_predictions(file_path="EM_raw.xlsx"):
    """读取并处理电力排放因子，生成逐年插值数据"""
    raw_data = pd.read_excel(file_path, sheet_name=None)
    full_years = np.arange(2020, 2061)
    predictions = {}

    for scenario, df in raw_data.items():
        # 假设第一列是城市名，变成Index
        # 你的EM_raw格式应该是：列头是年份，索引是城市？或者列头是城市，索引是年份？
        # 根据原代码逻辑：df.columns[0] set_index，说明第一列是年份
        df = df.set_index(df.columns[0]).astype(float)
        result_df = pd.DataFrame(index=full_years, columns=df.columns)
        
        for city in df.columns:
            known_years = df.index.values.astype(int)
            values = df[city].values
            
            # PCHIP 插值
            interpolator = PchipInterpolator(known_years, values, extrapolate=False)
            predicted = interpolator(full_years)
            
            # 后处理：保留4位小数，限制范围，强制单调递减
            predicted = np.round(predicted, 4)
            predicted = np.clip(predicted, np.min(values), np.max(values))
            for i in range(1, len(predicted)):
                predicted[i] = min(predicted[i], predicted[i-1] - 1e-4)
            
            result_df[city] = predicted
        
        predictions[scenario] = result_df
    return predictions

# 获取排放因子数据
# 请确保当前目录下有 EM_raw.xlsx
if os.path.exists("EM_raw.xlsx"):
    em_data = get_em_predictions()
else:
    print("警告：未找到 EM_raw.xlsx，请确保文件存在。")
    em_data = {} # 避免后续报错，实际运行时需要文件

def load_temp_data_grouped(scenario):
    """
    读取指定SSP情景的温度分布文件，并重组为易于查询的字典。
    文件格式：Year | Province_City/City | ... Temps ...
    返回字典结构: { (City, Year): {Temp: Hours, ...} }
    """
    if CLIMATE_MODEL:
        filename = os.path.join(
            CLIMATE_DATA_DIR,
            f"{CLIMATE_MODEL}_{scenario}_TempDist_3hr.xlsx"
        )
        city_col = "City"
    else:
        filename = f"Processed_Hourly_Stats_{scenario}_Ordered.xlsx"
        city_col = "Province_City"
    print(f"正在读取温度文件: {filename} ...")
    
    try:
        df = pd.read_excel(filename)
        if CLIMATE_MODEL:
            df = df[(df["Model"] == CLIMATE_MODEL) & (df["Scenario"] == scenario)].copy()
            df = df.drop(columns=[c for c in ["Model", "Scenario"] if c in df.columns])
        # 假设第一列是 Year, 第二列是 Province_City/City，后面是温度列
        
        # 将数据转换为长格式或者直接利用索引
        # 设置索引为 [Year, City] 加速查询
        df.set_index(['Year', city_col], inplace=True)
        
        # 将 DataFrame 转为字典，键为 (Year, City)，值为 Series (Temp -> Hours)
        # 这是一个巨大的字典，但查询速度极快
        # 注意：Excel中列名为数字（如 -38, -37...），需要确保类型匹配
        temp_dict = {}
        for idx, row in df.iterrows():
            # idx 是 (Year, City)
            # row 是该年该城市的温度分布
            # 过滤掉 0 值以节省空间 (可选)
            valid_hours = row[row > 0].to_dict()
            # 确保 key 是年份和城市名
            year, city = idx
            temp_dict[(year, city)] = valid_hours
            
        return temp_dict
        
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到。")
        return {}


# 省份-城市映射：销量文件按省份给出，这里映射到当前脚本使用的城市名称
CITY_TO_PROVINCE = {
    '沈阳': '辽宁',
    '长春': '吉林',
    '哈尔滨': '黑龙江',
    '北京': '北京',
    '天津': '天津',
    '石家庄': '河北',
    '太原': '山西',
    '呼和浩特': '内蒙古',
    '济南': '山东',
    '上海': '上海',
    '南京': '江苏',
    '杭州': '浙江',
    '合肥': '安徽',
    '福州': '福建',
    '南昌': '江西',
    '郑州': '河南',
    '武汉': '湖北',
    '长沙': '湖南',
    '重庆': '重庆',
    '成都': '四川',
    '广州': '广东',
    '南宁': '广西',
    '海口': '海南',
    '贵阳': '贵州',
    '昆明': '云南',
    '西安': '陕西',
    '兰州': '甘肃',
    '西宁': '青海',
    '银川': '宁夏',
    '乌鲁木齐': '新疆',
    '拉萨': '西藏'
}


def load_city_sales_lookup(file_path="Regression/National_AC_sales_estimation_AllSSP_weibull.xlsx"):
    """读取省级年度销量，按 SSP 情景 + 城市 + 年份建立三级查找表。"""
    if not os.path.exists(file_path):
        print(f"警告：未找到销量文件 {file_path}，将按 0 处理。")
        return {}

    sales_df = pd.read_excel(file_path, sheet_name="provincial_sales_flow")
    sales_df = sales_df[['SSP_Scenario', 'Province', 'Year', 'New_Sales_Units']].copy()
    sales_df['Year'] = sales_df['Year'].astype(int)
    sales_df['New_Sales_Units'] = sales_df['New_Sales_Units'].clip(lower=0)
    sales_df = sales_df[(sales_df['Year'] >= 2025) & (sales_df['Year'] <= 2050)]

    # 三级查找表: {ssp_scenario: {city: {year: volume}}}
    city_sales_lookup = {}
    for ssp_scen, grp in sales_df.groupby('SSP_Scenario'):
        piv = grp.pivot(index='Year', columns='Province', values='New_Sales_Units').fillna(0)
        scen_lookup = {}
        for city, province in CITY_TO_PROVINCE.items():
            if province in piv.columns:
                scen_lookup[city] = piv[province].to_dict()
            else:
                scen_lookup[city] = {}
        city_sales_lookup[str(ssp_scen)] = scen_lookup

    return city_sales_lookup


def compute_refrigerant_mix():
    """
    三个制冷剂替代政策情景下2025-2050年逐年占比。

    所有情景起点（2025）：HFC-32=85%, R410A=15%, HC-290=0%
    所有情景均在2029年前线性淡出R410A。

    BAU（HFC-32稳态）：2030年起保持100%HFC-32。
    MTP（渐进过渡 Moderate Transition Pathway）：
        2035年起引入HC-290，2050年达90%（线性），HFC-32填补余量。
        基于基加利修正案基准期（中国2020-2022）的平均GWP倒推，
        2029减25%、2035减40%、2040减59%，2045符1年账57%-77%之间。
    APD（加速替代 Accelerated Phase-Down）：
        2025起HC-290线性增长，2035年达100%；R410A仍于2029年清零。
    """
    mix = {}
    years_range = list(range(2025, 2051))
    for sc in ['BAU', 'MTP', 'APD']:
        rows = {}
        for y in years_range:
            # R410A: 线性 15%→0%（2025-2029）
            r410a = max(0.0, 0.15 * (2029 - y) / (2029 - 2025)) if y <= 2029 else 0.0
            if sc == 'BAU':
                r290 = 0.0
            elif sc == 'MTP':
                # HC-290从2035年起线性增加到90%@2050
                r290 = min(0.90, 0.90 * (y - 2035) / (2050 - 2035)) if y > 2035 else 0.0
            else:  # APD
                # HC-290从2025年线性增加到100%@2035
                r290 = min(1.0, (y - 2025) / (2035 - 2025))
            r32 = max(0.0, 1.0 - r410a - r290)
            rows[y] = {'R410A': r410a, 'R32': r32, 'R290': r290}
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = 'Year'
        mix[sc] = df
    return mix


city_sales_by_year = load_city_sales_lookup()
refrigerant_mix_by_scenario = compute_refrigerant_mix()

# SSP 情景名称映射（LCTR 用小写，销量文件用大写）
_SCENARIO_TO_SSP = {
    'ssp126': 'SSP126',
    'ssp245': 'SSP245',
    'ssp585': 'SSP585',
}


def get_city_sales_units(city, year, ssp_scenario='SSP245'):
    """获取某城市某年在特定 SSP 情景下的新售销量（台）。"""
    scen_lookup = city_sales_by_year.get(ssp_scenario, city_sales_by_year.get('SSP245', {}))
    return float(scen_lookup.get(city, {}).get(year, 0.0))


def get_refrigerant_shares(policy_scenario, year):
    """获取某情景某年的制冷剂占比。"""
    scenario_df = refrigerant_mix_by_scenario.get(policy_scenario)
    if scenario_df is None or year not in scenario_df.index:
        # 兼容底：默认BAU 2029+的全HFC-32
        return {'R410A': 0.0, 'R32': 1.0, 'R290': 0.0}
    row = scenario_df.loc[year]
    return {'R410A': float(row['R410A']), 'R32': float(row['R32']), 'R290': float(row['R290'])}
    
# ---------------------------------------------------------
# 3. 核心计算逻辑：仅制冷、线性负荷、动态EER
# ---------------------------------------------------------

def calculate_aec_cooling_only(refrigerant, year_prod, year_use, temp_dist_hours):
    """
    计算单台机组在特定年份的制冷耗电量 (kWh)
    
    参数:
        refrigerant: 'R32', 'R410A', 'R290'
        year_prod: 生产年份 (用于确定 APF_improved)
        year_use: 使用年份 (用于确定 life 和 decay)
        temp_dist_hours: 字典 {Temp: Hours}, 该城市该年的温度分布
    
    返回:
        AEC (kWh)
    """
    life = year_use - year_prod
    if life < 0: return 0
    
    # 1. 确定 EER 表
    base_eers = EER_data[refrigerant]
    
    # 2. 确定能效提升 (APF_improved)
    # 假设提升是加在 EER 值上的 (EER_new = EER_base + Improvement)
    improvement = APF_improved.get(year_prod, 0)
    
    # 3. 确定老化衰减系数 (能耗乘数)
    decay_efficiency = get_decay_factor(life) 
    # 注意：APF衰减意味着相同负荷下能耗增加。
    # 原逻辑：APF_new = APF * 0.941 -> Power = Load / APF_new = Load / (APF * 0.941) = (Load/APF) * (1/0.941)
    # 令 energy_multiplier = 1 / decay_efficiency
    energy_multiplier = 1.0 / decay_efficiency
    
    total_energy_kwh = 0.0
    
    # 4. 遍历温度区间 (24°C - 41°C)
    # 虽然 EER 表有到 42，温度分布文件可能有更高，但根据指令只计算 24-41
    for t in range(24, 42): 
        # 获取该温度下的小时数
        # Excel列名可能是整数或字符串，需注意匹配。这里假设为整数。
        hours = temp_dist_hours.get(t, 0)
        if hours == 0:
            continue
            
        # 计算线性负荷 (W)
        # 23度时0负荷, 35度时3500W
        load_w = (t - 23) * 3500 / (35 - 23)
        
        # 获取 EER
        # 如果温度超出 EER 表范围 (虽然这里限制了 24-41)，做个保护
        base_eer = base_eers.get(t, 3.0) # 默认值防报错
        current_eer = base_eer + improvement
        
        # 计算功率 (W)
        power_w = load_w / current_eer
        
        # 累加能耗 (Wh)
        total_energy_kwh += (power_w * hours)
        
    # 转换为 kWh 并应用老化系数
    return (total_energy_kwh / 1000.0) * energy_multiplier

# ==========================================
# 4. 主模拟循环
# ==========================================

# 定义维度
years_prod = list(range(2025, 2051))  # 26年：2025-2050（10年寿命下覆盖至2060年运行终止）
years_use = list(range(2025, 2060))   # 最晚使用年份：2050+9=2059
lifespan = 10
scenarios = ['ssp126', 'ssp245', 'ssp585']
policy_scenarios = ['BAU', 'MTP', 'APD']
refrigerants = ['R410A', 'R32', 'R290'] # 替换 R1 为 R290
types = ['P', 'R', 'E']

# 充注量定义 (kg)
charge_amount = {
    'R410A': 0.762,
    'R32': 0.696,
    'R290': 0.367
}

# 制造与回收排放因子 (kgCO2eq/kg_ref)
rfm_factors = {'R410A': 10.7, 'R32': 7.2, 'R290': 0.05} 
RFD = 2.1 # 全局制冷剂处理排放因子

# 制造与回收的原材料排放 (kgCO2eq)
# MM_sum (制造): 
MM_sum = (15 + 28) * (0.46 * 1.8 + 0.12 * 12.6 + 0.19 * 3.8 + 0.23 * 2.8)
# RM_sum (回收): 
RM_sum = 0.07 * (15 + 28) * (0.46 + 0.12 + 0.19) + 0.15 * (15 + 28) * 0.23

# 初始化 LCTR 存储数组 (PK)
# 维度: [年份(生产), 地点, 情景, 制冷剂]
# 注意：原代码最后输出的是基于“生产年份”的 LCTR 总和？
# 原代码逻辑：LCTR_array[year] 存储的是该年生产的机组全生命周期 LCTR。
LCTR_array = xr.DataArray(
    data=np.zeros((len(years_prod), len(locations), len(scenarios), len(refrigerants))),
    dims=['年份', '地点', '情景', '制冷剂'],
    coords={
        '年份': years_prod,
        '地点': locations,
        '情景': scenarios,
        '制冷剂': refrigerants
    }
)

# 开始循环
print("开始进行全生命周期模拟计算...")

for scenario in scenarios:
    print(f"当前情景: {scenario}")
    
    # 1. 预读取该情景下的温度数据 (加速 IO)
    # 返回格式: {(Year_Use, City): {Temp: Hours}}
    temp_data_scenario = load_temp_data_grouped(scenario)
    
    if not temp_data_scenario:
        print(f"跳过情景 {scenario} (数据缺失)")
        continue

    for location in locations:
        # 获取该城市该情景下的排放因子序列 (Series)
        try:
            city_em_series = em_data[scenario][location]
        except KeyError:
            print(f"警告: 缺失 {location} 的排放因子数据")
            continue
            
        for refrigerant in refrigerants:
            # 获取制冷剂特定参数
            C = charge_amount[refrigerant]
            RFM = rfm_factors[refrigerant]
            
            for year_p in years_prod:
                # 针对每一台在 year_p 生产的空调，计算其全生命周期(15年)的总温升贡献
                total_lctr_pk = 0.0
                
                # 确定年泄漏率 ALR (基于生产年份判定还是使用年份？原代码基于 year_p)
                # 原逻辑: if year < 2041 (指 year_production). 
                if year_p < 2041:
                    ALR = 0.05 - 0.002 * (year_p - 2025)
                else:
                    ALR = 0.02
                
                # 循环生命周期
                for life in range(lifespan):
                    year_u = year_p + life # 使用年份
                    horizon = 2100 - year_u
                    
                    # 超过2100年不计入 AGTP (或 AGTP=0)
                    if horizon < 1: continue
                    if horizon > 100: horizon = 100 # 防止越界，虽不太可能
                    
                    # 1. 获取 AGTP 值
                    AGTP_CO2 = AGTP_data['CO2'][horizon]
                    
                    if refrigerant == 'R410A':
                        AGTP_Ref = (AGTP_data['HFC-125'][horizon] + AGTP_data['HFC-32'][horizon]) / 2
                    elif refrigerant == 'R32':
                        AGTP_Ref = AGTP_data['HFC-32'][horizon]
                    elif refrigerant == 'R290':
                        AGTP_Ref = AGTP_data['HC-290'][horizon]
                    
                    # 2. 计算当年能耗 (AEC)
                    # 需查找 temp_data_scenario[(year_u, location)]
                    temp_dist = temp_data_scenario.get((year_u, location), {})
                    if not temp_dist:
                        # 可能是年份超出Excel范围 (如2060以后)
                        AEC = 0
                    else:
                        AEC = calculate_aec_cooling_only(refrigerant, year_p, year_u, temp_dist)
                    
                    # 3. 获取当年电力排放因子
                    # 如果年份超出 EM 预测范围 (2060)，取最后一年或0? 假设 EM data 足够
                    if year_u in city_em_series.index:
                        EM = city_em_series[year_u]
                    else:
                        EM = city_em_series.iloc[-1] # 兜底
                    
                    # 4. 计算回收率 EOL (End of Life Direct Emission Rate)
                    # 原逻辑: 0.97 - 0.01 * (year_u - 2025)
                    EOL = 0.97 - 0.01 * (year_u - 2025)
                    if EOL < 0: EOL = 0 # 保护
                    
                    # 5. 计算各阶段排放的 AGTP 贡献 (Temp Change)
                    # 单位: kg * (K/kg) = K
                    
                    # 直接排放 (Direct)
                    # 生产阶段(Life=0): C * ALR
                    # 运行阶段(Life=1..13): C * ALR
                    # 报废阶段(Life=14): C * (ALR + EOL)
                    
                    # 间接排放 (Indirect)
                    # 生产阶段: MM_sum + C*(1+ALR)*RFM + AEC*EM
                    # 运行阶段: C*ALR*RFM + AEC*EM
                    # 报废阶段: RM_sum + C*ALR*RFM + AEC*EM + C*(1-EOL)*RFD
                    
                    # 简化逻辑：分年累加
                    
                    direct_emission_kg = 0.0
                    indirect_emission_kgCO2 = 0.0
                    
                    if life == 0: # 生产年 (包含第一年运行)
                        direct_emission_kg = C * ALR
                        indirect_emission_kgCO2 = MM_sum + C * (1 + ALR) * RFM + AEC * EM
                    
                    elif life == lifespan - 1: # 报废年 (Life=14)
                        direct_emission_kg = C * (ALR + EOL)
                        indirect_emission_kgCO2 = RM_sum + C * ALR * RFM + AEC * EM + C * (1 - EOL) * RFD
                    
                    else: # 中间运行年
                        direct_emission_kg = C * ALR
                        indirect_emission_kgCO2 = C * ALR * RFM + AEC * EM
                    
                    # 当年排放贡献的温升 (Delta T)
                    delta_T = (direct_emission_kg * AGTP_Ref) + (indirect_emission_kgCO2 * AGTP_CO2)
                    
                    total_lctr_pk += delta_T
                
                # 存入 DataArray, 转换为 pK (1e12)
                LCTR_array.loc[year_p, location, scenario, refrigerant] = total_lctr_pk * 1e12

print("计算完成，准备汇总与导出...")


# ==========================================
# 提取特定城市 LCTR 随生产年份变化趋势
# ==========================================

print("开始提取北京和昆明的 LCTR 趋势数据...")

# 1. 定义筛选条件
target_cities = ['北京', '昆明']
# scenarios 和 refrigerants 沿用全局变量
# years_prod 沿用全局变量 (2025-2050)

# 2. 准备数据容器
data_list = []

# 3. 提取数据
for city in target_cities:
    for scenario in scenarios:
        for refrigerant in refrigerants:
            # 获取该组合下的时间序列数据 (Series)
            # LCTR_array 维度: [年份, 地点, 情景, 制冷剂]
            # .loc[:, city, scenario, refrigerant] 返回的是基于 '年份' 的一维数组
            lctr_series = LCTR_array.loc[:, city, scenario, refrigerant]
            
            # 遍历年份存入列表
            for year in years_prod:
                val = float(lctr_series.loc[year])
                data_list.append({
                    'City': city,
                    'Scenario': scenario,
                    'Refrigerant': refrigerant,
                    'Prod_Year': year,
                    'Unit_LCTR_pK': val
                })

# 4. 创建 DataFrame
df_trend = pd.DataFrame(data_list)

# 5. 导出 Excel
file_name_trend = with_climate_tag("LCTR_Trend_Beijing_Kunming.xlsx")
with pd.ExcelWriter(file_name_trend, engine='openpyxl') as writer:
    # Sheet 1: 原始长列表
    df_trend.to_excel(writer, sheet_name='All_Data', index=False)
    
    # Sheet 2: 透视表 (方便画图)
    # 行: 生产年份
    # 列: [城市, 情景, 制冷剂]
    pivot_trend = df_trend.pivot(
        index='Prod_Year',
        columns=['City', 'Scenario', 'Refrigerant'],
        values='Unit_LCTR_pK'
    )
    pivot_trend.to_excel(writer, sheet_name='Pivot_For_Plotting')

print(f"提取完成！数据已保存至 {file_name_trend}")
print("-" * 30)

# ==========================================
# 5. 结果汇总与输出
# ==========================================

print("计算完成，正在进行全国总量汇总与导出...")

# 计算全国年度销售总量的环境影响 (LCTR_sum_array)
# 含义：某一年生产的所有空调在全中国范围内造成的全生命周期温升总和
# 单位：pK → mK
LCTR_sum_array = xr.DataArray(
    data=np.zeros((len(years_prod), len(scenarios), len(policy_scenarios))),
    dims=['年份', '情景', '场景'],
    coords={'年份': years_prod, '情景': scenarios, '场景': policy_scenarios}
)

for year in years_prod:
    for scenario in scenarios:
        for policy_scenario in policy_scenarios:
            total_national_impact = 0.0
            shares = get_refrigerant_shares(policy_scenario, year)
            ssp_key = _SCENARIO_TO_SSP.get(scenario, 'SSP245')
            for location in locations:
                city_sales = get_city_sales_units(location, year, ssp_key)
                if city_sales == 0:
                    continue
                
                for refrigerant in refrigerants:
                    # 单机 LCTR (pK)
                    unit_lctr = float(LCTR_array.loc[year, location, scenario, refrigerant])

                    # 累加：单机影响 * 年度销量 * 制冷剂占比
                    total_national_impact += unit_lctr * city_sales * shares.get(refrigerant, 0.0)

            LCTR_sum_array.loc[year, scenario, policy_scenario] = total_national_impact / 1E9 #单位转成mK

# 导出函数
def simple_export(da, sum_da, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 1. 单机详细数据 (All_Data) - 保留单机数据供微观分析
        da.to_dataframe(name='Unit_LCTR_pK').reset_index().to_excel(
            writer, 
            sheet_name='Unit_LCTR_Detail', 
            index=False
        )
        
        # 2. 分情景的单机矩阵 (可选，便于查看各城市差异)
        for scenario in da.情景.values:
            scenario_data = da.sel(情景=scenario)
            df = scenario_data.to_dataframe(name='LCTR').reset_index()
            pivot_df = df.pivot(index='年份', columns=['地点', '制冷剂'], values='LCTR')
            # 限制Sheet名称长度
            sheet_name = f"Unit_{str(scenario)[:25]}"
            pivot_df.to_excel(writer, sheet_name=sheet_name)
        
        # 3. 全国总量数据 (National Total) - 这是最重要的结果
        # 行：生产年份，列：SSP + 场景
        sum_df = sum_da.to_dataframe(name='National_Total_LCTR_mK').unstack(['情景', '场景'])
        sum_df.to_excel(writer, sheet_name='National_Total_Impact')

output_filename = with_climate_tag("LCTR_Result_National_Total.xlsx")
simple_export(LCTR_array, LCTR_sum_array, output_filename)
print(f"结果已保存至 {output_filename}")

# ==========================================
# 额外任务：计算各省份年度总温升效应 (不求和)
# ==========================================

print("-" * 30)
print("开始计算各省份独立总温升效应...")

# 1. 初始化数组 (保持与单机 LCTR_array 相同的四维结构)
# 维度: [年份, 地点, 情景, 制冷剂]
LCTR_Provincial_Total = xr.DataArray(
    data=np.zeros((len(years_prod), len(locations), len(scenarios), len(refrigerants))),
    dims=['年份', '地点', '情景', '制冷剂'],
    coords={
        '年份': years_prod,
        '地点': locations,
        '情景': scenarios,
        '制冷剂': refrigerants
    }
)

# 2. 计算逻辑
# 遍历地点，将该地点的单机数据整体乘以该地点的销量（按 SSP 情景区分）
for location in locations:
    for year in years_prod:
        for scen in scenarios:
            ssp_key = _SCENARIO_TO_SSP.get(scen, 'SSP245')
            volume = get_city_sales_units(location, year, ssp_key)
            # Unit LCTR (pK) * Sales Volume = Provincial Total LCTR (pK)
            LCTR_Provincial_Total.loc[year, location, scen, :] = LCTR_array.loc[year, location, scen, :] * volume

# 3. 导出函数
def export_provincial_data(da, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Sheet 1: 所有数据的长列表格式 (方便后续用Python或Tableau处理)
        print("正在写入长格式数据...")
        da.to_dataframe(name='Provincial_Total_LCTR_pK').reset_index().to_excel(
            writer, 
            sheet_name='All_Data_Detail', 
            index=False
        )
        
        # Sheet 2-4: 分情景展示 (透视表格式)
        # 这种格式方便在Excel中直接画图：每一列是一个"省份-制冷剂"组合，每一行是年份
        for scenario in scenarios:
            print(f"正在处理情景 {scenario} ...")
            scenario_data = da.sel(情景=scenario)
            df = scenario_data.to_dataframe(name='LCTR').reset_index()
            
            # 透视：行=年份，列=[地点, 制冷剂]
            pivot_df = df.pivot(
                index='年份', 
                columns=['地点', '制冷剂'], 
                values='LCTR'
            )
            
            sheet_name = f"{scenario}_Provincial"
            pivot_df.to_excel(writer, sheet_name=sheet_name)

# 4. 执行导出
output_prov_filename = with_climate_tag("LCTR_Result_Provincial_Total.xlsx")
export_provincial_data(LCTR_Provincial_Total, output_prov_filename)

print(f"计算完成！各省份总量数据已保存至 {output_prov_filename}")
print("-" * 30)


# ==========================================
# LCTR 构成分析 (Breakdown Calculation)
# ==========================================

# 1. 设定分析目标
target_cities = ['哈尔滨', '北京', '上海', '昆明', '广州']
target_scenario = 'ssp245'
target_year_prod = 2025
refrigerants = ['R410A', 'R32', 'R290']

# 2. 更新参数 (基于您提供的最新数据)
charge_amount = {
    'R410A': 0.762,
    'R32': 0.696,
    'R290': 0.367
}

rfm_factors = {'R410A': 10.7, 'R32': 7.2, 'R290': 0.05} 
RFD = 2.1 

# 计算材料与制造排放 (固定值)
# (15 + 28) 可能是指室外机+室内机重量
weight_sum = 15 + 28 
MM_sum = weight_sum * (0.46 * 1.8 + 0.12 * 12.6 + 0.19 * 3.8 + 0.23 * 2.8)
RM_sum = 0.07 * weight_sum * (0.46 + 0.12 + 0.19) + 0.15 * weight_sum * 0.23

# 3. 准备结果容器
breakdown_results = []

print(f"开始计算 LCTR 构成分析...")
print(f"情景: {target_scenario}, 生产年份: {target_year_prod}")

# 预读取温度数据
temp_data_scenario = load_temp_data_grouped(target_scenario)

for location in target_cities:
    # 获取排放因子
    city_em_series = em_data[target_scenario][location]
    
    for refrigerant in refrigerants:
        C = charge_amount[refrigerant]
        RFM = rfm_factors[refrigerant]
        
        # 初始化三个部分的累加器
        lctr_direct_pk = 0.0   # 直接排放 (泄漏)
        lctr_energy_pk = 0.0   # 运行能耗
        lctr_embodied_pk = 0.0 # 隐含碳 (材料+制冷剂生产回收)
        
        # 确定泄漏率 ALR (2025年生产)
        if target_year_prod < 2041:
            ALR = 0.05 - 0.002 * (target_year_prod - 2025)
        else:
            ALR = 0.02
            
        # 循环生命周期
        for life in range(lifespan):
            year_u = target_year_prod + life
            horizon = 2100 - year_u
            
            if horizon < 1: continue
            if horizon > 100: horizon = 100
            
            # 获取 AGTP
            AGTP_CO2 = AGTP_data['CO2'][horizon]
            
            if refrigerant == 'R410A':
                # 确保这里 AGTP_data 包含 HFC-125 和 HFC-32
                AGTP_Ref = (AGTP_data['HFC-125'][horizon] + AGTP_data['HFC-32'][horizon]) / 2
            elif refrigerant == 'R32':
                AGTP_Ref = AGTP_data['HFC-32'][horizon]
            elif refrigerant == 'R290':
                # 确保 AGTP_data 包含 HC-290 (或使用您之前定义的 AGTP_R290_values)
                AGTP_Ref = AGTP_data['HC-290'][horizon] 

            # 计算 AEC
            temp_dist = temp_data_scenario.get((year_u, location), {})
            if not temp_dist:
                AEC = 0
            else:
                AEC = calculate_aec_cooling_only(refrigerant, target_year_prod, year_u, temp_dist)
            
            # 获取 EM
            if year_u in city_em_series.index:
                EM = city_em_series[year_u]
            else:
                EM = city_em_series.iloc[-1]
            
            # 计算 EOL
            EOL = 0.97 - 0.01 * (year_u - 2025)
            if EOL < 0: EOL = 0
            
            # === 分解排放量 (kg) ===
            
            # 1. 直接排放部分 (Refrigerant Leakage)
            if life == lifespan - 1:  # 报废年
                mass_direct = C * (ALR + EOL)
            else:
                mass_direct = C * ALR
            
            # 2. 运行能耗部分 (Operational Energy)
            mass_energy_co2 = AEC * EM
            
            # 3. 隐含碳部分 (Embodied: Material + Ref Production/Recycle)
            mass_embodied_co2 = 0.0
            
            if life == 0:
                # 初始制造 + 初始充注 + 第一年泄漏补充的生产排放
                mass_embodied_co2 = MM_sum + C * (1 + ALR) * RFM 
            elif life == lifespan - 1:  # 报废年
                # 回收处理 + 当年泄漏补充的生产排放 + 剩余制冷剂处理排放
                mass_embodied_co2 = RM_sum + C * ALR * RFM + C * (1 - EOL) * RFD
            else:
                # 中间年份泄漏补充的生产排放
                mass_embodied_co2 = C * ALR * RFM
                
            # === 转换为 LCTR (pK) 并累加 ===
            lctr_direct_pk += mass_direct * AGTP_Ref
            lctr_energy_pk += mass_energy_co2 * AGTP_CO2
            lctr_embodied_pk += mass_embodied_co2 * AGTP_CO2

        # 乘以 1e12 转换为 pK
        lctr_direct_pk *= 1e12
        lctr_energy_pk *= 1e12
        lctr_embodied_pk *= 1e12
        total_pk = lctr_direct_pk + lctr_energy_pk + lctr_embodied_pk
        
        # 存入列表
        breakdown_results.append({
            'City': location,
            'Refrigerant': refrigerant,
            'Direct_LCTR_pK': lctr_direct_pk,
            'Energy_LCTR_pK': lctr_energy_pk,
            'Embodied_LCTR_pK': lctr_embodied_pk,
            'Total_LCTR_pK': total_pk,
            # 计算百分比
            'Direct_Ratio': lctr_direct_pk / total_pk if total_pk else 0,
            'Energy_Ratio': lctr_energy_pk / total_pk if total_pk else 0,
            'Embodied_Ratio': lctr_embodied_pk / total_pk if total_pk else 0
        })

# 4. 导出结果
df_breakdown = pd.DataFrame(breakdown_results)
file_name = with_climate_tag("LCTR_Breakdown_2025_SSP245.xlsx")
df_breakdown.to_excel(file_name, index=False, float_format="%.4f")

print("-" * 50)
print(f"计算完成！结果已保存至 {file_name}")
print("-" * 50)
print("预览 (前5行):")
print(df_breakdown[['City', 'Refrigerant', 'Total_LCTR_pK', 'Direct_Ratio', 'Energy_Ratio']].head(6))



# ==========================================
# 3.4 Target Time Point Effect Analysis
# ==========================================

print("开始计算不同目标时间点(Target Year)的影响...")

# 1. 定义分析维度
target_years = list(range(2060, 2101, 5))  # [2060, 2065, ..., 2100]
target_scenario_micro = 'ssp245'
selected_cities = ['哈尔滨', '北京', '上海', '广州', '昆明']
selected_prod_years = [2025, 2035, 2045, 2050]

# 2. 准备数据容器

# 容器1：微观对比 (Micro) - 单机 LCTR
# 维度: [目标年, 城市, 生产年, 制冷剂] (仅针对 SSP245)
results_micro = []

# 容器2：宏观总量 (Macro) - 全国累积 LCTR
# 维度: [目标年, 情景, 制冷剂]
# 这里的"全国总量"定义为：2025-2045年生产的所有空调(21年*31省销量)的总效应
results_macro = []

# 3. 主循环
for t_year in target_years:
    print(f"正在计算目标年份: {t_year} ...")
    
    # --- 宏观统计累加器 (初始化为0) ---
    # 结构: {Scenario: {Refrigerant: Total_LCTR_pK}}
    macro_agg = {s: {r: 0.0 for r in refrigerants} for s in scenarios}
    
    # 遍历情景 (宏观需要所有情景，微观只需要SSP245)
    for scenario in scenarios:
        
        # 预读取温度数据 (如果之前已加载，这里可复用，否则重新加载)
        # 注意：这里为了稳妥重新加载，如果内存足够大之前应该已经有了
        # 假设上下文已有 temp_data_scenario (但那是最后一个循环的)，所以最好重新调用
        # 为了速度，我们在这里仅加载一次，服务于内部循环
        temp_data = load_temp_data_grouped(scenario)
        if not temp_data: continue
            
        # 遍历所有城市 (为了算宏观总量)
        for location in locations:
            # 获取排放因子
            city_em_series = em_data[scenario][location]
            
            # 是否是微观分析选中的城市？
            is_selected_city = (location in selected_cities) and (scenario == target_scenario_micro)
            
            for refrigerant in refrigerants:
                C = charge_amount[refrigerant]
                RFM = rfm_factors[refrigerant]
                
                # 遍历生产年份 2025-2045
                for year_p in years_prod:
                    # 获取该城市该年的销量（动态，来自Regression结果，按 SSP 情景区分）
                    volume = get_city_sales_units(location, year_p, _SCENARIO_TO_SSP.get(scenario, 'SSP245'))
                    
                    # --- 单机 LCTR 计算 ---
                    unit_lctr_pk = 0.0
                    
                    # 年泄漏率
                    if year_p < 2041:
                        ALR = 0.05 - 0.002 * (year_p - 2025)
                    else:
                        ALR = 0.02
                        
                    # 寿命循环
                    for life in range(lifespan):
                        year_u = year_p + life
                        
                        # 【关键修改】：Horizon 基于当前循环的 t_year
                        horizon = t_year - year_u
                        
                        # 只有在目标年之前发生的排放才会计入
                        # 且 AGTP 表只支持到 100 年 (假设)
                        if horizon < 1: continue 
                        if horizon > 100: horizon = 100 
                        
                        # 获取 AGTP
                        AGTP_CO2 = AGTP_data['CO2'][horizon]
                        if refrigerant == 'R410A':
                            AGTP_Ref = (AGTP_data['HFC-125'][horizon] + AGTP_data['HFC-32'][horizon]) / 2
                        elif refrigerant == 'R32':
                            AGTP_Ref = AGTP_data['HFC-32'][horizon]
                        elif refrigerant == 'R290':
                            AGTP_Ref = AGTP_data['HC-290'][horizon]
                            
                        # 获取 AEC
                        AEC = 0
                        temp_dist = temp_data.get((year_u, location), {})
                        if temp_dist:
                            AEC = calculate_aec_cooling_only(refrigerant, year_p, year_u, temp_dist)
                            
                        # 获取 EM
                        if year_u in city_em_series.index:
                            EM = city_em_series[year_u]
                        else:
                            EM = city_em_series.iloc[-1]
                            
                        # EOL
                        EOL = 0.97 - 0.01 * (year_u - 2025)
                        if EOL < 0: EOL = 0
                        
                        # 排放量计算 (kg)
                        if life == 0:
                            m_dir = C * ALR
                            m_ind = MM_sum + C * (1 + ALR) * RFM + AEC * EM
                        elif life == lifespan - 1:
                            m_dir = C * (ALR + EOL)
                            m_ind = RM_sum + C * ALR * RFM + AEC * EM + C * (1 - EOL) * RFD
                        else:
                            m_dir = C * ALR
                            m_ind = C * ALR * RFM + AEC * EM
                            
                        # 累加温升 (pK)
                        unit_lctr_pk += (m_dir * AGTP_Ref + m_ind * AGTP_CO2) * 1e12
                    
                    # --- 数据记录 ---
                    
                    # 1. 宏观累加 (Macro)
                    # 累加值 = 单机LCTR * 销量
                    macro_agg[scenario][refrigerant] += (unit_lctr_pk * volume)
                    
                    # 2. 微观记录 (Micro) - 仅限特定条件
                    if is_selected_city and (year_p in selected_prod_years):
                        results_micro.append({
                            'Target_Year': t_year,
                            'City': location,
                            'Prod_Year': year_p,
                            'Refrigerant': refrigerant,
                            'Unit_LCTR_pK': unit_lctr_pk
                        })
    
    # 每个目标年份结束后，记录宏观结果
    for s in scenarios:
        for r in refrigerants:
            results_macro.append({
                'Target_Year': t_year,
                'Scenario': s,
                'Refrigerant': r,
                'Total_National_LCTR_pK': macro_agg[s][r]
            })

# ==========================================
# 4. 数据导出
# ==========================================

file_name_target = with_climate_tag("LCTR_Target_Time_Analysis.xlsx")
print(f"计算完成，正在导出至 {file_name_target} ...")

with pd.ExcelWriter(file_name_target, engine='openpyxl') as writer:
    
    # 1. Micro Data (透视表格式)
    df_micro = pd.DataFrame(results_micro)
    # 透视：行=目标年，列=[城市, 生产年, 制冷剂]
    pivot_micro = df_micro.pivot(
        index='Target_Year',
        columns=['City', 'Prod_Year', 'Refrigerant'],
        values='Unit_LCTR_pK'
    )
    pivot_micro.to_excel(writer, sheet_name='Micro_City_Unit_Level')
    
    # 2. Macro Data (透视表格式)
    df_macro = pd.DataFrame(results_macro)
    # 透视：行=目标年，列=[情景, 制冷剂]
    pivot_macro = df_macro.pivot(
        index='Target_Year',
        columns=['Scenario', 'Refrigerant'],
        values='Total_National_LCTR_pK'
    )
    pivot_macro.to_excel(writer, sheet_name='Macro_National_Total')

print("完成！")

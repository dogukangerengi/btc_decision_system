# =============================================================================
# İNDİKATÖR KATEGORİLERİ VE TANIMLARI
# =============================================================================
# Amaç: 100+ teknik indikatörü kategorize etmek ve parametrelerini tanımlamak
#
# Kategoriler:
# 1. TREND       - Fiyat yönü ve trend gücü (MA'lar, ADX, Supertrend)
# 2. MOMENTUM    - Aşırı alım/satım ve momentum (RSI, Stochastic, MACD)
# 3. VOLATILITY  - Piyasa volatilitesi (ATR, Bollinger, Keltner)
# 4. VOLUME      - Hacim analizi (OBV, MFI, CMF)
# 5. COMPOSITE   - Birleşik sistemler (Ichimoku, Squeeze)
#
# İstatistiksel Not:
# - Aynı kategorideki indikatörler yüksek korelasyon gösterir (multicollinearity)
# - Feature selection'da kategori başına 1-2 indikatör seçmek optimal
# =============================================================================

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class IndicatorConfig:
    """Tek bir indikatörün yapılandırması."""
    name: str                              # pandas-ta fonksiyon adı
    display_name: str                       # Gösterim adı
    category: str                           # Kategori
    params: Dict[str, Any]                  # Parametreler
    output_columns: List[str]               # Çıktı kolonları
    description: str                        # Açıklama
    signal_type: str = "level"              # level, crossover, band


# =============================================================================
# TREND İNDİKATÖRLERİ (17 adet)
# =============================================================================

TREND_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig("sma", "SMA_20", "trend", {"length": 20}, ["SMA_20"],
                   "Simple MA 20 - Kısa vadeli trend filtresi", "crossover"),
    IndicatorConfig("sma", "SMA_50", "trend", {"length": 50}, ["SMA_50"],
                   "Simple MA 50 - Orta vadeli trend filtresi", "crossover"),
    IndicatorConfig("sma", "SMA_200", "trend", {"length": 200}, ["SMA_200"],
                   "Simple MA 200 - Uzun vadeli trend (bull/bear market)", "crossover"),
    IndicatorConfig("ema", "EMA_12", "trend", {"length": 12}, ["EMA_12"],
                   "EMA 12 - MACD fast component", "crossover"),
    IndicatorConfig("ema", "EMA_20", "trend", {"length": 20}, ["EMA_20"],
                   "EMA 20 - Kısa vadeli dinamik S/R", "crossover"),
    IndicatorConfig("ema", "EMA_26", "trend", {"length": 26}, ["EMA_26"],
                   "EMA 26 - MACD slow component", "crossover"),
    IndicatorConfig("ema", "EMA_50", "trend", {"length": 50}, ["EMA_50"],
                   "EMA 50 - Orta vadeli dinamik S/R", "crossover"),
    IndicatorConfig("wma", "WMA_20", "trend", {"length": 20}, ["WMA_20"],
                   "Weighted MA - Lineer ağırlıklı ortalama", "crossover"),
    IndicatorConfig("dema", "DEMA_20", "trend", {"length": 20}, ["DEMA_20"],
                   "Double EMA - Düşük lag", "crossover"),
    IndicatorConfig("tema", "TEMA_20", "trend", {"length": 20}, ["TEMA_20"],
                   "Triple EMA - Minimum lag", "crossover"),
    IndicatorConfig("hma", "HMA_20", "trend", {"length": 20}, ["HMA_20"],
                   "Hull MA - Çok düşük lag, overshooting riski", "crossover"),
    IndicatorConfig("kama", "KAMA", "trend", {"length": 10}, ["KAMA_10_2_30"],
                   "Kaufman Adaptive MA - Volatiliteye uyarlanır", "crossover"),
    IndicatorConfig("adx", "ADX", "trend", {"length": 14}, ["ADX_14", "DMP_14", "DMN_14"],
                   "ADX - Trend gücü (>25 güçlü trend, yön değil)", "level"),
    IndicatorConfig("aroon", "AROON", "trend", {"length": 25}, ["AROONU_25", "AROOND_25", "AROONOSC_25"],
                   "Aroon - Trend başlangıcı ve gücü", "crossover"),
    IndicatorConfig("psar", "PSAR", "trend", {"af0": 0.02, "af": 0.02, "max_af": 0.2},
                   ["PSARl_0.02_0.2", "PSARs_0.02_0.2"], "Parabolic SAR - Trailing stop", "level"),
    IndicatorConfig("supertrend", "SUPERTREND", "trend", {"length": 10, "multiplier": 3.0},
                   ["SUPERT_10_3.0", "SUPERTd_10_3.0"], "Supertrend - ATR bazlı trend", "level"),
    IndicatorConfig("vortex", "VORTEX", "trend", {"length": 14}, ["VTXP_14", "VTXN_14"],
                   "Vortex - Trend yönü crossover", "crossover"),
]


# =============================================================================
# MOMENTUM İNDİKATÖRLERİ (18 adet)
# =============================================================================

MOMENTUM_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig("rsi", "RSI_14", "momentum", {"length": 14}, ["RSI_14"],
                   "RSI 14 - Klasik momentum, 30-70 aşırı bölgeler", "level"),
    IndicatorConfig("rsi", "RSI_7", "momentum", {"length": 7}, ["RSI_7"],
                   "RSI 7 - Kısa periyot, daha hassas", "level"),
    IndicatorConfig("rsi", "RSI_21", "momentum", {"length": 21}, ["RSI_21"],
                   "RSI 21 - Uzun periyot, daha az sinyal", "level"),
    IndicatorConfig("stoch", "STOCH", "momentum", {"k": 14, "d": 3, "smooth_k": 3},
                   ["STOCHk_14_3_3", "STOCHd_14_3_3"], "Stochastic - Range-bound momentum", "crossover"),
    IndicatorConfig("stochrsi", "STOCHRSI", "momentum", {"length": 14, "rsi_length": 14, "k": 3, "d": 3},
                   ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"], "Stochastic RSI - Çok hassas", "crossover"),
    IndicatorConfig("willr", "WILLR", "momentum", {"length": 14}, ["WILLR_14"],
                   "Williams %R - Stochastic benzeri, -20/-80 ekstrem", "level"),
    IndicatorConfig("cci", "CCI", "momentum", {"length": 20}, ["CCI_20_0.015"],
                   "CCI - Mean deviation bazlı, +100/-100 ekstrem", "level"),
    IndicatorConfig("mom", "MOM", "momentum", {"length": 10}, ["MOM_10"],
                   "Momentum - Basit fiyat farkı", "level"),
    IndicatorConfig("roc", "ROC", "momentum", {"length": 10}, ["ROC_10"],
                   "Rate of Change - Yüzdesel değişim", "level"),
    IndicatorConfig("roc", "ROC_20", "momentum", {"length": 20}, ["ROC_20"],
                   "ROC 20 - Orta vadeli momentum", "level"),
    IndicatorConfig("ao", "AO", "momentum", {"fast": 5, "slow": 34}, ["AO_5_34"],
                   "Awesome Oscillator - Bill Williams", "level"),
    IndicatorConfig("macd", "MACD", "momentum", {"fast": 12, "slow": 26, "signal": 9},
                   ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"], "MACD - Trend momentum", "crossover"),
    IndicatorConfig("ppo", "PPO", "momentum", {"fast": 12, "slow": 26, "signal": 9},
                   ["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"], "PPO - Yüzdesel MACD", "crossover"),
    IndicatorConfig("tsi", "TSI", "momentum", {"fast": 13, "slow": 25, "signal": 13},
                   ["TSI_13_25_13", "TSIs_13_25_13"], "True Strength Index", "crossover"),
    IndicatorConfig("uo", "UO", "momentum", {"fast": 7, "medium": 14, "slow": 28}, ["UO_7_14_28"],
                   "Ultimate Oscillator - Multi-timeframe", "level"),
    IndicatorConfig("cmo", "CMO", "momentum", {"length": 14}, ["CMO_14"],
                   "Chande Momentum - RSI alternatifi", "level"),
    IndicatorConfig("fisher", "FISHER", "momentum", {"length": 9}, ["FISHERT_9_1", "FISHERTs_9_1"],
                   "Fisher Transform - Gaussian dönüşüm", "crossover"),
    IndicatorConfig("coppock", "COPPOCK", "momentum", {"length": 10, "fast": 11, "slow": 14},
                   ["COPC_11_14_10"], "Coppock Curve - Long-term momentum", "level"),
]


# =============================================================================
# VOLATİLİTE İNDİKATÖRLERİ (12 adet)
# =============================================================================

VOLATILITY_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig("atr", "ATR", "volatility", {"length": 14}, ["ATRr_14"],
                   "ATR - Volatilite ölçüsü, position sizing", "level"),
    IndicatorConfig("atr", "ATR_7", "volatility", {"length": 7}, ["ATRr_7"],
                   "ATR 7 - Kısa vadeli volatilite", "level"),
    IndicatorConfig("natr", "NATR", "volatility", {"length": 14}, ["NATR_14"],
                   "Normalized ATR - Yüzdesel volatilite", "level"),
    IndicatorConfig("bbands", "BBANDS", "volatility", {"length": 20, "std": 2.0},
                   ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"],
                   "Bollinger Bands - Volatilite bandları, squeeze", "band"),
    IndicatorConfig("bbands", "BBANDS_1STD", "volatility", {"length": 20, "std": 1.0},
                   ["BBL_20_1.0", "BBM_20_1.0", "BBU_20_1.0", "BBB_20_1.0", "BBP_20_1.0"],
                   "Bollinger 1 Std - Dar bantlar", "band"),
    IndicatorConfig("kc", "KC", "volatility", {"length": 20, "scalar": 1.5},
                   ["KCLe_20_1.5", "KCBe_20_1.5", "KCUe_20_1.5"],
                   "Keltner Channel - ATR bazlı bantlar", "band"),
    IndicatorConfig("donchian", "DONCHIAN", "volatility", {"lower_length": 20, "upper_length": 20},
                   ["DCL_20_20", "DCM_20_20", "DCU_20_20"], "Donchian Channel - Breakout", "band"),
    IndicatorConfig("massi", "MASSI", "volatility", {"fast": 9, "slow": 25}, ["MASSI_9_25"],
                   "Mass Index - Reversal bulge", "level"),
    IndicatorConfig("ui", "UI", "volatility", {"length": 14}, ["UI_14"],
                   "Ulcer Index - Downside risk", "level"),
    IndicatorConfig("accbands", "ACCBANDS", "volatility", {"length": 20},
                   ["ACCBL_20", "ACCBM_20", "ACCBU_20"], "Acceleration Bands", "band"),
    IndicatorConfig("rvi", "RVI", "volatility", {"length": 14}, ["RVI_14", "RVIs_14"],
                   "Relative Volatility Index", "crossover"),
    IndicatorConfig("true_range", "TR", "volatility", {}, ["TRUERANGE_1"],
                   "True Range - Tek bar volatilite", "level"),
]


# =============================================================================
# HACİM İNDİKATÖRLERİ (12 adet)
# =============================================================================

VOLUME_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig("obv", "OBV", "volume", {}, ["OBV"],
                   "On-Balance Volume - Kümülatif hacim", "level"),
    IndicatorConfig("ad", "AD", "volume", {}, ["AD"],
                   "Accumulation/Distribution - CLV weighted", "level"),
    IndicatorConfig("adosc", "ADOSC", "volume", {"fast": 3, "slow": 10}, ["ADOSC_3_10"],
                   "A/D Oscillator - A/D hattının MACD'si", "crossover"),
    IndicatorConfig("cmf", "CMF", "volume", {"length": 20}, ["CMF_20"],
                   "Chaikin Money Flow - Period bazlı A/D", "level"),
    IndicatorConfig("mfi", "MFI", "volume", {"length": 14}, ["MFI_14"],
                   "Money Flow Index - Volume-weighted RSI", "level"),
    IndicatorConfig("efi", "EFI", "volume", {"length": 13}, ["EFI_13"],
                   "Elder Force Index - Price change * volume", "level"),
    IndicatorConfig("nvi", "NVI", "volume", {"length": 1}, ["NVI_1"],
                   "Negative Volume Index - Down volume days", "level"),
    IndicatorConfig("pvi", "PVI", "volume", {"length": 1}, ["PVI_1"],
                   "Positive Volume Index - Up volume days", "level"),
    IndicatorConfig("pvol", "PVOL", "volume", {}, ["PVOL"],
                   "Price-Volume - Basit çarpım", "level"),
    IndicatorConfig("pvt", "PVT", "volume", {}, ["PVT"],
                   "Price Volume Trend - ROC weighted", "level"),
    IndicatorConfig("vwma", "VWMA", "volume", {"length": 20}, ["VWMA_20"],
                   "Volume Weighted MA", "crossover"),
]


# =============================================================================
# BİRLEŞİK İNDİKATÖRLER (5 adet)
# =============================================================================

COMPOSITE_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig("ichimoku", "ICHIMOKU", "composite", {"tenkan": 9, "kijun": 26, "senkou": 52},
                   ["ISA_9", "ISB_26", "ITS_9", "IKS_26", "ICS_26"],
                   "Ichimoku Cloud - Multi-component trend system", "crossover"),
    IndicatorConfig("squeeze", "SQUEEZE", "composite", 
                   {"bb_length": 20, "bb_std": 2.0, "kc_length": 20, "kc_scalar": 1.5},
                   ["SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO"],
                   "TTM Squeeze - Volatilite sıkışması", "level"),
    IndicatorConfig("qqe", "QQE", "composite", {"length": 14, "smooth": 5},
                   ["QQE_14_5_RSI", "QQEl_14_5", "QQEs_14_5"],
                   "Quantitative Qualitative Estimation", "crossover"),
]


# =============================================================================
# TÜM KATEGORİLER
# =============================================================================

ALL_INDICATORS: Dict[str, List[IndicatorConfig]] = {
    "trend": TREND_INDICATORS,
    "momentum": MOMENTUM_INDICATORS,
    "volatility": VOLATILITY_INDICATORS,
    "volume": VOLUME_INDICATORS,
    "composite": COMPOSITE_INDICATORS,
}


def get_all_indicators() -> List[IndicatorConfig]:
    """Tüm indikatörlerin düz listesini döndürür."""
    result = []
    for indicators in ALL_INDICATORS.values():
        result.extend(indicators)
    return result


def get_indicators_by_category(category: str) -> List[IndicatorConfig]:
    """Belirli kategorideki indikatörleri döndürür."""
    return ALL_INDICATORS.get(category, [])


def get_category_names() -> List[str]:
    """Kategori isimlerini döndürür."""
    return list(ALL_INDICATORS.keys())


def get_indicator_count() -> Dict[str, int]:
    """Her kategorideki indikatör sayısını döndürür."""
    return {cat: len(indicators) for cat, indicators in ALL_INDICATORS.items()}


# =============================================================================
# TEST KODU
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("İNDİKATÖR KATEGORİLERİ")
    print("=" * 60)
    
    counts = get_indicator_count()
    total = sum(counts.values())
    
    print(f"\nToplam: {total} indikatör\n")
    for cat, count in counts.items():
        print(f"  {cat.upper():<12}: {count:>2} indikatör")
    print("=" * 60)

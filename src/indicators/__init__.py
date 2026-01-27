# =============================================================================
# İNDİKATÖR MODÜLÜ (INDICATORS MODULE)
# =============================================================================
# Bu modül teknik indikatör hesaplama ve istatistiksel seçim sağlar.
#
# Bileşenler:
# - categories.py: 64+ indikatör tanımı ve parametreleri
# - calculator.py: pandas-ta ile indikatör hesaplama motoru
# - selector.py: İstatistiksel feature selection (IC, p-value, FDR)
#
# Kullanım:
# --------
# from indicators import IndicatorCalculator, IndicatorSelector
#
# calc = IndicatorCalculator()
# df = calc.calculate_all(ohlcv_df)
#
# selector = IndicatorSelector()
# scores = selector.evaluate_all_indicators(df)
# best = selector.select_best_indicators(scores)
# =============================================================================

from .categories import (
    IndicatorConfig,
    ALL_INDICATORS,
    get_all_indicators,
    get_indicators_by_category,
    get_category_names,
    get_indicator_count,
)

from .calculator import IndicatorCalculator

from .selector import IndicatorScore, IndicatorSelector


__all__ = [
    'IndicatorConfig',
    'ALL_INDICATORS',
    'get_all_indicators',
    'get_indicators_by_category',
    'get_category_names',
    'get_indicator_count',
    'IndicatorCalculator',
    'IndicatorScore',
    'IndicatorSelector',
]

__version__ = '1.0.0'

# =============================================================================
# BACKTEST MODÜLÜ
# =============================================================================
# Walk-forward validation, risk metrikleri ve timeframe seçimi.
#
# Kullanım:
# from backtest import DynamicBacktester, BacktestResult, TimeframeRanking
#
# backtester = DynamicBacktester()
# results = backtester.compare_timeframes(data_dict)
# ranking = backtester.select_best_timeframe(results)
# =============================================================================

from .backtester import (
    BacktestResult,
    TimeframeRanking,
    DynamicBacktester,
)

__all__ = [
    'BacktestResult',
    'TimeframeRanking',
    'DynamicBacktester',
]

__version__ = '1.0.0'

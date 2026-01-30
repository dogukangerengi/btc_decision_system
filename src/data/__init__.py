# =============================================================================
# VERİ MODÜLÜ (DATA MODULE)
# =============================================================================
# Veri çekme ve ön işleme modülleri.
#
# Kullanım:
# from data import DataFetcher, DataPreprocessor
#
# fetcher = DataFetcher(symbol="BTC/USDT")
# df = fetcher.fetch_ohlcv(timeframe="1h", limit=500)
#
# preprocessor = DataPreprocessor()
# df_clean = preprocessor.full_pipeline(df)
# =============================================================================

from .fetcher import DataFetcher
from .preprocessor import DataPreprocessor

__all__ = [
    'DataFetcher',
    'DataPreprocessor',
]

__version__ = '1.0.0'

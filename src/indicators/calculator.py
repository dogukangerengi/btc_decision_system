# =============================================================================
# İNDİKATÖR HESAPLAMA MOTORU (INDICATOR CALCULATOR)
# =============================================================================
# Amaç: pandas-ta kütüphanesi ile 100+ teknik indikatör hesaplamak
# 
# İstatistiksel Dikkat Noktaları:
# 1. Look-ahead bias: Tüm indikatörler sadece t ve öncesi veriyi kullanır
# 2. NaN handling: Rolling window başlangıcında NaN oluşur
# 3. Multicollinearity: Aynı kategorideki indikatörler yüksek korelasyonlu
# =============================================================================

import pandas as pd                          # Veri yapıları
import pandas_ta as ta                       # Teknik analiz kütüphanesi
import numpy as np                           # Sayısal hesaplamalar
from typing import Dict, List, Optional, Tuple, Any
import warnings                              # Uyarı yönetimi

# Kategori tanımlarını import et
from categories import (
    ALL_INDICATORS,
    IndicatorConfig,
    get_all_indicators,
    get_indicators_by_category,
    get_category_names,
)


class IndicatorCalculator:
    """
    Teknik indikatörleri hesaplayan sınıf.
    
    pandas-ta kütüphanesi üzerine wrapper. Her indikatör için:
    - Parametre validasyonu
    - NaN handling
    - Hata yönetimi
    - Çıktı standardizasyonu
    
    İstatistiksel Önem:
    ------------------
    - Tüm indikatörler SADECE geçmiş veriyi kullanır (look-ahead bias yok)
    - Rolling window hesaplamaları başlangıçta NaN üretir
    - Farklı uzunluktaki indikatörler farklı miktarda NaN üretir
    """
    
    def __init__(self, verbose: bool = True):
        """
        IndicatorCalculator başlatır.
        
        Parameters:
        ----------
        verbose : bool
            True ise hesaplama detayları yazdırılır
        """
        self.verbose = verbose
        
        # pandas-ta uyarılarını sustur
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    def calculate_single(
        self,
        df: pd.DataFrame,
        indicator: IndicatorConfig
    ) -> pd.DataFrame:
        """
        Tek bir indikatör hesaplar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (kolonlar: open, high, low, close, volume)
            
        indicator : IndicatorConfig
            İndikatör yapılandırması
        
        Returns:
        -------
        pd.DataFrame
            Hesaplanan indikatör kolonları
        """
        
        try:
            # pandas-ta fonksiyonunu çağır
            result = df.ta.__getattribute__(indicator.name)(**indicator.params)
            
            # Sonuç None olabilir
            if result is None:
                if self.verbose:
                    print(f"   ⚠ {indicator.display_name}: None döndü")
                return pd.DataFrame(index=df.index)
            
            # Bazı fonksiyonlar tuple döndürür (örn: ichimoku)
            # Bu durumda ilk elementi al (genellikle ana DataFrame)
            if isinstance(result, tuple):
                result = result[0] if len(result) > 0 else None
                if result is None:
                    return pd.DataFrame(index=df.index)
            
            # Series ise DataFrame'e çevir
            if isinstance(result, pd.Series):
                result = result.to_frame()
            
            # DataFrame değilse boş döndür
            if not isinstance(result, pd.DataFrame):
                if self.verbose:
                    print(f"   ⚠ {indicator.display_name}: Beklenmeyen tip {type(result)}")
                return pd.DataFrame(index=df.index)
            
            # Boş DataFrame kontrolü
            if result.empty:
                if self.verbose:
                    print(f"   ⚠ {indicator.display_name}: Boş sonuç")
                return pd.DataFrame(index=df.index)
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"   ✗ {indicator.display_name}: {str(e)[:50]}")
            return pd.DataFrame(index=df.index)
    
    def calculate_category(
        self,
        df: pd.DataFrame,
        category: str
    ) -> pd.DataFrame:
        """
        Bir kategorideki TÜM indikatörleri hesaplar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
            
        category : str
            Kategori: 'trend', 'momentum', 'volatility', 'volume', 'composite'
        
        Returns:
        -------
        pd.DataFrame
            Orijinal OHLCV + kategori indikatörleri
        """
        
        indicators = get_indicators_by_category(category)
        
        if not indicators:
            raise ValueError(f"Geçersiz kategori: {category}")
        
        if self.verbose:
            print(f"\n📊 {category.upper()} hesaplanıyor ({len(indicators)} indikatör)...")
        
        result_df = df.copy()
        success_count = 0
        
        for ind in indicators:
            ind_result = self.calculate_single(df, ind)
            
            if not ind_result.empty:
                for col in ind_result.columns:
                    if col not in result_df.columns:
                        result_df[col] = ind_result[col]
                success_count += 1
        
        if self.verbose:
            print(f"   ✓ {success_count}/{len(indicators)} başarılı")
        
        return result_df
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        TÜM kategorilerdeki indikatörleri hesaplar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
            
        categories : List[str], optional
            Hesaplanacak kategoriler. None ise tüm kategoriler.
        
        Returns:
        -------
        pd.DataFrame
            OHLCV + tüm indikatörler (100+ kolon)
        
        İstatistiksel Uyarı:
        -------------------
        - 100+ indikatör = yüksek boyutlu veri
        - Multiple testing correction gerekli (Bonferroni/FDR)
        - Feature selection zorunlu
        """
        
        if categories is None:
            categories = get_category_names()
        
        if self.verbose:
            print("=" * 60)
            print("TÜM İNDİKATÖRLER HESAPLANIYOR")
            print(f"Kategoriler: {categories}")
            print(f"Veri: {len(df)} bar")
            print("=" * 60)
        
        result_df = df.copy()
        
        for category in categories:
            category_df = self.calculate_category(df, category)
            
            # Yeni kolonları ekle
            new_cols = [c for c in category_df.columns if c not in result_df.columns]
            for col in new_cols:
                result_df[col] = category_df[col]
        
        # NaN istatistikleri
        indicator_cols = [c for c in result_df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        nan_pct = result_df[indicator_cols].isnull().mean() * 100
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("HESAPLAMA TAMAMLANDI")
            print(f"Toplam kolon: {len(result_df.columns)}")
            print(f"İndikatör sayısı: {len(indicator_cols)}")
            print(f"Ortalama NaN oranı: {nan_pct.mean():.1f}%")
            print("=" * 60)
        
        return result_df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temel fiyat özellikleri ekler (pandas-ta dışı).
        
        Eklenen özellikler:
        - log_return: Log getiri (toplamsal, normal dağılıma yakın)
        - simple_return: Basit yüzdesel getiri
        - range: High - Low (True Range değil, sadece bar range)
        - body: Close - Open (mum gövdesi)
        - body_pct: Gövde / Open * 100
        - upper_wick, lower_wick: Fitil uzunlukları
        - gap: Bugün Open - Dün Close
        """
        
        result_df = df.copy()
        
        # Log return: ln(P_t / P_{t-1})
        # İstatistiksel avantaj: toplamsal, yaklaşık normal dağılım
        result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        # Simple return: (P_t - P_{t-1}) / P_{t-1}
        result_df['simple_return'] = result_df['close'].pct_change()
        
        # Bar özellikleri
        result_df['range'] = result_df['high'] - result_df['low']
        result_df['body'] = result_df['close'] - result_df['open']
        result_df['body_pct'] = (result_df['body'] / result_df['open']) * 100
        
        # Fitiller
        result_df['upper_wick'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
        result_df['lower_wick'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        
        # Gap
        result_df['gap'] = result_df['open'] - result_df['close'].shift(1)
        result_df['gap_pct'] = (result_df['gap'] / result_df['close'].shift(1)) * 100
        
        # Volume özellikleri
        result_df['volume_sma_20'] = result_df['volume'].rolling(20).mean()
        result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma_20']
        
        # Close pozisyonu (0=Low'da, 1=High'da)
        result_df['hl_position'] = (result_df['close'] - result_df['low']) / (result_df['range'] + 1e-10)
        
        return result_df
    
    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Rolling istatistiksel özellikler ekler.
        
        Her window için:
        - ret_mean: Ortalama getiri
        - ret_std: Getiri volatilitesi (σ)
        - ret_skew: Asimetri (fat-tail left/right)
        - ret_kurt: Basıklık (fat-tail severity)
        - zscore: (Close - MA) / Std
        
        İstatistiksel Açıklama:
        ----------------------
        - Skewness < 0: Sol kuyruk uzun (crash riski)
        - Kurtosis > 3: Kalın kuyruk (extreme event riski)
        - Z-score: Mean-reversion sinyali için kullanışlı
        """
        
        result_df = df.copy()
        
        # Log return yoksa hesapla
        if 'log_return' not in result_df.columns:
            result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        returns = result_df['log_return']
        
        for w in windows:
            prefix = f"roll{w}_"
            
            # Getiri istatistikleri
            result_df[f'{prefix}ret_mean'] = returns.rolling(w).mean()
            result_df[f'{prefix}ret_std'] = returns.rolling(w).std()
            result_df[f'{prefix}ret_skew'] = returns.rolling(w).skew()
            result_df[f'{prefix}ret_kurt'] = returns.rolling(w).kurt()
            
            # Z-score: Fiyatın rolling dağılımdaki pozisyonu
            roll_mean = result_df['close'].rolling(w).mean()
            roll_std = result_df['close'].rolling(w).std()
            result_df[f'{prefix}zscore'] = (result_df['close'] - roll_mean) / (roll_std + 1e-10)
            
            # Min/Max
            result_df[f'{prefix}close_min'] = result_df['close'].rolling(w).min()
            result_df[f'{prefix}close_max'] = result_df['close'].rolling(w).max()
            
            # Percentile rank
            result_df[f'{prefix}pct_rank'] = result_df['close'].rolling(w).apply(
                lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5,
                raw=False
            )
        
        return result_df
    
    def add_forward_returns(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        İleri getiriler ekler (TARGET değişkenler).
        
        UYARI: Bu kolonlar SADECE backtest/train için kullanılmalı!
        Canlı sistemde bu bilgi mevcut değil (look-ahead bias).
        
        Parameters:
        ----------
        periods : List[int]
            İleri periyotlar (1=sonraki bar, 5=5 bar sonra, vb.)
        
        Returns:
        -------
        pd.DataFrame
            fwd_ret_N kolonları eklendi
        """
        
        result_df = df.copy()
        
        for p in periods:
            # İleri log getiri
            result_df[f'fwd_ret_{p}'] = np.log(
                result_df['close'].shift(-p) / result_df['close']
            )
            
            # İleri yön (binary: 1=up, 0=down)
            result_df[f'fwd_dir_{p}'] = (result_df[f'fwd_ret_{p}'] > 0).astype(int)
        
        return result_df
    
    def get_clean_data(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
        drop_forward: bool = True
    ) -> pd.DataFrame:
        """
        Analiz için temiz veri döndürür.
        
        Parameters:
        ----------
        dropna : bool
            True ise NaN içeren satırları kaldır
            
        drop_forward : bool
            True ise forward return kolonlarını kaldır (canlı sistem için)
        
        Returns:
        -------
        pd.DataFrame
            Temizlenmiş DataFrame
        """
        
        result_df = df.copy()
        
        # Forward return kolonlarını kaldır
        if drop_forward:
            fwd_cols = [c for c in result_df.columns if c.startswith('fwd_')]
            result_df = result_df.drop(columns=fwd_cols, errors='ignore')
        
        # NaN'ları kaldır
        if dropna:
            before_len = len(result_df)
            result_df = result_df.dropna()
            after_len = len(result_df)
            
            if self.verbose:
                print(f"NaN temizleme: {before_len} → {after_len} satır ({before_len - after_len} silindi)")
        
        return result_df


# =============================================================================
# TEST KODU
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("İNDİKATÖR CALCULATOR TEST")
    print("=" * 70)
    
    # Veri çekme
    import sys
    sys.path.insert(0, '../data')
    
    try:
        from fetcher import DataFetcher
        
        fetcher = DataFetcher(symbol="BTC/USDT")
        df = fetcher.fetch_ohlcv(timeframe="1h", limit=500)
        
        print(f"\nTest verisi: {len(df)} bar")
        
        # Calculator oluştur
        calc = IndicatorCalculator(verbose=True)
        
        # 1. Tek kategori
        print("\n[TEST 1] Tek kategori (momentum):")
        df_mom = calc.calculate_category(df, "momentum")
        mom_cols = [c for c in df_mom.columns if c not in df.columns]
        print(f"   Eklenen kolonlar: {len(mom_cols)}")
        
        # 2. Tüm kategoriler
        print("\n[TEST 2] Tüm kategoriler:")
        df_all = calc.calculate_all(df)
        
        # 3. Price features
        print("\n[TEST 3] Price features:")
        df_price = calc.add_price_features(df)
        price_cols = [c for c in df_price.columns if c not in df.columns]
        print(f"   Eklenen: {price_cols}")
        
        # 4. Rolling stats
        print("\n[TEST 4] Rolling stats:")
        df_roll = calc.add_rolling_stats(df, windows=[10, 20])
        roll_cols = [c for c in df_roll.columns if 'roll' in c]
        print(f"   Eklenen: {len(roll_cols)} kolon")
        
        # 5. Forward returns
        print("\n[TEST 5] Forward returns:")
        df_fwd = calc.add_forward_returns(df, periods=[1, 5])
        fwd_cols = [c for c in df_fwd.columns if 'fwd' in c]
        print(f"   Eklenen: {fwd_cols}")
        
        # 6. Sample output
        print("\n[TEST 6] Örnek çıktı (son 3 bar, seçili kolonlar):")
        sample_cols = ['close', 'RSI_14', 'MACD_12_26_9', 'ATRr_14', 'OBV']
        available = [c for c in sample_cols if c in df_all.columns]
        print(df_all[available].tail(3))
        
        print("\n" + "=" * 70)
        print("TÜM TESTLER TAMAMLANDI")
        print("=" * 70)
        
    except ImportError as e:
        print(f"Import hatası: {e}")
        print("Fetcher olmadan test yapılamıyor.")

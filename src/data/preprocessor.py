# =============================================================================
# VERİ ÖN İŞLEME MODÜLÜ (DATA PREPROCESSOR)
# =============================================================================
# Amaç: Ham OHLCV verisini analiz için hazırlamak
# - Eksik veri tespiti ve doldurma (imputation)
# - Outlier tespiti ve işleme
# - Return hesaplama
# - Veri normalizasyonu
# 
# İstatistiksel Not: Veri ön işleme, tüm downstream analizlerin kalitesini
# doğrudan etkiler. Özellikle look-ahead bias'a dikkat edilmelidir.
# =============================================================================

import pandas as pd                      # Veri manipülasyonu
import numpy as np                       # Sayısal hesaplamalar
from typing import Optional, Tuple, List, Dict  # Tip belirteçleri
from scipy import stats                  # İstatistiksel fonksiyonlar


class DataPreprocessor:
    """
    OHLCV verisini ön işleme tabi tutan sınıf.
    
    İstatistiksel Önem:
    ------------------
    1. Missing data handling: Bias önleme için kritik
    2. Outlier treatment: Robust istatistikler için gerekli
    3. Return calculation: Log vs simple return seçimi önemli
    4. Stationarity: Çoğu model durağan seri gerektirir
    """
    
    def __init__(self):
        """
        Preprocessor sınıfını başlatır.
        Stateless tasarım: Her method bağımsız çalışır.
        """
        pass
    
    # =========================================================================
    # EKSİK VERİ İŞLEME
    # =========================================================================
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",            # Doldurma yöntemi
        max_gap: int = 5                  # Maksimum ardışık eksik veri
    ) -> pd.DataFrame:
        """
        Eksik verileri tespit eder ve doldurur.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            OHLCV DataFrame
            
        method : str
            Doldurma yöntemi:
            - "ffill": Forward fill (önceki değerle doldur)
            - "bfill": Backward fill (sonraki değerle doldur)
            - "interpolate": Linear interpolasyon
            - "drop": Eksik satırları sil
            
        max_gap : int
            Ardışık eksik veri sayısı bu değeri aşarsa doldurma yapılmaz
            (Çok uzun gap'ler genellikle market closure'ı gösterir)
        
        Döndürür:
        --------
        pd.DataFrame
            Eksik değerleri işlenmiş DataFrame
        
        İstatistiksel Not:
        -----------------
        - Forward fill: Look-ahead bias riski YOK (önceki veri kullanılır)
        - Backward fill: Look-ahead bias riski VAR (gelecek veri kullanılır)
        - Interpolation: Kısmi look-ahead bias riski
        
        Backtest için SADECE forward fill önerilir.
        """
        
        df_clean = df.copy()              # Orijinal veriyi koru
        
        # Eksik değer istatistikleri
        missing_before = df_clean.isnull().sum().sum()
        
        if missing_before == 0:
            print("✓ Eksik değer bulunamadı")
            return df_clean
        
        print(f"⚠ {missing_before} eksik değer tespit edildi")
        
        # Ardışık eksik değer kontrolü
        # max_gap'ten uzun boşlukları işaretle
        for col in df_clean.columns:
            # Ardışık NaN gruplarını bul
            mask = df_clean[col].isnull()
            # Grup numaralarını ata
            groups = (mask != mask.shift()).cumsum()
            # Her grubun uzunluğunu hesapla
            group_sizes = mask.groupby(groups).transform('sum')
            # Çok uzun gap'leri işaretle (doldurmayacağız)
            long_gaps = (group_sizes > max_gap) & mask
            
            if long_gaps.any():
                print(f"  ⚠ {col}: {long_gaps.sum()} değer {max_gap}+ uzunluğunda gap içinde (doldurulmayacak)")
        
        # Doldurma yöntemi uygula
        if method == "ffill":
            # Forward fill: Önceki geçerli değerle doldur
            # limit parametresi max_gap kadar doldurmayı sınırlar
            df_clean = df_clean.ffill(limit=max_gap)
            
        elif method == "bfill":
            # Backward fill: Sonraki geçerli değerle doldur
            # DİKKAT: Look-ahead bias riski!
            print("  ⚠ UYARI: bfill look-ahead bias riski taşır!")
            df_clean = df_clean.bfill(limit=max_gap)
            
        elif method == "interpolate":
            # Linear interpolasyon
            # DİKKAT: Kısmi look-ahead bias riski
            print("  ⚠ UYARI: interpolate kısmi look-ahead bias riski taşır!")
            df_clean = df_clean.interpolate(method='linear', limit=max_gap)
            
        elif method == "drop":
            # Eksik satırları tamamen sil
            df_clean = df_clean.dropna()
            
        else:
            raise ValueError(f"Geçersiz method: {method}")
        
        # Sonuç istatistikleri
        missing_after = df_clean.isnull().sum().sum()
        print(f"✓ {missing_before - missing_after} eksik değer dolduruldu")
        print(f"  Kalan eksik: {missing_after}")
        
        return df_clean
    
    # =========================================================================
    # RETURN HESAPLAMA
    # =========================================================================
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        method: str = "log",              # Return hesaplama yöntemi
        periods: int = 1                  # Kaç periyot sonrası return
    ) -> pd.DataFrame:
        """
        Fiyat verisinden return (getiri) hesaplar.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            OHLCV DataFrame (en az 'close' kolonu olmalı)
            
        method : str
            Return hesaplama yöntemi:
            - "log": Logaritmik return (ln(P_t / P_{t-1}))
            - "simple": Basit return ((P_t - P_{t-1}) / P_{t-1})
            
        periods : int
            Forward return periyodu
            1 = bir sonraki bar'ın getirisi
            
        Döndürür:
        --------
        pd.DataFrame
            Orijinal kolonlar + 'returns' kolonu
        
        İstatistiksel Not:
        -----------------
        Log return avantajları:
        1. Toplamsal: r_total = r_1 + r_2 + ... + r_n
        2. Negatif simetri: -100% ile sınırlı değil
        3. Normal dağılıma daha yakın (genellikle)
        4. Volatilite hesaplamaları için daha uygun
        
        Simple return avantajları:
        1. Yorumlaması kolay (%5 return = %5 kazanç)
        2. Portföy return'ü doğrudan hesaplanabilir
        
        Backtest için genellikle log return tercih edilir.
        """
        
        df_with_returns = df.copy()
        
        if method == "log":
            # Log return: ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})
            # np.log kullanıyoruz (doğal logaritma)
            df_with_returns['returns'] = np.log(
                df_with_returns['close'] / df_with_returns['close'].shift(periods)
            )
            
        elif method == "simple":
            # Simple return: (P_t - P_{t-1}) / P_{t-1}
            # pct_change() bunu otomatik hesaplar
            df_with_returns['returns'] = df_with_returns['close'].pct_change(periods)
            
        else:
            raise ValueError(f"Geçersiz method: {method}. 'log' veya 'simple' kullanın.")
        
        # Forward return (gelecek getiri - sinyal değerlendirmesi için)
        # DİKKAT: Bu kolon backtest'te kullanılmalı, look-ahead bias'a dikkat!
        df_with_returns['forward_returns'] = df_with_returns['returns'].shift(-periods)
        
        return df_with_returns
    
    # =========================================================================
    # OUTLIER TESPİTİ VE İŞLEME
    # =========================================================================
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str = "returns",          # Outlier tespit edilecek kolon
        method: str = "zscore",           # Tespit yöntemi
        threshold: float = 3.0            # Eşik değer
    ) -> pd.Series:
        """
        Outlier'ları (aykırı değerler) tespit eder.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Veri DataFrame'i
            
        column : str
            Outlier tespiti yapılacak kolon
            
        method : str
            Tespit yöntemi:
            - "zscore": Z-skor (standart sapma bazlı)
            - "iqr": Interquartile Range (Q1-Q3 bazlı)
            - "mad": Median Absolute Deviation (robust)
            
        threshold : float
            Outlier eşiği
            - zscore için: genellikle 3.0 (3 sigma kuralı)
            - iqr için: genellikle 1.5 (Tukey's fence)
            - mad için: genellikle 3.5
        
        Döndürür:
        --------
        pd.Series
            Boolean mask (True = outlier)
        
        İstatistiksel Not:
        -----------------
        - Z-score: Normal dağılım varsayar, outlier'lara duyarlı
        - IQR: Dağılım agnostik, orta derecede robust
        - MAD: En robust yöntem, fat-tailed dağılımlar için ideal
        
        Finansal veri genellikle fat-tailed olduğu için MAD önerilir.
        """
        
        data = df[column].dropna()        # NaN'ları kaldır
        
        if method == "zscore":
            # Z-score: (x - mean) / std
            # |z| > threshold ise outlier
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = pd.Series(z_scores > threshold, index=data.index)
            
        elif method == "iqr":
            # IQR yöntemi: Q1 - threshold*IQR ile Q3 + threshold*IQR dışı
            Q1 = data.quantile(0.25)      # 1. çeyreklik (25. percentile)
            Q3 = data.quantile(0.75)      # 3. çeyreklik (75. percentile)
            IQR = Q3 - Q1                 # Interquartile range
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == "mad":
            # MAD: Median Absolute Deviation
            # Robust alternatif: median bazlı, outlier'lara dayanıklı
            median = data.median()
            # Her noktanın mediandan farkının mutlak değeri
            mad = np.median(np.abs(data - median))
            # MAD'ı standart sapma ölçeğine çevirmek için sabit
            # Normal dağılım için: MAD * 1.4826 ≈ std
            mad_scaled = mad * 1.4826
            
            # Modified z-score
            modified_z = np.abs((data - median) / mad_scaled)
            outlier_mask = pd.Series(modified_z > threshold, index=data.index)
            
        else:
            raise ValueError(f"Geçersiz method: {method}")
        
        # Tam index'e genişlet (NaN'lar False olarak)
        full_mask = pd.Series(False, index=df.index)
        full_mask[outlier_mask.index] = outlier_mask
        
        print(f"📊 Outlier tespiti ({method}, threshold={threshold}):")
        print(f"   Toplam outlier: {full_mask.sum()} / {len(df)} ({100*full_mask.mean():.2f}%)")
        
        return full_mask
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str = "returns",
        method: str = "winsorize",        # İşleme yöntemi
        limits: Tuple[float, float] = (0.01, 0.01)  # Winsorize limitleri
    ) -> pd.DataFrame:
        """
        Outlier'ları işler (temizler veya dönüştürür).
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Veri DataFrame'i
            
        column : str
            İşlenecek kolon
            
        method : str
            İşleme yöntemi:
            - "winsorize": Uç değerleri percentile değerlerine çek
            - "clip": Belirli min-max aralığına sınırla
            - "remove": Outlier satırları sil
            - "nan": Outlier'ları NaN yap
            
        limits : Tuple[float, float]
            Winsorize için alt ve üst percentile limitleri
            (0.01, 0.01) = %1 alt ve üst uçlar
        
        Döndürür:
        --------
        pd.DataFrame
            Outlier'ları işlenmiş DataFrame
        
        İstatistiksel Not:
        -----------------
        Winsorization tercih edilir çünkü:
        1. Veri kaybı yok (remove'un aksine)
        2. Uç değerlerin etkisi azaltılır
        3. Dağılımın genel yapısı korunur
        
        Dikkat: Gerçek piyasa crash'leri de "outlier" görünebilir.
        Bunları tamamen silmek unrealistic backtest'e yol açar.
        """
        
        df_processed = df.copy()
        
        if method == "winsorize":
            # Winsorization: Uç değerleri percentile değerlerine çek
            # scipy.stats.mstats.winsorize kullanılabilir ama manuel daha kontrollü
            lower_percentile = limits[0]
            upper_percentile = 1 - limits[1]
            
            lower_val = df_processed[column].quantile(lower_percentile)
            upper_val = df_processed[column].quantile(upper_percentile)
            
            # Clip: lower_val ile upper_val arasına sınırla
            df_processed[column] = df_processed[column].clip(lower_val, upper_val)
            
            print(f"✓ Winsorization uygulandı: [{lower_val:.4f}, {upper_val:.4f}]")
            
        elif method == "clip":
            # Manuel clip: Sabit değerlerle sınırla
            lower_val, upper_val = limits
            df_processed[column] = df_processed[column].clip(lower_val, upper_val)
            
        elif method == "remove":
            # Outlier'ları tespit et ve sil
            outlier_mask = self.detect_outliers(df, column)
            df_processed = df_processed[~outlier_mask]
            print(f"✓ {outlier_mask.sum()} outlier satır silindi")
            
        elif method == "nan":
            # Outlier'ları NaN yap
            outlier_mask = self.detect_outliers(df, column)
            df_processed.loc[outlier_mask, column] = np.nan
            print(f"✓ {outlier_mask.sum()} outlier NaN yapıldı")
            
        else:
            raise ValueError(f"Geçersiz method: {method}")
        
        return df_processed
    
    # =========================================================================
    # VOLATİLİTE HESAPLAMA
    # =========================================================================
    
    def calculate_volatility(
        self,
        df: pd.DataFrame,
        window: int = 20,                 # Rolling window boyutu
        method: str = "standard",         # Volatilite hesaplama yöntemi
        annualize: bool = True,           # Yıllıklaştır
        periods_per_year: int = 252 * 24  # Saatlik veri için
    ) -> pd.DataFrame:
        """
        Rolling volatilite hesaplar.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Returns kolonu içeren DataFrame
            
        window : int
            Rolling window boyutu (bar sayısı)
            
        method : str
            Volatilite hesaplama yöntemi:
            - "standard": Standart sapma
            - "parkinson": High-Low bazlı (daha verimli)
            - "garman_klass": OHLC bazlı (en verimli)
            
        annualize : bool
            True ise yıllıklaştırılmış volatilite döndür
            
        periods_per_year : int
            Bir yıldaki periyot sayısı
            1h veri için: 252 * 24 = 6048
            1d veri için: 252
        
        Döndürür:
        --------
        pd.DataFrame
            Orijinal kolonlar + 'volatility' kolonu
        
        İstatistiksel Not:
        -----------------
        Parkinson ve Garman-Klass volatility estimator'ları
        sadece close fiyatına dayanan standart sapmadan daha
        verimlidir (efficient). OHLC bilgisini kullanırlar.
        
        Verimliliği karşılaştırma:
        - Standard: baseline
        - Parkinson: ~5x daha efficient
        - Garman-Klass: ~8x daha efficient
        """
        
        df_vol = df.copy()
        
        if method == "standard":
            # Standart sapma: sqrt(variance of returns)
            # Rolling pencere üzerinde hesapla
            rolling_vol = df_vol['returns'].rolling(window=window).std()
            
        elif method == "parkinson":
            # Parkinson volatility: High-Low range bazlı
            # Formül: sqrt(1/(4*ln(2)) * (ln(H/L))^2)
            # Daha verimli çünkü intrabar bilgi kullanır
            log_hl = np.log(df_vol['high'] / df_vol['low'])
            parkinson_factor = 1 / (4 * np.log(2))
            rolling_vol = np.sqrt(
                parkinson_factor * (log_hl ** 2).rolling(window=window).mean()
            )
            
        elif method == "garman_klass":
            # Garman-Klass volatility: OHLC bazlı
            # En verimli estimator (drift-adjusted)
            log_hl = np.log(df_vol['high'] / df_vol['low'])
            log_co = np.log(df_vol['close'] / df_vol['open'])
            
            # Garman-Klass formülü
            gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
            rolling_vol = np.sqrt(gk.rolling(window=window).mean())
            
        else:
            raise ValueError(f"Geçersiz method: {method}")
        
        # Yıllıklaştırma
        if annualize:
            # Volatilite sqrt(T) ile ölçeklenir
            rolling_vol = rolling_vol * np.sqrt(periods_per_year)
        
        df_vol['volatility'] = rolling_vol
        
        return df_vol
    
    # =========================================================================
    # PIPELINE: TÜM ÖN İŞLEME ADIMLARI
    # =========================================================================
    
    def full_pipeline(
        self,
        df: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Tüm ön işleme adımlarını sırasıyla uygular.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Ham OHLCV DataFrame
            
        config : Dict, optional
            Özel yapılandırma. Varsayılan değerler kullanılır.
        
        Döndürür:
        --------
        pd.DataFrame
            Tam işlenmiş, analize hazır DataFrame
        
        Pipeline Adımları:
        -----------------
        1. Missing value handling (ffill)
        2. Return calculation (log returns)
        3. Outlier winsorization
        4. Volatility calculation (Garman-Klass)
        5. Forward return ekleme (sinyal değerlendirmesi için)
        """
        
        # Varsayılan konfigürasyon
        default_config = {
            'missing_method': 'ffill',
            'missing_max_gap': 5,
            'return_method': 'log',
            'return_periods': 1,
            'outlier_method': 'winsorize',
            'outlier_limits': (0.01, 0.01),
            'volatility_window': 20,
            'volatility_method': 'garman_klass',
        }
        
        # Config güncelle
        if config:
            default_config.update(config)
        cfg = default_config
        
        print("=" * 50)
        print("VERİ ÖN İŞLEME PIPELINE")
        print("=" * 50)
        
        # Adım 1: Missing values
        print("\n[1/4] Missing value işleme...")
        df_processed = self.handle_missing_values(
            df,
            method=cfg['missing_method'],
            max_gap=cfg['missing_max_gap']
        )
        
        # Adım 2: Returns
        print("\n[2/4] Return hesaplama...")
        df_processed = self.calculate_returns(
            df_processed,
            method=cfg['return_method'],
            periods=cfg['return_periods']
        )
        
        # Adım 3: Outliers
        print("\n[3/4] Outlier işleme...")
        df_processed = self.handle_outliers(
            df_processed,
            column='returns',
            method=cfg['outlier_method'],
            limits=cfg['outlier_limits']
        )
        
        # Adım 4: Volatility
        print("\n[4/4] Volatilite hesaplama...")
        df_processed = self.calculate_volatility(
            df_processed,
            window=cfg['volatility_window'],
            method=cfg['volatility_method']
        )
        
        # İlk birkaç NaN satırı kaldır (rolling hesaplamalardan)
        df_processed = df_processed.dropna()
        
        print("\n" + "=" * 50)
        print(f"✓ Pipeline tamamlandı: {len(df_processed)} satır hazır")
        print("=" * 50)
        
        return df_processed


# =============================================================================
# MODÜL TEST KODU
# =============================================================================

if __name__ == "__main__":
    
    # Test için örnek veri oluştur
    print("=" * 60)
    print("DATA PREPROCESSOR TEST")
    print("=" * 60)
    
    # Önce gerçek veri çekelim (fetcher modülünden)
    from fetcher import DataFetcher
    
    # Veri çek
    fetcher = DataFetcher(exchange_id="binance", symbol="BTC/USDT")
    df_raw = fetcher.fetch_ohlcv(timeframe="1h", limit=200)
    
    print(f"\nHam veri boyutu: {len(df_raw)}")
    print(df_raw.head())
    
    # Preprocessor test
    preprocessor = DataPreprocessor()
    
    # Full pipeline uygula
    df_processed = preprocessor.full_pipeline(df_raw)
    
    print(f"\nİşlenmiş veri boyutu: {len(df_processed)}")
    print("\nİşlenmiş veri kolonları:")
    print(df_processed.columns.tolist())
    print("\nSon 5 satır:")
    print(df_processed.tail())
    
    print("\n" + "=" * 60)
    print("TÜM TESTLER TAMAMLANDI")
    print("=" * 60)

# =============================================================================
# İSTATİSTİKSEL İNDİKATÖR SEÇİM MODÜLÜ (INDICATOR SELECTOR)
# =============================================================================
# Amaç: 100+ indikatör arasından istatistiksel olarak anlamlı olanları seçmek
#
# Metodoloji:
# 1. Information Coefficient (IC) - Spearman korelasyonu, rank-based, robust
# 2. IC t-testi - IC'nin sıfırdan farklı olup olmadığı
# 3. IC Stability - Rolling IC'nin tutarlılığı (IC_IR = mean(IC)/std(IC))
# 4. Bonferroni Correction - Multiple testing düzeltmesi (FWER control)
# 5. Benjamini-Hochberg FDR - Daha güçlü alternatif (FDR control)
#
# İstatistiksel Temel:
# - IC = Spearman(indicator_t, forward_return_{t+n})
# - IC > 0: İndikatör ve gelecek getiri aynı yönde
# - |IC| > 0.02 genellikle ekonomik olarak anlamlı kabul edilir
# - p < 0.05/N (Bonferroni) veya FDR-adjusted p < 0.05
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from categories import (
    ALL_INDICATORS,
    get_all_indicators,
    get_indicators_by_category,
    get_category_names,
)


@dataclass
class IndicatorScore:
    """
    Bir indikatörün istatistiksel değerlendirme sonucu.
    
    Attributes:
    ----------
    name : str
        İndikatör kolon adı
    category : str
        Kategori (trend, momentum, vb.)
    ic_mean : float
        Ortalama Information Coefficient (Spearman)
    ic_std : float
        IC'nin standart sapması
    ic_ir : float
        IC Information Ratio = ic_mean / ic_std (stability ölçüsü)
    ic_tstat : float
        IC t-istatistiği (H0: IC=0)
    p_value : float
        İki kuyruklu p-değeri
    p_value_adjusted : float
        Multiple testing düzeltmeli p-değeri
    is_significant : bool
        Düzeltilmiş p < 0.05 mi?
    n_observations : int
        Geçerli gözlem sayısı
    direction : str
        Sinyal yönü: 'bullish' (IC > 0) veya 'bearish' (IC < 0)
    """
    name: str
    category: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_tstat: float
    p_value: float
    p_value_adjusted: float
    is_significant: bool
    n_observations: int
    direction: str


class IndicatorSelector:
    """
    İstatistiksel olarak anlamlı indikatörleri seçen sınıf.
    
    Metodoloji:
    ----------
    1. Her indikatör için IC (Information Coefficient) hesapla
    2. IC'nin istatistiksel anlamlılığını test et (t-test)
    3. Multiple testing correction uygula (Bonferroni veya FDR)
    4. Her kategoriden en iyi 1-2 indikatör seç
    
    Neden IC (Information Coefficient)?
    ----------------------------------
    - Spearman korelasyonu kullanır → outlier'lara robust
    - Rank-based → non-linear ilişkileri yakalar
    - -1 ile +1 arası normalize → karşılaştırılabilir
    - Finans endüstri standardı (Barra, MSCI faktör modelleri)
    
    Neden Multiple Testing Correction?
    ---------------------------------
    - 100 indikatör test ediyoruz
    - α=0.05 ile 100 test → ~5 yanlış pozitif beklenir
    - Bonferroni: α_adj = 0.05/100 = 0.0005 (çok muhafazakar)
    - FDR (Benjamini-Hochberg): Daha güçlü, false discovery rate kontrol
    """
    
    # Minimum kabul edilebilir değerler
    MIN_IC = 0.02                    # Ekonomik anlamlılık eşiği
    MIN_IC_IR = 0.3                  # IC stability eşiği
    MIN_OBSERVATIONS = 100           # Minimum gözlem sayısı
    
    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = "fdr",    # 'bonferroni' veya 'fdr'
        verbose: bool = True
    ):
        """
        IndicatorSelector başlatır.
        
        Parameters:
        ----------
        alpha : float
            Anlamlılık düzeyi (default: 0.05)
            
        correction_method : str
            Multiple testing correction yöntemi:
            - 'bonferroni': Çok muhafazakar, FWER kontrol
            - 'fdr': Benjamini-Hochberg, FDR kontrol (önerilen)
            
        verbose : bool
            Detaylı çıktı göster
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.verbose = verbose
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def calculate_ic(
        self,
        indicator: pd.Series,
        forward_return: pd.Series
    ) -> Tuple[float, float]:
        """
        Tek bir indikatör için Information Coefficient hesaplar.
        
        IC = Spearman(indicator_t, return_{t+n})
        
        Parameters:
        ----------
        indicator : pd.Series
            İndikatör değerleri (t zamanında)
            
        forward_return : pd.Series
            İleri getiriler (t+n zamanında)
        
        Returns:
        -------
        Tuple[float, float]
            (ic_value, p_value)
        
        Neden Spearman?
        --------------
        - Pearson: Lineer ilişki varsayar, outlier'lara hassas
        - Spearman: Monotonic ilişki, rank-based, robust
        - Finans verisi: Fat-tailed, outlier'lı → Spearman tercih
        """
        
        # NaN'ları hizala ve temizle
        valid_mask = ~(indicator.isna() | forward_return.isna())
        ind_clean = indicator[valid_mask]
        ret_clean = forward_return[valid_mask]
        
        if len(ind_clean) < self.MIN_OBSERVATIONS:
            return np.nan, 1.0
        
        # Spearman korelasyonu
        try:
            ic, p_value = stats.spearmanr(ind_clean, ret_clean)
            return ic, p_value
        except Exception:
            return np.nan, 1.0
    
    def calculate_rolling_ic(
        self,
        indicator: pd.Series,
        forward_return: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Rolling IC hesaplar (IC stability analizi için).
        
        Parameters:
        ----------
        window : int
            Rolling window boyutu (varsayılan: 60 = ~2 ay @ günlük)
        
        Returns:
        -------
        pd.Series
            Her zaman noktası için IC değeri
        
        Kullanım:
        --------
        IC_mean = rolling_ic.mean()    # Ortalama IC
        IC_std = rolling_ic.std()      # IC volatilitesi
        IC_IR = IC_mean / IC_std       # Information Ratio (stability)
        """
        
        # NaN'ları hizala
        df = pd.DataFrame({
            'indicator': indicator,
            'forward_return': forward_return
        }).dropna()
        
        if len(df) < window + 10:
            return pd.Series(dtype=float)
        
        # Rolling Spearman
        def rolling_spearman(x):
            return stats.spearmanr(x['indicator'], x['forward_return'])[0]
        
        rolling_ic = df.rolling(window).apply(
            lambda x: stats.spearmanr(x, df.loc[x.index, 'forward_return'])[0] 
            if len(x) >= window else np.nan,
            raw=False
        )['indicator']
        
        return rolling_ic
    
    def evaluate_indicator(
        self,
        df: pd.DataFrame,
        indicator_col: str,
        target_col: str = 'fwd_ret_1',
        category: str = 'unknown'
    ) -> IndicatorScore:
        """
        Tek bir indikatörü değerlendirir.
        
        Parameters:
        ----------
        df : pd.DataFrame
            İndikatör ve forward return içeren DataFrame
            
        indicator_col : str
            İndikatör kolon adı
            
        target_col : str
            Hedef (forward return) kolon adı
            
        category : str
            İndikatör kategorisi
        
        Returns:
        -------
        IndicatorScore
            Değerlendirme sonucu
        """
        
        if indicator_col not in df.columns or target_col not in df.columns:
            return IndicatorScore(
                name=indicator_col, category=category,
                ic_mean=np.nan, ic_std=np.nan, ic_ir=np.nan,
                ic_tstat=np.nan, p_value=1.0, p_value_adjusted=1.0,
                is_significant=False, n_observations=0, direction='neutral'
            )
        
        indicator = df[indicator_col]
        forward_return = df[target_col]
        
        # NaN temizle
        valid_mask = ~(indicator.isna() | forward_return.isna())
        n_obs = valid_mask.sum()
        
        if n_obs < self.MIN_OBSERVATIONS:
            return IndicatorScore(
                name=indicator_col, category=category,
                ic_mean=np.nan, ic_std=np.nan, ic_ir=np.nan,
                ic_tstat=np.nan, p_value=1.0, p_value_adjusted=1.0,
                is_significant=False, n_observations=n_obs, direction='neutral'
            )
        
        # Rolling IC hesapla
        window = min(60, n_obs // 5)  # Adaptif window
        if window < 20:
            window = 20
        
        # Basit IC hesabı
        ic_mean, p_value = self.calculate_ic(indicator, forward_return)
        
        if np.isnan(ic_mean):
            return IndicatorScore(
                name=indicator_col, category=category,
                ic_mean=np.nan, ic_std=np.nan, ic_ir=np.nan,
                ic_tstat=np.nan, p_value=1.0, p_value_adjusted=1.0,
                is_significant=False, n_observations=n_obs, direction='neutral'
            )
        
        # Rolling IC için bootstrap yaklaşım
        n_bootstrap = min(100, n_obs // 10)
        bootstrap_ics = []
        
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(n_obs, size=n_obs, replace=True)
            try:
                ic_sample, _ = stats.spearmanr(
                    indicator.dropna().iloc[sample_idx[:len(indicator.dropna())]],
                    forward_return.dropna().iloc[sample_idx[:len(forward_return.dropna())]]
                )
                if not np.isnan(ic_sample):
                    bootstrap_ics.append(ic_sample)
            except:
                pass
        
        if len(bootstrap_ics) > 10:
            ic_std = np.std(bootstrap_ics)
        else:
            ic_std = abs(ic_mean) * 0.5  # Fallback
        
        # IC Information Ratio
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        # t-statistic (H0: IC = 0)
        ic_tstat = ic_mean / (ic_std / np.sqrt(n_obs)) if ic_std > 0 else 0
        
        # Yön
        direction = 'bullish' if ic_mean > 0 else 'bearish' if ic_mean < 0 else 'neutral'
        
        return IndicatorScore(
            name=indicator_col,
            category=category,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_tstat=ic_tstat,
            p_value=p_value,
            p_value_adjusted=p_value,  # Sonra düzeltilecek
            is_significant=False,       # Sonra belirlenecek
            n_observations=n_obs,
            direction=direction
        )
    
    def evaluate_all_indicators(
        self,
        df: pd.DataFrame,
        target_col: str = 'fwd_ret_1',
        indicator_cols: Optional[List[str]] = None
    ) -> List[IndicatorScore]:
        """
        TÜM indikatörleri değerlendirir ve sıralar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            İndikatörler ve forward return içeren DataFrame
            
        target_col : str
            Hedef kolon adı
            
        indicator_cols : List[str], optional
            Değerlendirilecek kolonlar. None ise otomatik tespit.
        
        Returns:
        -------
        List[IndicatorScore]
            Tüm indikatörlerin değerlendirmesi
        """
        
        # İndikatör kolonlarını tespit et
        if indicator_cols is None:
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'log_return', 'simple_return']
            exclude_prefixes = ['fwd_', 'roll']  # Forward ve rolling kolonlar hariç
            
            indicator_cols = [
                c for c in df.columns 
                if c not in exclude_cols 
                and not any(c.startswith(p) for p in exclude_prefixes)
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
            ]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"İNDİKATÖR DEĞERLENDİRME")
            print(f"{'='*60}")
            print(f"Değerlendirilecek: {len(indicator_cols)} indikatör")
            print(f"Hedef: {target_col}")
            print(f"Gözlem sayısı: {len(df)}")
            print(f"Multiple testing: {self.correction_method.upper()}")
        
        # Her indikatörü değerlendir
        scores: List[IndicatorScore] = []
        
        for col in indicator_cols:
            # Kategori tespiti
            category = self._detect_category(col)
            
            # Değerlendir
            score = self.evaluate_indicator(df, col, target_col, category)
            scores.append(score)
        
        # Multiple testing correction
        scores = self._apply_multiple_testing_correction(scores)
        
        # |IC| büyüklüğüne göre sırala
        scores.sort(key=lambda x: abs(x.ic_mean) if not np.isnan(x.ic_mean) else 0, reverse=True)
        
        if self.verbose:
            # Özet
            significant = sum(1 for s in scores if s.is_significant)
            print(f"\nAnlamlı indikatör: {significant}/{len(scores)}")
            print(f"Ortalama |IC|: {np.nanmean([abs(s.ic_mean) for s in scores]):.4f}")
        
        return scores
    
    def _apply_multiple_testing_correction(
        self,
        scores: List[IndicatorScore]
    ) -> List[IndicatorScore]:
        """
        Multiple testing correction uygular.
        
        Bonferroni:
        - α_adj = α / n_tests
        - Çok muhafazakar, FWER kontrol
        - p < α/n ise anlamlı
        
        Benjamini-Hochberg (FDR):
        - p-değerlerini sırala
        - Her p_i için: p_i < (i/n) * α ?
        - En büyük i'yi bul → o ve altındakiler anlamlı
        - Daha güçlü, false discovery rate kontrol
        """
        
        n_tests = len(scores)
        p_values = [s.p_value for s in scores]
        
        if self.correction_method == 'bonferroni':
            # Bonferroni: p_adj = p * n
            adjusted_alpha = self.alpha / n_tests
            
            for score in scores:
                score.p_value_adjusted = min(score.p_value * n_tests, 1.0)
                score.is_significant = (
                    score.p_value < adjusted_alpha and 
                    abs(score.ic_mean) >= self.MIN_IC and
                    not np.isnan(score.ic_mean)
                )
        
        elif self.correction_method == 'fdr':
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            
            # BH threshold hesapla
            n = len(sorted_p)
            bh_thresholds = (np.arange(1, n + 1) / n) * self.alpha
            
            # En büyük i'yi bul: p_i <= (i/n) * α
            significant_mask = sorted_p <= bh_thresholds
            
            if significant_mask.any():
                max_significant_idx = np.where(significant_mask)[0][-1]
            else:
                max_significant_idx = -1
            
            # Adjusted p-values (BH)
            adjusted_p = np.zeros(n)
            for i in range(n - 1, -1, -1):
                if i == n - 1:
                    adjusted_p[sorted_indices[i]] = min(sorted_p[i] * n / (i + 1), 1.0)
                else:
                    adjusted_p[sorted_indices[i]] = min(
                        sorted_p[i] * n / (i + 1),
                        adjusted_p[sorted_indices[i + 1]]
                    )
            
            # Score'ları güncelle
            for i, score in enumerate(scores):
                score.p_value_adjusted = adjusted_p[i]
                score.is_significant = (
                    i in sorted_indices[:max_significant_idx + 1] and
                    abs(score.ic_mean) >= self.MIN_IC and
                    not np.isnan(score.ic_mean)
                )
        
        return scores
    
    def _detect_category(self, col_name: str) -> str:
        """Kolon adından kategori tespit eder."""
        
        # Bilinen pattern'ler
        trend_patterns = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'HMA', 'KAMA', 
                         'ADX', 'AROON', 'PSAR', 'SUPER', 'VORTEX']
        momentum_patterns = ['RSI', 'STOCH', 'WILLR', 'CCI', 'MOM', 'ROC', 
                            'MACD', 'PPO', 'TSI', 'AO', 'CMO', 'FISHER']
        volatility_patterns = ['ATR', 'NATR', 'BB', 'KC', 'DONCH', 'MASSI', 
                              'UI', 'ACC', 'RVI', 'TRUE']
        volume_patterns = ['OBV', 'AD', 'CMF', 'MFI', 'EFI', 'NVI', 'PVI', 
                          'PVOL', 'PVT', 'VWMA', 'VWAP']
        
        col_upper = col_name.upper()
        
        for pattern in trend_patterns:
            if pattern in col_upper:
                return 'trend'
        for pattern in momentum_patterns:
            if pattern in col_upper:
                return 'momentum'
        for pattern in volatility_patterns:
            if pattern in col_upper:
                return 'volatility'
        for pattern in volume_patterns:
            if pattern in col_upper:
                return 'volume'
        
        return 'other'
    
    def select_best_indicators(
        self,
        scores: List[IndicatorScore],
        max_per_category: int = 2,
        min_ic: float = None,
        only_significant: bool = True
    ) -> Dict[str, List[IndicatorScore]]:
        """
        Her kategoriden en iyi indikatörleri seçer.
        
        Parameters:
        ----------
        scores : List[IndicatorScore]
            Değerlendirilmiş indikatörler
            
        max_per_category : int
            Kategori başına maksimum indikatör sayısı
            
        min_ic : float, optional
            Minimum |IC| eşiği
            
        only_significant : bool
            Sadece istatistiksel olarak anlamlı olanları seç
        
        Returns:
        -------
        Dict[str, List[IndicatorScore]]
            Kategori → seçilen indikatörler
        """
        
        if min_ic is None:
            min_ic = self.MIN_IC
        
        selected: Dict[str, List[IndicatorScore]] = {}
        
        for score in scores:
            # Filtreleme
            if np.isnan(score.ic_mean):
                continue
            if abs(score.ic_mean) < min_ic:
                continue
            if only_significant and not score.is_significant:
                continue
            
            # Kategoriye ekle
            if score.category not in selected:
                selected[score.category] = []
            
            if len(selected[score.category]) < max_per_category:
                selected[score.category].append(score)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SEÇİLEN İNDİKATÖRLER")
            print(f"{'='*60}")
            
            for category, indicators in selected.items():
                print(f"\n📊 {category.upper()}:")
                for ind in indicators:
                    sig_mark = "✓" if ind.is_significant else "○"
                    print(f"   {sig_mark} {ind.name:<25} IC={ind.ic_mean:+.4f} (p={ind.p_value_adjusted:.4f})")
        
        return selected
    
    def get_summary_report(
        self,
        scores: List[IndicatorScore],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Değerlendirme özet raporu oluşturur.
        
        Returns:
        -------
        pd.DataFrame
            Sıralanmış indikatör değerlendirme tablosu
        """
        
        data = []
        for s in scores[:top_n]:
            data.append({
                'İndikatör': s.name,
                'Kategori': s.category,
                'IC': s.ic_mean,
                'IC_Std': s.ic_std,
                'IC_IR': s.ic_ir,
                't-stat': s.ic_tstat,
                'p-value': s.p_value,
                'p-adj': s.p_value_adjusted,
                'Anlamlı': '✓' if s.is_significant else '',
                'Yön': s.direction,
                'N': s.n_observations
            })
        
        return pd.DataFrame(data)


# =============================================================================
# TEST KODU
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("İNDİKATÖR SELECTOR TEST")
    print("=" * 70)
    
    import sys
    sys.path.insert(0, '../data')
    
    try:
        from fetcher import DataFetcher
        from calculator import IndicatorCalculator
        
        # 1. Veri çek
        print("\n[1] Veri çekiliyor...")
        fetcher = DataFetcher(symbol="BTC/USDT")
        df = fetcher.fetch_ohlcv(timeframe="1h", limit=1000)
        print(f"    {len(df)} bar çekildi")
        
        # 2. İndikatörler hesapla
        print("\n[2] İndikatörler hesaplanıyor...")
        calc = IndicatorCalculator(verbose=False)
        df_indicators = calc.calculate_all(df)
        df_indicators = calc.add_forward_returns(df_indicators, periods=[1, 5, 10])
        print(f"    {len(df_indicators.columns)} kolon oluşturuldu")
        
        # 3. Selector oluştur
        print("\n[3] İndikatörler değerlendiriliyor...")
        selector = IndicatorSelector(
            alpha=0.05,
            correction_method='fdr',
            verbose=True
        )
        
        scores = selector.evaluate_all_indicators(
            df_indicators,
            target_col='fwd_ret_5'  # 5-bar ileri getiri
        )
        
        # 4. En iyi indikatörleri seç
        print("\n[4] En iyiler seçiliyor...")
        best = selector.select_best_indicators(
            scores,
            max_per_category=2,
            only_significant=False  # Test için tümünü göster
        )
        
        # 5. Özet rapor
        print("\n[5] Özet Rapor (Top 15):")
        report = selector.get_summary_report(scores, top_n=15)
        print(report.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("TEST TAMAMLANDI")
        print("=" * 70)
        
    except ImportError as e:
        print(f"Import hatası: {e}")
        print("Gerekli modüller yüklenemedi.")

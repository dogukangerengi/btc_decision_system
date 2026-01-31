# =============================================================================
# BTC DİNAMİK KARAR DESTEK SİSTEMİ - ANA ORKESTRASYON
# =============================================================================
# Amaç: Tüm modülleri birleştirip saatlik analiz döngüsü çalıştırmak
#
# Akış:
# 1. Veri Çekme (DataFetcher) - Multi-timeframe OHLCV
# 2. İndikatör Hesaplama (IndicatorCalculator) - 60+ indikatör
# 3. İstatistiksel Seçim (IndicatorSelector) - IC, p-value, FDR
# 4. IC Bazlı Timeframe Seçimi - Karar destek için optimize
# 5. Rapor Oluşturma - Telegram bildirimi
#
# v1.2.0 Güncelleme:
# - Backtest bazlı TF seçimi → IC bazlı TF seçimi
# - Karar destek sistemine uygun metrikler
# - Sharpe/WinRate yerine IC gücü ve tutarlılığı
#
# Çalışma Modu:
# - Tek seferlik: python main.py
# - Sürekli (saatlik): python main.py --schedule
# =============================================================================

import sys
import os
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# =============================================================================
# .ENV DOSYASINI YÜKLE (Telegram token'ları için)
# =============================================================================
from dotenv import load_dotenv

# Proje kök dizinindeki .env dosyasını bul ve yükle
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent  # main.py -> src -> project_root
ENV_FILE = PROJECT_ROOT / '.env'

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Alternatif: src dizininde .env varsa
    load_dotenv(CURRENT_FILE.parent / '.env')

# =============================================================================
# PATH AYARLARI
# =============================================================================
# Tüm modül klasörlerini Python path'ine ekle

SRC_DIR = CURRENT_FILE.parent              # src klasörü

# Her modül klasörünü ayrı ayrı ekle (internal import'lar için)
for subdir in ['data', 'indicators', 'backtest', 'notifications']:
    module_path = SRC_DIR / subdir
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# =============================================================================
# MODÜL İMPORTLARI
# =============================================================================

from fetcher import DataFetcher
from calculator import IndicatorCalculator
from selector import IndicatorSelector, IndicatorScore
from telegram_notifier import TelegramNotifier, AnalysisReport

# =============================================================================
# LOGGİNG AYARLARI
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# IC BAZLI TF SIRALAMA DATACLASS
# =============================================================================

@dataclass
class ICTimeframeScore:
    """Bir timeframe'in IC bazlı skoru."""
    timeframe: str
    top_ic: float                    # En güçlü |IC| değeri
    top_ic_indicator: str            # En güçlü IC'ye sahip indikatör
    avg_ic: float                    # Ortalama |IC|
    significant_count: int           # Anlamlı indikatör sayısı (|IC| > 0.02)
    total_count: int                 # Toplam test edilen indikatör
    consistency: float               # IC tutarlılığı (0-1, aynı yönde olanların oranı)
    dominant_direction: str          # Baskın yön: 'LONG', 'SHORT', 'NEUTRAL'
    composite_score: float           # Toplam skor (0-100)
    market_regime: str               # Piyasa rejimi


@dataclass
class ICTimeframeRanking:
    """IC bazlı timeframe sıralaması."""
    rankings: List[ICTimeframeScore]     # Sıralı TF skorları
    best_timeframe: str                   # En iyi TF
    market_regime: str                    # Genel piyasa rejimi
    confidence: float                     # Seçim güveni (0-100)


# =============================================================================
# YAPILANDIRMA
# =============================================================================

class Config:
    """Sistem yapılandırması."""
    
    # İşlem çifti
    SYMBOL = "BTC/USDT"
    
    # Day trading için aktif timeframe'ler
    TIMEFRAMES = {
        '5m':  {'bars': 2000, 'description': 'Scalping, entry/exit timing'},
        '15m': {'bars': 1500, 'description': 'Ana day trading TF'},
        '30m': {'bars': 1000, 'description': 'Trend konfirmasyonu'},
        '1h':  {'bars': 1000, 'description': 'İntraday trend'},
        '2h':  {'bars': 750,  'description': 'Swing noktaları'},
        '4h':  {'bars': 500,  'description': 'Büyük resim, major S/R'},
    }
    
    # İndikatör hesaplama kategorileri
    INDICATOR_CATEGORIES = ['trend', 'momentum', 'volatility', 'volume']
    
    # İstatistiksel seçim parametreleri
    SELECTOR_ALPHA = 0.05              # Anlamlılık düzeyi
    SELECTOR_METHOD = 'fdr'            # Multiple testing correction
    MAX_INDICATORS_PER_CATEGORY = 2    # Kategori başına max indikatör
    
    # IC bazlı TF seçim ağırlıkları
    IC_WEIGHT_TOP_IC = 0.40            # En güçlü IC ağırlığı
    IC_WEIGHT_AVG_IC = 0.25            # Ortalama IC ağırlığı
    IC_WEIGHT_COUNT = 0.15             # Anlamlı indikatör sayısı ağırlığı
    IC_WEIGHT_CONSISTENCY = 0.20       # Tutarlılık ağırlığı
    
    # Forward return hedefi (IC hesabı için)
    FORWARD_RETURN_PERIODS = [1, 5, 10, 20]
    TARGET_PERIOD = 5                   # Ana hedef: 5 bar sonrası
    
    # Scheduler
    SCHEDULE_INTERVAL_MINUTES = 60     # Saatlik çalışma
    
    # Telegram (env var'dan okunacak)
    TELEGRAM_ENABLED = True


# =============================================================================
# ANA ANALİZ SINIFI
# =============================================================================

class BTCDecisionSystem:
    """
    BTC Dinamik Karar Destek Sistemi.
    
    Tüm analiz pipeline'ını yöneten ana sınıf.
    
    v1.2.0: IC bazlı TF seçimi (karar destek için optimize)
    """
    
    def __init__(self, config: Config = None, verbose: bool = True):
        """
        Sistemi başlatır.
        
        Parameters:
        ----------
        config : Config
            Yapılandırma objesi
        verbose : bool
            Detaylı çıktı
        """
        self.config = config or Config()
        self.verbose = verbose
        
        # Modül instance'ları
        self.fetcher = DataFetcher(symbol=self.config.SYMBOL)
        self.calculator = IndicatorCalculator(verbose=False)
        self.selector = IndicatorSelector(
            alpha=self.config.SELECTOR_ALPHA,
            correction_method=self.config.SELECTOR_METHOD,
            verbose=False
        )
        self.notifier = TelegramNotifier()
        
        # Sonuçlar
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.indicator_scores: Dict[str, List[IndicatorScore]] = {}
        self.ic_ranking: ICTimeframeRanking = None
        self.current_price: float = 0.0
        
        logger.info(f"BTCDecisionSystem v1.2.0 başlatıldı - {self.config.SYMBOL}")
    
    # =========================================================================
    # ADIM 1: VERİ ÇEKME
    # =========================================================================
    
    def fetch_all_data(self) -> bool:
        """
        Tüm timeframe'ler için veri çeker.
        
        Returns:
        -------
        bool
            Başarılı ise True
        """
        logger.info("=" * 60)
        logger.info("ADIM 1: VERİ ÇEKME")
        logger.info("=" * 60)
        
        self.data_dict = {}
        
        for tf, params in self.config.TIMEFRAMES.items():
            try:
                bars = params['bars']
                logger.info(f"  {tf}: {bars} bar çekiliyor...")
                
                df = self.fetcher.fetch_max_ohlcv(timeframe=tf, max_bars=bars, progress=False)
                
                if df is not None and len(df) > 100:
                    self.data_dict[tf] = df
                    logger.info(f"  {tf}: ✓ {len(df)} bar ({df.index[0].date()} → {df.index[-1].date()})")
                else:
                    logger.warning(f"  {tf}: ✗ Yetersiz veri")
                    
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        # Güncel fiyat (en kısa TF'den)
        if self.data_dict:
            shortest_tf = min(self.data_dict.keys(), key=lambda x: self._tf_to_minutes(x))
            self.current_price = self.data_dict[shortest_tf]['close'].iloc[-1]
            logger.info(f"\n  💰 Güncel Fiyat: ${self.current_price:,.2f}")
        
        return len(self.data_dict) > 0
    
    # =========================================================================
    # ADIM 2: İNDİKATÖR HESAPLAMA
    # =========================================================================
    
    def calculate_indicators(self) -> bool:
        """
        Tüm timeframe'ler için indikatör hesaplar.
        
        Returns:
        -------
        bool
            Başarılı ise True
        """
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 2: İNDİKATÖR HESAPLAMA")
        logger.info("=" * 60)
        
        for tf, df in self.data_dict.items():
            try:
                logger.info(f"  {tf}: İndikatörler hesaplanıyor...")
                
                # Tüm kategorileri hesapla
                df_with_indicators = self.calculator.calculate_all(
                    df, 
                    categories=self.config.INDICATOR_CATEGORIES
                )
                
                # Price features ekle
                df_with_indicators = self.calculator.add_price_features(df_with_indicators)
                
                # Rolling stats ekle
                df_with_indicators = self.calculator.add_rolling_stats(
                    df_with_indicators, 
                    windows=[10, 20, 50]
                )
                
                # Forward returns ekle (IC hesabı için)
                df_with_indicators = self.calculator.add_forward_returns(
                    df_with_indicators,
                    periods=self.config.FORWARD_RETURN_PERIODS
                )
                
                self.data_dict[tf] = df_with_indicators
                
                n_indicators = len([c for c in df_with_indicators.columns 
                                   if c not in ['open', 'high', 'low', 'close', 'volume']])
                logger.info(f"  {tf}: ✓ {n_indicators} kolon oluşturuldu")
                
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        return True
    
    # =========================================================================
    # ADIM 3: İSTATİSTİKSEL İNDİKATÖR SEÇİMİ
    # =========================================================================
    
    def select_indicators(self) -> bool:
        """
        Her timeframe için istatistiksel olarak anlamlı indikatörleri seçer.
        
        Returns:
        -------
        bool
            Başarılı ise True
        """
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 3: İSTATİSTİKSEL İNDİKATÖR SEÇİMİ (IC Analizi)")
        logger.info("=" * 60)
        
        target_col = f'fwd_ret_{self.config.TARGET_PERIOD}'
        
        for tf, df in self.data_dict.items():
            try:
                logger.info(f"  {tf}: IC analizi yapılıyor...")
                
                # Tüm indikatörleri değerlendir
                scores = self.selector.evaluate_all_indicators(
                    df,
                    target_col=target_col
                )
                
                self.indicator_scores[tf] = scores
                
                # Anlamlı indikatör sayısı
                significant = [s for s in scores if abs(s.ic_mean) > 0.02 and not np.isnan(s.ic_mean)]
                logger.info(f"  {tf}: ✓ {len(significant)}/{len(scores)} anlamlı indikatör")
                
                # En güçlü IC'yi logla
                if significant:
                    top_ic = max(significant, key=lambda x: abs(x.ic_mean))
                    logger.info(f"  {tf}: En güçlü IC: {top_ic.name} = {top_ic.ic_mean:+.4f}")
                
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        return True
    
    # =========================================================================
    # ADIM 4: IC BAZLI TİMEFRAME SEÇİMİ
    # =========================================================================
    
    def select_timeframe_by_ic(self) -> bool:
        """
        IC değerlerine göre en uygun timeframe'i seçer.
        
        Karar Destek İçin Optimize:
        - Backtest performansı DEĞİL, sinyal gücü önemli
        - En güçlü |IC| = En güvenilir indikatörler
        - Tutarlılık = Net yön (LONG veya SHORT)
        
        Skor Formülü:
        Score = (top_ic × 40) + (avg_ic × 25) + (count × 15) + (consistency × 20)
        
        Returns:
        -------
        bool
            Başarılı ise True
        """
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 4: IC BAZLI TİMEFRAME SEÇİMİ")
        logger.info("=" * 60)
        
        tf_scores: List[ICTimeframeScore] = []
        
        for tf, scores in self.indicator_scores.items():
            if not scores:
                continue
            
            # Sadece ana kategorilerdeki anlamlı IC'ler
            valid_categories = ['trend', 'momentum', 'volatility', 'volume']
            significant = [s for s in scores 
                          if abs(s.ic_mean) > 0.02 
                          and not np.isnan(s.ic_mean)
                          and s.category in valid_categories]
            
            if not significant:
                continue
            
            # === METRİKLER ===
            
            # 1. En güçlü IC
            top_ic_score = max(significant, key=lambda x: abs(x.ic_mean))
            top_ic = abs(top_ic_score.ic_mean)
            top_ic_indicator = top_ic_score.name
            
            # 2. Ortalama |IC|
            avg_ic = np.mean([abs(s.ic_mean) for s in significant])
            
            # 3. Anlamlı indikatör sayısı
            significant_count = len(significant)
            total_count = len(scores)
            
            # 4. Tutarlılık (aynı yönde olanların oranı)
            positive_ic = sum(1 for s in significant if s.ic_mean > 0)
            negative_ic = sum(1 for s in significant if s.ic_mean < 0)
            consistency = max(positive_ic, negative_ic) / len(significant)
            
            # 5. Baskın yön
            if negative_ic > positive_ic * 1.5:
                dominant_direction = 'SHORT'
            elif positive_ic > negative_ic * 1.5:
                dominant_direction = 'LONG'
            else:
                dominant_direction = 'NEUTRAL'
            
            # 6. Piyasa rejimi (ADX bazlı)
            market_regime = self._detect_regime(tf)
            
            # === COMPOSİTE SKOR ===
            # Normalize et (0-100 arası)
            
            # Top IC: 0.02-0.40 arası → 0-100 puan
            top_ic_norm = min((top_ic - 0.02) / 0.38 * 100, 100)
            
            # Avg IC: 0.02-0.15 arası → 0-100 puan
            avg_ic_norm = min((avg_ic - 0.02) / 0.13 * 100, 100)
            
            # Count: 10-60 arası → 0-100 puan
            count_norm = min(significant_count / 50 * 100, 100)
            
            # Consistency: 0.5-1.0 arası → 0-100 puan
            consistency_norm = (consistency - 0.5) / 0.5 * 100
            consistency_norm = max(0, min(consistency_norm, 100))
            
            # Ağırlıklı toplam
            composite = (
                top_ic_norm * self.config.IC_WEIGHT_TOP_IC +
                avg_ic_norm * self.config.IC_WEIGHT_AVG_IC +
                count_norm * self.config.IC_WEIGHT_COUNT +
                consistency_norm * self.config.IC_WEIGHT_CONSISTENCY
            )
            
            # Rejim bazlı ayarlama
            if market_regime == 'ranging':
                composite *= 0.85  # Ranging'de trend sinyalleri zayıf
            elif market_regime == 'volatile':
                composite *= 0.80  # Volatil'de belirsizlik yüksek
            
            tf_score = ICTimeframeScore(
                timeframe=tf,
                top_ic=top_ic,
                top_ic_indicator=top_ic_indicator,
                avg_ic=avg_ic,
                significant_count=significant_count,
                total_count=total_count,
                consistency=consistency,
                dominant_direction=dominant_direction,
                composite_score=composite,
                market_regime=market_regime
            )
            
            tf_scores.append(tf_score)
            
            logger.info(f"  {tf}: Top IC={top_ic:.3f} ({top_ic_indicator[:15]}) | "
                       f"Avg={avg_ic:.3f} | N={significant_count} | "
                       f"Dir={dominant_direction} | Skor={composite:.1f}")
        
        if not tf_scores:
            logger.error("  Hiçbir TF için IC skoru hesaplanamadı!")
            return False
        
        # Composite skora göre sırala
        tf_scores.sort(key=lambda x: x.composite_score, reverse=True)
        
        # En iyi TF
        best = tf_scores[0]
        
        # Genel piyasa rejimi (çoğunluk)
        regime_counts = {}
        for ts in tf_scores:
            regime_counts[ts.market_regime] = regime_counts.get(ts.market_regime, 0) + 1
        overall_regime = max(regime_counts, key=regime_counts.get)
        
        # Ranking oluştur
        self.ic_ranking = ICTimeframeRanking(
            rankings=tf_scores,
            best_timeframe=best.timeframe,
            market_regime=overall_regime,
            confidence=best.composite_score
        )
        
        logger.info(f"\n  🏆 En iyi timeframe: {best.timeframe}")
        logger.info(f"  📊 En güçlü IC: {best.top_ic:.4f} ({best.top_ic_indicator})")
        logger.info(f"  🎯 Baskın yön: {best.dominant_direction}")
        logger.info(f"  ↔️ Piyasa rejimi: {overall_regime}")
        logger.info(f"  📈 Skor: {best.composite_score:.1f}/100")
        
        return True
    
    def _detect_regime(self, timeframe: str) -> str:
        """
        Piyasa rejimini tespit eder.
        
        ADX bazlı:
        - ADX > 25: Trending
        - ADX < 20: Ranging
        - Else: Transitioning
        """
        if timeframe not in self.data_dict:
            return 'unknown'
        
        df = self.data_dict[timeframe]
        
        # ADX kontrolü
        if 'ADX_14' in df.columns:
            adx = df['ADX_14'].iloc[-1]
            dmp = df.get('DMP_14', pd.Series([50])).iloc[-1] if 'DMP_14' in df.columns else 50
            dmn = df.get('DMN_14', pd.Series([50])).iloc[-1] if 'DMN_14' in df.columns else 50
        else:
            # ADX yoksa basit volatilite kontrolü
            returns = df['close'].pct_change().tail(50)
            vol = returns.std() * 100
            if vol > 3:
                return 'volatile'
            elif vol < 1:
                return 'ranging'
            return 'transitioning'
        
        # ADX bazlı rejim
        if adx > 25:
            if dmp > dmn:
                return 'trending_up'
            else:
                return 'trending_down'
        elif adx < 20:
            # Volatilite kontrolü
            atr_col = 'ATRr_14' if 'ATRr_14' in df.columns else None
            if atr_col and df[atr_col].iloc[-1] / df['close'].iloc[-1] > 0.03:
                return 'volatile'
            return 'ranging'
        else:
            return 'transitioning'
    
    # =========================================================================
    # ADIM 5: RAPOR OLUŞTURMA
    # =========================================================================
    
    def generate_report(self) -> AnalysisReport:
        """
        Analiz raporu oluşturur.
        
        IC Bazlı Yaklaşım:
        - TF seçimi IC skoruna göre
        - Güven skoru IC gücü ve tutarlılığına göre
        
        Returns:
        -------
        AnalysisReport
            Telegram'a gönderilecek rapor
        """
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 5: RAPOR OLUŞTURMA")
        logger.info("=" * 60)
        
        # En iyi TF (IC bazlı seçim)
        best_tf = self.ic_ranking.best_timeframe
        best_score = self.ic_ranking.rankings[0]
        
        # Sinyal yönü (IC bazlı)
        direction = best_score.dominant_direction
        
        # Aktif indikatörler ve IC değerleri
        active_indicators, indicator_details = self._get_active_indicators_with_ic(best_tf)
        
        # IC bazlı güven skoru
        confidence = self._calculate_ic_confidence(best_tf)
        
        # Notlar
        notes = self._generate_notes_ic_based(best_tf)
        
        # Rapor oluştur
        report = AnalysisReport(
            symbol=self.config.SYMBOL,
            price=self.current_price,
            recommended_timeframe=best_tf,
            market_regime=self.ic_ranking.market_regime,
            direction=direction,
            confidence_score=confidence,
            active_indicators=active_indicators,
            indicator_details=indicator_details,
            notes=notes
        )
        
        logger.info(f"  ✓ Rapor oluşturuldu")
        logger.info(f"  📊 TF: {best_tf} | Yön: {direction} | Güven: {confidence:.0f}")
        
        return report
    
    def _calculate_ic_confidence(self, timeframe: str) -> float:
        """
        IC bazlı güven skoru hesaplar.
        """
        if timeframe not in self.indicator_scores:
            return 50.0
        
        scores = self.indicator_scores[timeframe]
        
        # Sadece ana kategorilerdeki anlamlı indikatörler
        valid_categories = ['trend', 'momentum', 'volatility', 'volume']
        significant = [s for s in scores 
                      if abs(s.ic_mean) > 0.02 
                      and not np.isnan(s.ic_mean)
                      and s.category in valid_categories]
        
        if not significant:
            return 40.0
        
        # 1. Anlamlı indikatör sayısı katkısı (max 30 puan)
        n_significant = len(significant)
        count_score = min(n_significant / 20 * 30, 30)
        
        # 2. Ortalama |IC| katkısı (max 40 puan)
        avg_ic = np.mean([abs(s.ic_mean) for s in significant])
        ic_score = min((avg_ic - 0.02) / 0.08 * 40, 40)
        ic_score = max(ic_score, 0)
        
        # 3. IC tutarlılığı katkısı (max 30 puan)
        positive_ic = sum(1 for s in significant if s.ic_mean > 0)
        negative_ic = sum(1 for s in significant if s.ic_mean < 0)
        
        if n_significant > 0:
            consistency = max(positive_ic, negative_ic) / n_significant
            consistency_score = consistency * 30
        else:
            consistency_score = 15
        
        # Toplam (ham skor)
        total = count_score + ic_score + consistency_score
        
        # Piyasa rejimi ayarlaması
        if self.ic_ranking:
            regime = self.ic_ranking.market_regime
            
            if regime == 'ranging':
                total *= 0.75
            elif regime == 'volatile':
                total *= 0.70
            elif regime == 'transitioning':
                total *= 0.85
        
        return min(max(total, 0), 100)
    
    def _get_active_indicators_with_ic(self, timeframe: str) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """
        Aktif indikatörleri ve IC değerlerini döndürür.
        """
        active = {}
        ic_details = {}
        
        if timeframe not in self.indicator_scores:
            return active, ic_details
        
        scores = self.indicator_scores[timeframe]
        valid_categories = ['trend', 'momentum', 'volatility', 'volume']
        category_scores = {cat: [] for cat in valid_categories}
        
        def get_base_indicator(name: str) -> str:
            """İndikatörün ana adını döndürür (duplicate önleme için)."""
            groups = {
                'AROON': ['AROONU', 'AROOND', 'AROONOSC'],
                'STOCH': ['STOCHRSIk', 'STOCHRSId', 'STOCHk', 'STOCHd'],
                'MACD': ['MACDh', 'MACDs', 'MACD_'],
                'PPO': ['PPOh', 'PPOs', 'PPO_'],
                'TSI': ['TSIs', 'TSI_'],
                'BB': ['BBU_', 'BBM_', 'BBL_', 'BBB_', 'BBP_'],
                'KC': ['KCUe', 'KCBe', 'KCLe'],
                'DC': ['DCU_', 'DCM_', 'DCL_'],
                'PSAR': ['PSARl', 'PSARs'],
                'SUPERTREND': ['SUPERTs', 'SUPERTl', 'SUPERTd', 'SUPERT_'],
                'ICHIMOKU': ['ITS_', 'IKS_', 'ISA_', 'ISB_', 'ICS_'],
                'DI': ['DMP_', 'DMN_'],
                'VORTEX': ['VTXP', 'VTXN'],
                'FISHER': ['FISHERTs', 'FISHERT'],
                'RVI': ['RVIs', 'RVI_'],
                'QQE': ['QQEl', 'QQEs', 'QQE_'],
                'COPC': ['COPC'],
            }
            
            for group_name, patterns in groups.items():
                for pattern in patterns:
                    if name.startswith(pattern):
                        return group_name
            
            return name.split('_')[0]
        
        # Tüm anlamlı indikatörleri topla ve kategorilere ayır
        all_significant = []
        for score in scores:
            cat = score.category.lower() if score.category else 'other'
            if cat not in valid_categories:
                continue
            if abs(score.ic_mean) > 0.02 and not np.isnan(score.ic_mean):
                category_scores[cat].append(score)
                all_significant.append(score)
        
        # EN GÜÇLÜ indikatörü bul
        top_indicator = None
        if all_significant:
            top_indicator = max(all_significant, key=lambda x: abs(x.ic_mean))
        
        # Her kategoriden seç
        for cat in valid_categories:
            if not category_scores[cat]:
                continue
            
            sorted_scores = sorted(
                category_scores[cat], 
                key=lambda x: abs(x.ic_mean), 
                reverse=True
            )
            
            # Unique gruplardan seç
            selected = []
            used_groups = set()
            
            for s in sorted_scores:
                base_name = get_base_indicator(s.name)
                
                if base_name not in used_groups:
                    selected.append(s)
                    used_groups.add(base_name)
                    
                    if len(selected) >= 2:
                        break
            
            if selected:
                active[cat] = [s.name for s in selected]
                for s in selected:
                    ic_details[s.name] = s.ic_mean
        
        # EN GÜÇLÜ indikatörü kategorisine ekle (eğer zaten yoksa)
        if top_indicator:
            top_cat = top_indicator.category.lower() if top_indicator.category else 'other'
            if top_cat in valid_categories:
                if top_cat not in active:
                    active[top_cat] = []
                
                if top_indicator.name not in active[top_cat]:
                    active[top_cat].insert(0, top_indicator.name)
                    ic_details[top_indicator.name] = top_indicator.ic_mean
                    
                    if len(active[top_cat]) > 2:
                        removed = active[top_cat].pop()
                        if removed in ic_details and removed != top_indicator.name:
                            del ic_details[removed]
        
        return active, ic_details
    
    def _generate_notes_ic_based(self, timeframe: str) -> str:
        """IC bazlı notlar oluşturur."""
        notes = []
        
        if self.ic_ranking and self.ic_ranking.rankings:
            best = self.ic_ranking.rankings[0]
            
            # Yön gücü
            if best.dominant_direction == 'SHORT' and best.consistency > 0.7:
                notes.append("📉 İndikatörler güçlü SHORT yönünde")
            elif best.dominant_direction == 'LONG' and best.consistency > 0.7:
                notes.append("📈 İndikatörler güçlü LONG yönünde")
            elif best.consistency < 0.6:
                notes.append("↔️ Karışık sinyal - dikkatli ol")
            
            # En güçlü IC
            if best.top_ic > 0.15:
                ind_name = best.top_ic_indicator.split('_')[0]
                notes.append(f"⭐ En güçlü: {ind_name} (IC={best.top_ic:.2f})")
        
        # Piyasa rejimi
        if self.ic_ranking:
            if self.ic_ranking.market_regime == 'volatile':
                notes.append("⚡ Yüksek volatilite")
            elif self.ic_ranking.market_regime == 'transitioning':
                notes.append("🔄 Geçiş dönemi")
            elif self.ic_ranking.market_regime == 'ranging':
                notes.append("📊 Yatay piyasa")
        
        return " | ".join(notes) if notes else ""
    
    # =========================================================================
    # ADIM 6: TELEGRAM BİLDİRİMİ
    # =========================================================================
    
    def send_notification(self, report: AnalysisReport) -> bool:
        """Telegram bildirimi gönderir."""
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 6: TELEGRAM BİLDİRİMİ")
        logger.info("=" * 60)
        
        if not self.config.TELEGRAM_ENABLED:
            logger.info("  Telegram devre dışı")
            return True
        
        if not self.notifier.is_configured():
            logger.warning("  Telegram yapılandırılmamış (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
            # Console'a yazdır
            print("\n" + "-" * 50)
            print("TELEGRAM MESAJI (yapılandırılmadığı için gönderilmedi):")
            print("-" * 50)
            import re
            msg = self.notifier.format_analysis_report(report)
            clean_msg = re.sub(r'<[^>]+>', '', msg)
            print(clean_msg)
            print("-" * 50)
            return True
        
        try:
            success = self.notifier.send_report_sync(report)
            if success:
                logger.info("  ✓ Telegram bildirimi gönderildi")
            else:
                logger.error("  ✗ Telegram bildirimi gönderilemedi")
            return success
        except Exception as e:
            logger.error(f"  ✗ Telegram hatası: {e}")
            return False
    
    # =========================================================================
    # ANA ÇALIŞTIRMA
    # =========================================================================
    
    def run_analysis(self) -> Optional[AnalysisReport]:
        """
        Tam analiz döngüsünü çalıştırır.
        
        Returns:
        -------
        AnalysisReport
            Oluşturulan rapor (veya None)
        """
        start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"🚀 BTC DECISION SYSTEM v1.2.0 - ANALİZ BAŞLADI")
        logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        try:
            # Adım 1: Veri çekme
            if not self.fetch_all_data():
                logger.error("Veri çekme başarısız!")
                return None
            
            # Adım 2: İndikatör hesaplama
            if not self.calculate_indicators():
                logger.error("İndikatör hesaplama başarısız!")
                return None
            
            # Adım 3: İstatistiksel seçim
            if not self.select_indicators():
                logger.error("İndikatör seçimi başarısız!")
                return None
            
            # Adım 4: IC bazlı TF seçimi
            if not self.select_timeframe_by_ic():
                logger.error("TF seçimi başarısız!")
                return None
            
            # Adım 5: Rapor oluşturma
            report = self.generate_report()
            
            # Adım 6: Telegram bildirimi
            self.send_notification(report)
            
            # Özet
            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 70)
            logger.info(f"✅ ANALİZ TAMAMLANDI - {elapsed:.1f} saniye")
            logger.info("=" * 70)
            
            return report
            
        except Exception as e:
            logger.exception(f"Analiz hatası: {e}")
            return None
    
    # =========================================================================
    # YARDIMCI METODLAR
    # =========================================================================
    
    def _tf_to_minutes(self, tf: str) -> int:
        """Timeframe'i dakikaya çevirir."""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '8h': 480, '12h': 720, '1d': 1440
        }
        return mapping.get(tf, 60)
    
    def print_summary(self):
        """Detaylı özet yazdırır."""
        
        print("\n" + "=" * 70)
        print("IC BAZLI TIMEFRAME SIRALAMASI")
        print("=" * 70)
        
        if self.ic_ranking and self.ic_ranking.rankings:
            print(f"\n{'TF':<6} {'Top IC':<10} {'Avg IC':<10} {'N':<6} {'Dir':<8} {'Rejim':<12} {'Skor':<8}")
            print("-" * 70)
            
            for ts in self.ic_ranking.rankings:
                marker = "→" if ts.timeframe == self.ic_ranking.best_timeframe else " "
                print(f"{marker}{ts.timeframe:<5} {ts.top_ic:<10.4f} {ts.avg_ic:<10.4f} "
                      f"{ts.significant_count:<6} {ts.dominant_direction:<8} "
                      f"{ts.market_regime:<12} {ts.composite_score:<8.1f}")
            
            print("\n" + "=" * 70)
            best = self.ic_ranking.rankings[0]
            print(f"🏆 ÖNERİLEN: {best.timeframe}")
            print(f"   En güçlü sinyal: {best.top_ic_indicator} (IC={best.top_ic:+.4f})")
            print(f"   Baskın yön: {best.dominant_direction}")
            print(f"   Güven skoru: {best.composite_score:.0f}/100")


# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler(system: BTCDecisionSystem, interval_minutes: int = 60):
    """Belirtilen aralıkla analizi tekrarlar."""
    logger.info(f"Scheduler başlatıldı - Her {interval_minutes} dakikada bir çalışacak")
    
    while True:
        try:
            system.run_analysis()
            system.print_summary()
            
            next_run = datetime.now() + timedelta(minutes=interval_minutes)
            logger.info(f"\n⏰ Sonraki çalışma: {next_run.strftime('%H:%M:%S')}")
            
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("\nScheduler durduruldu (Ctrl+C)")
            break
        except Exception as e:
            logger.exception(f"Scheduler hatası: {e}")
            time.sleep(60)


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================

def main():
    """Ana giriş noktası."""
    
    parser = argparse.ArgumentParser(
        description='BTC Dinamik Karar Destek Sistemi v1.2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py              # Tek seferlik analiz
  python main.py --schedule   # Saatlik sürekli çalışma
  python main.py --interval 30  # 30 dakikada bir
        """
    )
    
    parser.add_argument(
        '--schedule', '-s',
        action='store_true',
        help='Sürekli çalışma modu (saatlik)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Çalışma aralığı (dakika, varsayılan: 60)'
    )
    
    parser.add_argument(
        '--no-telegram', '-nt',
        action='store_true',
        help='Telegram bildirimlerini devre dışı bırak'
    )
    
    parser.add_argument(
        '--symbol', '-sym',
        type=str,
        default='BTC/USDT',
        help='İşlem çifti (varsayılan: BTC/USDT). Örnek: ETH/USDT, SOL/USDT'
    )
    
    args = parser.parse_args()
    
    # Yapılandırma
    config = Config()
    if args.no_telegram:
        config.TELEGRAM_ENABLED = False
    
    # Symbol değiştir
    config.SYMBOL = args.symbol.upper()
    
    # Sistem oluştur
    system = BTCDecisionSystem(config=config, verbose=True)
    
    if args.schedule:
        run_scheduler(system, interval_minutes=args.interval)
    else:
        report = system.run_analysis()
        if report:
            system.print_summary()


if __name__ == "__main__":
    main()

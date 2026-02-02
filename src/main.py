# =============================================================================
# BTC DİNAMİK KARAR DESTEK SİSTEMİ - ANA ORKESTRASYON v2.0
# =============================================================================
# Güncellemeler:
# - Gerçek anlık fiyat (ticker'dan)
# - Kategori bazlı top indikatörler
# - Kompakt TF sıralaması
# - Yeni Telegram format desteği
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
# .ENV DOSYASINI YÜKLE
# =============================================================================
from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
ENV_FILE = PROJECT_ROOT / '.env'

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv(CURRENT_FILE.parent / '.env')

# =============================================================================
# PATH AYARLARI
# =============================================================================
SRC_DIR = CURRENT_FILE.parent

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
    top_ic: float
    top_ic_indicator: str
    avg_ic: float
    significant_count: int
    total_count: int
    consistency: float
    dominant_direction: str
    composite_score: float
    market_regime: str


@dataclass
class ICTimeframeRanking:
    """IC bazlı timeframe sıralaması."""
    rankings: List[ICTimeframeScore]
    best_timeframe: str
    market_regime: str
    confidence: float


# =============================================================================
# YAPILANDIRMA
# =============================================================================
class Config:
    """Sistem yapılandırması."""
    
    SYMBOL = "BTC/USDT"
    
    TIMEFRAMES = {
        '5m':  {'bars': 2000, 'description': 'Scalping'},
        '15m': {'bars': 1500, 'description': 'Day trading'},
        '30m': {'bars': 1000, 'description': 'Trend konfirmasyonu'},
        '1h':  {'bars': 1000, 'description': 'İntraday trend'},
        '2h':  {'bars': 750,  'description': 'Swing noktaları'},
        '4h':  {'bars': 500,  'description': 'Büyük resim'},
    }
    
    INDICATOR_CATEGORIES = ['trend', 'momentum', 'volatility', 'volume']
    
    SELECTOR_ALPHA = 0.05
    SELECTOR_METHOD = 'fdr'
    MAX_INDICATORS_PER_CATEGORY = 2
    
    IC_WEIGHT_TOP_IC = 0.40
    IC_WEIGHT_AVG_IC = 0.25
    IC_WEIGHT_COUNT = 0.15
    IC_WEIGHT_CONSISTENCY = 0.20
    
    FORWARD_RETURN_PERIODS = [1, 5, 10, 20]
    TARGET_PERIOD = 5
    
    SCHEDULE_INTERVAL_MINUTES = 60
    TELEGRAM_ENABLED = True


# =============================================================================
# ANA ANALİZ SINIFI
# =============================================================================
class BTCDecisionSystem:
    """BTC Dinamik Karar Destek Sistemi v2.0"""
    
    def __init__(self, config: Config = None, verbose: bool = True):
        self.config = config or Config()
        self.verbose = verbose
        
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
        self.change_24h: float = 0.0  # YENİ
        
        logger.info(f"BTCDecisionSystem v2.0 başlatıldı - {self.config.SYMBOL}")
    
    # =========================================================================
    # ADIM 1: VERİ ÇEKME
    # =========================================================================
    def fetch_all_data(self) -> bool:
        """Tüm timeframe'ler için veri çeker."""
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
                    logger.info(f"  {tf}: ✓ {len(df)} bar")
                else:
                    logger.warning(f"  {tf}: ✗ Yetersiz veri")
                    
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        # === GERÇEK ANLIK FİYAT (YENİ) ===
        try:
            price_info = self.fetcher.get_latest_price()
            self.current_price = price_info['last']
            self.change_24h = price_info.get('percentage', 0) or 0
            logger.info(f"\n  💰 Güncel Fiyat: ${self.current_price:,.2f} ({self.change_24h:+.2f}% 24h)")
        except Exception as e:
            logger.warning(f"  Ticker alınamadı: {e}")
            self.change_24h = 0
            if self.data_dict:
                shortest_tf = min(self.data_dict.keys(), key=lambda x: self._tf_to_minutes(x))
                self.current_price = self.data_dict[shortest_tf]['close'].iloc[-1]
                logger.info(f"\n  💰 Fiyat (son bar): ${self.current_price:,.2f}")
        
        return len(self.data_dict) > 0
    
    # =========================================================================
    # ADIM 2: İNDİKATÖR HESAPLAMA
    # =========================================================================
    def calculate_indicators(self) -> bool:
        """Tüm timeframe'ler için indikatör hesaplar."""
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 2: İNDİKATÖR HESAPLAMA")
        logger.info("=" * 60)
        
        for tf, df in self.data_dict.items():
            try:
                logger.info(f"  {tf}: İndikatörler hesaplanıyor...")
                
                df_with_indicators = self.calculator.calculate_all(
                    df, 
                    categories=self.config.INDICATOR_CATEGORIES
                )
                df_with_indicators = self.calculator.add_price_features(df_with_indicators)
                df_with_indicators = self.calculator.add_rolling_stats(df_with_indicators, windows=[10, 20, 50])
                df_with_indicators = self.calculator.add_forward_returns(
                    df_with_indicators,
                    periods=self.config.FORWARD_RETURN_PERIODS
                )
                
                self.data_dict[tf] = df_with_indicators
                
                n_indicators = len([c for c in df_with_indicators.columns 
                                   if c not in ['open', 'high', 'low', 'close', 'volume']])
                logger.info(f"  {tf}: ✓ {n_indicators} kolon")
                
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        return True
    
    # =========================================================================
    # ADIM 3: İSTATİSTİKSEL İNDİKATÖR SEÇİMİ
    # =========================================================================
    def select_indicators(self) -> bool:
        """Her timeframe için istatistiksel olarak anlamlı indikatörleri seçer."""
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 3: İSTATİSTİKSEL İNDİKATÖR SEÇİMİ")
        logger.info("=" * 60)
        
        target_col = f'fwd_ret_{self.config.TARGET_PERIOD}'
        
        for tf, df in self.data_dict.items():
            try:
                logger.info(f"  {tf}: IC analizi yapılıyor...")
                
                scores = self.selector.evaluate_all_indicators(df, target_col=target_col)
                self.indicator_scores[tf] = scores
                
                significant = [s for s in scores if abs(s.ic_mean) > 0.02 and not np.isnan(s.ic_mean)]
                logger.info(f"  {tf}: ✓ {len(significant)}/{len(scores)} anlamlı")
                
                if significant:
                    top_ic = max(significant, key=lambda x: abs(x.ic_mean))
                    logger.info(f"  {tf}: Top IC: {top_ic.name} = {top_ic.ic_mean:+.4f}")
                
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        return True
    
    # =========================================================================
    # ADIM 4: IC BAZLI TİMEFRAME SEÇİMİ
    # =========================================================================
    def select_timeframe_by_ic(self) -> bool:
        """IC değerlerine göre en uygun timeframe'i seçer."""
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 4: IC BAZLI TİMEFRAME SEÇİMİ")
        logger.info("=" * 60)
        
        tf_scores: List[ICTimeframeScore] = []
        
        for tf, scores in self.indicator_scores.items():
            if not scores:
                continue
            
            valid_categories = ['trend', 'momentum', 'volatility', 'volume']
            significant = [s for s in scores 
                          if abs(s.ic_mean) > 0.02 
                          and not np.isnan(s.ic_mean)
                          and s.category in valid_categories]
            
            if not significant:
                continue
            
            # Metrikler
            top_ic_score = max(significant, key=lambda x: abs(x.ic_mean))
            top_ic = abs(top_ic_score.ic_mean)
            top_ic_indicator = top_ic_score.name
            avg_ic = np.mean([abs(s.ic_mean) for s in significant])
            significant_count = len(significant)
            total_count = len(scores)
            
            positive_ic = sum(1 for s in significant if s.ic_mean > 0)
            negative_ic = sum(1 for s in significant if s.ic_mean < 0)
            consistency = max(positive_ic, negative_ic) / len(significant)
            
            if negative_ic > positive_ic * 1.5:
                dominant_direction = 'SHORT'
            elif positive_ic > negative_ic * 1.5:
                dominant_direction = 'LONG'
            else:
                dominant_direction = 'NEUTRAL'
            
            market_regime = self._detect_regime(tf)
            
            # Composite skor
            top_ic_norm = min((top_ic - 0.02) / 0.38 * 100, 100)
            avg_ic_norm = min((avg_ic - 0.02) / 0.13 * 100, 100)
            count_norm = min(significant_count / 50 * 100, 100)
            consistency_norm = max(0, min((consistency - 0.5) / 0.5 * 100, 100))
            
            composite = (
                top_ic_norm * self.config.IC_WEIGHT_TOP_IC +
                avg_ic_norm * self.config.IC_WEIGHT_AVG_IC +
                count_norm * self.config.IC_WEIGHT_COUNT +
                consistency_norm * self.config.IC_WEIGHT_CONSISTENCY
            )
            
            if market_regime == 'ranging':
                composite *= 0.85
            elif market_regime == 'volatile':
                composite *= 0.80
            elif market_regime == 'transitioning':
                composite *= 0.90
            
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
            
            logger.info(f"  {tf}: IC={top_ic:.3f} | Dir={dominant_direction} | Skor={composite:.1f}")
        
        if not tf_scores:
            logger.error("  Hiçbir TF için IC skoru hesaplanamadı!")
            return False
        
        tf_scores.sort(key=lambda x: x.composite_score, reverse=True)
        best = tf_scores[0]
        
        regime_counts = {}
        for ts in tf_scores:
            regime_counts[ts.market_regime] = regime_counts.get(ts.market_regime, 0) + 1
        overall_regime = max(regime_counts, key=regime_counts.get)
        
        self.ic_ranking = ICTimeframeRanking(
            rankings=tf_scores,
            best_timeframe=best.timeframe,
            market_regime=overall_regime,
            confidence=best.composite_score
        )
        
        logger.info(f"\n  🏆 En iyi: {best.timeframe} | {best.dominant_direction} | {best.composite_score:.0f}/100")
        
        return True
    
    def _detect_regime(self, timeframe: str) -> str:
        """Piyasa rejimini tespit eder."""
        if timeframe not in self.data_dict:
            return 'unknown'
        
        df = self.data_dict[timeframe]
        
        if 'ADX_14' in df.columns:
            adx = df['ADX_14'].iloc[-1]
            dmp = df.get('DMP_14', pd.Series([50])).iloc[-1] if 'DMP_14' in df.columns else 50
            dmn = df.get('DMN_14', pd.Series([50])).iloc[-1] if 'DMN_14' in df.columns else 50
        else:
            returns = df['close'].pct_change().tail(50)
            vol = returns.std() * 100
            if vol > 3:
                return 'volatile'
            elif vol < 1:
                return 'ranging'
            return 'transitioning'
        
        if adx > 25:
            return 'trending_up' if dmp > dmn else 'trending_down'
        elif adx < 20:
            atr_col = 'ATRr_14' if 'ATRr_14' in df.columns else None
            if atr_col and df[atr_col].iloc[-1] / df['close'].iloc[-1] > 0.03:
                return 'volatile'
            return 'ranging'
        else:
            return 'transitioning'
    
    # =========================================================================
    # ADIM 5: RAPOR OLUŞTURMA (YENİ FORMAT)
    # =========================================================================
    def generate_report(self) -> AnalysisReport:
        """Analiz raporu oluşturur - YENİ KOMPAKT FORMAT."""
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 5: RAPOR OLUŞTURMA")
        logger.info("=" * 60)
        
        best_tf = self.ic_ranking.best_timeframe
        best_score = self.ic_ranking.rankings[0]
        direction = best_score.dominant_direction
        
        # Aktif indikatörler (eski format için uyumluluk)
        active_indicators, indicator_details = self._get_active_indicators_with_ic(best_tf)
        
        # === KATEGORİ TOPS (YENİ) ===
        category_tops = self._get_category_tops(best_tf)
        
        # === TF SIRALAMASI (YENİ) ===
        tf_rankings = []
        for ts in self.ic_ranking.rankings[:4]:
            tf_rankings.append({
                'tf': ts.timeframe,
                'score': ts.composite_score,
                'direction': ts.dominant_direction
            })
        
        confidence = self._calculate_ic_confidence(best_tf)
        notes = self._generate_notes_ic_based(best_tf)
        
        report = AnalysisReport(
            symbol=self.config.SYMBOL,
            price=self.current_price,
            recommended_timeframe=best_tf,
            market_regime=self.ic_ranking.market_regime,
            direction=direction,
            confidence_score=confidence,
            active_indicators=active_indicators,
            indicator_details=indicator_details,
            category_tops=category_tops,      # YENİ
            tf_rankings=tf_rankings,          # YENİ
            notes=notes,
            change_24h=self.change_24h        # YENİ
        )
        
        logger.info(f"  ✓ Rapor oluşturuldu")
        logger.info(f"  📊 TF: {best_tf} | Yön: {direction} | Güven: {confidence:.0f}")
        
        return report
    
    def _get_category_tops(self, timeframe: str) -> Dict[str, dict]:
        """Her kategoriden en güçlü IC'li indikatörü döndürür."""
        category_tops = {}
        
        if timeframe not in self.indicator_scores:
            return category_tops
        
        scores = self.indicator_scores[timeframe]
        valid_categories = ['trend', 'momentum', 'volatility', 'volume']
        
        for cat in valid_categories:
            cat_scores = [s for s in scores 
                         if s.category == cat 
                         and abs(s.ic_mean) > 0.02 
                         and not np.isnan(s.ic_mean)]
            
            if cat_scores:
                best = max(cat_scores, key=lambda x: abs(x.ic_mean))
                category_tops[cat] = {
                    'name': best.name,
                    'ic': best.ic_mean
                }
        
        return category_tops
    
    def _calculate_ic_confidence(self, timeframe: str) -> float:
        """IC bazlı güven skoru hesaplar."""
        if timeframe not in self.indicator_scores:
            return 50.0
        
        scores = self.indicator_scores[timeframe]
        valid_categories = ['trend', 'momentum', 'volatility', 'volume']
        significant = [s for s in scores 
                      if abs(s.ic_mean) > 0.02 
                      and not np.isnan(s.ic_mean)
                      and s.category in valid_categories]
        
        if not significant:
            return 40.0
        
        n_significant = len(significant)
        count_score = min(n_significant / 20 * 30, 30)
        
        avg_ic = np.mean([abs(s.ic_mean) for s in significant])
        ic_score = max(0, min((avg_ic - 0.02) / 0.08 * 40, 40))
        
        positive_ic = sum(1 for s in significant if s.ic_mean > 0)
        negative_ic = sum(1 for s in significant if s.ic_mean < 0)
        consistency = max(positive_ic, negative_ic) / n_significant if n_significant > 0 else 0.5
        consistency_score = consistency * 30
        
        total = count_score + ic_score + consistency_score
        
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
        """Aktif indikatörleri ve IC değerlerini döndürür."""
        active = {}
        ic_details = {}
        
        if timeframe not in self.indicator_scores:
            return active, ic_details
        
        scores = self.indicator_scores[timeframe]
        valid_categories = ['trend', 'momentum', 'volatility', 'volume']
        
        for cat in valid_categories:
            cat_scores = [s for s in scores 
                         if s.category == cat 
                         and abs(s.ic_mean) > 0.02 
                         and not np.isnan(s.ic_mean)]
            
            if cat_scores:
                sorted_scores = sorted(cat_scores, key=lambda x: abs(x.ic_mean), reverse=True)
                selected = sorted_scores[:2]
                
                active[cat] = [s.name for s in selected]
                for s in selected:
                    ic_details[s.name] = s.ic_mean
        
        return active, ic_details
    
    def _generate_notes_ic_based(self, timeframe: str) -> str:
        """IC bazlı notlar oluşturur."""
        notes = []
        
        if self.ic_ranking and self.ic_ranking.rankings:
            best = self.ic_ranking.rankings[0]
            
            if best.dominant_direction == 'SHORT' and best.consistency > 0.7:
                notes.append("📉 Güçlü SHORT sinyalleri")
            elif best.dominant_direction == 'LONG' and best.consistency > 0.7:
                notes.append("📈 Güçlü LONG sinyalleri")
            elif best.consistency < 0.6:
                notes.append("↔️ Karışık sinyaller")
            
            if best.top_ic > 0.15:
                ind_name = best.top_ic_indicator.split('_')[0]
                notes.append(f"⭐ Top: {ind_name} (IC={best.top_ic:.2f})")
        
        if self.ic_ranking:
            if self.ic_ranking.market_regime == 'volatile':
                notes.append("⚡ Yüksek volatilite")
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
            logger.warning("  Telegram yapılandırılmamış")
            print("\n" + "-" * 50)
            print("TELEGRAM MESAJI (gönderilmedi):")
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
                logger.error("  ✗ Telegram gönderilemedi")
            return success
        except Exception as e:
            logger.error(f"  ✗ Telegram hatası: {e}")
            return False
    
    # =========================================================================
    # ANA ÇALIŞTIRMA
    # =========================================================================
    def run_analysis(self) -> Optional[AnalysisReport]:
        """Tam analiz döngüsünü çalıştırır."""
        start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"🚀 BTC DECISION SYSTEM v2.0 - ANALİZ BAŞLADI")
        logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        try:
            if not self.fetch_all_data():
                logger.error("Veri çekme başarısız!")
                return None
            
            if not self.calculate_indicators():
                logger.error("İndikatör hesaplama başarısız!")
                return None
            
            if not self.select_indicators():
                logger.error("İndikatör seçimi başarısız!")
                return None
            
            if not self.select_timeframe_by_ic():
                logger.error("TF seçimi başarısız!")
                return None
            
            report = self.generate_report()
            self.send_notification(report)
            
            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 70)
            logger.info(f"✅ ANALİZ TAMAMLANDI - {elapsed:.1f} saniye")
            logger.info("=" * 70)
            
            return report
            
        except Exception as e:
            logger.exception(f"Analiz hatası: {e}")
            return None
    
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
            print(f"\n{'TF':<6} {'Top IC':<10} {'Dir':<8} {'Rejim':<12} {'Skor':<8}")
            print("-" * 50)
            
            for ts in self.ic_ranking.rankings:
                marker = "→" if ts.timeframe == self.ic_ranking.best_timeframe else " "
                print(f"{marker}{ts.timeframe:<5} {ts.top_ic:<10.4f} {ts.dominant_direction:<8} "
                      f"{ts.market_regime:<12} {ts.composite_score:<8.1f}")
            
            print("\n" + "=" * 70)
            best = self.ic_ranking.rankings[0]
            print(f"🏆 ÖNERİLEN: {best.timeframe} | {best.dominant_direction} | {best.composite_score:.0f}/100")


# =============================================================================
# SCHEDULER
# =============================================================================
def run_scheduler(system: BTCDecisionSystem, interval_minutes: int = 60):
    """Belirtilen aralıkla analizi tekrarlar."""
    logger.info(f"Scheduler başlatıldı - Her {interval_minutes} dakikada bir")
    
    while True:
        try:
            system.run_analysis()
            system.print_summary()
            
            next_run = datetime.now() + timedelta(minutes=interval_minutes)
            logger.info(f"\n⏰ Sonraki: {next_run.strftime('%H:%M:%S')}")
            
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("\nScheduler durduruldu")
            break
        except Exception as e:
            logger.exception(f"Scheduler hatası: {e}")
            time.sleep(60)


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================
def main():
    """Ana giriş noktası."""
    parser = argparse.ArgumentParser(description='BTC Karar Destek Sistemi v2.0')
    
    parser.add_argument('--schedule', '-s', action='store_true', help='Sürekli çalışma modu')
    parser.add_argument('--interval', '-i', type=int, default=60, help='Çalışma aralığı (dakika)')
    parser.add_argument('--no-telegram', '-nt', action='store_true', help='Telegram devre dışı')
    parser.add_argument('--symbol', '-sym', type=str, default='BTC/USDT', help='İşlem çifti')
    
    args = parser.parse_args()
    
    config = Config()
    if args.no_telegram:
        config.TELEGRAM_ENABLED = False
    config.SYMBOL = args.symbol.upper()
    
    system = BTCDecisionSystem(config=config, verbose=True)
    
    if args.schedule:
        run_scheduler(system, interval_minutes=args.interval)
    else:
        report = system.run_analysis()
        if report:
            system.print_summary()


if __name__ == "__main__":
    main()

# =============================================================================
# BTC DİNAMİK KARAR DESTEK SİSTEMİ - ANA ORKESTRASYON
# =============================================================================
# Amaç: Tüm modülleri birleştirip saatlik analiz döngüsü çalıştırmak
#
# Akış:
# 1. Veri Çekme (DataFetcher) - Multi-timeframe OHLCV
# 2. İndikatör Hesaplama (IndicatorCalculator) - 60+ indikatör
# 3. İstatistiksel Seçim (IndicatorSelector) - IC, p-value, FDR
# 4. Dinamik Backtest (DynamicBacktester) - Walk-forward validation
# 5. Timeframe Seçimi - Composite scoring
# 6. Rapor Oluşturma - Telegram bildirimi
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
from backtester import DynamicBacktester, BacktestResult, TimeframeRanking
from telegram_notifier import TelegramNotifier, AnalysisReport
from utils.plotter import AnalysisPlotter

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
    
    # Backtest parametreleri
    BACKTEST_TRAIN_RATIO = 0.7         # %70 train, %30 test
    BACKTEST_N_WALKS = 5               # Walk-forward adım sayısı
    BACKTEST_MIN_TRADES = 30           # Minimum işlem sayısı
    
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
        self.backtester = DynamicBacktester(
            train_ratio=self.config.BACKTEST_TRAIN_RATIO,
            n_walks=self.config.BACKTEST_N_WALKS,
            min_trades=self.config.BACKTEST_MIN_TRADES,
            verbose=False
        )
        self.notifier = TelegramNotifier()
        self.plotter = AnalysisPlotter()
        
        # Sonuçlar
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.indicator_scores: Dict[str, List[IndicatorScore]] = {}
        self.backtest_results: List[BacktestResult] = []
        self.timeframe_ranking: TimeframeRanking = None
        self.best_indicators: Dict[str, List[IndicatorScore]] = {}
        self.current_price: float = 0.0
        
        logger.info(f"BTCDecisionSystem başlatıldı - {self.config.SYMBOL}")
    
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
                
                df = self.fetcher.fetch_max_ohlcv(timeframe=tf, max_bars=bars)
                
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
        logger.info("ADIM 3: İSTATİSTİKSEL İNDİKATÖR SEÇİMİ")
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
                
                # En iyileri seç
                best = self.selector.select_best_indicators(
                    scores,
                    max_per_category=self.config.MAX_INDICATORS_PER_CATEGORY,
                    only_significant=False  # Düşük volatilite dönemlerinde bile sinyal al
                )
                
                # Anlamlı indikatör sayısı
                significant = sum(1 for s in scores if s.is_significant)
                logger.info(f"  {tf}: ✓ {significant}/{len(scores)} anlamlı indikatör")
                
                # En güçlü IC'yi logla
                if scores:
                    top_ic = max(scores, key=lambda x: abs(x.ic_mean) if not np.isnan(x.ic_mean) else 0)
                    logger.info(f"  {tf}: En güçlü IC: {top_ic.name} = {top_ic.ic_mean:.4f}")
                
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
        
        return True
    
    # =========================================================================
    # ADIM 4: DİNAMİK BACKTEST (Multi-Indicator Composite)
    # =========================================================================
    
    def run_backtests(self) -> bool:
        """
        Tüm timeframe'ler için IC-based composite backtest yapar.
        
        Yeni Mantık:
        -----------
        1. Her TF için IC analizi ile seçilen indikatörleri kullan
        2. Multi-indicator composite sinyal üret
        3. Walk-forward validation ile test et
        
        Returns:
        -------
        bool
            Başarılı ise True
        """
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 4: DİNAMİK BACKTEST (Multi-Indicator)")
        logger.info("=" * 60)
        
        self.backtest_results = []
        
        try:
            for tf, df in self.data_dict.items():
                # Bu TF için IC skorlarını al
                scores = self.indicator_scores.get(tf, [])
                
                if not scores:
                    logger.warning(f"  {tf}: IC skorları bulunamadı, atlanıyor")
                    continue
                
                # Kullanılan indikatörleri logla
                best_inds = self.backtester._select_best_for_signal(scores)
                ind_names = [x[0] for x in best_inds[:4]]  # İlk 4 tanesini göster
                logger.info(f"  {tf}: Composite sinyal → {', '.join(ind_names)}...")
                
                # Composite backtest yap
                result = self.backtester.run_composite_backtest(
                    df=df,
                    indicator_scores=scores,
                    timeframe=tf,
                    threshold=0.3  # Sinyal eşiği
                )
                
                self.backtest_results.append(result)
                
                logger.info(f"  {tf}: Sharpe={result.sharpe_ratio:.2f} | "
                           f"WR={result.win_rate:.1f}% | "
                           f"DD={result.max_drawdown:.1f}%")
            
            if not self.backtest_results:
                logger.error("  Hiçbir TF için backtest yapılamadı!")
                return False
            
            # En iyi timeframe'i seç
            self.timeframe_ranking = self.backtester.select_best_timeframe(
                self.backtest_results
            )
            
            logger.info(f"\n  🏆 En iyi timeframe: {self.timeframe_ranking.best_timeframe}")
            logger.info(f"  📊 Piyasa rejimi: {self.timeframe_ranking.market_regime}")
            logger.info(f"  🎯 Güven: {self.timeframe_ranking.confidence:.0f}/100")
            
            return True
            
        except Exception as e:
            logger.error(f"  Backtest hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =========================================================================
    # ADIM 5: RAPOR OLUŞTURMA
    # =========================================================================
    
    def generate_report(self) -> AnalysisReport:
        """
        Analiz raporu oluşturur.
        
        Returns:
        -------
        AnalysisReport
            Telegram'a gönderilecek rapor
        """
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 5: RAPOR OLUŞTURMA")
        logger.info("=" * 60)
        
        # En iyi TF'nin backtest sonucu
        best_tf = self.timeframe_ranking.best_timeframe
        best_result = next(
            (r for r in self.backtest_results if r.timeframe == best_tf),
            None
        )
        
        # Sinyal yönü belirleme
        direction = self._determine_direction(best_tf)
        
        # Aktif indikatörler (kategorilere göre)
        active_indicators = self._get_active_indicators(best_tf)
        
        # Risk metrikleri
        risk_metrics = {}
        if best_result:
            risk_metrics = {
                'sharpe': best_result.sharpe_ratio,
                'max_dd': best_result.max_drawdown,
                'win_rate': best_result.win_rate
            }
        
        # Notlar
        notes = self._generate_notes(best_result)
        
        # Rapor oluştur
        report = AnalysisReport(
            symbol=self.config.SYMBOL,
            price=self.current_price,
            recommended_timeframe=best_tf,
            market_regime=self.timeframe_ranking.market_regime,
            direction=direction,
            confidence_score=self.timeframe_ranking.confidence,
            active_indicators=active_indicators,
            risk_metrics=risk_metrics,
            notes=notes
        )
        
        logger.info(f"  ✓ Rapor oluşturuldu")
        logger.info(f"  📊 TF: {best_tf} | Yön: {direction} | Güven: {report.confidence_score:.0f}")
        
        return report
    
    def _determine_direction(self, timeframe: str) -> str:
        """Sinyal yönünü belirler."""
        
        if timeframe not in self.indicator_scores:
            return "NEUTRAL"
        
        scores = self.indicator_scores[timeframe]
        
        # Trend kategorisindeki anlamlı indikatörlerin IC ortalaması
        trend_scores = [s for s in scores if s.category == 'trend' and s.is_significant]
        
        if not trend_scores:
            # Tüm anlamlı indikatörlerin IC ortalaması
            significant_scores = [s for s in scores if s.is_significant]
            if significant_scores:
                avg_ic = np.mean([s.ic_mean for s in significant_scores])
            else:
                return "NEUTRAL"
        else:
            avg_ic = np.mean([s.ic_mean for s in trend_scores])
        
        # IC > 0.05: LONG, IC < -0.05: SHORT, else NEUTRAL
        if avg_ic > 0.05:
            return "LONG"
        elif avg_ic < -0.05:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def _get_active_indicators(self, timeframe: str) -> Dict[str, List[str]]:
        """
        Aktif indikatörleri kategorilere göre döndürür.
        Her kategoriden en yüksek IC'ye sahip max 2 indikatör.
        """
        
        active = {}
        
        if timeframe not in self.indicator_scores:
            return active
        
        scores = self.indicator_scores[timeframe]
        
        # Sadece ana kategoriler (other hariç)
        valid_categories = ['trend', 'momentum', 'volatility', 'volume']
        
        # Her kategori için skorları grupla
        category_scores = {cat: [] for cat in valid_categories}
        
        for score in scores:
            # Kategori kontrolü
            cat = score.category.lower() if score.category else 'other'
            
            # Sadece valid kategorileri al
            if cat not in valid_categories:
                continue
            
            # Anlamlı IC kontrolü
            if abs(score.ic_mean) > 0.02 and not np.isnan(score.ic_mean):
                category_scores[cat].append(score)
        
        # Her kategoriden en iyi 2'yi seç (|IC| bazında)
        for cat in valid_categories:
            if category_scores[cat]:
                # IC mutlak değerine göre sırala
                sorted_scores = sorted(
                    category_scores[cat], 
                    key=lambda x: abs(x.ic_mean), 
                    reverse=True
                )
                # Max 2 indikatör
                active[cat] = [s.name for s in sorted_scores[:2]]
        
        return active
    
    def _generate_notes(self, result: BacktestResult) -> str:
        """Uyarı notları oluşturur."""
        
        notes = []
        
        if result:
            if result.sharpe_ratio < 0:
                notes.append("⚠️ Negatif Sharpe - dikkatli olun")
            if result.max_drawdown < -15:
                notes.append("⚠️ Yüksek drawdown riski")
            if result.win_rate < 50:
                notes.append("⚠️ Düşük win rate")
        
        if self.timeframe_ranking:
            if self.timeframe_ranking.market_regime == 'volatile':
                notes.append("⚡ Yüksek volatilite - pozisyon boyutunu küçült")
            elif self.timeframe_ranking.market_regime == 'transitioning':
                notes.append("🔄 Geçiş dönemi - net trend yok")
        
        return " | ".join(notes) if notes else ""
    
    # =========================================================================
    # ADIM 6: TELEGRAM BİLDİRİMİ
    # =========================================================================
    
    def send_notification(self, report: AnalysisReport) -> bool:
        """Telegram bildirimi (Metin + Grafik) gönderir."""
        logger.info("\n" + "=" * 60)
        logger.info("ADIM 6: TELEGRAM BİLDİRİMİ")
        logger.info("=" * 60)
        
        if not self.config.TELEGRAM_ENABLED:
            logger.info("  Telegram devre dışı")
            return True
        
        if not self.notifier.is_configured():
            logger.warning("  Telegram yapılandırılmamış (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
            return True
        
        try:
            # 1. Önce Metin Raporunu Gönder
            self.notifier.send_report_sync(report)
            
            # 2. Grafiği Oluştur ve Gönder
            tf = report.recommended_timeframe
            if tf in self.data_dict:
                df = self.data_dict[tf]
                
                # Sadece aktif indikatör isimlerini düz liste yap
                flat_indicators = {}
                for cat, inds in report.active_indicators.items():
                    flat_indicators[cat] = inds

                # Grafiği çiz
                logger.info(f"  📊 {tf} için grafik oluşturuluyor...")
                
                # Plotter ile resmi oluştur
                image_buf = self.plotter.create_analysis_chart(
                    df, 
                    report.symbol, 
                    tf, 
                    flat_indicators
                )
                
                # Resmi gönder (Yeni senkron metod ile)
                self.notifier.send_chart_sync(
                    photo_file=image_buf, 
                    caption=f"📊 {report.symbol} - {tf} Grafik Analizi"
                )
                logger.info("  📸 Grafik gönderildi")

            return True
            
        except Exception as e:
            logger.error(f"  ✗ Bildirim hatası: {e}")
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
        logger.info(f"🚀 BTC DECISION SYSTEM - ANALİZ BAŞLADI")
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
            
            # Adım 4: Backtest
            if not self.run_backtests():
                logger.error("Backtest başarısız!")
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
        print("DETAYLI ÖZET")
        print("=" * 70)
        
        # Backtest sonuçları tablosu
        if self.backtest_results:
            print("\n📊 TIMEFRAME KARŞILAŞTIRMA:")
            summary = self.backtester.get_summary_table(self.backtest_results)
            print(summary.to_string(index=False))
        
        # Timeframe sıralaması
        if self.timeframe_ranking:
            print("\n🏆 TIMEFRAME SIRALAMASSI:")
            for tf, score in self.timeframe_ranking.rankings:
                marker = "→" if tf == self.timeframe_ranking.best_timeframe else " "
                print(f"  {marker} {tf}: {score:.1f} puan")
        
        # Öneri
        if self.timeframe_ranking:
            print("\n" + self.timeframe_ranking.recommendation)


# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler(system: BTCDecisionSystem, interval_minutes: int = 60):
    """
    Belirtilen aralıkla analizi tekrarlar.
    
    Parameters:
    ----------
    system : BTCDecisionSystem
        Analiz sistemi
    interval_minutes : int
        Çalışma aralığı (dakika)
    """
    logger.info(f"Scheduler başlatıldı - Her {interval_minutes} dakikada bir çalışacak")
    
    while True:
        try:
            # Analizi çalıştır
            system.run_analysis()
            system.print_summary()
            
            # Bir sonraki çalışmaya kadar bekle
            next_run = datetime.now() + timedelta(minutes=interval_minutes)
            logger.info(f"\n⏰ Sonraki çalışma: {next_run.strftime('%H:%M:%S')}")
            
            # Saat başına hizala (opsiyonel)
            # wait_seconds = (60 - datetime.now().minute) * 60 - datetime.now().second
            wait_seconds = interval_minutes * 60
            
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            logger.info("\nScheduler durduruldu (Ctrl+C)")
            break
        except Exception as e:
            logger.exception(f"Scheduler hatası: {e}")
            time.sleep(60)  # Hata durumunda 1 dakika bekle


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================

# src/main.py dosyasının en altındaki main fonksiyonunu bununla değiştir:

def main():
    """Ana giriş noktası."""
    
    parser = argparse.ArgumentParser(
        description='Kripto Dinamik Karar Destek Sistemi', # İsmi güncelledik
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --symbol ETH/USDT  # ETH analizi yap
  python main.py --symbol SOL/USDT --no-telegram # SOL analizi yap, bildirim gönderme
  python main.py --schedule --interval 30 # Varsayılan (BTC) ile 30 dk'da bir çalış
        """
    )
    
    # YENİ EKLENEN KISIM: Sembol argümanı
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='BTC/USDT',
        help='Analiz edilecek işlem çifti (Örn: ETH/USDT, SOL/USDT)'
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
    
    args = parser.parse_args()
    
    # Yapılandırma
    config = Config()
    
    # YENİ EKLENEN KISIM: Config'i argümanla güncelleme
    # Kullanıcı terminalden ne girdiyse (örn: ETH/USDT), config'i eziyoruz.
    config.SYMBOL = args.symbol.upper() 
    
    if args.no_telegram:
        config.TELEGRAM_ENABLED = False
    
    # Sistem oluştur
    system = BTCDecisionSystem(config=config, verbose=True)
    
    if args.schedule:
        # Sürekli çalışma modu
        run_scheduler(system, interval_minutes=args.interval)
    else:
        # Tek seferlik çalışma
        report = system.run_analysis()
        if report:
            system.print_summary()


if __name__ == "__main__":
    main()

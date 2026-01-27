# =============================================================================
# VERİ ÇEKME MODÜLÜ (DATA FETCHER) - v2.0
# =============================================================================
# Amaç: CCXT kütüphanesi ile kripto borsalarından OHLCV verisi çekmek
# Güncelleme: Tüm timeframe'ler + maksimum veri çekme desteği
# 
# İstatistiksel Not: Daha fazla veri = daha güvenilir backtest sonuçları
# Ancak çok eski veri piyasa rejim değişikliklerini içerebilir (non-stationarity)
# =============================================================================

import ccxt                                    # Kripto borsa API'leri için unified interface
import pandas as pd                            # Zaman serisi veri yapısı
import numpy as np                             # Sayısal hesaplamalar
from datetime import datetime, timedelta, timezone  # Zaman işlemleri (timezone-aware)
from typing import Optional, List, Dict, Tuple      # Tip belirteçleri (type hints)
import time                                    # Rate limiting için bekleme
from pathlib import Path                       # Dosya yolu işlemleri


class DataFetcher:
    """
    Kripto borsalarından OHLCV verisi çeken sınıf.
    
    Güncelleme v2.0:
    - Tüm timeframe'ler eklendi (5m, 30m, 2h dahil)
    - Maksimum veri çekme (Binance limitine kadar)
    - Veri kaydetme/yükleme fonksiyonları
    - Gelişmiş hata yönetimi
    
    İstatistiksel Önem:
    - Backtest için yeterli veri: minimum 1000 bar önerilir
    - Walk-forward validation için: minimum 30 günlük out-of-sample
    - Rejim değişikliği riski: 6 aydan eski veriye dikkat
    """
    
    # -------------------------------------------------------------------------
    # TÜM DESTEKLENEN ZAMAN DİLİMLERİ
    # -------------------------------------------------------------------------
    # Binance'in desteklediği tüm timeframe'ler
    # Dakika cinsinden karşılıkları (volatilite ölçekleme için gerekli)
    
    TIMEFRAME_MINUTES: Dict[str, int] = {
        "1m": 1,          # 1 dakika   - Scalping, çok gürültülü
        "3m": 3,          # 3 dakika   - Kısa vadeli scalping
        "5m": 5,          # 5 dakika   - Kısa vadeli trading ⭐ YENİ
        "15m": 15,        # 15 dakika  - Day trading standardı
        "30m": 30,        # 30 dakika  - Orta-kısa vade ⭐ YENİ
        "1h": 60,         # 1 saat     - Day trading / Swing
        "2h": 120,        # 2 saat     - Orta vade ⭐ YENİ
        "4h": 240,        # 4 saat     - Swing trading için ideal
        "6h": 360,        # 6 saat     - Orta-uzun vade
        "8h": 480,        # 8 saat     - Pozisyon trading
        "12h": 720,       # 12 saat    - Pozisyon trading
        "1d": 1440,       # 1 gün      - Pozisyon / HODLing
        "3d": 4320,       # 3 gün      - Uzun vade
        "1w": 10080,      # 1 hafta    - Uzun vade trend
    }
    
    # -------------------------------------------------------------------------
    # BİNANCE VERİ LİMİTLERİ
    # -------------------------------------------------------------------------
    # Binance API limitleri ve önerilen çekme stratejisi
    
    BINANCE_LIMITS = {
        'max_candles_per_request': 1000,      # Tek istekte maksimum mum
        'rate_limit_per_minute': 1200,        # Dakikada maksimum istek
        'recommended_delay': 0.1,             # İstekler arası bekleme (saniye)
    }
    
    # -------------------------------------------------------------------------
    # HER TIMEFRAME İÇİN ÖNERİLEN VERİ MİKTARI (Day Trading Optimize)
    # -------------------------------------------------------------------------
    # Day trading için daha fazla veri = daha güvenilir backtest
    # Kısa timeframe'lerde gürültü fazla, bu yüzden daha çok sample gerekli
    
    RECOMMENDED_BARS: Dict[str, int] = {
        "1m": 10000,      # ~7 gün (scalping analizi için)
        "3m": 7000,       # ~14 gün
        "5m": 5000,       # ~17 gün ⭐ Day trading kısa vade
        "15m": 4000,      # ~42 gün (~6 hafta) ⭐ Day trading ana TF
        "30m": 3000,      # ~62 gün (~2 ay) ⭐ Trend konfirmasyonu
        "1h": 2000,       # ~83 gün (~3 ay) ⭐ İntraday trend
        "2h": 1500,       # ~125 gün (~4 ay) ⭐ Swing noktaları
        "4h": 1000,       # ~166 gün (~5.5 ay) ⭐ Büyük resim
        "6h": 750,        # ~187 gün
        "8h": 600,        # ~200 gün
        "12h": 500,       # ~250 gün
        "1d": 365,        # 1 yıl
        "3d": 200,        # ~600 gün
        "1w": 104,        # 2 yıl
    }
    
    # -------------------------------------------------------------------------
    # AKTİF TİMEFRAME'LER (Day Trading için optimize edilmiş)
    # -------------------------------------------------------------------------
    # Not: Binance 10m desteklemiyor, en yakın alternatifler kullanıldı
    # Multi-resolution analiz: Kısa (5m-15m), Orta (30m-1h), Uzun (2h-4h)
    
    ACTIVE_TIMEFRAMES: List[str] = [
        "5m",             # Kısa vade - Entry/Exit timing, scalping
        "15m",            # Kısa vade - Day trading ana timeframe
        "30m",            # Orta vade - Trend konfirmasyonu
        "1h",             # Orta vade - İntraday trend yapısı
        "2h",             # Uzun vade - Swing noktaları
        "4h",             # Uzun vade - Büyük resim, major S/R
    ]
    
    def __init__(
        self,
        exchange_id: str = "binance",         # Varsayılan borsa
        symbol: str = "BTC/USDT",             # Varsayılan işlem çifti
        sandbox: bool = False                  # Test modu
    ):
        """
        DataFetcher sınıfını başlatır.
        
        Parametreler:
        ------------
        exchange_id : str
            CCXT borsa ID'si (binance, bybit, okx, vb.)
            
        symbol : str
            İşlem çifti (BTC/USDT, ETH/USDT, vb.)
            
        sandbox : bool
            True ise test ortamı kullanılır
        """
        
        # Borsa nesnesini oluştur
        self.exchange = getattr(ccxt, exchange_id)({
            'sandbox': sandbox,
            'enableRateLimit': True,          # Otomatik rate limiting
            'options': {
                'defaultType': 'spot',
            }
        })
        
        self.symbol = symbol
        self.exchange_id = exchange_id
        
        # Market bilgilerini yükle
        self._load_markets()
    
    def _load_markets(self) -> None:
        """Borsa market bilgilerini yükler."""
        try:
            self.exchange.load_markets()
            print(f"✓ {self.exchange_id.upper()} borsası bağlantısı başarılı")
            print(f"  Toplam {len(self.exchange.markets)} market mevcut")
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Ağ hatası: {e}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"Borsa hatası: {e}")
    
    def fetch_ohlcv(
        self,
        timeframe: str = "1h",
        limit: int = 1000,                    # Binance max: 1000
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Belirtilen timeframe için OHLCV verisi çeker.
        
        Parametreler:
        ------------
        timeframe : str
            Zaman dilimi (TIMEFRAME_MINUTES'daki değerlerden biri)
            
        limit : int
            Çekilecek maksimum bar sayısı (Binance max: 1000)
            
        since : int, optional
            Başlangıç zamanı (Unix timestamp, milisaniye)
        
        Döndürür:
        --------
        pd.DataFrame
            Kolonlar: timestamp(index), open, high, low, close, volume
        """
        
        # Timeframe geçerliliğini kontrol et
        if timeframe not in self.TIMEFRAME_MINUTES:
            valid_tfs = list(self.TIMEFRAME_MINUTES.keys())
            raise ValueError(f"Geçersiz timeframe: {timeframe}. Geçerli: {valid_tfs}")
        
        # Sembol kontrolü
        if self.symbol not in self.exchange.markets:
            raise ValueError(f"{self.symbol} bu borsada mevcut değil")
        
        # Limit kontrolü (Binance max 1000)
        limit = min(limit, self.BINANCE_LIMITS['max_candles_per_request'])
        
        try:
            ohlcv_raw = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )
            
            if not ohlcv_raw:
                raise ValueError(f"{self.symbol} için veri bulunamadı")
            
            # DataFrame oluştur
            df = pd.DataFrame(
                ohlcv_raw,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Timestamp'i timezone-aware datetime'a çevir
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = None
            
            # Veri tiplerini optimize et
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            return df
            
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Ağ hatası (veri çekme): {e}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"Borsa hatası (veri çekme): {e}")
    
    def fetch_max_ohlcv(
        self,
        timeframe: str = "1h",
        max_bars: Optional[int] = None,       # None = önerilen miktar
        progress: bool = True                  # İlerleme göster
    ) -> pd.DataFrame:
        """
        Belirtilen timeframe için MAKSİMUM veri çeker.
        
        Binance'in 1000 bar limitini GERİYE DOĞRU pagination ile aşar.
        Rate limiting otomatik uygulanır.
        
        Parametreler:
        ------------
        timeframe : str
            Zaman dilimi
            
        max_bars : int, optional
            Çekilecek maksimum bar sayısı
            None ise RECOMMENDED_BARS değeri kullanılır
            
        progress : bool
            True ise ilerleme durumu yazdırılır
        
        Döndürür:
        --------
        pd.DataFrame
            Birleştirilmiş OHLCV DataFrame
        
        İstatistiksel Not:
        -----------------
        Daha fazla veri:
        + Daha güvenilir backtest (larger sample size)
        + Daha iyi out-of-sample validation
        - Piyasa rejim değişikliği riski (non-stationarity)
        - İşlem maliyeti yapısı değişmiş olabilir
        
        Öneri: 3-6 ay veri optimal trade-off sağlar
        """
        
        # Hedef bar sayısını belirle
        if max_bars is None:
            max_bars = self.RECOMMENDED_BARS.get(timeframe, 1000)
        
        # Timeframe dakika değeri
        tf_minutes = self.TIMEFRAME_MINUTES[timeframe]
        
        # Tahmini gün sayısı
        estimated_days = (max_bars * tf_minutes) / (60 * 24)
        
        if progress:
            print(f"\n📊 {self.symbol} | {timeframe} | Hedef: {max_bars} bar (~{estimated_days:.1f} gün)")
        
        # =====================================================================
        # GERİYE DOĞRU PAGİNATION STRATEJİSİ
        # =====================================================================
        # Binance'de 'since' = "bu tarihten SONRA" demek
        # Bu yüzden geçmiş tarihten başlayıp ileri doğru çekiyoruz
        # =====================================================================
        
        # Başlangıç tarihini hesapla (şu an - tahmini süre - buffer)
        # Buffer: Hafta sonları/tatiller için ekstra %20
        buffer_factor = 1.2
        start_time = datetime.now(timezone.utc) - timedelta(minutes=int(max_bars * tf_minutes * buffer_factor))
        since_ms = int(start_time.timestamp() * 1000)
        
        all_data: List[pd.DataFrame] = []
        total_fetched = 0
        chunk_size = self.BINANCE_LIMITS['max_candles_per_request']
        current_since = since_ms
        
        while total_fetched < max_bars:
            # Kalan bar sayısı
            remaining = max_bars - total_fetched
            fetch_limit = min(chunk_size, remaining)
            
            try:
                # Chunk çek (since parametresi ile geçmişten başla)
                df_chunk = self.fetch_ohlcv(
                    timeframe=timeframe,
                    limit=fetch_limit,
                    since=current_since
                )
                
                if df_chunk.empty:
                    if progress:
                        print(f"   ⚠ Veri sonu (toplam: {total_fetched})")
                    break
                
                all_data.append(df_chunk)
                total_fetched += len(df_chunk)
                
                # Sonraki chunk için: SON bar'ın timestamp'i + 1ms
                # İLERİ DOĞRU gidiyoruz (geçmişten şu ana)
                last_ts = df_chunk.index[-1]
                current_since = int(last_ts.timestamp() * 1000) + 1
                
                if progress:
                    print(f"   → {total_fetched}/{max_bars} bar çekildi ({100*total_fetched/max_bars:.1f}%)")
                
                # Eğer beklenen miktardan az geldiyse, daha fazla veri yok
                if len(df_chunk) < fetch_limit:
                    if progress:
                        print(f"   ✓ Veri sonu ulaşıldı")
                    break
                
                # Şu ana ulaştıysak dur
                if last_ts >= datetime.now(timezone.utc) - timedelta(minutes=tf_minutes):
                    if progress:
                        print(f"   ✓ Güncel veriye ulaşıldı")
                    break
                
                # Rate limiting
                time.sleep(self.BINANCE_LIMITS['recommended_delay'])
                
            except Exception as e:
                if progress:
                    print(f"   ⚠ Hata (devam ediliyor): {e}")
                # Hata durumunda kısa bekleme ve devam
                time.sleep(0.5)
                break
        
        if not all_data:
            raise ValueError(f"{timeframe} için veri çekilemedi")
        
        # Tüm chunk'ları birleştir
        df_combined = pd.concat(all_data)
        
        # Duplicate'leri kaldır
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        
        # Kronolojik sırala
        df_combined = df_combined.sort_index()
        
        # İstenen bar sayısına kırp (fazla çekmiş olabiliriz)
        if len(df_combined) > max_bars:
            df_combined = df_combined.tail(max_bars)
        
        if progress:
            actual_days = (df_combined.index[-1] - df_combined.index[0]).days
            print(f"   ✓ Toplam: {len(df_combined)} bar | {actual_days} gün | "
                  f"{df_combined.index[0].strftime('%Y-%m-%d')} → {df_combined.index[-1].strftime('%Y-%m-%d')}")
        
        return df_combined
    
    def fetch_all_timeframes(
        self,
        timeframes: Optional[List[str]] = None,   # None = ACTIVE_TIMEFRAMES
        max_bars_override: Optional[int] = None,  # Her timeframe için aynı bar sayısı
        save_to_disk: bool = False,               # CSV olarak kaydet
        data_dir: str = "data"                    # Kayıt klasörü
    ) -> Dict[str, pd.DataFrame]:
        """
        TÜM aktif timeframe'ler için veri çeker.
        
        Parametreler:
        ------------
        timeframes : List[str], optional
            Çekilecek timeframe listesi
            None ise ACTIVE_TIMEFRAMES kullanılır
            
        max_bars_override : int, optional
            Her timeframe için sabit bar sayısı
            None ise RECOMMENDED_BARS kullanılır
            
        save_to_disk : bool
            True ise veriler CSV olarak kaydedilir
            
        data_dir : str
            Kayıt klasörü yolu
        
        Döndürür:
        --------
        Dict[str, pd.DataFrame]
            Anahtar: timeframe, Değer: OHLCV DataFrame
        
        Kullanım:
        --------
        >>> fetcher = DataFetcher()
        >>> all_data = fetcher.fetch_all_timeframes()
        >>> print(all_data.keys())  # dict_keys(['5m', '15m', '30m', '1h', '2h', '4h', '1d'])
        """
        
        # Timeframe listesi
        if timeframes is None:
            timeframes = self.ACTIVE_TIMEFRAMES
        
        print("=" * 60)
        print(f"📥 TÜM TIMEFRAME'LER İÇİN VERİ ÇEKİLİYOR")
        print(f"   Symbol: {self.symbol}")
        print(f"   Timeframe'ler: {timeframes}")
        print("=" * 60)
        
        data_dict: Dict[str, pd.DataFrame] = {}
        
        for tf in timeframes:
            try:
                # Bar sayısını belirle
                bars = max_bars_override if max_bars_override else self.RECOMMENDED_BARS.get(tf, 1000)
                
                # Veri çek
                df = self.fetch_max_ohlcv(
                    timeframe=tf,
                    max_bars=bars,
                    progress=True
                )
                
                data_dict[tf] = df
                
                # Disk'e kaydet (opsiyonel)
                if save_to_disk:
                    self._save_to_csv(df, tf, data_dir)
                
            except Exception as e:
                print(f"\n   ✗ {tf} için hata: {e}")
                continue
            
            # Timeframe'ler arası bekleme
            time.sleep(0.5)
        
        # Özet tablo
        self._print_summary(data_dict)
        
        return data_dict
    
    def _save_to_csv(
        self,
        df: pd.DataFrame,
        timeframe: str,
        data_dir: str
    ) -> None:
        """Veriyi CSV dosyasına kaydeder."""
        
        # Klasör oluştur
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Dosya adı: BTC_USDT_1h_20240125.csv
        symbol_clean = self.symbol.replace("/", "_")
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{symbol_clean}_{timeframe}_{date_str}.csv"
        filepath = Path(data_dir) / filename
        
        df.to_csv(filepath)
        print(f"   💾 Kaydedildi: {filepath}")
    
    def _print_summary(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Çekilen verilerin özetini yazdırır."""
        
        print("\n" + "=" * 60)
        print("📊 VERİ ÇEKİM ÖZETİ")
        print("=" * 60)
        print(f"{'Timeframe':<10} {'Bars':<10} {'Başlangıç':<12} {'Bitiş':<12} {'Gün':<6}")
        print("-" * 60)
        
        for tf, df in data_dict.items():
            start = df.index[0].strftime('%Y-%m-%d')
            end = df.index[-1].strftime('%Y-%m-%d')
            days = (df.index[-1] - df.index[0]).days
            print(f"{tf:<10} {len(df):<10} {start:<12} {end:<12} {days:<6}")
        
        print("=" * 60)
    
    def fetch_multi_timeframe(
        self,
        timeframes: List[str] = ["15m", "1h", "4h"],
        limit: int = 500,
        delay: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Birden fazla timeframe için OHLCV verisi çeker (basit versiyon).
        Geriye uyumluluk için korunmuştur.
        """
        
        data_dict: Dict[str, pd.DataFrame] = {}
        
        for tf in timeframes:
            print(f"  → {tf} verisi çekiliyor...", end=" ")
            
            try:
                df = self.fetch_ohlcv(timeframe=tf, limit=limit)
                data_dict[tf] = df
                print(f"✓ ({len(df)} bar)")
            except Exception as e:
                print(f"✗ Hata: {e}")
                continue
            
            time.sleep(delay)
        
        return data_dict
    
    def fetch_historical(
        self,
        timeframe: str = "1h",
        days: int = 30,
        chunk_size: int = 1000
    ) -> pd.DataFrame:
        """
        Belirtilen gün sayısı kadar geçmiş veriyi çeker.
        Geriye uyumluluk için korunmuştur.
        """
        
        tf_minutes = self.TIMEFRAME_MINUTES[timeframe]
        total_bars_needed = int((days * 24 * 60) / tf_minutes)
        
        return self.fetch_max_ohlcv(
            timeframe=timeframe,
            max_bars=total_bars_needed,
            progress=True
        )
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Çekilen verinin kalitesini doğrular.
        
        Kontroller:
        1. Missing values
        2. OHLC tutarlılığı (High >= Open/Close, Low <= Open/Close)
        3. Volume anomalileri
        4. Zaman sürekliliği
        
        Döndürür:
        --------
        Dict[str, any]
            Veri kalite metrikleri
        """
        
        validation_results = {}
        
        # 1. Toplam satır sayısı
        validation_results['total_rows'] = len(df)
        
        # 2. Missing value kontrolü
        missing_counts = df.isnull().sum().to_dict()
        validation_results['missing_values'] = missing_counts
        validation_results['has_missing'] = any(v > 0 for v in missing_counts.values())
        
        # 3. OHLC tutarlılık kontrolü
        high_valid = (df['high'] >= df['open']) & (df['high'] >= df['close'])
        low_valid = (df['low'] <= df['open']) & (df['low'] <= df['close'])
        ohlc_invalid_count = (~high_valid | ~low_valid).sum()
        validation_results['ohlc_invalid_rows'] = int(ohlc_invalid_count)
        
        # 4. Volume kontrolü
        zero_volume = (df['volume'] == 0).sum()
        negative_volume = (df['volume'] < 0).sum()
        validation_results['zero_volume_rows'] = int(zero_volume)
        validation_results['negative_volume_rows'] = int(negative_volume)
        
        # 5. Zaman aralığı
        validation_results['start_date'] = df.index.min().strftime('%Y-%m-%d %H:%M')
        validation_results['end_date'] = df.index.max().strftime('%Y-%m-%d %H:%M')
        
        # 6. Temel istatistikler
        validation_results['price_range'] = {
            'min': float(df['low'].min()),
            'max': float(df['high'].max()),
            'last': float(df['close'].iloc[-1])
        }
        
        # 7. Volume istatistikleri
        validation_results['volume_stats'] = {
            'mean': float(df['volume'].mean()),
            'median': float(df['volume'].median()),
            'std': float(df['volume'].std())
        }
        
        # 8. Gap analizi (eksik bar tespiti)
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            expected_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            gaps = (time_diffs > expected_diff * 1.5).sum()
            validation_results['detected_gaps'] = int(gaps)
        else:
            validation_results['detected_gaps'] = 0
        
        # Genel geçerlilik
        validation_results['is_valid'] = (
            not validation_results['has_missing'] and
            ohlc_invalid_count == 0 and
            negative_volume == 0
        )
        
        return validation_results
    
    def get_latest_price(self) -> Dict[str, float]:
        """Güncel fiyat bilgisini çeker (ticker)."""
        
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            return {
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
        except Exception as e:
            raise ValueError(f"Ticker çekme hatası: {e}")
    
    def get_available_timeframes(self) -> List[str]:
        """Kullanılabilir tüm timeframe'leri döndürür."""
        return list(self.TIMEFRAME_MINUTES.keys())
    
    def get_active_timeframes(self) -> List[str]:
        """Aktif (analiz için kullanılan) timeframe'leri döndürür."""
        return self.ACTIVE_TIMEFRAMES.copy()


# =============================================================================
# MODÜL TEST KODU
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("DATA FETCHER v2.0 TEST")
    print("=" * 70)
    
    # DataFetcher örneği oluştur
    fetcher = DataFetcher(
        exchange_id="binance",
        symbol="BTC/USDT"
    )
    
    # 1. Kullanılabilir timeframe'leri göster
    print("\n[1] Kullanılabilir Timeframe'ler:")
    print(f"   Tümü: {fetcher.get_available_timeframes()}")
    print(f"   Aktif: {fetcher.get_active_timeframes()}")
    
    # 2. Güncel fiyat
    print("\n[2] Güncel Fiyat:")
    price = fetcher.get_latest_price()
    print(f"   BTC/USDT: ${price['last']:,.2f}")
    print(f"   24h Değişim: {price['change_24h']:.2f}%")
    
    # 3. Tek timeframe maksimum veri çekme testi
    print("\n[3] Tek Timeframe Maksimum Veri (1h):")
    df_1h = fetcher.fetch_max_ohlcv(timeframe="1h", max_bars=500)
    print(f"   Son 5 bar:\n{df_1h.tail()}")
    
    # 4. Veri doğrulama
    print("\n[4] Veri Doğrulama:")
    validation = fetcher.validate_data(df_1h)
    print(f"   Toplam: {validation['total_rows']} bar")
    print(f"   Geçerli: {validation['is_valid']}")
    print(f"   Gap sayısı: {validation['detected_gaps']}")
    
    # 5. Tüm aktif timeframe'ler için veri çekme (küçük miktar - test için)
    print("\n[5] Day Trading Timeframe'leri (test - 100 bar):")
    all_data = fetcher.fetch_all_timeframes(
        timeframes=["5m", "15m", "30m", "1h", "2h", "4h"],  # Day trading TF'ler
        max_bars_override=100,            # Küçük miktar (test için)
        save_to_disk=False
    )
    
    print("\n" + "=" * 70)
    print("TÜM TESTLER TAMAMLANDI")
    print("=" * 70)

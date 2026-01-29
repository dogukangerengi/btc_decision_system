# =============================================================================
# TELEGRAM BİLDİRİM MODÜLÜ
# =============================================================================
# Amaç: Analiz sonuçlarını Telegram üzerinden bildirmek
#
# Özellikler:
# - Async messaging (python-telegram-bot v20+)
# - Rate limiting (Telegram API limitleri)
# - Formatlı mesajlar (HTML)
# - Hata yönetimi ve retry mekanizması
#
# Kurulum:
# 1. @BotFather'dan bot oluştur, token al
# 2. Bot'u gruba ekle veya direkt mesaj at
# 3. Chat ID'yi öğren
# 4. .env dosyasına TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ekle
# =============================================================================

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
import re

# python-telegram-bot v20+ async API kullanıyor
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter

# Logging ayarları
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """
    Telegram'a gönderilecek analiz raporu.
    
    Attributes:
    ----------
    symbol : str
        İşlem çifti (örn: BTC/USDT)
    price : float
        Güncel fiyat
    recommended_timeframe : str
        Önerilen zaman dilimi
    market_regime : str
        Piyasa rejimi (trending_up, trending_down, ranging, volatile)
    direction : str
        Sinyal yönü (LONG, SHORT, NEUTRAL)
    confidence_score : float
        Güven skoru (0-100)
    active_indicators : Dict[str, List[str]]
        Aktif indikatörler (kategori → indikatör listesi)
    risk_metrics : Dict[str, float]
        Risk metrikleri (Sharpe, MaxDD, vb.)
    timestamp : datetime
        Analiz zamanı
    notes : str
        Ek notlar
    """
    symbol: str
    price: float
    recommended_timeframe: str
    market_regime: str
    direction: str
    confidence_score: float
    active_indicators: Dict[str, List[str]]
    risk_metrics: Dict[str, float]
    timestamp: datetime = None
    notes: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramNotifier:
    """
    Telegram bildirim gönderen sınıf.
    
    Kullanım:
    --------
    notifier = TelegramNotifier(token="...", chat_id="...")
    await notifier.send_analysis_report(report)
    
    veya senkron:
    notifier.send_report_sync(report)
    """
    
    # Telegram API rate limitleri
    MAX_MESSAGE_LENGTH = 4096          # Maksimum mesaj uzunluğu
    RATE_LIMIT_DELAY = 1.0             # İstekler arası minimum bekleme (saniye)
    MAX_RETRIES = 3                    # Maksimum yeniden deneme
    
    # Emoji mapping
    REGIME_EMOJI = {
        'trending_up': '📈',
        'trending_down': '📉',
        'ranging': '↔️',
        'volatile': '⚡',
        'transitioning': '🔄',
        'unknown': '❓'
    }
    
    DIRECTION_EMOJI = {
        'LONG': '🟢',
        'SHORT': '🔴',
        'NEUTRAL': '⚪'
    }
    
    # İndikatör isim dönüşüm tablosu (TradingView uyumlu)
    INDICATOR_NAMES = {
        # Trend
        'SUPERTs_10_3.0': 'Supertrend',
        'SUPERTl_10_3.0': 'Supertrend',
        'SUPERTd_10_3.0': 'Supertrend',
        'EMA_12': 'EMA (12)',
        'EMA_20': 'EMA (20)',
        'EMA_26': 'EMA (26)',
        'EMA_50': 'EMA (50)',
        'SMA_20': 'SMA (20)',
        'SMA_50': 'SMA (50)',
        'SMA_200': 'SMA (200)',
        'TEMA_20': 'TEMA (20)',
        'DEMA_20': 'DEMA (20)',
        'WMA_20': 'WMA (20)',
        'HMA_20': 'Hull MA (20)',
        'KAMA_20': 'KAMA (20)',
        'ADX_14': 'ADX (14)',
        'DMP_14': 'DI+ (14)',
        'DMN_14': 'DI- (14)',
        'PSARl_0.02_0.2': 'Parabolic SAR',
        'PSARs_0.02_0.2': 'Parabolic SAR',
        'AROON_25': 'Aroon (25)',
        'AROONu_25': 'Aroon Up',
        'AROONd_25': 'Aroon Down',
        
        # Momentum
        'RSI_7': 'RSI (7)',
        'RSI_14': 'RSI (14)',
        'RSI_21': 'RSI (21)',
        'MACD_12_26_9': 'MACD',
        'MACDh_12_26_9': 'MACD Histogram',
        'MACDs_12_26_9': 'MACD Signal',
        'STOCHk_14_3_3': 'Stochastic %K',
        'STOCHd_14_3_3': 'Stochastic %D',
        'STOCHRSIk_14_14_3_3': 'StochRSI',
        'STOCHRSId_14_14_3_3': 'StochRSI %D',
        'CCI_20_0.015': 'CCI (20)',
        'WILLR_14': 'Williams %R',
        'MOM_10': 'Momentum (10)',
        'ROC_10': 'ROC (10)',
        'ROC_20': 'ROC (20)',
        'AO_5_34': 'Awesome Osc',
        'PPO_12_26_9': 'PPO',
        'PPOh_12_26_9': 'PPO Histogram',
        'TSI_13_25_13': 'TSI',
        'UO_7_14_28': 'Ultimate Osc',
        
        # Volatility
        'ATRr_14': 'ATR (14)',
        'ATRr_7': 'ATR (7)',
        'NATR_14': 'NATR (14)',
        'BBU_20_2.0': 'BB Upper',
        'BBM_20_2.0': 'BB Middle',
        'BBL_20_2.0': 'BB Lower',
        'BBB_20_2.0': 'BB Width',
        'BBP_20_2.0': 'BB %B',
        'KCUe_20_1.5': 'Keltner Upper',
        'KCBe_20_1.5': 'Keltner Basis',
        'KCLe_20_1.5': 'Keltner Lower',
        'DCU_20_20': 'Donchian Upper',
        'DCM_20_20': 'Donchian Middle',
        'DCL_20_20': 'Donchian Lower',
        
        # Volume
        'OBV': 'OBV',
        'AD': 'A/D Line',
        'PVT': 'Price Volume Trend',
        'MFI_14': 'MFI (14)',
        'CMF_20': 'CMF (20)',
        'ADOSC_3_10': 'Chaikin Osc',
        'EFI_13': 'Elder Force',
        'VWMA_20': 'VWMA (20)',
        
        # Composite
        'ITS_9': 'Ichimoku Tenkan',
        'IKS_26': 'Ichimoku Kijun',
        'ISA_9': 'Ichimoku Span A',
        'ISB_26': 'Ichimoku Span B',
        'SQZ_20_2.0_20_1.5': 'Squeeze Mom',
    }
    
    def _format_indicator_name(self, raw_name: str) -> str:
        """
        Ham indikatör ismini TradingView uyumlu isme çevirir.
        
        Örnek:
        - SUPERTs_10_3.0 → Supertrend
        - CCI_20_0.015 → CCI (20)
        - KCUe_20_1.5 → Keltner Upper
        """
        # Direkt eşleşme varsa kullan
        if raw_name in self.INDICATOR_NAMES:
            return self.INDICATOR_NAMES[raw_name]
        
        # Pattern matching ile dönüşüm dene
        name = raw_name
        
        # Supertrend pattern
        if name.startswith('SUPER'):
            return 'Supertrend'
        
        # EMA/SMA pattern
        for prefix in ['EMA_', 'SMA_', 'WMA_', 'TEMA_', 'DEMA_', 'HMA_', 'KAMA_']:
            if name.startswith(prefix):
                period = name.replace(prefix, '')
                return f"{prefix[:-1]} ({period})"
        
        # RSI pattern
        if name.startswith('RSI_'):
            period = name.replace('RSI_', '')
            return f"RSI ({period})"
        
        # ATR pattern
        if name.startswith('ATR'):
            return 'ATR (14)'
        
        # CCI pattern  
        if name.startswith('CCI_'):
            parts = name.split('_')
            return f"CCI ({parts[1]})" if len(parts) > 1 else 'CCI'
        
        # Stochastic pattern
        if 'STOCH' in name:
            if 'RSI' in name:
                return 'StochRSI'
            return 'Stochastic'
        
        # BB pattern
        if name.startswith('BB'):
            if 'U' in name: return 'BB Upper'
            if 'L' in name: return 'BB Lower'
            if 'M' in name: return 'BB Middle'
            return 'Bollinger Bands'
        
        # KC pattern
        if name.startswith('KC'):
            if 'U' in name: return 'Keltner Upper'
            if 'L' in name: return 'Keltner Lower'
            if 'B' in name: return 'Keltner Basis'
            return 'Keltner Channel'
        
        # DC pattern
        if name.startswith('DC'):
            if 'U' in name: return 'Donchian Upper'
            if 'L' in name: return 'Donchian Lower'
            if 'M' in name: return 'Donchian Middle'
            return 'Donchian'
        
        # MACD pattern
        if name.startswith('MACD'):
            if 'h' in name: return 'MACD Histogram'
            if 's' in name: return 'MACD Signal'
            return 'MACD'
        
        # PSAR pattern
        if name.startswith('PSAR'):
            return 'Parabolic SAR'
        
        # Williams %R
        if name.startswith('WILLR'):
            return 'Williams %R'
        
        # Ichimoku
        if name.startswith('I') and name[1] in ['T', 'K', 'S']:
            patterns = {'ITS': 'Ichimoku Tenkan', 'IKS': 'Ichimoku Kijun', 
                       'ISA': 'Ichimoku Span A', 'ISB': 'Ichimoku Span B'}
            for p, n in patterns.items():
                if name.startswith(p):
                    return n
        
        # Bilinmeyen - olduğu gibi döndür ama daha temiz
        return name.split('_')[0] if '_' in name else name
    
    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        parse_mode: str = "HTML"
    ):
        """
        TelegramNotifier başlatır.
        
        Parameters:
        ----------
        token : str
            Telegram Bot Token (@BotFather'dan alınır)
            None ise TELEGRAM_BOT_TOKEN env var kullanılır
            
        chat_id : str
            Hedef chat ID (grup veya kullanıcı)
            None ise TELEGRAM_CHAT_ID env var kullanılır
            
        parse_mode : str
            Mesaj formatı: "HTML" veya "Markdown"
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.parse_mode = ParseMode.HTML if parse_mode == "HTML" else ParseMode.MARKDOWN
        
        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN tanımlanmamış!")
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID tanımlanmamış!")
        
        # Bot instance (lazy initialization)
        self._bot = None
    
    @property
    def bot(self) -> Bot:
        """Bot instance'ı döndürür (lazy init)."""
        if self._bot is None and self.token:
            self._bot = Bot(token=self.token)
        return self._bot
    
    def is_configured(self) -> bool:
        """Bot'un doğru yapılandırılıp yapılandırılmadığını kontrol eder."""
        return bool(self.token and self.chat_id)
    
    # =========================================================================
    # MESAJ FORMATLAMA
    # =========================================================================
    
    def format_analysis_report(self, report: AnalysisReport) -> str:
        """
        Analiz raporunu formatlı HTML mesajına çevirir.
        """
        
        # Emoji'ler
        regime_emoji = self.REGIME_EMOJI.get(report.market_regime, '❓')
        direction_emoji = self.DIRECTION_EMOJI.get(report.direction, '⚪')
        
        # Güven skoru gösterimi
        if report.confidence_score >= 70:
            confidence_bar = "🟢🟢🟢"
        elif report.confidence_score >= 50:
            confidence_bar = "🟡🟡"
        else:
            confidence_bar = "🔴"
        
        # Mesaj oluştur
        msg = f"""<b>🔔 {report.symbol} ANALİZ RAPORU</b>
━━━━━━━━━━━━━━━━━━━━━━━

💰 <b>Fiyat:</b> ${report.price:,.2f}
⏰ <b>Zaman:</b> {report.timestamp.strftime('%Y-%m-%d %H:%M')} UTC

<b>📊 ÖNERİLEN TIMEFRAME:</b> {report.recommended_timeframe}
{regime_emoji} <b>Piyasa Rejimi:</b> {report.market_regime}
{direction_emoji} <b>Sinyal:</b> {report.direction}
🎯 <b>Güven Skoru:</b> {report.confidence_score:.0f}/100 {confidence_bar}

"""
        
        # Aktif indikatörler (max 2 per kategori, formatted names)
        if report.active_indicators:
            msg += "<b>📈 AKTİF İNDİKATÖRLER:</b>\n"
            
            # Kategori sıralaması (other hariç)
            category_order = ['trend', 'momentum', 'volatility', 'volume']
            
            for category in category_order:
                if category in report.active_indicators:
                    indicators = report.active_indicators[category]
                    if indicators:
                        # Max 2 indikatör al ve isimlerini dönüştür
                        formatted = [self._format_indicator_name(ind) for ind in indicators[:2]]
                        # Duplicate isimleri kaldır
                        formatted = list(dict.fromkeys(formatted))
                        ind_str = ", ".join(formatted)
                        
                        # Kategori emoji'leri
                        cat_emoji = {
                            'trend': '📊',
                            'momentum': '⚡',
                            'volatility': '📉',
                            'volume': '📶'
                        }
                        emoji = cat_emoji.get(category, '•')
                        msg += f"{emoji} <i>{category.title()}</i>: {ind_str}\n"
            
            msg += "\n"
        
        # Risk metrikleri
        if report.risk_metrics:
            msg += "<b>⚠️ RİSK METRİKLERİ:</b>\n"
            
            if 'sharpe' in report.risk_metrics:
                sharpe = report.risk_metrics['sharpe']
                sharpe_icon = "✅" if sharpe > 1 else "⚠️" if sharpe > 0 else "❌"
                msg += f"• Sharpe Ratio: {sharpe:.2f} {sharpe_icon}\n"
            
            if 'max_dd' in report.risk_metrics:
                max_dd = report.risk_metrics['max_dd']
                dd_icon = "✅" if max_dd > -10 else "⚠️" if max_dd > -20 else "❌"
                msg += f"• Max Drawdown: {max_dd:.1f}% {dd_icon}\n"
            
            if 'win_rate' in report.risk_metrics:
                wr = report.risk_metrics['win_rate']
                wr_icon = "✅" if wr > 55 else "⚠️" if wr > 50 else "❌"
                msg += f"• Win Rate: {wr:.1f}% {wr_icon}\n"
            
            msg += "\n"
        
        # Notlar
        if report.notes:
            msg += f"📝 <b>Not:</b> {report.notes}\n"
        
        # Footer
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "<i>🤖 BTC Decision System v1.0</i>"
        
        return msg.strip()
    
    def format_simple_alert(
        self,
        title: str,
        message: str,
        alert_type: str = "info"
    ) -> str:
        """Basit alert mesajı formatlar."""
        icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }
        icon = icons.get(alert_type, 'ℹ️')
        return f"{icon} <b>{title}</b>\n\n{message}"
    
    def format_price_alert(
        self,
        symbol: str,
        price: float,
        change_pct: float,
        timeframe: str = "1h"
    ) -> str:
        """Fiyat değişim alert'i formatlar."""
        
        if change_pct >= 0:
            emoji = "🟢" if change_pct > 2 else "📈"
            direction = "+"
        else:
            emoji = "🔴" if change_pct < -2 else "📉"
            direction = ""
        
        return f"""{emoji} <b>{symbol} Fiyat Alert</b>

💰 Fiyat: ${price:,.2f}
📊 Değişim ({timeframe}): {direction}{change_pct:.2f}%
⏰ {datetime.now().strftime('%H:%M:%S')} UTC"""
    
    # =========================================================================
    # MESAJ GÖNDERME
    # =========================================================================
    
    async def send_message(
        self,
        text: str,
        disable_notification: bool = False
    ) -> bool:
        """
        Async mesaj gönderir.
        
        Returns:
        -------
        bool
            Başarılı ise True
        """
        
        if not self.is_configured():
            logger.error("Telegram yapılandırılmamış! Token ve Chat ID gerekli.")
            return False
        
        # Mesaj uzunluğu kontrolü
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(f"Mesaj çok uzun ({len(text)} karakter), kırpılıyor...")
            text = text[:self.MAX_MESSAGE_LENGTH - 100] + "\n\n<i>... (kırpıldı)</i>"
        
        # Retry mekanizması
        for attempt in range(self.MAX_RETRIES):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=self.parse_mode,
                    disable_notification=disable_notification
                )
                logger.info(f"Mesaj başarıyla gönderildi (attempt {attempt + 1})")
                return True
                
            except RetryAfter as e:
                # Rate limit - bekle ve tekrar dene
                wait_time = e.retry_after + 1
                logger.warning(f"Rate limit! {wait_time} saniye bekleniyor...")
                await asyncio.sleep(wait_time)
                
            except TelegramError as e:
                logger.error(f"Telegram hatası (attempt {attempt + 1}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RATE_LIMIT_DELAY * (attempt + 1))
                else:
                    return False
        
        return False
    
    async def send_analysis_report(
        self,
        report: AnalysisReport,
        silent: bool = False
    ) -> bool:
        """Analiz raporu gönderir (async)."""
        message = self.format_analysis_report(report)
        return await self.send_message(message, disable_notification=silent)
    
    async def send_alert(
        self,
        title: str,
        message: str,
        alert_type: str = "info"
    ) -> bool:
        """Basit alert gönderir (async)."""
        formatted = self.format_simple_alert(title, message, alert_type)
        return await self.send_message(formatted)
    
    async def send_chart(self, photo_file, caption: str = "") -> bool:
        """Grafik/Resim gönderir (async)."""
        if not self.is_configured():
            return False
        
        try:
            # Dosya imlecini başa al (önlem olarak)
            if hasattr(photo_file, 'seek'):
                photo_file.seek(0)
                
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_file,
                caption=caption,
                parse_mode=self.parse_mode
            )
            return True
        except Exception as e:
            logger.error(f"Grafik gönderme hatası: {e}")
            return False

    def send_chart_sync(self, photo_file, caption: str = "") -> bool:
        """Senkron grafik gönderme wrapper'ı."""
        try:
            return asyncio.run(self.send_chart(photo_file, caption))
        finally:
            self._bot = None  # <--- BU SATIR KRİTİK (Hata düzeltici)
    
    # =========================================================================
    # SENKRON WRAPPER'LAR
    # =========================================================================
    
    def send_message_sync(self, text: str, disable_notification: bool = False) -> bool:
        """Senkron mesaj gönderme wrapper'ı."""
        return asyncio.run(self.send_message(text, disable_notification))
    
    def send_report_sync(self, report: AnalysisReport, silent: bool = False) -> bool:
        """Senkron rapor gönderme wrapper'ı."""
        try:
            return asyncio.run(self.send_analysis_report(report, silent))
        finally:
            self._bot = None  # <--- BU SATIR KRİTİK (Hata düzeltici)
    
    def send_alert_sync(
        self,
        title: str,
        message: str,
        alert_type: str = "info"
    ) -> bool:
        """Senkron alert gönderme wrapper'ı."""
        return asyncio.run(self.send_alert(title, message, alert_type))
    
    # =========================================================================
    # YARDIMCI METODLAR
    # =========================================================================
    
    async def test_connection(self) -> bool:
        """Bot bağlantısını test eder."""
        if not self.is_configured():
            logger.error("Bot yapılandırılmamış!")
            return False
        
        try:
            me = await self.bot.get_me()
            logger.info(f"Bot bağlantısı başarılı: @{me.username}")
            return True
        except TelegramError as e:
            logger.error(f"Bot bağlantı hatası: {e}")
            return False
    
    def test_connection_sync(self) -> bool:
        """Senkron bağlantı testi."""
        return asyncio.run(self.test_connection())


# =============================================================================
# FACTORY FONKSİYONU
# =============================================================================

def create_notifier_from_env() -> TelegramNotifier:
    """Environment variable'lardan TelegramNotifier oluşturur."""
    return TelegramNotifier(
        token=os.getenv('TELEGRAM_BOT_TOKEN'),
        chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )


# =============================================================================
# TEST KODU
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("TELEGRAM NOTIFIER TEST")
    print("=" * 60)
    
    # Test raporu oluştur
    test_report = AnalysisReport(
        symbol="BTC/USDT",
        price=97250.00,
        recommended_timeframe="4h",
        market_regime="transitioning",
        direction="NEUTRAL",
        confidence_score=58,
        active_indicators={
            "trend": ["SUPERTREND", "EMA_50"],
            "momentum": ["RSI_14", "MACD"],
            "volatility": ["ATR_14", "BBANDS"],
            "volume": ["OBV", "MFI"]
        },
        risk_metrics={
            "sharpe": -0.26,
            "max_dd": -11.2,
            "win_rate": 52.7
        },
        notes="Piyasa geçiş döneminde, dikkatli olun."
    )
    
    # Notifier oluştur
    notifier = TelegramNotifier()
    
    # Formatlı mesajı göster
    print("\n📨 FORMATLI MESAJ:")
    print("-" * 60)
    formatted_msg = notifier.format_analysis_report(test_report)
    # HTML tag'lerini temizle (console için)
    clean_msg = re.sub(r'<[^>]+>', '', formatted_msg)
    print(clean_msg)
    print("-" * 60)
    
    # Bağlantı testi
    if notifier.is_configured():
        print("\n🔌 Bağlantı testi yapılıyor...")
        if notifier.test_connection_sync():
            print("✅ Bot bağlantısı başarılı!")
            
            # Gerçek mesaj gönder
            print("\n📤 Test mesajı gönderiliyor...")
            success = notifier.send_report_sync(test_report)
            if success:
                print("✅ Mesaj gönderildi!")
            else:
                print("❌ Mesaj gönderilemedi!")
        else:
            print("❌ Bot bağlantısı başarısız!")
    else:
        print("\n⚠️ Telegram yapılandırılmamış!")
        print("   Lütfen .env dosyasına ekleyin:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        print("\n💡 Bot oluşturma:")
        print("   1. Telegram'da @BotFather'a git")
        print("   2. /newbot komutu ile bot oluştur")
        print("   3. Token'ı kopyala")
        print("\n💡 Chat ID bulma:")
        print("   1. Bot'a bir mesaj at")
        print("   2. https://api.telegram.org/bot<TOKEN>/getUpdates")
        print("   3. 'chat':{'id': XXXXX} kısmındaki ID'yi al")
    
    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI")
    print("=" * 60)

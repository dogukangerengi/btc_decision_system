# =============================================================================
# TELEGRAM BİLDİRİM MODÜLÜ - v2.0
# =============================================================================
# Güncellemeler:
# - Kategori bazlı indikatör gösterimi (her kategoriden 1 top)
# - Kompakt format (sadece 2 TF sıralaması)
# - Gerçek fiyat + 24h değişim desteği
# - Kısa indikatör isimleri
# =============================================================================

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import os

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter

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
    symbol : str - İşlem çifti (örn: BTC/USDT)
    price : float - Güncel fiyat
    recommended_timeframe : str - Önerilen zaman dilimi
    market_regime : str - Piyasa rejimi
    direction : str - Sinyal yönü (LONG, SHORT, NEUTRAL)
    confidence_score : float - Güven skoru (0-100)
    active_indicators : Dict[str, List[str]] - Aktif indikatörler
    indicator_details : Dict[str, float] - İndikatör IC değerleri
    category_tops : Dict[str, dict] - Her kategoriden en iyi indikatör (YENİ)
    tf_rankings : List[dict] - TF sıralaması (YENİ)
    change_24h : float - 24 saatlik değişim % (YENİ)
    """
    symbol: str
    price: float
    recommended_timeframe: str
    market_regime: str
    direction: str
    confidence_score: float
    active_indicators: Dict[str, List[str]]
    indicator_details: Dict[str, float] = field(default_factory=dict)
    category_tops: Dict[str, dict] = field(default_factory=dict)
    tf_rankings: List[dict] = field(default_factory=list)
    timestamp: datetime = None
    notes: str = ""
    change_24h: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramNotifier:
    """Telegram bildirim gönderen sınıf - v2.0"""
    
    MAX_MESSAGE_LENGTH = 4096
    RATE_LIMIT_DELAY = 1.0
    MAX_RETRIES = 3
    
    # İndikatör kısa adları mapping
    INDICATOR_SHORTCUTS = {
        # Trend
        'AROONU_25': 'Aroon↑', 'AROOND_25': 'Aroon↓', 'AROONOSC_25': 'AroonOsc',
        'SUPERTs_10_3.0': 'SuperT', 'SUPERTl_10_3.0': 'SuperT', 'SUPERTd_10_3.0': 'SuperT',
        'SUPERT_10_3.0': 'SuperT',
        'PSARs_0.02_0.2': 'PSAR', 'PSARl_0.02_0.2': 'PSAR',
        'VTXP_14': 'Vortex+', 'VTXN_14': 'Vortex-',
        'ADX_14': 'ADX', 'DMP_14': 'DI+', 'DMN_14': 'DI-',
        'KAMA_10_2_30': 'KAMA',
        
        # Momentum
        'COPC_11_14_10': 'Coppock',
        'STOCHRSIk_14_14_3_3': 'StochRSI', 'STOCHRSId_14_14_3_3': 'StochRSI',
        'STOCHk_14_3_3': 'Stoch%K', 'STOCHd_14_3_3': 'Stoch%D',
        'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'MACDh', 'MACDs_12_26_9': 'MACDs',
        'PPO_12_26_9': 'PPO', 'PPOh_12_26_9': 'PPOh', 'PPOs_12_26_9': 'PPOs',
        'TSI_13_25_13': 'TSI', 'TSIs_13_25_13': 'TSIs',
        'FISHERT_9_1': 'Fisher', 'FISHERTs_9_1': 'Fisher',
        'UO_7_14_28': 'UltOsc', 'AO_5_34': 'AwesomeOsc',
        'CCI_20_0.015': 'CCI', 'WILLR_14': 'WillR', 'CMO_14': 'CMO',
        
        # Volatility
        'MASSI_9_25': 'MassIdx',
        'BBU_20_2.0': 'BB↑', 'BBM_20_2.0': 'BB', 'BBL_20_2.0': 'BB↓',
        'BBB_20_2.0': 'BB%B', 'BBP_20_2.0': 'BB%P',
        'BBU_20_1.0': 'BB1↑', 'BBL_20_1.0': 'BB1↓',
        'KCUe_20_1.5': 'KC↑', 'KCBe_20_1.5': 'KC', 'KCLe_20_1.5': 'KC↓',
        'DCU_20_20': 'DC↑', 'DCM_20_20': 'DC', 'DCL_20_20': 'DC↓',
        'ATRr_14': 'ATR', 'ATRr_7': 'ATR7', 'NATR_14': 'NATR',
        'RVI_14': 'RVI', 'RVIs_14': 'RVIs',
        'UI_14': 'Ulcer', 'TRUERANGE_1': 'TR',
        'ACCBU_20': 'AccB↑', 'ACCBL_20': 'AccB↓',
        
        # Volume
        'CMF_20': 'CMF', 'ADOSC_3_10': 'ChaikinOsc', 'MFI_14': 'MFI',
        'EFI_13': 'ElderForce', 'VWMA_20': 'VWMA',
        'OBV': 'OBV', 'AD': 'A/D', 'PVT': 'PVT',
        'NVI_1': 'NVI', 'PVI_1': 'PVI', 'PVOL': 'PVol',
        
        # Composite
        'SQZ_20_2.0_20_1.5': 'Squeeze',
        'QQE_14_5_RSI': 'QQE', 'QQEl_14_5': 'QQEl', 'QQEs_14_5': 'QQEs',
        'ISA_9': 'Ichi-A', 'ISB_26': 'Ichi-B', 'ITS_9': 'Ichi-T',
        'IKS_26': 'Ichi-K', 'ICS_26': 'Ichi-C',
    }
    
    # Kategori emoji ve isimleri
    CATEGORY_INFO = {
        'trend': ('📊', 'Trend'),
        'momentum': ('⚡', 'Momentum'),
        'volatility': ('📉', 'Volatilite'),
        'volume': ('📶', 'Hacim')
    }
    
    # Rejim mapping
    REGIME_MAP = {
        'trending_up': '📈 Trend↑',
        'trending_down': '📉 Trend↓',
        'ranging': '↔️ Yatay',
        'volatile': '⚡ Volatil',
        'transitioning': '🔄 Geçiş',
        'unknown': '❓'
    }
    
    def __init__(self, token: str = None, chat_id: str = None):
        """
        TelegramNotifier başlatır.
        
        Parameters:
        ----------
        token : str - Bot token (veya TELEGRAM_BOT_TOKEN env var)
        chat_id : str - Chat ID (veya TELEGRAM_CHAT_ID env var)
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self._bot = None
    
    @property
    def bot(self) -> Bot:
        """Lazy bot initialization."""
        if self._bot is None and self.token:
            self._bot = Bot(token=self.token)
        return self._bot
    
    def is_configured(self) -> bool:
        """Telegram yapılandırılmış mı kontrol et."""
        return bool(self.token and self.chat_id)
    
    def _shorten_indicator(self, name: str) -> str:
        """
        İndikatör adını kısaltır.
        
        Öncelik:
        1. INDICATOR_SHORTCUTS dict'te varsa onu kullan
        2. Bilinen prefix'lere göre kısalt
        3. Son çare: ilk 8 karakteri al
        """
        # Direkt mapping varsa kullan
        if name in self.INDICATOR_SHORTCUTS:
            return self.INDICATOR_SHORTCUTS[name]
        
        # Bilinen prefix'ler için kısaltma kuralları
        prefix_rules = {
            'EMA_': lambda n: f"EMA{n.split('_')[1]}",
            'SMA_': lambda n: f"SMA{n.split('_')[1]}",
            'WMA_': lambda n: f"WMA{n.split('_')[1]}",
            'HMA_': lambda n: f"HMA{n.split('_')[1]}",
            'DEMA_': lambda n: f"DEMA{n.split('_')[1]}",
            'TEMA_': lambda n: f"TEMA{n.split('_')[1]}",
            'RSI_': lambda n: f"RSI{n.split('_')[1]}",
            'ROC_': lambda n: f"ROC{n.split('_')[1]}",
            'MOM_': lambda n: f"MOM{n.split('_')[1]}",
            'CCI_': lambda n: f"CCI{n.split('_')[1]}",
        }
        
        for prefix, formatter in prefix_rules.items():
            if name.startswith(prefix):
                try:
                    return formatter(name)
                except:
                    pass
        
        # Varsayılan: ilk kısmı al (max 8 karakter)
        parts = name.split('_')
        return parts[0][:8] if parts else name[:8]
    
    def format_analysis_report(self, report: AnalysisReport) -> str:
        """
        Analiz raporunu Telegram mesajı olarak formatlar.
        
        YENİ KOMPAKT FORMAT:
        - Fiyat + 24h değişim
        - TF + Yön + Güven + Rejim
        - 4 kategoriden birer top indikatör (IC değeriyle)
        - Sadece ilk 2 TF sıralaması
        """
        
        # === HEADER ===
        dir_emoji = "🟢" if report.direction == "LONG" else "🔴" if report.direction == "SHORT" else "⚪"
        change_emoji = "📈" if report.change_24h > 0 else "📉" if report.change_24h < 0 else "➡️"
        
        # Güven barı
        score = report.confidence_score
        if score >= 70:
            conf_bar = "🟢🟢🟢"
        elif score >= 50:
            conf_bar = "🟡🟡"
        else:
            conf_bar = "🔴"
        
        # === KATEGORİ SİNYALLERİ ===
        category_lines = ""
        
        # Öncelik: category_tops (yeni format)
        if report.category_tops:
            for cat, info in self.CATEGORY_INFO.items():
                if cat in report.category_tops:
                    ind = report.category_tops[cat]
                    ic_val = ind['ic']
                    ic_sign = "+" if ic_val > 0 else ""
                    short_name = self._shorten_indicator(ind['name'])
                    category_lines += f"\n{info[0]} {info[1]}: {short_name} ({ic_sign}{ic_val:.2f})"
        
        # Fallback: eski active_indicators format
        elif report.active_indicators:
            for cat, info in self.CATEGORY_INFO.items():
                if cat in report.active_indicators and report.active_indicators[cat]:
                    ind_name = report.active_indicators[cat][0]
                    ic_val = report.indicator_details.get(ind_name, 0)
                    ic_sign = "+" if ic_val > 0 else ""
                    short_name = self._shorten_indicator(ind_name)
                    category_lines += f"\n{info[0]} {info[1]}: {short_name} ({ic_sign}{ic_val:.2f})"
        
        # === TF SIRALAMASI (sadece ilk 2) ===
        tf_lines = ""
        if report.tf_rankings:
            for r in report.tf_rankings[:2]:
                marker = "→" if r['tf'] == report.recommended_timeframe else "  "
                d_emoji = "🟢" if r['direction'] == "LONG" else "🔴" if r['direction'] == "SHORT" else "⚪"
                tf_lines += f"\n{marker}{r['tf']}: {r['score']:.0f} {d_emoji}"
        
        # === MESAJ OLUŞTUR ===
        msg = f"""🔔 <b>{report.symbol} ANALİZ</b>
━━━━━━━━━━━━━━━━━━━━━

💰 Fiyat: ${report.price:,.2f} ({change_emoji}{report.change_24h:+.1f}%)

📊 TF: <b>{report.recommended_timeframe}</b> | {dir_emoji} <b>{report.direction}</b>
🎯 Güven: {score:.0f}/100 {conf_bar}
📍 Rejim: {self.REGIME_MAP.get(report.market_regime, '❓')}"""
        
        # Kategori sinyalleri ekle
        if category_lines:
            msg += f"\n\n⭐ <b>Kategori Sinyalleri:</b>{category_lines}"
        
        # TF sıralaması ekle
        if tf_lines:
            msg += f"\n\n📋 <b>TF Sıralaması:</b>{tf_lines}"
        
        # Notlar varsa ekle
        if report.notes:
            msg += f"\n\n📝 {report.notes}"
        
        # Footer
        msg += f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⏰ {report.timestamp.strftime('%Y-%m-%d %H:%M')}"
        
        return msg.strip()
    
    def format_simple_alert(self, title: str, message: str, alert_type: str = "info") -> str:
        """Basit uyarı mesajı formatla."""
        icons = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'success': '✅'}
        return f"{icons.get(alert_type, 'ℹ️')} <b>{title}</b>\n\n{message}"
    
    async def send_message(self, text: str, disable_notification: bool = False) -> bool:
        """
        Telegram mesajı gönder (async).
        
        Parameters:
        ----------
        text : str - Mesaj içeriği
        disable_notification : bool - Sessiz bildirim
        
        Returns:
        -------
        bool - Başarılı ise True
        """
        if not self.is_configured():
            logger.error("Telegram yapılandırılmamış!")
            return False
        
        # Mesaj uzunluk kontrolü
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n<i>...</i>"
        
        # Retry logic
        for attempt in range(self.MAX_RETRIES):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_notification=disable_notification
                )
                logger.info("Mesaj gönderildi")
                return True
            except RetryAfter as e:
                logger.warning(f"Rate limit, {e.retry_after}s bekleniyor...")
                await asyncio.sleep(e.retry_after + 1)
            except TelegramError as e:
                logger.error(f"Telegram hatası: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RATE_LIMIT_DELAY * (attempt + 1))
        
        return False
    
    async def send_analysis_report(self, report: AnalysisReport, silent: bool = False) -> bool:
        """Analiz raporu gönder (async)."""
        message = self.format_analysis_report(report)
        return await self.send_message(message, disable_notification=silent)
    
    async def send_alert(self, title: str, message: str, alert_type: str = "info") -> bool:
        """Uyarı mesajı gönder (async)."""
        formatted = self.format_simple_alert(title, message, alert_type)
        return await self.send_message(formatted)
    
    # === SYNC WRAPPERS ===
    # main.py gibi sync kodlardan çağrılabilmesi için
    
    def send_message_sync(self, text: str, disable_notification: bool = False) -> bool:
        """Sync message gönder."""
        return asyncio.run(self.send_message(text, disable_notification))
    
    def send_report_sync(self, report: AnalysisReport, silent: bool = False) -> bool:
        """Sync rapor gönder."""
        return asyncio.run(self.send_analysis_report(report, silent))
    
    def send_alert_sync(self, title: str, message: str, alert_type: str = "info") -> bool:
        """Sync uyarı gönder."""
        return asyncio.run(self.send_alert(title, message, alert_type))
    
    async def test_connection(self) -> bool:
        """Bot bağlantısını test et."""
        if not self.is_configured():
            return False
        try:
            me = await self.bot.get_me()
            logger.info(f"Bot bağlantısı başarılı: @{me.username}")
            return True
        except TelegramError as e:
            logger.error(f"Bağlantı hatası: {e}")
            return False
    
    def test_connection_sync(self) -> bool:
        """Sync bağlantı testi."""
        return asyncio.run(self.test_connection())


def create_notifier_from_env() -> TelegramNotifier:
    """Ortam değişkenlerinden notifier oluştur."""
    return TelegramNotifier()

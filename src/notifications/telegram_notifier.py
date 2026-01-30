# =============================================================================
# TELEGRAM BİLDİRİM MODÜLÜ
# =============================================================================
# Amaç: Analiz sonuçlarını Telegram üzerinden bildirmek
#
# Özellikler:
# - Async messaging (python-telegram-bot v20+)
# - Rate limiting (Telegram API limitleri)
# - Formatlı mesajlar (HTML)
# - IC bazlı sinyal gücü (backtest metrikleri yerine)
# =============================================================================

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
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
        Güven skoru (0-100) - IC bazlı
    active_indicators : Dict[str, List[str]]
        Aktif indikatörler (kategori → indikatör listesi)
    indicator_details : Dict[str, float]
        İndikatör IC değerleri (indikatör_adı → IC)
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
    indicator_details: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = None
    notes: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TelegramNotifier:
    """Telegram bildirim gönderen sınıf."""
    
    MAX_MESSAGE_LENGTH = 4096
    RATE_LIMIT_DELAY = 1.0
    MAX_RETRIES = 3
    
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
    
    INDICATOR_NAMES = {
        'SUPERTs_10_3.0': 'Supertrend', 'SUPERTl_10_3.0': 'Supertrend',
        'SUPERTd_10_3.0': 'Supertrend', 'SUPERT_10_3.0': 'Supertrend',
        'EMA_12': 'EMA (12)', 'EMA_20': 'EMA (20)', 'EMA_26': 'EMA (26)', 'EMA_50': 'EMA (50)',
        'SMA_20': 'SMA (20)', 'SMA_50': 'SMA (50)', 'SMA_200': 'SMA (200)',
        'TEMA_20': 'TEMA (20)', 'DEMA_20': 'DEMA (20)', 'WMA_20': 'WMA (20)',
        'HMA_20': 'Hull MA (20)', 'KAMA_10_2_30': 'KAMA',
        'ADX_14': 'ADX (14)', 'DMP_14': 'DI+ (14)', 'DMN_14': 'DI- (14)',
        'PSARl_0.02_0.2': 'Parabolic SAR', 'PSARs_0.02_0.2': 'Parabolic SAR',
        'AROONU_25': 'Aroon Up', 'AROOND_25': 'Aroon Down', 'AROONOSC_25': 'Aroon Osc',
        'VTXP_14': 'Vortex+', 'VTXN_14': 'Vortex-',
        'RSI_7': 'RSI (7)', 'RSI_14': 'RSI (14)', 'RSI_21': 'RSI (21)',
        'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'MACD Histogram', 'MACDs_12_26_9': 'MACD Signal',
        'STOCHk_14_3_3': 'Stochastic %K', 'STOCHd_14_3_3': 'Stochastic %D',
        'STOCHRSIk_14_14_3_3': 'StochRSI', 'STOCHRSId_14_14_3_3': 'StochRSI %D',
        'CCI_20_0.015': 'CCI (20)', 'WILLR_14': 'Williams %R',
        'MOM_10': 'Momentum (10)', 'ROC_10': 'ROC (10)', 'ROC_20': 'ROC (20)',
        'AO_5_34': 'Awesome Osc', 'PPO_12_26_9': 'PPO', 'TSI_13_25_13': 'TSI',
        'UO_7_14_28': 'Ultimate Osc', 'CMO_14': 'CMO', 'FISHERT_9_1': 'Fisher Transform',
        'ATRr_14': 'ATR (14)', 'ATRr_7': 'ATR (7)', 'NATR_14': 'NATR (14)',
        'BBU_20_2.0': 'BB Upper', 'BBM_20_2.0': 'BB Middle', 'BBL_20_2.0': 'BB Lower',
        'BBB_20_2.0': 'BB Width', 'BBP_20_2.0': 'BB %B',
        'KCUe_20_1.5': 'Keltner Upper', 'KCBe_20_1.5': 'Keltner Basis', 'KCLe_20_1.5': 'Keltner Lower',
        'DCU_20_20': 'Donchian Upper', 'DCM_20_20': 'Donchian Middle', 'DCL_20_20': 'Donchian Lower',
        'OBV': 'OBV', 'AD': 'A/D Line', 'PVT': 'PVT', 'MFI_14': 'MFI (14)',
        'CMF_20': 'CMF (20)', 'ADOSC_3_10': 'Chaikin Osc', 'EFI_13': 'Elder Force', 'VWMA_20': 'VWMA (20)',
    }
    
    def _format_indicator_name(self, raw_name: str) -> str:
        if raw_name in self.INDICATOR_NAMES:
            return self.INDICATOR_NAMES[raw_name]
        if raw_name.startswith('SUPER'): return 'Supertrend'
        for prefix in ['EMA_', 'SMA_', 'WMA_', 'TEMA_', 'DEMA_', 'HMA_', 'RSI_', 'ATR']:
            if raw_name.startswith(prefix):
                period = raw_name.replace(prefix, '').split('_')[0]
                return f"{prefix.rstrip('_')} ({period})"
        if raw_name.startswith('CCI_'):
            parts = raw_name.split('_')
            return f"CCI ({parts[1]})" if len(parts) > 1 else 'CCI'
        if 'STOCH' in raw_name: return 'StochRSI' if 'RSI' in raw_name else 'Stochastic'
        if raw_name.startswith('BB'): return 'Bollinger Bands'
        if raw_name.startswith('KC'): return 'Keltner Channel'
        if raw_name.startswith('DC'): return 'Donchian'
        if raw_name.startswith('MACD'): return 'MACD'
        if raw_name.startswith('PSAR'): return 'Parabolic SAR'
        if raw_name.startswith('WILLR'): return 'Williams %R'
        return raw_name.split('_')[0] if '_' in raw_name else raw_name
    
    def __init__(self, token: str = None, chat_id: str = None, parse_mode: str = "HTML"):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.parse_mode = ParseMode.HTML if parse_mode == "HTML" else ParseMode.MARKDOWN
        self._bot = None
    
    @property
    def bot(self) -> Bot:
        if self._bot is None and self.token:
            self._bot = Bot(token=self.token)
        return self._bot
    
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)
    
    def format_analysis_report(self, report: AnalysisReport) -> str:
        regime_emoji = self.REGIME_EMOJI.get(report.market_regime, '❓')
        direction_emoji = self.DIRECTION_EMOJI.get(report.direction, '⚪')
        
        if report.confidence_score >= 70:
            confidence_bar = "🟢🟢🟢"
        elif report.confidence_score >= 50:
            confidence_bar = "🟡🟡"
        else:
            confidence_bar = "🔴"
        
        msg = f"""<b>🔔 {report.symbol} ANALİZ RAPORU</b>
━━━━━━━━━━━━━━━━━━━━━━━

💰 <b>Fiyat:</b> ${report.price:,.2f}
⏰ <b>Zaman:</b> {report.timestamp.strftime('%Y-%m-%d %H:%M')} UTC

<b>📊 ÖNERİLEN TIMEFRAME:</b> {report.recommended_timeframe}
{regime_emoji} <b>Piyasa Rejimi:</b> {report.market_regime}
{direction_emoji} <b>Baskın Yön:</b> {report.direction}
🎯 <b>Sinyal Gücü:</b> {report.confidence_score:.0f}/100 {confidence_bar}

"""
        
        if report.active_indicators:
            msg += "<b>📈 AKTİF İNDİKATÖRLER:</b>\n"
            category_order = ['trend', 'momentum', 'volatility', 'volume']
            cat_emoji = {'trend': '📊', 'momentum': '⚡', 'volatility': '📉', 'volume': '📶'}
            
            for category in category_order:
                if category in report.active_indicators:
                    indicators = report.active_indicators[category]
                    if indicators:
                        formatted_list = []
                        for ind in indicators[:2]:
                            formatted_name = self._format_indicator_name(ind)
                            if report.indicator_details and ind in report.indicator_details:
                                ic_val = report.indicator_details[ind]
                                if isinstance(ic_val, (int, float)):
                                    formatted_list.append(f"{formatted_name} ({ic_val:+.2f})")
                                else:
                                    formatted_list.append(formatted_name)
                            else:
                                formatted_list.append(formatted_name)
                        formatted_list = list(dict.fromkeys(formatted_list))
                        ind_str = ", ".join(formatted_list)
                        emoji = cat_emoji.get(category, '•')
                        msg += f"{emoji} <i>{category.title()}</i>: {ind_str}\n"
            msg += "\n"
        
        if report.notes:
            msg += f"📝 <b>Not:</b> {report.notes}\n\n"
        
        msg += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "<i>🤖 BTC Decision System v1.0</i>"
        
        return msg.strip()
    
    def format_simple_alert(self, title: str, message: str, alert_type: str = "info") -> str:
        icons = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'success': '✅'}
        icon = icons.get(alert_type, 'ℹ️')
        return f"{icon} <b>{title}</b>\n\n{message}"
    
    async def send_message(self, text: str, disable_notification: bool = False) -> bool:
        if not self.is_configured():
            logger.error("Telegram yapılandırılmamış!")
            return False
        
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[:self.MAX_MESSAGE_LENGTH - 100] + "\n\n<i>... (kırpıldı)</i>"
        
        for attempt in range(self.MAX_RETRIES):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id, text=text,
                    parse_mode=self.parse_mode, disable_notification=disable_notification
                )
                logger.info(f"Mesaj başarıyla gönderildi")
                return True
            except RetryAfter as e:
                await asyncio.sleep(e.retry_after + 1)
            except TelegramError as e:
                logger.error(f"Telegram hatası: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RATE_LIMIT_DELAY * (attempt + 1))
        return False
    
    async def send_analysis_report(self, report: AnalysisReport, silent: bool = False) -> bool:
        message = self.format_analysis_report(report)
        return await self.send_message(message, disable_notification=silent)
    
    async def send_alert(self, title: str, message: str, alert_type: str = "info") -> bool:
        formatted = self.format_simple_alert(title, message, alert_type)
        return await self.send_message(formatted)
    
    def send_message_sync(self, text: str, disable_notification: bool = False) -> bool:
        return asyncio.run(self.send_message(text, disable_notification))
    
    def send_report_sync(self, report: AnalysisReport, silent: bool = False) -> bool:
        return asyncio.run(self.send_analysis_report(report, silent))
    
    def send_alert_sync(self, title: str, message: str, alert_type: str = "info") -> bool:
        return asyncio.run(self.send_alert(title, message, alert_type))
    
    async def test_connection(self) -> bool:
        if not self.is_configured():
            return False
        try:
            me = await self.bot.get_me()
            logger.info(f"Bot bağlantısı başarılı: @{me.username}")
            return True
        except TelegramError as e:
            logger.error(f"Bot bağlantı hatası: {e}")
            return False
    
    def test_connection_sync(self) -> bool:
        return asyncio.run(self.test_connection())


def create_notifier_from_env() -> TelegramNotifier:
    return TelegramNotifier(
        token=os.getenv('TELEGRAM_BOT_TOKEN'),
        chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )

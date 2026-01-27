# =============================================================================
# NOTIFICATIONS MODÜLÜ
# =============================================================================
# Telegram bildirim sistemi.
#
# Kullanım:
# from notifications import TelegramNotifier, AnalysisReport
#
# notifier = TelegramNotifier()
# report = AnalysisReport(symbol="BTC/USDT", price=97000, ...)
# notifier.send_report_sync(report)
# =============================================================================

from .telegram_notifier import (
    AnalysisReport,
    TelegramNotifier,
    create_notifier_from_env,
)

__all__ = [
    'AnalysisReport',
    'TelegramNotifier',
    'create_notifier_from_env',
]

__version__ = '1.0.0'

# =============================================================================
# TELEGRAM BOT - IC ANALİZ (SADECE ANALİZ)
# =============================================================================
# Komutlar:
# /analiz [COIN] - IC analizi
# /fiyat [COIN]  - Anlık fiyat
# /liste         - Desteklenen coinler
# =============================================================================

import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Path ayarları
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR / 'data'))
sys.path.insert(0, str(CURRENT_DIR / 'indicators'))

# .env yükle
from dotenv import load_dotenv
load_dotenv(CURRENT_DIR.parent / '.env')

# Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

# Analiz modülleri
from fetcher import DataFetcher
from calculator import IndicatorCalculator
from selector import IndicatorSelector

# Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# =============================================================================
# YAPILANDIRMA
# =============================================================================

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

def get_supported_coins() -> List[str]:
    """Binance'daki tüm USDT çiftlerini döndürür."""
    try:
        import ccxt
        exchange = ccxt.binance()
        exchange.load_markets()
        
        # USDT çiftlerini filtrele
        coins = []
        for symbol in exchange.markets:
            if symbol.endswith('/USDT') and ':' not in symbol:  # Spot only
                coin = symbol.replace('/USDT', '')
                coins.append(coin)
        
        return sorted(coins)
    except Exception as e:
        logger.error(f"Coin listesi alınamadı: {e}")
        # Fallback liste
        return ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'MATIC']

# Başlangıçta bir kez çek
SUPPORTED_COINS = get_supported_coins()
logger.info(f"Desteklenen coin sayısı: {len(SUPPORTED_COINS)}")

# main.py ile aynı parametreler
TIMEFRAMES = {
    '5m':  2000,
    '15m': 1500,
    '30m': 1000,
    '1h':  1000,
    '2h':  750,
    '4h':  500
}

IC_WEIGHTS = {
    'top_ic': 0.40,
    'avg_ic': 0.25,
    'count': 0.15,
    'consistency': 0.20
}

executor = ThreadPoolExecutor(max_workers=6)

# =============================================================================
# ANALİZ FONKSİYONU
# =============================================================================

def run_analysis(symbol: str) -> Optional[Dict]:
    """Tek coin için tam IC analizi yapar."""
    
    try:
        fetcher = DataFetcher(symbol=f"{symbol}/USDT")
        calculator = IndicatorCalculator(verbose=False)
        selector = IndicatorSelector(alpha=0.05, correction_method='fdr', verbose=False)
        
        # Güncel fiyat ve 24h bilgileri
        try:
            ticker = fetcher.get_latest_price()
            current_price = ticker['last']
            change_24h = ticker.get('change_24h', 0) or 0
            high_24h = ticker.get('high_24h', 0) or 0
            low_24h = ticker.get('low_24h', 0) or 0
        except:
            current_price = 0
            change_24h = 0
            high_24h = 0
            low_24h = 0
        
        tf_scores = []
        all_indicator_scores = {}
        
        for tf, bars in TIMEFRAMES.items():
            try:
                df = fetcher.fetch_max_ohlcv(timeframe=tf, max_bars=bars, progress=False)
                if df is None or len(df) < 200:
                    continue
                
                df = calculator.calculate_all(df, categories=['trend', 'momentum', 'volatility', 'volume'])
                df = calculator.add_forward_returns(df, periods=[1, 5, 10])
                
                scores = selector.evaluate_all_indicators(df, target_col='fwd_ret_5')
                all_indicator_scores[tf] = scores
                
                significant = [s for s in scores 
                              if abs(s.ic_mean) > 0.02 
                              and s.category in ['trend', 'momentum', 'volatility', 'volume']]
                
                if not significant:
                    continue
                
                top_ic = max(abs(s.ic_mean) for s in significant)
                avg_ic = sum(abs(s.ic_mean) for s in significant) / len(significant)
                
                positive = sum(1 for s in significant if s.ic_mean > 0)
                negative = sum(1 for s in significant if s.ic_mean < 0)
                consistency = max(positive, negative) / len(significant)
                
                if negative > positive * 1.5:
                    direction = 'SHORT'
                elif positive > negative * 1.5:
                    direction = 'LONG'
                else:
                    direction = 'NEUTRAL'
                
                top_norm = min((top_ic - 0.02) / 0.38 * 100, 100)
                avg_norm = min((avg_ic - 0.02) / 0.13 * 100, 100)
                count_norm = min(len(significant) / 50 * 100, 100)
                cons_norm = max(0, (consistency - 0.5) / 0.5 * 100)
                
                composite = (
                    top_norm * IC_WEIGHTS['top_ic'] +
                    avg_norm * IC_WEIGHTS['avg_ic'] +
                    count_norm * IC_WEIGHTS['count'] +
                    cons_norm * IC_WEIGHTS['consistency']
                )
                
                tf_scores.append({
                    'tf': tf,
                    'score': composite,
                    'direction': direction,
                    'top_ic': top_ic,
                    'avg_ic': avg_ic,
                    'count': len(significant),
                    'consistency': consistency
                })
                
            except Exception as e:
                logger.warning(f"{tf} hatası: {e}")
                continue
        
        if not tf_scores:
            return None
        
        tf_scores.sort(key=lambda x: x['score'], reverse=True)
        best = tf_scores[0]
        
        # Kategori tops
        category_tops = {}
        if best['tf'] in all_indicator_scores:
            scores = all_indicator_scores[best['tf']]
            for cat in ['trend', 'momentum', 'volatility', 'volume']:
                cat_scores = [s for s in scores if s.category == cat and abs(s.ic_mean) > 0.02]
                if cat_scores:
                    top = max(cat_scores, key=lambda x: abs(x.ic_mean))
                    category_tops[cat] = {'name': top.name, 'ic': top.ic_mean}
        
        # Rejim
        regime = 'unknown'
        atr_value = 0
        atr_pct = 0
        
        if best['tf'] in all_indicator_scores:
            scores = all_indicator_scores[best['tf']]
            adx_scores = [s for s in scores if 'ADX' in s.name]
            if adx_scores:
                adx_ic = adx_scores[0].ic_mean
                if abs(adx_ic) > 0.1:
                    regime = 'trending'
                else:
                    regime = 'ranging'
        
        # ATR değerini al (risk hesabı için)
        if best['tf'] in all_indicator_scores:
            # En son veriyi kullanarak ATR hesapla
            try:
                df = fetcher.fetch_ohlcv(timeframe=best['tf'], limit=20)
                if df is not None and len(df) > 14:
                    # ATR hesapla (14 periyot)
                    high = df['high']
                    low = df['low']
                    close = df['close']
                    
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr_value = tr.rolling(14).mean().iloc[-1]
                    atr_pct = (atr_value / current_price * 100) if current_price > 0 else 0
            except:
                pass
        
        return {
            'symbol': f"{symbol}/USDT",
            'price': current_price,
            'change_24h': change_24h,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'atr': atr_value,
            'atr_pct': atr_pct,
            'best_tf': best['tf'],
            'direction': best['direction'],
            'confidence': best['score'],
            'regime': regime,
            'category_tops': category_tops,
            'tf_rankings': tf_scores[:4]
        }
        
    except Exception as e:
        logger.error(f"Analiz hatası ({symbol}): {e}")
        return None


# =============================================================================
# MESAJ FORMATLAMA
# =============================================================================

INDICATOR_SHORTCUTS = {
    'PSARl_0.02_0.2': 'PSAR', 'PSARs_0.02_0.2': 'PSAR',
    'SUPERT_10_3.0': 'SuperT', 'SUPERTd_10_3.0': 'SuperT',
    'AROONU_25': 'AroonU', 'AROOND_25': 'AroonD', 'AROONOSC_25': 'AroonO',
    'ADX_14': 'ADX', 'DMP_14': 'DI+', 'DMN_14': 'DI-',
    'VTXP_14': 'Vortex+', 'VTXN_14': 'Vortex-',
    'KAMA_10_2_30': 'KAMA',
    'RSI_14': 'RSI14', 'RSI_7': 'RSI7', 'RSI_21': 'RSI21',
    'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'MACDh', 'MACDs_12_26_9': 'MACDs',
    'STOCHk_14_3_3': 'Stoch%K', 'STOCHd_14_3_3': 'Stoch%D',
    'STOCHRSIk_14_14_3_3': 'StochRSI',
    'CCI_20_0.015': 'CCI', 'WILLR_14': 'WillR',
    'MOM_10': 'Mom', 'ROC_10': 'ROC10', 'ROC_20': 'ROC20',
    'AO_5_34': 'AO', 'PPO_12_26_9': 'PPO',
    'TSI_13_25_13': 'TSI', 'UO_7_14_28': 'UO',
    'CMO_14': 'CMO', 'FISHERT_9_1': 'Fisher',
    'COPC_11_14_10': 'Coppock',
    'ATRr_14': 'ATR14', 'ATRr_7': 'ATR7', 'NATR_14': 'NATR',
    'BBU_20_2.0': 'BBU', 'BBM_20_2.0': 'BBM', 'BBL_20_2.0': 'BBL',
    'BBB_20_2.0': 'BBW', 'BBP_20_2.0': 'BB%B',
    'KCUe_20_1.5': 'KCU', 'KCBe_20_1.5': 'KCB', 'KCLe_20_1.5': 'KCL',
    'MASSI_9_25': 'MassIdx', 'UI_14': 'Ulcer',
    'RVI_14': 'RVI', 'TRUERANGE_1': 'TR',
    'OBV': 'OBV', 'AD': 'A/D', 'PVT': 'PVT',
    'CMF_20': 'CMF', 'MFI_14': 'MFI',
    'ADOSC_3_10': 'ChaikinO', 'EFI_13': 'EFI',
    'NVI_1': 'NVI', 'PVI_1': 'PVI', 'VWMA_20': 'VWMA',
}


def shorten_indicator(name: str) -> str:
    """İndikatör adını kısaltır."""
    if name in INDICATOR_SHORTCUTS:
        return INDICATOR_SHORTCUTS[name]
    
    prefixes = {
        'EMA_': 'EMA', 'SMA_': 'SMA', 'WMA_': 'WMA',
        'TEMA_': 'TEMA', 'DEMA_': 'DEMA', 'HMA_': 'HMA',
    }
    for prefix, short in prefixes.items():
        if name.startswith(prefix):
            period = name.replace(prefix, '').split('_')[0]
            return f"{short}{period}"
    
    return name.split('_')[0][:8]


def format_analysis_message(result: Dict) -> str:
    """Analiz sonucunu Telegram mesajı olarak formatlar."""
    
    dir_emoji = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪'}.get(result['direction'], '⚪')
    regime_emoji = {'trending': '📈', 'ranging': '↔️', 'volatile': '⚡'}.get(result['regime'], '🔄')
    
    conf = result['confidence']
    if conf >= 70:
        conf_bar = "🟢🟢🟢"
    elif conf >= 50:
        conf_bar = "🟡🟡"
    else:
        conf_bar = "🔴"
    
    change = result.get('change_24h', 0)
    change_str = f"📈+{change:.1f}%" if change >= 0 else f"📉{change:.1f}%"
    
    msg = f"""🔔 <b>{result['symbol']} ANALİZ</b>
💰 Fiyat: ${result['price']:,.2f} ({change_str})
📊 TF: {result['best_tf']} | {dir_emoji} {result['direction']}
🎯 Güven: {conf:.0f}/100 {conf_bar}
📍 Rejim: {regime_emoji} {result['regime']}

"""
    
    if result.get('category_tops'):
        msg += "⭐ <b>Kategori Sinyalleri:</b>\n"
        cat_emoji = {'trend': '📊', 'momentum': '⚡', 'volatility': '📉', 'volume': '📶'}
        
        for cat in ['trend', 'momentum', 'volatility', 'volume']:
            if cat in result['category_tops']:
                ind = result['category_tops'][cat]
                short_name = shorten_indicator(ind['name'])
                ic = ind['ic']
                msg += f"{cat_emoji[cat]} {cat.title()}: {short_name} ({ic:+.2f})\n"
        msg += "\n"
    
    if result.get('tf_rankings'):
        msg += "📋 <b>TF Sıralaması:</b>\n"
        for i, tf_info in enumerate(result['tf_rankings'][:2]):
            marker = "→" if i == 0 else " "
            dir_mini = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪'}.get(tf_info['direction'], '⚪')
            msg += f"{marker}{tf_info['tf']}: {tf_info['score']:.0f} {dir_mini}\n"
        msg += "\n"
    
    # Risk bilgisi
    atr_pct = result.get('atr_pct', 0)
    atr = result.get('atr', 0)
    high_24h = result.get('high_24h', 0)
    low_24h = result.get('low_24h', 0)
    price = result.get('price', 0)
    
    if atr_pct > 0:
        stop_distance = atr * 1.5  # 1.5x ATR stop
        
        # Volatilite seviyesi
        if atr_pct > 4:
            vol_emoji = "🔴"
            vol_text = "Yüksek"
        elif atr_pct > 2:
            vol_emoji = "🟡"
            vol_text = "Normal"
        else:
            vol_emoji = "🟢"
            vol_text = "Düşük"
        
        msg += f"""⚠️ <b>Risk Bilgisi:</b>
{vol_emoji} Volatilite: %{atr_pct:.1f} ({vol_text})
🛑 Stop Mesafesi: ${stop_distance:,.0f} (1.5x ATR)
📏 24h Range: ${low_24h:,.0f} - ${high_24h:,.0f}"""
    
    return msg


# =============================================================================
# TELEGRAM KOMUTLARI
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Başlangıç mesajı."""
    msg = """🤖 <b>BTC Decision Bot</b>

IC bazlı kripto analiz botu.

<b>Komutlar:</b>
/analiz [COIN] - IC analizi
/fiyat [COIN] - Anlık fiyat
/liste - Desteklenen coinler

<b>Örnekler:</b>
<code>/analiz BTC</code>
<code>/fiyat ETH</code>
<code>/a XRP</code>"""
    
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_liste(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Desteklenen coinleri listeler."""
    total = len(SUPPORTED_COINS)
    # İlk 30 tanesini göster
    sample = ", ".join(SUPPORTED_COINS[:30])
    msg = f"""📋 <b>Desteklenen Coinler</b>

Toplam: <b>{total}</b> coin (Binance USDT çiftleri)

<b>Örnekler:</b>
<code>{sample}...</code>

💡 Herhangi bir USDT çiftini analiz edebilirsin:
<code>/analiz BTC</code>
<code>/analiz PEPE</code>
<code>/analiz ARB</code>"""
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def cmd_analiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """IC analizi yapar."""
    
    if context.args:
        symbol = context.args[0].upper()
    else:
        symbol = 'BTC'
    
    if symbol not in SUPPORTED_COINS:
        # Belki küçük/büyük harf sorunu vardır, tekrar kontrol et
        symbol_check = symbol.upper()
        if symbol_check not in SUPPORTED_COINS:
            await update.message.reply_text(
                f"❌ {symbol} Binance'da bulunamadı.\n"
                f"Doğru yazdığından emin ol (örn: BTC, ETH, PEPE)"
            )
            return
        symbol = symbol_check
    
    loading_msg = await update.message.reply_text(
        f"⏳ {symbol} analiz ediliyor (~30-60 saniye)..."
    )
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_analysis, symbol)
        
        if result is None:
            await loading_msg.edit_text(f"❌ {symbol} için veri alınamadı.")
            return
        
        analysis_msg = format_analysis_message(result)
        await loading_msg.edit_text(analysis_msg, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")
        await loading_msg.edit_text(f"❌ Hata: {str(e)[:100]}")


async def cmd_fiyat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Anlık fiyat gösterir."""
    
    if context.args:
        symbol = context.args[0].upper()
    else:
        symbol = 'BTC'
    
    try:
        fetcher = DataFetcher(symbol=f"{symbol}/USDT")
        ticker = fetcher.get_latest_price()
        
        price = ticker['last']
        change = ticker.get('change_24h', 0) or 0
        high = ticker.get('high_24h', 0)
        low = ticker.get('low_24h', 0)
        
        change_emoji = "📈" if change >= 0 else "📉"
        
        msg = f"""💰 <b>{symbol}/USDT</b>

Fiyat: <code>${price:,.2f}</code>
24h: {change_emoji} {change:+.2f}%
High: ${high:,.2f}
Low: ${low:,.2f}"""
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Hata: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN bulunamadı!")
        return
    
    logger.info("Bot başlatılıyor...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("liste", cmd_liste))
    app.add_handler(CommandHandler("analiz", cmd_analiz))
    app.add_handler(CommandHandler("a", cmd_analiz))
    app.add_handler(CommandHandler("fiyat", cmd_fiyat))
    app.add_handler(CommandHandler("f", cmd_fiyat))
    
    logger.info("Bot hazır!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

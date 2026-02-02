# =============================================================================
# KRİPTO KARAR DESTEK SİSTEMİ - STREAMLIT WEB DASHBOARD
# =============================================================================
# Çalıştırma: streamlit run app_fixed.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# PATH AYARLARI
# =============================================================================
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
for subdir in ['data', 'indicators', 'notifications']:
    module_path = CURRENT_DIR / subdir
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from fetcher import DataFetcher
from calculator import IndicatorCalculator
from selector import IndicatorSelector, IndicatorScore

# =============================================================================
# SAYFA YAPILANDIRMASI
# =============================================================================

st.set_page_config(
    page_title="🚀 Kripto Karar Destek",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SABİTLER
# =============================================================================

SUPPORTED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
]

TIMEFRAME_CONFIG = {
    '5m':  {'bars': 500, 'desc': 'Scalping', 'minutes': 5},
    '15m': {'bars': 400, 'desc': 'Day Trading', 'minutes': 15},
    '30m': {'bars': 300, 'desc': 'Kısa Swing', 'minutes': 30},
    '1h':  {'bars': 250, 'desc': 'Intraday', 'minutes': 60},
    '2h':  {'bars': 200, 'desc': 'Swing', 'minutes': 120},
    '4h':  {'bars': 150, 'desc': 'Position', 'minutes': 240},
}

# =============================================================================
# DATACLASS
# =============================================================================

@dataclass
class TimeframeAnalysis:
    """Bir timeframe'in analiz sonucu."""
    timeframe: str
    top_ic: float
    top_indicator: str
    avg_ic: float
    significant_count: int
    consistency: float
    direction: str
    composite_score: float
    regime: str
    scores: List[IndicatorScore]

# =============================================================================
# CACHED FONKSİYONLAR
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, timeframe: str, bars: int):
    """OHLCV verisini 5 dakika cache ile çeker."""
    try:
        fetcher = DataFetcher(symbol=symbol)
        return fetcher.fetch_ohlcv(timeframe=timeframe, limit=bars)
    except Exception as e:
        return None

@st.cache_resource
def get_calculator():
    """Singleton calculator."""
    return IndicatorCalculator(verbose=False)

@st.cache_resource
def get_selector():
    """Singleton selector."""
    return IndicatorSelector(alpha=0.05, correction_method='fdr', verbose=False)

# =============================================================================
# ANALİZ FONKSİYONLARI
# =============================================================================

def fetch_all_parallel(symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """Paralel veri çekme."""
    def fetch_single(tf):
        bars = TIMEFRAME_CONFIG[tf]['bars']
        return (tf, fetch_ohlcv_cached(symbol, tf, bars))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda tf: fetch_single(tf), timeframes))
    
    return {tf: df for tf, df in results if df is not None and len(df) > 50}


def detect_regime(df: pd.DataFrame) -> str:
    """ADX bazlı piyasa rejimi tespiti."""
    if 'ADX_14' not in df.columns:
        return 'unknown'
    
    adx = df['ADX_14'].iloc[-1]
    if pd.isna(adx):
        return 'unknown'
    
    dmp = df.get('DMP_14', pd.Series([50])).iloc[-1] if 'DMP_14' in df.columns else 50
    dmn = df.get('DMN_14', pd.Series([50])).iloc[-1] if 'DMN_14' in df.columns else 50
    
    if adx > 25:
        return 'trending_up' if dmp > dmn else 'trending_down'
    elif adx < 20:
        return 'ranging'
    return 'transitioning'


def analyze_timeframe(df: pd.DataFrame, tf: str, calc, sel, fwd: int):
    """Tek TF analizi."""
    try:
        # İndikatörler
        df_ind = calc.calculate_all(df, categories=['trend', 'momentum', 'volatility', 'volume'])
        df_ind = calc.add_price_features(df_ind)
        df_ind = calc.add_forward_returns(df_ind, periods=[1, 5, 10])
        
        # IC analizi
        scores = sel.evaluate_all_indicators(df_ind, target_col=f'fwd_ret_{fwd}')
        
        # Anlamlı olanlar
        valid_cats = ['trend', 'momentum', 'volatility', 'volume']
        sig = [s for s in scores if abs(s.ic_mean) > 0.02 and not np.isnan(s.ic_mean) and s.category in valid_cats]
        
        if not sig:
            return None
        
        # Metrikler
        top = max(sig, key=lambda x: abs(x.ic_mean))
        avg_ic = np.mean([abs(s.ic_mean) for s in sig])
        
        pos = sum(1 for s in sig if s.ic_mean > 0)
        neg = sum(1 for s in sig if s.ic_mean < 0)
        cons = max(pos, neg) / len(sig)
        
        direction = 'SHORT' if neg > pos * 1.5 else 'LONG' if pos > neg * 1.5 else 'NEUTRAL'
        regime = detect_regime(df_ind)
        
        # Composite skor
        top_norm = min((abs(top.ic_mean) - 0.02) / 0.38 * 100, 100)
        avg_norm = min((avg_ic - 0.02) / 0.13 * 100, 100)
        cnt_norm = min(len(sig) / 50 * 100, 100)
        cons_norm = max(0, min((cons - 0.5) / 0.5 * 100, 100))
        
        score = (top_norm * 0.40 + avg_norm * 0.25 + cnt_norm * 0.15 + cons_norm * 0.20)
        
        if regime == 'ranging':
            score *= 0.85
        elif regime == 'volatile':
            score *= 0.80
        elif regime == 'transitioning':
            score *= 0.90
        
        return TimeframeAnalysis(tf, abs(top.ic_mean), top.name, avg_ic, len(sig), cons, direction, score, regime, scores)
    except Exception as e:
        return None


def run_analysis(symbol: str, tfs: List[str], fwd: int):
    """Tam analiz pipeline."""
    progress = st.progress(0, text="Veriler çekiliyor...")
    start = time.time()
    
    data = fetch_all_parallel(symbol, tfs)
    progress.progress(30, text="İndikatörler hesaplanıyor...")
    
    if not data:
        st.error("Veri çekilemedi!")
        return {}, [], 0.0
    
    first_tf = min(data.keys(), key=lambda x: TIMEFRAME_CONFIG[x]['minutes'])
    price = data[first_tf]['close'].iloc[-1]
    
    calc, sel = get_calculator(), get_selector()
    analyses = []
    
    for i, (tf, df) in enumerate(data.items()):
        progress.progress(30 + int((i+1)/len(data)*60), text=f"{tf} analiz ediliyor...")
        result = analyze_timeframe(df, tf, calc, sel, fwd)
        if result:
            analyses.append(result)
    
    analyses.sort(key=lambda x: x.composite_score, reverse=True)
    progress.progress(100, text="Tamamlandı!")
    st.success(f"✅ Analiz: {time.time()-start:.1f}s")
    
    return data, analyses, price

# =============================================================================
# GRAFİKLER
# =============================================================================

def price_chart(df: pd.DataFrame, symbol: str, tf: str):
    """Candlestick grafik."""
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#00d26a', decreasing_line_color='#ff4757'
    ))
    fig.update_layout(
        title=f'{symbol} - {tf}', template='plotly_dark', height=400,
        xaxis_rangeslider_visible=False, showlegend=False
    )
    return fig


def tf_chart(analyses: List[TimeframeAnalysis]):
    """TF karşılaştırma bar chart."""
    colors = ['#00d26a' if a.direction=='LONG' else '#ff4757' if a.direction=='SHORT' else '#ffa502' for a in analyses]
    fig = go.Figure(go.Bar(
        x=[a.timeframe for a in analyses],
        y=[a.composite_score for a in analyses],
        marker_color=colors,
        text=[f"{a.composite_score:.0f}" for a in analyses],
        textposition='outside'
    ))
    fig.update_layout(
        title='Timeframe Skorları', template='plotly_dark', height=300, yaxis_range=[0, 100]
    )
    return fig


def indicator_chart(scores: List[IndicatorScore], n: int = 20):
    """Top indikatör bar chart."""
    valid = sorted([s for s in scores if not np.isnan(s.ic_mean)], key=lambda x: abs(x.ic_mean), reverse=True)[:n]
    colors = ['#00d26a' if s.ic_mean > 0 else '#ff4757' for s in valid]
    
    fig = go.Figure(go.Bar(
        y=[s.name[:20] for s in valid],
        x=[s.ic_mean for s in valid],
        orientation='h', marker_color=colors,
        text=[f"{s.ic_mean:+.3f}" for s in valid], textposition='outside'
    ))
    fig.update_layout(
        title=f'Top {n} İndikatör (IC)', template='plotly_dark', height=500,
        yaxis=dict(autorange="reversed")
    )
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================

def sidebar():
    """Kullanıcı ayarları."""
    st.sidebar.markdown("## ⚙️ Ayarlar")
    
    symbol = st.sidebar.selectbox("📈 Parite", SUPPORTED_PAIRS, index=0)
    custom = st.sidebar.text_input("Özel Parite", placeholder="PEPE/USDT")
    if custom:
        symbol = custom.upper()
    
    st.sidebar.markdown("---")
    
    tfs = st.sidebar.multiselect(
        "⏱️ Timeframe'ler",
        list(TIMEFRAME_CONFIG.keys()),
        default=['15m', '30m', '1h', '2h', '4h'],
        format_func=lambda x: f"{x} ({TIMEFRAME_CONFIG[x]['desc']})"
    )
    
    st.sidebar.markdown("---")
    
    # İleri ayarlar
    with st.sidebar.expander("🔧 İleri Ayarlar"):
        fwd = st.slider("Forward Period", 1, 20, 5)
    
    st.sidebar.markdown("---")
    run = st.sidebar.button("🚀 Analiz Başlat", type="primary", use_container_width=True)
    
    return {'symbol': symbol, 'tfs': tfs, 'fwd': fwd, 'run': run}

# =============================================================================
# SONUÇLAR
# =============================================================================

def show_results():
    """Analiz sonuçlarını göster."""
    analyses = st.session_state['analyses']
    data = st.session_state['data']
    price = st.session_state['price']
    symbol = st.session_state['symbol']
    best = analyses[0]
    
    st.markdown("---")
    
    # Metrikler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Fiyat", f"${price:,.2f}")
    c2.metric("📊 En İyi TF", best.timeframe, f"Skor: {best.composite_score:.0f}")
    
    dir_emoji = "🟢" if best.direction=="LONG" else "🔴" if best.direction=="SHORT" else "⚪"
    c3.metric("🎯 Yön", f"{dir_emoji} {best.direction}", f"Tutarlılık: {best.consistency:.0%}")
    
    regime_map = {'trending_up': '📈 Trend↑', 'trending_down': '📉 Trend↓', 'ranging': '↔️ Yatay', 
                  'volatile': '⚡ Volatil', 'transitioning': '🔄 Geçiş', 'unknown': '❓'}
    c4.metric("📍 Rejim", regime_map.get(best.regime, '❓'), f"Top IC: {best.top_ic:.3f}")
    
    st.markdown("---")
    
    # Tablar
    t1, t2, t3, t4 = st.tabs(["📊 TF Karşılaştırma", "📈 Fiyat", "🎯 İndikatörler", "📋 Detay"])
    
    with t1:
        st.plotly_chart(tf_chart(analyses), use_container_width=True)
        st.dataframe(pd.DataFrame([{
            'TF': a.timeframe, 'Skor': f"{a.composite_score:.1f}", 'Top IC': f"{a.top_ic:.4f}",
            'Top İnd': a.top_indicator[:25], 'N': a.significant_count, 'Yön': a.direction
        } for a in analyses]), use_container_width=True, hide_index=True)
    
    with t2:
        if best.timeframe in data:
            st.plotly_chart(price_chart(data[best.timeframe], symbol, best.timeframe), use_container_width=True)
    
    with t3:
        st.plotly_chart(indicator_chart(best.scores), use_container_width=True)
        st.markdown("**IC Yorumu:** 🟢 IC>0 = LONG, 🔴 IC<0 = SHORT, |IC|>0.05 = Güçlü")
    
    with t4:
        tf_sel = st.selectbox("TF Seç", [a.timeframe for a in analyses])
        sel_a = next((a for a in analyses if a.timeframe == tf_sel), None)
        if sel_a:
            df = pd.DataFrame([{
                'İndikatör': s.name, 'Kategori': s.category, 'IC': s.ic_mean,
                't-stat': s.ic_tstat, 'p-value': s.p_value, 'N': s.n_observations
            } for s in sel_a.scores[:50] if not np.isnan(s.ic_mean)])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Özet
    if best.direction == 'LONG' and best.composite_score >= 60:
        st.success(f"**📈 LONG** | TF: {best.timeframe} | Güven: {best.composite_score:.0f}/100 | Top: {best.top_indicator[:20]} (IC={best.top_ic:+.3f})")
    elif best.direction == 'SHORT' and best.composite_score >= 60:
        st.error(f"**📉 SHORT** | TF: {best.timeframe} | Güven: {best.composite_score:.0f}/100 | Top: {best.top_indicator[:20]} (IC={best.top_ic:+.3f})")
    else:
        st.warning(f"**↔️ NEUTRAL** | TF: {best.timeframe} | Güven: {best.composite_score:.0f}/100 | Karışık sinyal")

# =============================================================================
# MAIN
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🚀 Kripto Karar Destek Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#888;">IC Bazlı İstatistiksel Analiz</p>', unsafe_allow_html=True)
    
    settings = sidebar()
    
    # fwd değişkeni expander dışında tanımlı değilse varsayılan ata
    if 'fwd' not in settings or settings['fwd'] is None:
        settings['fwd'] = 5
    
    if not settings['tfs']:
        st.warning("⚠️ En az bir timeframe seçin!")
        return
    
    if settings['run']:
        st.markdown("---")
        data, analyses, price = run_analysis(settings['symbol'], settings['tfs'], settings['fwd'])
        
        if analyses:
            st.session_state['analyses'] = analyses
            st.session_state['data'] = data
            st.session_state['price'] = price
            st.session_state['symbol'] = settings['symbol']
    
    if 'analyses' in st.session_state and st.session_state['analyses']:
        show_results()

if __name__ == "__main__":
    main()

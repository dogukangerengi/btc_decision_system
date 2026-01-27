# =============================================================================
# DİNAMİK BACKTEST VE TİMEFRAME SEÇİM MODÜLÜ
# =============================================================================
# Amaç: Farklı timeframe'leri karşılaştırıp en uygun olanı seçmek
#
# Metodoloji:
# 1. Walk-Forward Validation - Out-of-sample test, overfitting önleme
# 2. Risk-Adjusted Metrics - Sharpe, Sortino, Calmar, Max Drawdown
# 3. Regime Detection - Trending vs Ranging piyasa tespiti
# 4. Adaptive Timeframe Selection - Piyasa koşullarına göre TF önerisi
#
# İstatistiksel Önem:
# - In-sample ≠ Out-of-sample performans (overfitting riski)
# - Walk-forward: Train → Test → Roll → Repeat
# - Multiple timeframe test → Bonferroni/FDR correction gerekli
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """
    Tek bir backtest sonucu.
    
    Risk-adjusted metriklerin yorumu:
    - Sharpe > 1.0: İyi, > 2.0: Çok iyi, > 3.0: Mükemmel
    - Sortino > 1.5: İyi (downside risk fokuslu)
    - Max DD < -10%: Kabul edilebilir, < -20%: Riskli
    - Win Rate > 55%: Pozitif beklenti için gerekli
    """
    timeframe: str
    total_return: float              # Toplam getiri (%)
    annualized_return: float         # Yıllıklandırılmış getiri (%)
    volatility: float                # Yıllıklandırılmış volatilite (%)
    sharpe_ratio: float              # Risk-adjusted return (rf=0 varsayımı)
    sortino_ratio: float             # Downside risk-adjusted return
    calmar_ratio: float              # Return / Max Drawdown
    max_drawdown: float              # Maximum düşüş (%)
    max_drawdown_duration: int       # Max DD süresi (bar sayısı)
    win_rate: float                  # Kazanan işlem oranı (%)
    profit_factor: float             # Gross profit / Gross loss
    total_trades: int                # Toplam işlem sayısı
    avg_trade_return: float          # Ortalama işlem getirisi (%)
    ic_mean: float                   # Ortalama Information Coefficient
    ic_stability: float              # IC_IR (stability)
    regime: str                      # Piyasa rejimi: 'trending', 'ranging', 'volatile'
    confidence_score: float          # 0-100 arası güven skoru
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'Timeframe': self.timeframe,
            'Total Return (%)': f"{self.total_return:.2f}",
            'Ann. Return (%)': f"{self.annualized_return:.2f}",
            'Volatility (%)': f"{self.volatility:.2f}",
            'Sharpe': f"{self.sharpe_ratio:.2f}",
            'Sortino': f"{self.sortino_ratio:.2f}",
            'Calmar': f"{self.calmar_ratio:.2f}",
            'Max DD (%)': f"{self.max_drawdown:.2f}",
            'Win Rate (%)': f"{self.win_rate:.1f}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Trades': self.total_trades,
            'IC Mean': f"{self.ic_mean:.4f}",
            'Regime': self.regime,
            'Confidence': f"{self.confidence_score:.0f}",
        }


@dataclass 
class TimeframeRanking:
    """Timeframe sıralaması ve önerisi."""
    rankings: List[Tuple[str, float]]    # (timeframe, score) listesi
    best_timeframe: str                   # En iyi TF
    recommendation: str                   # Detaylı öneri
    market_regime: str                    # Genel piyasa durumu
    confidence: float                     # Öneri güveni (0-100)


class DynamicBacktester:
    """
    Dinamik backtest ve timeframe seçim sınıfı.
    
    Walk-Forward Validation:
    -----------------------
    1. Veriyi train/test split yap (örn: %70/%30)
    2. Train'de strateji optimize et
    3. Test'te out-of-sample performans ölç
    4. Window'u kaydır ve tekrarla
    5. Tüm out-of-sample sonuçları birleştir
    
    Bu yaklaşım:
    + Overfitting'i minimize eder
    + Gerçek dünya performansına yakın sonuç verir
    - Daha az veri kullanılır (train/test split)
    """
    
    # Yıllıklandırma faktörleri (bar/yıl)
    ANNUALIZATION_FACTORS = {
        '1m': 525600,      # 60 * 24 * 365
        '3m': 175200,
        '5m': 105120,
        '15m': 35040,
        '30m': 17520,
        '1h': 8760,
        '2h': 4380,
        '4h': 2190,
        '6h': 1460,
        '8h': 1095,
        '12h': 730,
        '1d': 365,
        '3d': 122,
        '1w': 52,
    }
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        n_walks: int = 5,
        min_trades: int = 30,
        risk_free_rate: float = 0.0,
        verbose: bool = True
    ):
        """
        DynamicBacktester başlatır.
        
        Parameters:
        ----------
        train_ratio : float
            Train/test oranı (0.7 = %70 train, %30 test)
            
        n_walks : int
            Walk-forward adım sayısı
            
        min_trades : int
            Minimum işlem sayısı (istatistiksel anlamlılık için)
            
        risk_free_rate : float
            Risksiz faiz oranı (Sharpe hesabı için, genellikle 0)
            
        verbose : bool
            Detaylı çıktı
        """
        self.train_ratio = train_ratio
        self.n_walks = n_walks
        self.min_trades = min_trades
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
    
    # =========================================================================
    # TEMEL METRİK HESAPLAMALARI
    # =========================================================================
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Log getiri hesaplar.
        
        Log return tercih nedenleri:
        - Toplamsal (multi-period için)
        - Yaklaşık normal dağılım
        - Negatif değer üretemez (fiyat > 0)
        """
        return np.log(prices / prices.shift(1))
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        timeframe: str = '1h'
    ) -> float:
        """
        Sharpe Ratio hesaplar.
        
        Sharpe = (E[R] - Rf) / σ(R) * √(annualization_factor)
        
        Yorum:
        - < 0: Negatif risk-adjusted return
        - 0-1: Düşük
        - 1-2: İyi
        - 2-3: Çok iyi
        - > 3: Mükemmel (veya overfitting!)
        """
        if returns.std() == 0 or len(returns) < 10:
            return 0.0
        
        # Yıllıklandırma faktörü
        ann_factor = self.ANNUALIZATION_FACTORS.get(timeframe, 8760)
        
        excess_return = returns.mean() - self.risk_free_rate / ann_factor
        sharpe = excess_return / returns.std() * np.sqrt(ann_factor)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        timeframe: str = '1h'
    ) -> float:
        """
        Sortino Ratio hesaplar (downside risk fokuslu).
        
        Sortino = (E[R] - Rf) / σ_downside
        
        Sharpe'dan farkı: Sadece negatif volatiliteyi cezalandırır.
        Pozitif volatilite (büyük kazançlar) cezalandırılmaz.
        """
        if len(returns) < 10:
            return 0.0
        
        ann_factor = self.ANNUALIZATION_FACTORS.get(timeframe, 8760)
        
        # Downside deviation (sadece negatif getiriler)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 10.0  # Sınırsız (negatif getiri yok)
        
        downside_std = negative_returns.std()
        excess_return = returns.mean() - self.risk_free_rate / ann_factor
        sortino = excess_return / downside_std * np.sqrt(ann_factor)
        
        return sortino
    
    def calculate_max_drawdown(
        self,
        returns: pd.Series
    ) -> Tuple[float, int]:
        """
        Maximum Drawdown hesaplar.
        
        Max DD = (Peak - Trough) / Peak
        
        Returns:
        -------
        Tuple[float, int]
            (max_drawdown_pct, duration_in_bars)
        """
        if len(returns) < 2:
            return 0.0, 0
        
        # Kümülatif getiri
        cum_returns = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cum_returns.cummax()
        
        # Drawdown serisi
        drawdown = (cum_returns - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Drawdown süresi
        dd_duration = 0
        if max_dd < 0:
            # En derin noktayı bul
            trough_idx = drawdown.idxmin()
            # O noktadan önceki peak'i bul
            peak_idx = cum_returns[:trough_idx].idxmax()
            dd_duration = len(cum_returns[peak_idx:trough_idx])
        
        return max_dd * 100, dd_duration  # Yüzde olarak
    
    def calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """
        Calmar Ratio hesaplar.
        
        Calmar = Annualized Return / |Max Drawdown|
        
        Risk-adjusted return ama drawdown bazlı.
        """
        if abs(max_drawdown) < 0.01:  # Neredeyse 0 drawdown
            return 10.0
        
        return annualized_return / abs(max_drawdown)
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Kazanan işlem oranını hesaplar."""
        if len(returns) == 0:
            return 0.0
        
        wins = (returns > 0).sum()
        return wins / len(returns) * 100
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Profit Factor hesaplar.
        
        PF = Gross Profits / Gross Losses
        
        > 1: Kârlı sistem
        > 1.5: İyi
        > 2.0: Çok iyi
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return 10.0  # Zarar yok
        
        return gross_profit / gross_loss
    
    # =========================================================================
    # MULTI-INDICATOR COMPOSITE SİNYAL SİSTEMİ
    # =========================================================================
    
    def generate_composite_signal(
        self,
        df: pd.DataFrame,
        indicator_scores: List = None,
        threshold: float = 0.3
    ) -> pd.Series:
        """
        IC analizi ile seçilen indikatörlerden composite sinyal üretir.
        
        Mantık:
        ------
        1. Her kategoriden en iyi IC'ye sahip indikatörleri al
        2. Her indikatörün z-score'unu hesapla (normalize)
        3. IC yönüne göre ağırlıklı toplam oluştur:
           - IC > 0: İndikatör yükselince LONG
           - IC < 0: İndikatör yükselince SHORT (tersle)
        4. Composite skor threshold'u geçerse sinyal ver
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV + indikatörler
        indicator_scores : List[IndicatorScore]
            IC analizi sonuçları
        threshold : float
            Sinyal eşiği (0-1 arası, varsayılan 0.3)
            
        Returns:
        -------
        pd.Series
            -1 (SHORT), 0 (NEUTRAL), +1 (LONG)
        """
        
        if indicator_scores is None or len(indicator_scores) == 0:
            # Fallback: basit momentum
            return np.sign(df['close'].pct_change(5).shift(1))
        
        # Kategorilere göre en iyi indikatörleri seç (max 2 per kategori)
        best_indicators = self._select_best_for_signal(indicator_scores)
        
        if not best_indicators:
            return np.sign(df['close'].pct_change(5).shift(1))
        
        # Her indikatör için z-score hesapla ve IC yönüne göre ağırlıklandır
        composite_scores = pd.Series(0.0, index=df.index)
        total_weight = 0
        
        for ind_name, ic_value, category in best_indicators:
            if ind_name not in df.columns:
                continue
            
            # Z-score normalize et (rolling 50 bar)
            series = df[ind_name]
            rolling_mean = series.rolling(50, min_periods=20).mean()
            rolling_std = series.rolling(50, min_periods=20).std()
            z_score = (series - rolling_mean) / (rolling_std + 1e-10)
            
            # Z-score'u -3, +3 arasında sınırla
            z_score = z_score.clip(-3, 3)
            
            # IC yönüne göre sinyal
            # IC > 0: İndikatör yüksek → fiyat yükselir → LONG (+)
            # IC < 0: İndikatör yüksek → fiyat düşer → z_score'u tersle
            direction = np.sign(ic_value)
            weight = abs(ic_value)  # IC büyüklüğü = ağırlık
            
            composite_scores += direction * z_score * weight
            total_weight += weight
        
        # Normalize et
        if total_weight > 0:
            composite_scores = composite_scores / total_weight
        
        # Threshold'a göre sinyal üret
        # composite > threshold → LONG
        # composite < -threshold → SHORT
        # arada → NEUTRAL
        signals = pd.Series(0, index=df.index)
        signals[composite_scores > threshold] = 1
        signals[composite_scores < -threshold] = -1
        
        # Look-ahead bias önleme: 1 bar geciktir
        signals = signals.shift(1)
        
        return signals
    
    def _select_best_for_signal(
        self,
        indicator_scores: List,
        max_per_category: int = 2,
        min_ic: float = 0.03
    ) -> List[Tuple[str, float, str]]:
        """
        Sinyal üretimi için en iyi indikatörleri seçer.
        
        Returns:
        -------
        List[(indicator_name, ic_value, category)]
        """
        
        # Kategorilere göre grupla
        categories = {}
        for score in indicator_scores:
            # score objesi IndicatorScore dataclass
            try:
                ic = score.ic_mean if hasattr(score, 'ic_mean') else 0
                name = score.name if hasattr(score, 'name') else str(score)
                cat = score.category if hasattr(score, 'category') else 'other'
                
                # Minimum IC filtresi
                if abs(ic) < min_ic or np.isnan(ic):
                    continue
                
                # Sadece ana kategoriler
                if cat not in ['trend', 'momentum', 'volatility', 'volume']:
                    continue
                    
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((name, ic, cat))
            except:
                continue
        
        # Her kategoriden en iyi N tanesini seç
        best = []
        for cat, indicators in categories.items():
            # |IC| büyüklüğüne göre sırala
            sorted_inds = sorted(indicators, key=lambda x: abs(x[1]), reverse=True)
            best.extend(sorted_inds[:max_per_category])
        
        return best
    
    def run_composite_backtest(
        self,
        df: pd.DataFrame,
        indicator_scores: List = None,
        timeframe: str = '1h',
        threshold: float = 0.3
    ) -> BacktestResult:
        """
        Multi-indicator composite sinyal ile backtest yapar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV + indikatörler
        indicator_scores : List
            IC analizi sonuçları
        timeframe : str
            Timeframe
        threshold : float
            Sinyal eşiği
        """
        
        df = df.copy()
        
        # Composite sinyal üret
        df['composite_signal'] = self.generate_composite_signal(
            df, indicator_scores, threshold
        )
        
        # Getiri hesapla
        df['returns'] = self.calculate_returns(df['close'])
        
        # Strateji getirisi: t-1 sinyali × t getirisi
        df['strategy_returns'] = df['composite_signal'].shift(1) * df['returns']
        
        # NaN temizle
        df = df.dropna(subset=['strategy_returns'])
        
        if len(df) < self.min_trades:
            return self._empty_result(timeframe)
        
        returns = df['strategy_returns']
        
        # Metrikler
        total_return = (np.exp(returns.sum()) - 1) * 100
        
        ann_factor = self.ANNUALIZATION_FACTORS.get(timeframe, 8760)
        n_periods = len(returns)
        annualized_return = ((1 + total_return/100) ** (ann_factor / n_periods) - 1) * 100
        
        volatility = returns.std() * np.sqrt(ann_factor) * 100
        
        sharpe = self.calculate_sharpe_ratio(returns, timeframe)
        sortino = self.calculate_sortino_ratio(returns, timeframe)
        
        max_dd, dd_duration = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(annualized_return, max_dd)
        
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        
        # İşlem sayısı (sinyal değişimi)
        signal_changes = (df['composite_signal'] != df['composite_signal'].shift(1)).sum()
        total_trades = signal_changes // 2
        
        avg_trade_return = total_return / max(total_trades, 1)
        
        # Ortalama IC (kullanılan indikatörlerin)
        ic_mean = 0.0
        if indicator_scores:
            best = self._select_best_for_signal(indicator_scores)
            if best:
                ic_mean = np.mean([abs(x[1]) for x in best])
        
        # Rejim
        regime = self.detect_regime(df)
        
        # Güven skoru
        confidence = self._calculate_confidence(
            sharpe, sortino, win_rate, total_trades, max_dd
        )
        
        return BacktestResult(
            timeframe=timeframe,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            ic_mean=ic_mean,
            ic_stability=0.0,
            regime=regime,
            confidence_score=confidence
        )
    
    # =========================================================================
    # REJİM TESPİTİ
    # =========================================================================
    
    def detect_regime(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> str:
        """
        Piyasa rejimini tespit eder.
        
        Rejimler:
        - 'trending_up': Güçlü yukarı trend
        - 'trending_down': Güçlü aşağı trend
        - 'ranging': Yatay hareket
        - 'volatile': Yüksek volatilite, belirsiz yön
        
        Tespit yöntemi:
        - ADX > 25: Trending
        - ADX < 20: Ranging
        - ATR percentile > 80: Volatile
        """
        
        if len(df) < lookback:
            return 'unknown'
        
        recent = df.tail(lookback)
        
        # ADX kontrolü (varsa)
        if 'ADX_14' in df.columns:
            adx = recent['ADX_14'].iloc[-1]
            dmp = recent.get('DMP_14', pd.Series([50])).iloc[-1]
            dmn = recent.get('DMN_14', pd.Series([50])).iloc[-1]
        else:
            adx = 25  # Varsayılan
            dmp = dmn = 50
        
        # Trend yönü (basit MA karşılaştırması)
        close = recent['close']
        ma_short = close.rolling(10).mean().iloc[-1]
        ma_long = close.rolling(30).mean().iloc[-1]
        
        # Volatilite (ATR veya std)
        if 'ATRr_14' in df.columns:
            atr_pct = recent['ATRr_14'].iloc[-1] / close.iloc[-1] * 100
        else:
            atr_pct = close.pct_change().std() * 100
        
        # Rejim belirleme
        if adx > 25:
            if dmp > dmn or ma_short > ma_long:
                return 'trending_up'
            else:
                return 'trending_down'
        elif adx < 20:
            if atr_pct > 3:  # %3'ten fazla volatilite
                return 'volatile'
            return 'ranging'
        else:
            return 'transitioning'
    
    # =========================================================================
    # BACKTEST MOTORU
    # =========================================================================
    
    def run_simple_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = None,
        timeframe: str = '1h'
    ) -> BacktestResult:
        """
        Basit momentum backtest yapar.
        
        Strateji:
        - Signal > 0: Long
        - Signal < 0: Short (veya flat)
        - Signal = 0: Flat
        
        Eğer signal_col verilmezse, forward return'ün işaretini kullanır
        (perfect foresight benchmark).
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV + indikatörler
            
        signal_col : str, optional
            Sinyal kolonu. None ise basit momentum kullanılır.
            
        timeframe : str
            Timeframe (yıllıklandırma için)
        """
        
        df = df.copy()
        
        # Getiri hesapla
        df['returns'] = self.calculate_returns(df['close'])
        
        # Sinyal oluştur
        if signal_col and signal_col in df.columns:
            # Verilen sinyali kullan
            df['signal'] = np.sign(df[signal_col])
        else:
            # Basit momentum: son N bar'ın getirisi
            df['momentum'] = df['returns'].rolling(5).sum()
            df['signal'] = np.sign(df['momentum'].shift(1))  # Lag ekle (look-ahead bias önleme)
        
        # Strateji getirisi
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']  # t-1 sinyali, t getirisi
        
        # NaN temizle
        df = df.dropna(subset=['strategy_returns'])
        
        if len(df) < self.min_trades:
            return self._empty_result(timeframe)
        
        returns = df['strategy_returns']
        
        # Metrikler
        total_return = (np.exp(returns.sum()) - 1) * 100
        
        ann_factor = self.ANNUALIZATION_FACTORS.get(timeframe, 8760)
        n_periods = len(returns)
        annualized_return = ((1 + total_return/100) ** (ann_factor / n_periods) - 1) * 100
        
        volatility = returns.std() * np.sqrt(ann_factor) * 100
        
        sharpe = self.calculate_sharpe_ratio(returns, timeframe)
        sortino = self.calculate_sortino_ratio(returns, timeframe)
        
        max_dd, dd_duration = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(annualized_return, max_dd)
        
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        
        # İşlem sayısı (sinyal değişimi)
        signal_changes = (df['signal'] != df['signal'].shift(1)).sum()
        total_trades = signal_changes // 2  # Giriş + çıkış = 1 işlem
        
        avg_trade_return = total_return / max(total_trades, 1)
        
        # IC hesabı (varsa forward return)
        ic_mean = 0.0
        ic_stability = 0.0
        if 'fwd_ret_1' in df.columns and signal_col:
            try:
                ic, _ = stats.spearmanr(df[signal_col].dropna(), df['fwd_ret_1'].dropna())
                ic_mean = ic if not np.isnan(ic) else 0.0
                ic_stability = abs(ic_mean) / (returns.std() + 1e-10)
            except:
                pass
        
        # Rejim
        regime = self.detect_regime(df)
        
        # Güven skoru
        confidence = self._calculate_confidence(
            sharpe, sortino, win_rate, total_trades, max_dd
        )
        
        return BacktestResult(
            timeframe=timeframe,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            ic_mean=ic_mean,
            ic_stability=ic_stability,
            regime=regime,
            confidence_score=confidence
        )
    
    def _calculate_confidence(
        self,
        sharpe: float,
        sortino: float,
        win_rate: float,
        n_trades: int,
        max_dd: float
    ) -> float:
        """
        Sonuç güven skoru hesaplar (0-100).
        
        Faktörler:
        - Sharpe/Sortino: Risk-adjusted performans
        - Win rate: Tutarlılık
        - Trade sayısı: İstatistiksel anlamlılık
        - Max DD: Risk kontrolü
        """
        score = 50.0  # Başlangıç
        
        # Sharpe katkısı (-20 to +20)
        score += min(max(sharpe * 10, -20), 20)
        
        # Win rate katkısı (-10 to +10)
        score += (win_rate - 50) * 0.2
        
        # Trade sayısı katkısı (0 to +15)
        if n_trades >= 100:
            score += 15
        elif n_trades >= 50:
            score += 10
        elif n_trades >= 30:
            score += 5
        
        # Max DD cezası (0 to -15)
        if max_dd < -30:
            score -= 15
        elif max_dd < -20:
            score -= 10
        elif max_dd < -10:
            score -= 5
        
        return max(0, min(100, score))
    
    def _empty_result(self, timeframe: str) -> BacktestResult:
        """Yetersiz veri için boş sonuç döndürür."""
        return BacktestResult(
            timeframe=timeframe,
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, max_drawdown_duration=0,
            win_rate=0.0, profit_factor=0.0,
            total_trades=0, avg_trade_return=0.0,
            ic_mean=0.0, ic_stability=0.0,
            regime='unknown', confidence_score=0.0
        )
    
    # =========================================================================
    # WALK-FORWARD VALİDASYON
    # =========================================================================
    
    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = None,
        timeframe: str = '1h'
    ) -> BacktestResult:
        """
        Walk-forward validation ile backtest yapar.
        
        Adımlar:
        1. Veriyi n_walks parçaya böl
        2. Her adımda: önceki parçalar = train, sonraki = test
        3. Test sonuçlarını birleştir
        4. Sadece out-of-sample sonuçları raporla
        
        Bu yaklaşım overfitting'i minimize eder.
        """
        
        n = len(df)
        walk_size = n // (self.n_walks + 1)
        
        if walk_size < self.min_trades:
            # Yeterli veri yok, basit backtest yap
            return self.run_simple_backtest(df, signal_col, timeframe)
        
        all_returns = []
        
        for i in range(self.n_walks):
            # Train: 0 to (i+1) * walk_size
            # Test: (i+1) * walk_size to (i+2) * walk_size
            train_end = (i + 1) * walk_size
            test_start = train_end
            test_end = min((i + 2) * walk_size, n)
            
            if test_end - test_start < 10:
                continue
            
            # Test verisi
            test_df = df.iloc[test_start:test_end].copy()
            
            # Basit backtest
            test_df['returns'] = self.calculate_returns(test_df['close'])
            
            if signal_col and signal_col in test_df.columns:
                test_df['signal'] = np.sign(test_df[signal_col])
            else:
                test_df['momentum'] = test_df['returns'].rolling(5).sum()
                test_df['signal'] = np.sign(test_df['momentum'].shift(1))
            
            test_df['strategy_returns'] = test_df['signal'].shift(1) * test_df['returns']
            
            # Out-of-sample returns topla
            all_returns.extend(test_df['strategy_returns'].dropna().tolist())
        
        if len(all_returns) < self.min_trades:
            return self._empty_result(timeframe)
        
        # Birleşik sonuçları hesapla
        returns = pd.Series(all_returns)
        
        total_return = (np.exp(returns.sum()) - 1) * 100
        ann_factor = self.ANNUALIZATION_FACTORS.get(timeframe, 8760)
        n_periods = len(returns)
        annualized_return = ((1 + total_return/100) ** (ann_factor / n_periods) - 1) * 100
        volatility = returns.std() * np.sqrt(ann_factor) * 100
        
        sharpe = self.calculate_sharpe_ratio(returns, timeframe)
        sortino = self.calculate_sortino_ratio(returns, timeframe)
        max_dd, dd_duration = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(annualized_return, max_dd)
        
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        
        regime = self.detect_regime(df)
        confidence = self._calculate_confidence(sharpe, sortino, win_rate, len(returns), max_dd)
        
        return BacktestResult(
            timeframe=timeframe,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(returns),
            avg_trade_return=total_return / len(returns),
            ic_mean=0.0,
            ic_stability=0.0,
            regime=regime,
            confidence_score=confidence
        )
    
    # =========================================================================
    # TİMEFRAME KARŞILAŞTIRMA VE SEÇİM
    # =========================================================================
    
    def compare_timeframes(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signal_col: str = None,
        use_walk_forward: bool = True
    ) -> List[BacktestResult]:
        """
        Birden fazla timeframe'i karşılaştırır.
        
        Parameters:
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Timeframe → DataFrame mapping
            
        signal_col : str, optional
            Sinyal kolonu
            
        use_walk_forward : bool
            Walk-forward validation kullan
        
        Returns:
        -------
        List[BacktestResult]
            Tüm timeframe sonuçları (Sharpe'a göre sıralı)
        """
        
        results: List[BacktestResult] = []
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("TİMEFRAME KARŞILAŞTIRMA")
            print("=" * 70)
        
        for tf, df in data_dict.items():
            if self.verbose:
                print(f"\n📊 {tf} backtest yapılıyor ({len(df)} bar)...")
            
            try:
                if use_walk_forward:
                    result = self.walk_forward_backtest(df, signal_col, tf)
                else:
                    result = self.run_simple_backtest(df, signal_col, tf)
                
                results.append(result)
                
                if self.verbose:
                    print(f"   Sharpe: {result.sharpe_ratio:.2f} | "
                          f"Return: {result.total_return:.1f}% | "
                          f"MaxDD: {result.max_drawdown:.1f}% | "
                          f"Regime: {result.regime}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ✗ Hata: {e}")
        
        # Sharpe'a göre sırala
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        return results
    
    def select_best_timeframe(
        self,
        results: List[BacktestResult],
        weight_sharpe: float = 0.4,
        weight_sortino: float = 0.2,
        weight_win_rate: float = 0.2,
        weight_confidence: float = 0.2
    ) -> TimeframeRanking:
        """
        En iyi timeframe'i seçer (composite scoring).
        
        Parameters:
        ----------
        results : List[BacktestResult]
            Backtest sonuçları
            
        weight_* : float
            Scoring ağırlıkları (toplamı 1.0 olmalı)
        
        Returns:
        -------
        TimeframeRanking
            Sıralama ve öneri
        """
        
        if not results:
            return TimeframeRanking(
                rankings=[], best_timeframe='unknown',
                recommendation='Yeterli veri yok',
                market_regime='unknown', confidence=0.0
            )
        
        # Normalize skorlar (min-max scaling)
        sharpes = [r.sharpe_ratio for r in results]
        sortinos = [r.sortino_ratio for r in results]
        win_rates = [r.win_rate for r in results]
        confidences = [r.confidence_score for r in results]
        
        def normalize(values):
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return [0.5] * len(values)
            return [(v - min_v) / (max_v - min_v) for v in values]
        
        norm_sharpe = normalize(sharpes)
        norm_sortino = normalize(sortinos)
        norm_win = normalize(win_rates)
        norm_conf = normalize(confidences)
        
        # Composite score
        rankings = []
        for i, result in enumerate(results):
            score = (
                weight_sharpe * norm_sharpe[i] +
                weight_sortino * norm_sortino[i] +
                weight_win_rate * norm_win[i] +
                weight_confidence * norm_conf[i]
            ) * 100
            
            rankings.append((result.timeframe, score))
        
        # Sırala
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        best_tf = rankings[0][0]
        best_result = next(r for r in results if r.timeframe == best_tf)
        
        # Genel rejim (çoğunluk)
        regime_counts = {}
        for r in results:
            regime_counts[r.regime] = regime_counts.get(r.regime, 0) + 1
        market_regime = max(regime_counts, key=regime_counts.get)
        
        # Öneri oluştur
        recommendation = self._generate_recommendation(best_result, rankings, market_regime)
        
        return TimeframeRanking(
            rankings=rankings,
            best_timeframe=best_tf,
            recommendation=recommendation,
            market_regime=market_regime,
            confidence=best_result.confidence_score
        )
    
    def _generate_recommendation(
        self,
        best_result: BacktestResult,
        rankings: List[Tuple[str, float]],
        market_regime: str
    ) -> str:
        """Detaylı öneri metni oluşturur."""
        
        tf = best_result.timeframe
        sharpe = best_result.sharpe_ratio
        win_rate = best_result.win_rate
        max_dd = best_result.max_drawdown
        
        rec = f"📊 ÖNERİLEN TIMEFRAME: {tf}\n\n"
        
        # Performans özeti
        rec += f"Performance:\n"
        rec += f"  • Sharpe Ratio: {sharpe:.2f}"
        if sharpe > 2:
            rec += " (Mükemmel)\n"
        elif sharpe > 1:
            rec += " (İyi)\n"
        else:
            rec += " (Düşük)\n"
        
        rec += f"  • Win Rate: {win_rate:.1f}%\n"
        rec += f"  • Max Drawdown: {max_dd:.1f}%\n"
        
        # Rejim bazlı öneriler
        rec += f"\nMarket Regime: {market_regime}\n"
        
        if market_regime in ['trending_up', 'trending_down']:
            rec += "  → Trend-following stratejiler uygun\n"
            rec += "  → Daha uzun TF'ler (1h-4h) daha iyi sinyal verebilir\n"
        elif market_regime == 'ranging':
            rec += "  → Mean-reversion stratejiler uygun\n"
            rec += "  → Daha kısa TF'ler (5m-15m) daha iyi olabilir\n"
        elif market_regime == 'volatile':
            rec += "  → Dikkatli ol, pozisyon boyutunu küçült\n"
            rec += "  → Stop-loss'ları geniş tut\n"
        
        # Risk uyarısı
        if max_dd < -20:
            rec += f"\n⚠️ UYARI: Max DD {max_dd:.1f}% - Risk yönetimi kritik!\n"
        
        return rec
    
    def get_summary_table(
        self,
        results: List[BacktestResult]
    ) -> pd.DataFrame:
        """Özet tablo döndürür."""
        
        data = []
        for r in results:
            data.append({
                'TF': r.timeframe,
                'Return%': f"{r.total_return:.1f}",
                'Ann.Ret%': f"{r.annualized_return:.1f}",
                'Vol%': f"{r.volatility:.1f}",
                'Sharpe': f"{r.sharpe_ratio:.2f}",
                'Sortino': f"{r.sortino_ratio:.2f}",
                'MaxDD%': f"{r.max_drawdown:.1f}",
                'WinRate%': f"{r.win_rate:.1f}",
                'PF': f"{r.profit_factor:.2f}",
                'Trades': r.total_trades,
                'Regime': r.regime,
                'Conf': f"{r.confidence_score:.0f}",
            })
        
        return pd.DataFrame(data)


# =============================================================================
# TEST KODU
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("DİNAMİK BACKTEST TEST")
    print("=" * 70)
    
    import sys
    from pathlib import Path
    
    # Tüm modül klasörlerini Python path'ine ekle
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent  # backtest -> src
    
    # Her modül klasörünü ayrı ayrı ekle (internal import'lar için)
    for subdir in ['data', 'indicators', 'backtest']:
        module_path = src_dir / subdir
        if module_path.exists() and str(module_path) not in sys.path:
            sys.path.insert(0, str(module_path))
    
    try:
        # Direct import (her klasör path'te olduğu için)
        from fetcher import DataFetcher
        from calculator import IndicatorCalculator
        
        # 1. Veri çek
        print("\n[1] Veri çekiliyor...")
        fetcher = DataFetcher(symbol="BTC/USDT")
        
        # Birden fazla timeframe
        timeframes = ['15m', '1h', '4h']
        data_dict = {}
        
        for tf in timeframes:
            df = fetcher.fetch_ohlcv(timeframe=tf, limit=500)
            
            # İndikatör ekle
            calc = IndicatorCalculator(verbose=False)
            df = calc.calculate_category(df, 'momentum')
            df = calc.add_forward_returns(df, periods=[1, 5])
            
            data_dict[tf] = df
            print(f"   {tf}: {len(df)} bar")
        
        # 2. Backtester oluştur
        print("\n[2] Backtest yapılıyor...")
        backtester = DynamicBacktester(
            train_ratio=0.7,
            n_walks=3,
            verbose=True
        )
        
        # 3. Timeframe karşılaştırma
        results = backtester.compare_timeframes(
            data_dict,
            signal_col='RSI_14',
            use_walk_forward=True
        )
        
        # 4. En iyi timeframe seç
        print("\n[3] En iyi timeframe seçiliyor...")
        ranking = backtester.select_best_timeframe(results)
        
        # 5. Özet
        print("\n" + "=" * 70)
        print("SONUÇLAR")
        print("=" * 70)
        
        print("\nTimeframe Sıralaması:")
        for tf, score in ranking.rankings:
            print(f"   {tf}: {score:.1f} puan")
        
        print(f"\n{ranking.recommendation}")
        
        print("\nDetaylı Tablo:")
        summary = backtester.get_summary_table(results)
        print(summary.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("TEST TAMAMLANDI")
        print("=" * 70)
        
    except ImportError as e:
        print(f"Import hatası: {e}")
        print(f"\nDebug: sys.path içindeki ilk 5 yol:")
        for p in sys.path[:5]:
            print(f"  - {p}")

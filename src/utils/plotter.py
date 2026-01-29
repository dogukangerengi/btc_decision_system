import matplotlib.pyplot as plt
import pandas as pd
import io

class AnalysisPlotter:
    """Analiz sonuçlarını görselleştiren sınıf."""
    
    def __init__(self):
        # Grafik stili ayarları - 'dark_background' modern bir görünüm sağlar
        try:
            plt.style.use('dark_background')
        except:
            # Eğer stil bulunamazsa varsayılanı kullan
            plt.style.use('default')
        
    def create_analysis_chart(self, df: pd.DataFrame, symbol: str, timeframe: str, active_indicators: dict) -> io.BytesIO:
        """
        Fiyat grafiği ve aktif indikatörleri çizer.
        Resmi bellekte (buffer) tutar, diske kaydetmez (hız için).
        """
        # Son 100 barı al (grafik çok sıkışık olmasın diye)
        plot_data = df.tail(100).copy()
        
        # 1. Ana Fiyat Grafiği Oluştur (Üst kısım büyük, alt kısım küçük)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Fiyatı çiz
        ax1.plot(plot_data.index, plot_data['close'], label='Fiyat', color='white', linewidth=1)
        
        # Trend ve Volatilite İndikatörlerini Ana Grafiğe (ax1) Ekle
        # (Bollinger, SMA, EMA vb. fiyatın üzerine çizilir)
        for cat in ['trend', 'volatility']:
            if cat in active_indicators:
                for ind_name in active_indicators[cat]:
                    if ind_name in plot_data.columns:
                        # Renkleri otomatik döngüye sokmak yerine basitçe belirgin renkler kullanıyoruz
                        ax1.plot(plot_data.index, plot_data[ind_name], label=ind_name, 
                                 alpha=0.7, linewidth=1, linestyle='--')
        
        ax1.set_title(f"{symbol} - {timeframe} Analiz Grafiği", fontsize=14, color='gold')
        ax1.set_ylabel("Fiyat ($)")
        ax1.legend(loc='upper left', fontsize='small')
        ax1.grid(True, alpha=0.2)
        
        # 2. Alt İndikatör Grafiği (Momentum vb.)
        # (RSI, MACD gibi osilatörler alta çizilir)
        if 'momentum' in active_indicators:
            for ind_name in active_indicators['momentum']:
                if ind_name in plot_data.columns:
                     ax2.plot(plot_data.index, plot_data[ind_name], label=ind_name, color='lime')
                     
                     # RSI ise referans çizgileri (30-70) ekle
                     if 'RSI' in ind_name.upper():
                         ax2.axhline(70, color='red', linestyle=':', alpha=0.5)
                         ax2.axhline(30, color='green', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel("Momentum")
        ax2.legend(loc='upper left', fontsize='small')
        ax2.grid(True, alpha=0.2)
        
        # Tarih formatını düzenle (x ekseni)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Resmi diske yazmak yerine belleğe (RAM) kaydet
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close(fig) # Belleği temizle
        
        return buf
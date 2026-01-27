# 🚀 BTC Dinamik Karar Destek Sistemi

Bitcoin için saatlik çalışan, istatistiksel olarak güçlü bir trading karar destek sistemi.

## 📋 Özellikler

- **Multi-Timeframe Analiz**: 5m, 15m, 30m, 1h, 2h, 4h
- **60+ Teknik İndikatör**: Trend, Momentum, Volatilite, Hacim
- **İstatistiksel Seçim**: Information Coefficient, p-value, FDR correction
- **Walk-Forward Backtest**: Out-of-sample validation, overfitting önleme
- **Risk Metrikleri**: Sharpe, Sortino, Calmar, Max Drawdown
- **Telegram Bildirimleri**: Formatlı analiz raporları

## 🏗️ Proje Yapısı

```
btc_decision_system/
├── src/
│   ├── main.py                 # Ana orkestrasyon
│   ├── data/
│   │   ├── __init__.py
│   │   └── fetcher.py          # Binance veri çekme
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── categories.py       # İndikatör tanımları
│   │   ├── calculator.py       # İndikatör hesaplama
│   │   └── selector.py         # İstatistiksel seçim
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtester.py       # Walk-forward backtest
│   └── notifications/
│       ├── __init__.py
│       └── telegram_notifier.py # Telegram bildirimleri
├── .env                        # API anahtarları (oluşturulacak)
├── requirements.txt
└── README.md
```

## ⚡ Hızlı Başlangıç

### 1. Kurulum

```bash
# Projeyi klonla veya indir
cd btc_decision_system

# Sanal ortam oluştur ve aktifle
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Çalıştırma

```bash
cd src

# Tek seferlik analiz
python main.py

# Saatlik sürekli çalışma
python main.py --schedule

# 30 dakikada bir
python main.py --interval 30

# Telegram olmadan
python main.py --no-telegram
```

## 📊 Örnek Çıktı

```
🔔 BTC/USDT ANALİZ RAPORU
━━━━━━━━━━━━━━━━━━━━━━━

💰 Fiyat: $89,602.24
⏰ Zaman: 2026-01-25 23:04 UTC

📊 ÖNERİLEN TIMEFRAME: 2h
🔄 Piyasa Rejimi: transitioning
🔴 Sinyal: SHORT
🎯 Güven Skoru: 76/100 🟢🟢🟢

📈 AKTİF İNDİKATÖRLER:
• Trend: SUPERTs_10_3.0, TEMA_20
• Momentum: CCI, WILLR, RSI
• Volume: PVT, OBV, AD

⚠️ RİSK METRİKLERİ:
• Sharpe Ratio: 1.53 ✅
• Max Drawdown: -10.3% ⚠️
• Win Rate: 53.4% ⚠️

━━━━━━━━━━━━━━━━━━━━━━━
🤖 BTC Decision System v1.0
```

## 🔧 Telegram Kurulumu (Opsiyonel)

### 1. Bot Oluşturma
1. Telegram'da [@BotFather](https://t.me/BotFather) aç
2. `/newbot` komutu gönder
3. Bot adı ve kullanıcı adı belirle
4. Token'ı kopyala

### 2. Chat ID Bulma
1. Oluşturduğun bot'a bir mesaj at
2. Tarayıcıda aç: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. `"chat":{"id": XXXXXX}` kısmındaki sayıyı kopyala

### 3. .env Dosyası
```bash
# .env dosyası oluştur
echo "TELEGRAM_BOT_TOKEN=your_token_here" >> .env
echo "TELEGRAM_CHAT_ID=your_chat_id_here" >> .env
```

## 📈 Metrik Yorumlama

| Metrik | İyi | Orta | Kötü |
|--------|-----|------|------|
| Sharpe Ratio | > 1.0 | 0 - 1.0 | < 0 |
| Sortino Ratio | > 1.5 | 0.5 - 1.5 | < 0.5 |
| Max Drawdown | > -10% | -10% to -20% | < -20% |
| Win Rate | > 55% | 50% - 55% | < 50% |
| IC (Information Coefficient) | > 0.05 | 0.02 - 0.05 | < 0.02 |

## 🔬 İstatistiksel Metodoloji

### Information Coefficient (IC)
```
IC = Spearman(indicator_t, return_{t+n})
```
- Rank-based korelasyon (outlier'lara robust)
- |IC| > 0.02: Ekonomik olarak anlamlı
- Multiple testing correction: Benjamini-Hochberg FDR

### Walk-Forward Validation
```
[=== Train ===][Test]
    [=== Train ===][Test]
        [=== Train ===][Test]
```
- Overfitting önleme
- Out-of-sample performans ölçümü
- Gerçek dünya simülasyonu

## ⚠️ Uyarılar

1. **Yatırım tavsiyesi değildir** - Karar destek sistemidir
2. **Geçmiş performans gelecek sonuçları garanti etmez**
3. **Risk yönetimi sizin sorumluluğunuzdadır**
4. **Paper trading ile test edin**

## 📝 Lisans

MIT License - Kişisel kullanım için serbesttir.

---

**Geliştirici**: Doğukan Gerengi  
**Versiyon**: 1.0.0  
**Son Güncelleme**: Ocak 2026

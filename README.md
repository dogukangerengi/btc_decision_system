# 🚀 BTC Dinamik Karar Destek Sistemi

Bitcoin için saatlik çalışan, IC (Information Coefficient) bazlı istatistiksel trading karar destek sistemi.

## 📋 Özellikler

- **Multi-Timeframe Analiz**: 5m, 15m, 30m, 1h, 2h, 4h
- **60+ Teknik İndikatör**: Trend, Momentum, Volatilite, Hacim
- **IC Bazlı İstatistiksel Seçim**: Spearman korelasyonu, p-value, FDR correction
- **Dinamik Güven Skoru**: Piyasa rejimine göre otomatik ayarlanan sinyal gücü
- **Walk-Forward Backtest**: Out-of-sample validation, overfitting önleme
- **Telegram Bildirimleri**: IC değerleri ile formatlı analiz raporları

## 🏗️ Proje Yapısı

```
btc_decision_system/
├── src/
│   ├── main.py                 # Ana orkestrasyon
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py          # Binance veri çekme
│   │   └── preprocessor.py     # Veri ön işleme
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── categories.py       # İndikatör tanımları
│   │   ├── calculator.py       # İndikatör hesaplama
│   │   └── selector.py         # İstatistiksel seçim (IC)
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtester.py       # Walk-forward backtest
│   └── notifications/
│       ├── __init__.py
│       └── telegram_notifier.py # Telegram bildirimleri
├── config/
│   └── settings.yaml           # Yapılandırma dosyası
├── logs/                       # Log dosyaları
├── .env                        # API anahtarları (oluşturulacak)
├── requirements.txt
├── setup_scheduler.sh          # Otomatik çalışma scripti
└── README.md
```

## ⚡ Hızlı Başlangıç

### 1. Kurulum

```bash
# Projeyi klonla
git clone https://github.com/kullanici/btc_decision_system.git
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

### 3. Otomatik Çalışma (macOS LaunchAgent)

```bash
# Kurulum
./setup_scheduler.sh install

# Durum kontrolü
./setup_scheduler.sh status

# Manuel çalıştırma
./setup_scheduler.sh run

# Telegram testi
./setup_scheduler.sh test

# Kaldırma
./setup_scheduler.sh uninstall
```

## 📊 Örnek Çıktı

```
🔔 BTC/USDT ANALİZ RAPORU
━━━━━━━━━━━━━━━━━━━━━━━

💰 Fiyat: $104,250.00
⏰ Zaman: 2026-01-30 14:00 UTC

📊 ÖNERİLEN TIMEFRAME: 30m
↔️ Piyasa Rejimi: ranging
🔴 Baskın Yön: SHORT
🎯 Sinyal Gücü: 61/100 🟡🟡

📈 AKTİF İNDİKATÖRLER:
📊 Trend: Aroon Down (+0.13), Supertrend (-0.10)
⚡ Momentum: Coppock (-0.18), ROC (20) (-0.15)
📉 Volatility: UI (+0.13), Bollinger Bands (-0.12)
📶 Volume: CMF (20) (-0.18), Chaikin Osc (-0.14)

📝 Not: 📉 İndikatörler güçlü SHORT yönünde | ⭐ En güçlü: COPC

━━━━━━━━━━━━━━━━━━━━━━━
🤖 BTC Decision System v1.0
```

## 🔬 İstatistiksel Metodoloji

### Information Coefficient (IC)

```
IC = Spearman(indicator_t, return_{t+n})
```

- **Spearman korelasyonu**: Rank-based, outlier'lara robust
- **|IC| > 0.02**: Ekonomik olarak anlamlı
- **IC > 0**: İndikatör yükselince fiyat yükselir (LONG)
- **IC < 0**: İndikatör yükselince fiyat düşer (SHORT)

### Güven Skoru Hesaplama

Güven skoru üç faktörden oluşur:

| Faktör | Ağırlık | Açıklama |
|--------|---------|----------|
| Anlamlı İndikatör Sayısı | 30 puan | Daha fazla = daha güvenilir |
| Ortalama \|IC\| | 40 puan | Daha yüksek = daha güçlü sinyal |
| IC Tutarlılığı | 30 puan | Aynı yönde = daha net sinyal |

### Piyasa Rejimi Ayarlaması

| Rejim | Çarpan | Açıklama |
|-------|--------|----------|
| Trending (up/down) | 1.00 | Trend sinyalleri güvenilir |
| Transitioning | 0.85 | Belirsizlik var |
| Ranging | 0.75 | Trend sinyalleri yanıltıcı |
| Volatile | 0.70 | Her sinyal riskli |

### Multiple Testing Correction

```
Benjamini-Hochberg FDR: p_adjusted = p * (n / rank)
```

- 60+ indikatör test ediliyor
- FDR correction ile yanlış pozitif oranı kontrol altında

### Walk-Forward Validation

```
[=== Train ===][Test]
    [=== Train ===][Test]
        [=== Train ===][Test]
```

- Overfitting önleme
- Out-of-sample performans ölçümü
- Gerçek dünya simülasyonu

## 📈 IC Değeri Yorumlama

| IC Aralığı | Anlam | Aksiyon |
|------------|-------|---------|
| > +0.10 | Çok güçlü pozitif | Güçlü LONG sinyali |
| +0.05 to +0.10 | Güçlü pozitif | LONG sinyali |
| +0.02 to +0.05 | Zayıf pozitif | Hafif LONG eğilimi |
| -0.02 to +0.02 | Anlamsız | Sinyal yok |
| -0.05 to -0.02 | Zayıf negatif | Hafif SHORT eğilimi |
| -0.10 to -0.05 | Güçlü negatif | SHORT sinyali |
| < -0.10 | Çok güçlü negatif | Güçlü SHORT sinyali |

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
cat > .env << EOF
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF
```

## 🎯 Sistem Özellikleri

### Otomatik En Güçlü İndikatör Seçimi

Sistem, tüm kategoriler arasından en yüksek |IC| değerine sahip indikatörü otomatik olarak ilgili kategoriye ekler. Bu sayede en güçlü sinyal her zaman görünür.

### Duplicate Filtreleme

Aynı indikatör grubunun farklı çıktıları (örn: MACD, MACDh, MACDs) tek bir indikatör olarak sayılır. Her kategoriden gerçekten farklı 2 indikatör seçilir.

### Rejim Bazlı Güven Ayarlaması

Ranging veya volatile piyasalarda trend sinyalleri otomatik olarak düşük güvenle işaretlenir. Bu, yanıltıcı sinyallerin önüne geçer.

## ⚠️ Uyarılar

1. **Yatırım tavsiyesi değildir** - Karar destek sistemidir
2. **IC değerleri göreceli performans gösterir** - Mutlak başarı garantisi değil
3. **Geçmiş performans gelecek sonuçları garanti etmez**
4. **Risk yönetimi sizin sorumluluğunuzdadır**
5. **Paper trading ile test edin**

## 🔄 Güncelleme Geçmişi

### v1.1.0 (Ocak 2026)
- IC bazlı güven skoru sistemi
- Piyasa rejimine göre otomatik güven ayarlaması
- En güçlü indikatör otomatik ekleme
- Duplicate indikatör filtreleme
- Telegram'da IC değerleri gösterimi

### v1.0.0 (Ocak 2026)
- İlk sürüm
- Multi-timeframe analiz
- Walk-forward backtest
- Telegram bildirimleri

## 📝 Lisans

MIT License - Kişisel kullanım için serbesttir.

---

**Geliştirici**: Doğukan Gerengi  
**Versiyon**: 1.1.0  
**Son Güncelleme**: Ocak 2026

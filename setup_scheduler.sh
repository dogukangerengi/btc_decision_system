#!/bin/bash
# =============================================================================
# BTC Decision System - Otomatik Çalışma Kurulum Script'i
# =============================================================================
# Kullanım:
#   ./setup_scheduler.sh install    # Kurulum
#   ./setup_scheduler.sh uninstall  # Kaldırma
#   ./setup_scheduler.sh status     # Durum kontrolü
#   ./setup_scheduler.sh logs       # Logları göster
#   ./setup_scheduler.sh run        # Manuel çalıştır
# =============================================================================

# Renkli çıktı
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Dizinler
PROJECT_DIR="$HOME/btc_decision_system"
PLIST_NAME="com.btc.decision.system.plist"
PLIST_SRC="$PROJECT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
LOG_DIR="$PROJECT_DIR/logs"

# Log dizini oluştur
mkdir -p "$LOG_DIR"

case "$1" in
    install)
        echo -e "${GREEN}📦 BTC Decision System Scheduler Kurulumu${NC}"
        echo "================================================"
        
        # Plist dosyası var mı kontrol et
        if [ ! -f "$PLIST_SRC" ]; then
            echo -e "${RED}❌ Hata: $PLIST_SRC bulunamadı${NC}"
            exit 1
        fi
        
        # Eski job'ı durdur (varsa)
        launchctl unload "$PLIST_DST" 2>/dev/null
        
        # Plist'i kopyala
        cp "$PLIST_SRC" "$PLIST_DST"
        echo -e "${GREEN}✓ Plist kopyalandı${NC}"
        
        # LaunchAgent'ı yükle
        launchctl load "$PLIST_DST"
        echo -e "${GREEN}✓ LaunchAgent yüklendi${NC}"
        
        echo ""
        echo -e "${GREEN}✅ Kurulum tamamlandı!${NC}"
        echo "   Sistem her saat başı çalışacak (XX:00)"
        echo ""
        echo "   Durum kontrolü: ./setup_scheduler.sh status"
        echo "   Loglar: ./setup_scheduler.sh logs"
        ;;
        
    uninstall)
        echo -e "${YELLOW}🗑️  Scheduler kaldırılıyor...${NC}"
        
        # Durdur ve kaldır
        launchctl unload "$PLIST_DST" 2>/dev/null
        rm -f "$PLIST_DST"
        
        echo -e "${GREEN}✅ Scheduler kaldırıldı${NC}"
        ;;
        
    status)
        echo -e "${GREEN}📊 Scheduler Durumu${NC}"
        echo "===================="
        
        if launchctl list | grep -q "com.btc.decision.system"; then
            echo -e "${GREEN}✓ Scheduler AKTİF${NC}"
            launchctl list | grep "com.btc.decision.system"
        else
            echo -e "${YELLOW}○ Scheduler PASİF${NC}"
        fi
        
        echo ""
        echo "Son çalışma logları:"
        if [ -f "$LOG_DIR/cron.log" ]; then
            tail -20 "$LOG_DIR/cron.log"
        else
            echo "  (henüz log yok)"
        fi
        ;;
        
    logs)
        echo -e "${GREEN}📜 Son Loglar${NC}"
        echo "============="
        
        if [ -f "$LOG_DIR/cron.log" ]; then
            tail -50 "$LOG_DIR/cron.log"
        else
            echo "  (henüz log yok)"
        fi
        ;;
        
    run)
        echo -e "${GREEN}🚀 Manuel Çalıştırma${NC}"
        echo "==================="
        
        cd "$PROJECT_DIR/src"
        source "$PROJECT_DIR/venv/bin/activate"
        python main.py
        ;;
        
    test)
        echo -e "${GREEN}🧪 Telegram Testi${NC}"
        echo "================="
        
        cd "$PROJECT_DIR/src"
        source "$PROJECT_DIR/venv/bin/activate"
        python -c "
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
env_file = Path('$PROJECT_DIR/.env')
if env_file.exists():
    load_dotenv(env_file)
    print('✓ .env dosyası yüklendi')

from notifications.telegram_notifier import TelegramNotifier
notifier = TelegramNotifier()
if notifier.is_configured():
    print('✓ Token ve Chat ID bulundu')
    if notifier.test_connection_sync():
        success = notifier.send_alert_sync('🧪 Test', 'BTC Decision System bağlantısı başarılı!', 'success')
        print('✅ Telegram testi başarılı!' if success else '❌ Mesaj gönderilemedi')
    else:
        print('❌ Bot bağlantısı başarısız')
else:
    print('❌ Telegram yapılandırılmamış')
    print('   TELEGRAM_BOT_TOKEN:', 'VAR' if notifier.token else 'YOK')
    print('   TELEGRAM_CHAT_ID:', 'VAR' if notifier.chat_id else 'YOK')
"
        ;;
        
    *)
        echo "BTC Decision System Scheduler"
        echo ""
        echo "Kullanım: $0 {install|uninstall|status|logs|run|test}"
        echo ""
        echo "Komutlar:"
        echo "  install    - Saatlik otomatik çalışmayı kur"
        echo "  uninstall  - Otomatik çalışmayı kaldır"
        echo "  status     - Durum ve son logları göster"
        echo "  logs       - Tüm logları göster"
        echo "  run        - Manuel olarak çalıştır"
        echo "  test       - Telegram bağlantısını test et"
        exit 1
        ;;
esac

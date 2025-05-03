import os
import subprocess
import sys
import time

def check_requirements():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder ve yükler"""
    print("Gerekli kütüphaneler kontrol ediliyor...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Kütüphaneler başarıyla yüklendi!")
    except subprocess.CalledProcessError:
        print("Kütüphane yüklemede hata oluştu! Lütfen manuel olarak requirements.txt dosyasındaki kütüphaneleri yükleyin.")
        return False
    
    return True

def train_models():
    """Modelleri eğitir"""
    print("\n=== MODEL EĞİTİMİ BAŞLIYOR ===\n")
    
    # Klasik model eğitimi
    print("1. Klasik modeller (Naive Bayes ve Lojistik Regresyon) eğitiliyor...")
    try:
        os.system(f"{sys.executable} haber_siniflandirma.py")
        print("Klasik modeller başarıyla eğitildi!")
    except Exception as e:
        print(f"Klasik model eğitiminde hata: {e}")
        return False
    
    # BERT modeli eğitimi
    print("\n2. BERT modeli eğitiliyor...")
    print("Not: Bu işlem GPU varsa daha hızlı tamamlanacaktır, yoksa uzun sürebilir...")
    
    choice = input("BERT modelini eğitmek istiyor musunuz? (e/h): ")
    if choice.lower() == 'e':
        try:
            os.system(f"{sys.executable} bert_siniflandirma.py")
            print("BERT modeli başarıyla eğitildi!")
        except Exception as e:
            print(f"BERT model eğitiminde hata: {e}")
            return False
    else:
        print("BERT model eğitimi atlandı.")
    
    return True

def start_web_interface():
    """Web arayüzünü başlatır"""
    print("\n=== WEB ARAYÜZÜ BAŞLATILIYOR ===\n")
    
    try:
        os.system(f"{sys.executable} -m streamlit run web_arayuz.py")
    except Exception as e:
        print(f"Web arayüzü başlatılırken hata: {e}")
        return False
    
    return True

def main():
    """Ana işlev"""
    print("=== TÜRKÇE HABER SINIFLANDIRMA PROJESİ ===\n")
    
    print("Bu program modelleri eğitip web arayüzünü başlatacaktır.\n")
    
    # 1. Gereksinimleri kontrol et
    if not check_requirements():
        print("Program gereksinimleri karşılanamadı.")
        return
    
    # 2. Veri setini kontrol et
    if not os.path.exists("veri/TurkishHeadlines.csv"):
        print("Veri seti bulunamadı! Lütfen 'veri' klasöründe TurkishHeadlines.csv dosyasının varlığını kontrol edin.")
        return
    
    # 3. Kullanıcıya seçenek sun
    print("\nLütfen yapmak istediğiniz işlemi seçin:")
    print("1. Modelleri eğit")
    print("2. Web arayüzünü başlat")
    print("3. Hem modelleri eğit hem de web arayüzünü başlat")
    
    choice = input("\nSeçiminiz (1/2/3): ")
    
    if choice == "1":
        train_models()
    elif choice == "2":
        if not (os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("nb_model.pkl") and os.path.exists("lr_model.pkl")):
            print("Uyarı: Klasik model dosyaları bulunamadı. Web arayüzünde bazı özellikler çalışmayabilir.")
            cont = input("Devam etmek istiyor musunuz? (e/h): ")
            if cont.lower() != 'e':
                return
        start_web_interface()
    elif choice == "3":
        if train_models():
            time.sleep(2)  # Biraz bekle
            start_web_interface()
    else:
        print("Geçersiz seçim!")

if __name__ == "__main__":
    main() 
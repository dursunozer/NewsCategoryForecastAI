import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import os

# NLTK'nin gerekli verileri indirme işlemini gerçekleştirir
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    # Ayrıca punkt_tab'ı kontrol eder
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK veri paketleri indiriliyor...")
    nltk.download('punkt')
    nltk.download('stopwords')
    try:
        nltk.download('punkt_tab')
    except:
        # Eğer paket bulunamazsa, özel indirme işlemi uygular
        print("punkt_tab için alternatif çözüm uygulanıyor...")
        # Tokenizer'ı direkt olarak word_tokenize'ı kullanacak şekilde değiştirir
    print("İndirme tamamlandı.")
    
# Türkçe durak kelimelerini (stopwords) yükler
try:
    tr_stop_words = set(stopwords.words('turkish'))
except LookupError:
    print("Türkçe stopwords indiriliyor...")
    nltk.download('stopwords')
    tr_stop_words = set(stopwords.words('turkish'))

# Özel tokenize fonksiyonu (punkt_tab sorunu çözümü için geliştirilmiştir)
def safe_tokenize(text):
    """Güvenli tokenizasyon fonksiyonu, hata durumunda alternatif yöntem kullanır"""
    if isinstance(text, str):
        try:
            return word_tokenize(text)
        except LookupError:
            # Basit tokenizasyon: boşluklara göre ayırma işlemi yapar
            return text.split()
    return []

# Sayfa başlığını ve genel yapılandırmayı ayarlar
st.set_page_config(page_title="Haber Kategori Tahmini", page_icon="📰", layout="wide")
st.title("Haber Kategori Tahmini")
st.markdown("Bu uygulama, haber başlıklarını kategorilerine göre sınıflandırır.")

# Metin temizleme işlemleri
def metin_temizle(text):
    """Metni temizleme: özel karakterleri kaldırır ve küçük harfe çevirir"""
    if isinstance(text, str):
        # Küçük harfe çevirme işlemi yapar
        text = text.lower()
        # Noktalama işaretlerini kaldırır
        text = re.sub(r'[^\w\s]', '', text)
        # Gereksiz boşlukları kaldırır
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Durak kelimeleri kaldırır
def durak_kelimeleri_kaldir(text):
    """Türkçe durak kelimelerini metinden çıkarır ve temiz metin döndürür"""
    if isinstance(text, str):
        tokens = safe_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in tr_stop_words]
        return ' '.join(filtered_tokens)
    return ""

# BERT için metin temizler
def bert_metin_temizle(text):
    """BERT için metni temizler: özel karakterleri düzenler ve gereksiz boşlukları kaldırır"""
    if isinstance(text, str):
        # Gereksiz karakterleri kaldırır
        text = re.sub(r'[^\w\s.?!,]', '', text)
        # Gereksiz boşlukları kaldırır
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Modelleri yükleme fonksiyonları
@st.cache_resource
def load_tfidf_nb_model():
    """TF-IDF ve Naive Bayes modellerini yükler ve hazır hale getirir"""
    try:
        # Model ve vektörleştirici dosyalarını kontrol eder
        if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('nb_model.pkl'):
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            with open('nb_model.pkl', 'rb') as f:
                nb_model = pickle.load(f)
            return tfidf_vectorizer, nb_model
        else:
            st.warning("TF-IDF ve Naive Bayes model dosyaları bulunamadı. Önce modelleri eğitmelisiniz.")
            return None, None
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None

@st.cache_resource
def load_tfidf_lr_model():
    """TF-IDF ve Lojistik Regresyon modellerini yükler ve kullanıma hazırlar"""
    try:
        # Model ve vektörleştirici dosyalarını kontrol eder
        if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('lr_model.pkl'):
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            with open('lr_model.pkl', 'rb') as f:
                lr_model = pickle.load(f)
            return tfidf_vectorizer, lr_model
        else:
            st.warning("TF-IDF ve Lojistik Regresyon model dosyaları bulunamadı. Önce modelleri eğitmelisiniz.")
            return None, None
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None

@st.cache_resource
def load_bert_model():
    """BERT modelini yükler ve tahmin için hazır hale getirir"""
    try:
        # BERT model dosyasını kontrol eder
        if os.path.exists('bert_haber_siniflandirici.pth') and os.path.exists('model_siniflar.txt'):
            # Sınıfları dosyadan okur ve listeye dönüştürür
            label_classes = []
            with open('model_siniflar.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        label_classes.append(parts[1])
            
            # GPU veya CPU cihazını belirler
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Tokenizer modelini yükler
            tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
            
            # BERT sınıflandırma modelini yükler
            model = AutoModelForSequenceClassification.from_pretrained(
                "dbmdz/bert-base-turkish-cased",
                num_labels=len(label_classes)
            )
            
            # Eğitilmiş model ağırlıklarını yükler ve cihaza taşır
            model.load_state_dict(torch.load('bert_haber_siniflandirici.pth', map_location=device))
            model.to(device)
            model.eval()  # Değerlendirme moduna alır
            
            return tokenizer, model, label_classes, device
        else:
            st.warning("BERT model dosyaları bulunamadı. Önce modeli eğitmelisiniz.")
            return None, None, None, None
    except Exception as e:
        st.error(f"BERT model yükleme hatası: {e}")
        return None, None, None, None

# Tahmin fonksiyonları
def predict_with_nb(text, vectorizer, model):
    """Naive Bayes ile metni sınıflandırır ve sonuçları döndürür"""
    if vectorizer is None or model is None:
        return "Model yüklenemedi. Lütfen önce modeli eğitin."
    
    # Metin önişleme adımlarını uygular
    clean_text = metin_temizle(text)
    processed_text = durak_kelimeleri_kaldir(clean_text)
    
    # TF-IDF ile vektörleştirir ve tahmin yapar
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    # Olasılıkları hesaplayarak en yüksek olasılığı belirler
    probs = model.predict_proba(text_vector)[0]
    max_prob = max(probs) * 100
    
    return prediction, max_prob

def predict_with_lr(text, vectorizer, model):
    """Lojistik Regresyon ile metni sınıflandırır ve sonuçları döndürür"""
    if vectorizer is None or model is None:
        return "Model yüklenemedi. Lütfen önce modeli eğitin."
    
    # Metin önişleme adımlarını uygular
    clean_text = metin_temizle(text)
    processed_text = durak_kelimeleri_kaldir(clean_text)
    
    # TF-IDF ile vektörleştirir ve tahmin yapar
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    # Olasılıkları hesaplayarak en yüksek olasılığı belirler
    probs = model.predict_proba(text_vector)[0]
    max_prob = max(probs) * 100
    
    return prediction, max_prob

def predict_with_bert(text, tokenizer, model, label_classes, device):
    """BERT ile derin öğrenme tabanlı sınıflandırma yapar ve sonuçları döndürür"""
    if tokenizer is None or model is None or label_classes is None:
        return "BERT model yüklenemedi. Lütfen önce modeli eğitin."
    
    # BERT için özel metin temizleme işlemi uygular
    clean_text = bert_metin_temizle(text)
    
    # BERT tokenizer ile metni modelin anlayacağı formata dönüştürür
    encoding = tokenizer(
        clean_text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Tensörleri uygun cihaza (GPU/CPU) taşır
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Modeli kullanarak tahmin yapar
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_prob, preds = torch.max(probs, dim=1)
    
    # Tahmin edilen sınıfı ve olasılığı döndürür
    predicted_class = label_classes[preds.item()]
    probability = max_prob.item() * 100
    
    return predicted_class, probability

# Arayüz sekmelerini oluşturur
tab1, tab2, tab3 = st.tabs(["Tahmin", "Veri Seti", "Hakkında"])

# Tahmin sekmesi içeriği
with tab1:
    st.header("Haber Başlığı Tahmin Et")
    
    # Kullanıcı girişi alanını oluşturur
    user_input = st.text_area("Haber başlığı girin:", height=100)
    
    # Üç modeli ayrı sütunlarda gösterir
    col1, col2, col3 = st.columns(3)
    
    # Naive Bayes sütunu
    with col1:
        if st.button("Naive Bayes ile Tahmin Et"):
            if user_input:
                # Naive Bayes modelini yükler
                tfidf_vectorizer, nb_model = load_tfidf_nb_model()
                
                if tfidf_vectorizer is not None and nb_model is not None:
                    # Tahmin işlemini gerçekleştirir
                    prediction, probability = predict_with_nb(user_input, tfidf_vectorizer, nb_model)
                    
                    # Sonuçları kullanıcıya gösterir
                    st.success(f"Tahmin: **{prediction}**")
                    st.info(f"Güven: {probability:.2f}%")
                else:
                    st.error("Naive Bayes modeli yüklenemedi. Lütfen önce modeli eğitin.")
            else:
                st.warning("Lütfen bir haber başlığı girin.")
    
    # Lojistik Regresyon sütunu
    with col2:
        if st.button("Lojistik Regresyon ile Tahmin Et"):
            if user_input:
                # Lojistik Regresyon modelini yükler
                tfidf_vectorizer, lr_model = load_tfidf_lr_model()
                
                if tfidf_vectorizer is not None and lr_model is not None:
                    # Tahmin işlemini gerçekleştirir
                    prediction, probability = predict_with_lr(user_input, tfidf_vectorizer, lr_model)
                    
                    # Sonuçları kullanıcıya gösterir
                    st.success(f"Tahmin: **{prediction}**")
                    st.info(f"Güven: {probability:.2f}%")
                else:
                    st.error("Lojistik Regresyon modeli yüklenemedi. Lütfen önce modeli eğitin.")
            else:
                st.warning("Lütfen bir haber başlığı girin.")
    
    # BERT sütunu
    with col3:
        if st.button("BERT ile Tahmin Et"):
            if user_input:
                # BERT modelini yükler
                tokenizer, model, label_classes, device = load_bert_model()
                
                if tokenizer is not None and model is not None and label_classes is not None:
                    # BERT ile tahmin işlemini gerçekleştirir
                    prediction, probability = predict_with_bert(user_input, tokenizer, model, label_classes, device)
                    
                    # Sonuçları kullanıcıya gösterir
                    st.success(f"Tahmin: **{prediction}**")
                    st.info(f"Güven: {probability:.2f}%")
                else:
                    st.error("BERT modeli yüklenemedi. Lütfen önce modeli eğitin.")
            else:
                st.warning("Lütfen bir haber başlığı girin.")
    
    # Örnek haber başlıklarını kullanıcıya sunar
    st.subheader("Örnek Haber Başlıkları")
    st.markdown("""
    1. "Türkiyenin ekonomik büyümesi beklentilerin üzerinde gerçekleşti"
    2. "Galatasaray deplasmanda Fenerbahçeyi 2-1 mağlup etti"
    3. "Sağlık Bakanlığı yeni Covid-19 Hastalığı tedbirlerini açıkladı"
    4. "Cumhurbaşkanı yeni kabineyi açıkladı"
    5. "Bilim insanları yeni bir gezegen keşfetti"
    """)

# Veri Seti sekmesi içeriği
with tab2:
    st.header("Veri Seti Görselleri")
    
    try:
        # Kategori dağılımı grafiğini gösterir
        st.subheader("Kategori Dağılımı")
        if os.path.exists("kategori_dagilimi.png"):
            st.image("kategori_dagilimi.png", caption="Haber Kategorilerinin Dağılımı")
        else:
            # Eğer dosya yoksa, veri içi bir grafik oluşturur
            category_counts = pd.Series({
                "Ekonomi": 5,
                "Magazin": 5,
                "Sağlık": 5,
                "Siyaset": 5,
                "Spor": 5,
                "Teknoloji": 5,
                "Yaşam": 5
            })
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
            ax.set_title('Haber Kategorilerinin Dağılımı')
            ax.set_xlabel('Haber Sayısı')
            ax.set_ylabel('Kategoriler')
            st.pyplot(fig)
        
        # Kelime sayısı dağılımını gösterir
        st.subheader("Kelime Sayısı Dağılımı")
        if os.path.exists("kelime_sayisi_dagilimi.png"):
            st.image("kelime_sayisi_dagilimi.png", caption="Haberlerdeki Kelime Sayısı Dağılımı")
        
        # Kategori-kelime sayısı ilişkisini görselleştirir
        st.subheader("Kategori-Kelime Sayısı İlişkisi")
        if os.path.exists("kategori_kelime_sayisi.png"):
            st.image("kategori_kelime_sayisi.png", caption="Kategorilere Göre Ortalama Kelime Sayıları")
        
        # BERT eğitim metriklerini gösterir
        st.subheader("BERT Eğitim Metrikleri")
        if os.path.exists("bert_training_history.png"):
            st.image("bert_training_history.png", caption="BERT Modelinin Eğitim Sürecindeki Metrikler")
        
        # BERT karışıklık matrisini görselleştirir
        st.subheader("BERT Karışıklık Matrisi")
        if os.path.exists("bert_confusion_matrix.png"):
            st.image("bert_confusion_matrix.png", caption="BERT Modelinin Karışıklık Matrisi")
        
        # Aşırı öğrenme analizini gösterir
        st.subheader("Aşırı Öğrenme (Overfitting) Analizi")
        if os.path.exists("bert_overfitting_analysis.png"):
            st.image("bert_overfitting_analysis.png", caption="BERT Modelinin Aşırı Öğrenme Analizi")
        
        # Klasik modellerin karışıklık matrisini görselleştirir
        st.subheader("Klasik Modellerin Karışıklık Matrisi")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Klasik Modellerin Karışıklık Matrisi")
            
    except Exception as e:
        st.error(f"Görsel yükleme hatası: {e}")
        st.info("Görseller yüklenirken bir hata oluştu. Lütfen 'Tahmin' sekmesine geçerek modeli kullanın.")

# Hakkında sekmesi içeriği
with tab3:
    st.header("Proje Hakkında")
    
    st.markdown("""
    ## Türkçe Haber Başlıklarını Sınıflandırma Projesi
    
    Bu proje, Türkçe haber başlıklarını kategorilerine göre sınıflandırmak için kullanılan yapay zeka modellerini içermektedir.
    
    ### Kullanılan Modeller
    
    1. **Geleneksel Makine Öğrenmesi Modelleri**:
       - Naive Bayes: Olasılık tabanlı sınıflandırma algoritması
       - Lojistik Regresyon: Doğrusal model tabanlı sınıflandırma yöntemi
    
    2. **Derin Öğrenme Modeli**:
       - BERT (Bidirectional Encoder Representations from Transformers): Son teknoloji NLP modeli
    ```bash
    !.. NOT: BERT modeli yeterli veri seti olmadığı için bazı durumlar Overfitting olabilir..!
    ```
        
    
    ### Kullanılan Teknolojiler
    
    - Python: Ana programlama dili
    - Pandas, NumPy: Veri manipülasyonu ve analizi
    - Scikit-Learn: Geleneksel makine öğrenmesi algoritmaları
    - PyTorch: Derin öğrenme framework'ü
    - Transformers (Hugging Face): BERT modeli implementasyonu
    - NLTK: Doğal dil işleme kütüphanesi
    - Streamlit: Web arayüzü oluşturma
    """)
    
    st.sidebar.title("Model Bilgileri")
    st.sidebar.markdown("""
    **Geleneksel Modeller**:
    - TF-IDF vektörleştirme: Metinleri sayısal özelliklere dönüştürür
    - Naive Bayes: Bayes teoremi tabanlı sınıflandırıcı
    - Lojistik Regresyon: Doğrusal sınıflandırıcı
    
    **Derin Öğrenme**:
    - BERT (Türkçe model): Çift yönlü transformers mimarisi
    """)
    
    # Model durumunu kontrol eder ve bilgi verir
    st.sidebar.title("Model Durumu")
    
    # NB model durumunu kontrol eder
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('nb_model.pkl'):
        st.sidebar.success("✅ Naive Bayes modeli yüklü")
    else:
        st.sidebar.error("❌ Naive Bayes modeli yüklü değil")
    
    # LR model durumunu kontrol eder
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('lr_model.pkl'):
        st.sidebar.success("✅ Lojistik Regresyon modeli yüklü")
    else:
        st.sidebar.error("❌ Lojistik Regresyon modeli yüklü değil")
    
    # BERT model durumunu kontrol eder
    if os.path.exists('bert_haber_siniflandirici.pth') and os.path.exists('model_siniflar.txt'):
        st.sidebar.success("✅ BERT modeli yüklü")
    else:
        st.sidebar.error("❌ BERT modeli yüklü değil") 
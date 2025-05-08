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

# NLTK'nin gerekli verileri indirme
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK veri paketleri indiriliyor...")
    nltk.download('punkt')
    nltk.download('stopwords')
    print("İndirme tamamlandı.")

# Türkçe stopwords
try:
    tr_stop_words = set(stopwords.words('turkish'))
except LookupError:
    print("Türkçe stopwords indiriliyor...")
    nltk.download('stopwords')
    tr_stop_words = set(stopwords.words('turkish'))

# Sayfa başlığı
st.set_page_config(page_title="Haber Kategori Tahmini", page_icon="📰", layout="wide")
st.title("Haber Kategori Tahmini")
st.markdown("Bu uygulama, haber başlıklarını kategorilerine göre sınıflandırır.")

# Metin temizleme işlemleri
def metin_temizle(text):
    """Metni temizleme: özel karakterleri kaldırma ve küçük harfe çevirme"""
    if isinstance(text, str):
        # Küçük harfe çevirme
        text = text.lower()
        # Noktalama işaretlerini kaldırma
        text = re.sub(r'[^\w\s]', '', text)
        # Gereksiz boşlukları kaldırma
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Durak kelimeleri kaldırma
def durak_kelimeleri_kaldir(text):
    """Türkçe durak kelimelerini metinden çıkarma"""
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in tr_stop_words]
        return ' '.join(filtered_tokens)
    return ""

# BERT için metin temizleme
def bert_metin_temizle(text):
    """Metni temizleme: özel karakterleri kaldırma ve düzenleme"""
    if isinstance(text, str):
        # Gereksiz karakterleri kaldırma
        text = re.sub(r'[^\w\s.?!,]', '', text)
        # Gereksiz boşlukları kaldırma
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Modelleri yükleme fonksiyonları
@st.cache_resource
def load_tfidf_nb_model():
    """TF-IDF ve Naive Bayes modellerini yükleme"""
    try:
        # Model ve vektörleştirici dosyalarını kontrol et
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
    """TF-IDF ve Lojistik Regresyon modellerini yükleme"""
    try:
        # Model ve vektörleştirici dosyalarını kontrol et
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
    """BERT modelini yükleme"""
    try:
        # BERT model dosyasını kontrol et
        if os.path.exists('bert_haber_siniflandirici.pth') and os.path.exists('model_siniflar.txt'):
            # Sınıfları yükle
            label_classes = []
            with open('model_siniflar.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        label_classes.append(parts[1])
            
            # Cihazı belirle
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Tokenizer yükle
            tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
            
            # Model yükle
            model = AutoModelForSequenceClassification.from_pretrained(
                "dbmdz/bert-base-turkish-cased",
                num_labels=len(label_classes)
            )
            
            # Eğitilmiş model ağırlıklarını yükle
            model.load_state_dict(torch.load('bert_haber_siniflandirici.pth', map_location=device))
            model.to(device)
            model.eval()
            
            return tokenizer, model, label_classes, device
        else:
            st.warning("BERT model dosyaları bulunamadı. Önce modeli eğitmelisiniz.")
            return None, None, None, None
    except Exception as e:
        st.error(f"BERT model yükleme hatası: {e}")
        return None, None, None, None

# Tahmin fonksiyonları
def predict_with_nb(text, vectorizer, model):
    """Naive Bayes ile tahmin et"""
    if vectorizer is None or model is None:
        return "Model yüklenemedi. Lütfen önce modeli eğitin."
    
    # Metin önişleme
    clean_text = metin_temizle(text)
    processed_text = durak_kelimeleri_kaldir(clean_text)
    
    # Vektörleştirme ve tahmin
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    # Olasılıkları al
    probs = model.predict_proba(text_vector)[0]
    max_prob = max(probs) * 100
    
    return prediction, max_prob

def predict_with_lr(text, vectorizer, model):
    """Lojistik Regresyon ile tahmin et"""
    if vectorizer is None or model is None:
        return "Model yüklenemedi. Lütfen önce modeli eğitin."
    
    # Metin önişleme
    clean_text = metin_temizle(text)
    processed_text = durak_kelimeleri_kaldir(clean_text)
    
    # Vektörleştirme ve tahmin
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    # Olasılıkları al
    probs = model.predict_proba(text_vector)[0]
    max_prob = max(probs) * 100
    
    return prediction, max_prob

def predict_with_bert(text, tokenizer, model, label_classes, device):
    """BERT ile tahmin et"""
    if tokenizer is None or model is None or label_classes is None:
        return "BERT model yüklenemedi. Lütfen önce modeli eğitin."
    
    # Metin önişleme
    clean_text = bert_metin_temizle(text)
    
    # Tokenleştirme
    encoding = tokenizer(
        clean_text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Tensörleri cihaza taşı
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Tahmin
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_prob, preds = torch.max(probs, dim=1)
    
    # Tahmin edilen sınıfı ve olasılığı döndür
    predicted_class = label_classes[preds.item()]
    probability = max_prob.item() * 100
    
    return predicted_class, probability

# Ana sayfa
tab1, tab2, tab3 = st.tabs(["Tahmin", "Veri Seti", "Hakkında"])

with tab1:
    st.header("Haber Başlığı Tahmin Et")
    
    # Kullanıcı girişi
    user_input = st.text_area("Haber başlığı girin:", height=100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Naive Bayes ile Tahmin Et"):
            if user_input:
                # Modelleri yükle
                tfidf_vectorizer, nb_model = load_tfidf_nb_model()
                
                if tfidf_vectorizer is not None and nb_model is not None:
                    # Tahmin yap
                    prediction, probability = predict_with_nb(user_input, tfidf_vectorizer, nb_model)
                    
                    # Sonuçları göster
                    st.success(f"Tahmin: **{prediction}**")
                    st.info(f"Güven: {probability:.2f}%")
                else:
                    st.error("Naive Bayes modeli yüklenemedi. Lütfen önce modeli eğitin.")
            else:
                st.warning("Lütfen bir haber başlığı girin.")
    
    with col2:
        if st.button("Lojistik Regresyon ile Tahmin Et"):
            if user_input:
                # Modelleri yükle
                tfidf_vectorizer, lr_model = load_tfidf_lr_model()
                
                if tfidf_vectorizer is not None and lr_model is not None:
                    # Tahmin yap
                    prediction, probability = predict_with_lr(user_input, tfidf_vectorizer, lr_model)
                    
                    # Sonuçları göster
                    st.success(f"Tahmin: **{prediction}**")
                    st.info(f"Güven: {probability:.2f}%")
                else:
                    st.error("Lojistik Regresyon modeli yüklenemedi. Lütfen önce modeli eğitin.")
            else:
                st.warning("Lütfen bir haber başlığı girin.")
    
    with col3:
        if st.button("BERT ile Tahmin Et"):
            if user_input:
                # BERT modelini yükle
                tokenizer, model, label_classes, device = load_bert_model()
                
                if tokenizer is not None and model is not None and label_classes is not None:
                    # Tahmin yap
                    prediction, probability = predict_with_bert(user_input, tokenizer, model, label_classes, device)
                    
                    # Sonuçları göster
                    st.success(f"Tahmin: **{prediction}**")
                    st.info(f"Güven: {probability:.2f}%")
                else:
                    st.error("BERT modeli yüklenemedi. Lütfen önce modeli eğitin.")
            else:
                st.warning("Lütfen bir haber başlığı girin.")
    
    # Örnek haber başlıkları
    st.subheader("Örnek Haber Başlıkları")
    st.markdown("""
    1. "Türkiyenin ekonomik büyümesi beklentilerin üzerinde gerçekleşti"
    2. "Galatasaray deplasmanda Fenerbahçeyi 2-1 mağlup etti"
    3. "Sağlık Bakanlığı yeni Covid-19 Hastalığı tedbirlerini açıkladı"
    4. "Cumhurbaşkanı yeni kabineyi açıkladı"
    5. "Bilim insanları yeni bir gezegen keşfetti"
    """)

with tab2:
    st.header("Veri Seti")
    
    try:
        # Veri setini yükle ve göster
        if os.path.exists('veri/TurkishHeadlines.csv'):
            df = pd.read_csv('veri/TurkishHeadlines.csv')
            
            # Veri seti bilgileri
            st.subheader("Veri Seti Özeti")
            st.write(f"Toplam kayıt sayısı: {df.shape[0]}")
            
            # Kategorileri göster
            st.subheader("Kategori Dağılımı")
            category_counts = df['ETIKET'].value_counts()
            
            # Grafik oluştur
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
            ax.set_title('Haber Kategorilerinin Dağılımı')
            ax.set_xlabel('Haber Sayısı')
            ax.set_ylabel('Kategoriler')
            st.pyplot(fig)
            
            # Örnek verileri göster
            st.subheader("Kategorilere Göre Haberler")
            
            # Her kategoriden 5'er tane örnek seçme
            for kategori in sorted(df['ETIKET'].unique()):
                with st.expander(f"{kategori} Kategorisi Haberleri", expanded=True):
                    kategori_df = df[df['ETIKET'] == kategori].head(5)
                    for _, row in kategori_df.iterrows():
                        st.write(f"**{row['HABERLER']}**")
        else:
            st.warning("Veri seti bulunamadı. Lütfen 'veri/TurkishHeadlines.csv' dosyasının mevcut olduğundan emin olun.")
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")

with tab3:
    st.header("Proje Hakkında")
    
    st.markdown("""
    ## Türkçe Haber Başlıklarını Sınıflandırma Projesi
    
    Bu proje, Türkçe haber başlıklarını kategorilerine göre sınıflandırmak için kullanılan yapay zeka modellerini içermektedir.
    
    ### Kullanılan Modeller
    
    1. **Geleneksel Makine Öğrenmesi Modelleri**:
       - Naive Bayes
       - Lojistik Regresyon
    
    2. **Derin Öğrenme Modeli**:
       - BERT (Bidirectional Encoder Representations from Transformers)
    ```bash
    !.. NOT: BERT modeli yeterli veri seti olmadığı için bazı durumlar Overfitting olabilir..!
    ```
        
    
    ### Kullanılan Teknolojiler
    
    - Python
    - Pandas, NumPy
    - Scikit-Learn
    - PyTorch
    - Transformers (Hugging Face)
    - NLTK
    - Streamlit
    """)
    
    st.sidebar.title("Model Bilgileri")
    st.sidebar.markdown("""
    **Geleneksel Modeller**:
    - TF-IDF vektörleştirme
    - Naive Bayes
    - Lojistik Regresyon
    
    **Derin Öğrenme**:
    - BERT (Türkçe model)
    """)
    
    # Model durumunu kontrol et
    st.sidebar.title("Model Durumu")
    
    # NB model durumu
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('nb_model.pkl'):
        st.sidebar.success("✅ Naive Bayes modeli yüklü")
    else:
        st.sidebar.error("❌ Naive Bayes modeli yüklü değil")
    
    # LR model durumu
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('lr_model.pkl'):
        st.sidebar.success("✅ Lojistik Regresyon modeli yüklü")
    else:
        st.sidebar.error("❌ Lojistik Regresyon modeli yüklü değil")
    
    # BERT model durumu
    if os.path.exists('bert_haber_siniflandirici.pth') and os.path.exists('model_siniflar.txt'):
        st.sidebar.success("✅ BERT modeli yüklü")
    else:
        st.sidebar.error("❌ BERT modeli yüklü değil") 
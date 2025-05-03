import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import pickle

# NLTK gerekli veri setlerini indirme
nltk.download('punkt')
nltk.download('stopwords')

# Türkçe stopwords oluşturma
tr_stop_words = set(stopwords.words('turkish'))

# Veri yükleme
def veri_yukle(dosya_yolu):
    """CSV dosyasından veri yükleme fonksiyonu"""
    return pd.read_csv(dosya_yolu)

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

# Veri analizi ve görselleştirme
def veri_analizi(df):
    """Verileri analiz etme ve görselleştirme"""
    # Kategori dağılımı
    plt.figure(figsize=(12, 6))
    sns.countplot(y='ETIKET', data=df)
    plt.title('Haber Kategorilerinin Dağılımı')
    plt.xlabel('Haber Sayısı')
    plt.ylabel('Kategoriler')
    plt.tight_layout()
    plt.savefig('kategori_dagilimi.png')
    
    # Kelime sayısı dağılımı
    df['kelime_sayisi'] = df['HABERLER'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(12, 6))
    sns.histplot(df['kelime_sayisi'], bins=50)
    plt.title('Haber Başlıklarındaki Kelime Sayısı Dağılımı')
    plt.xlabel('Kelime Sayısı')
    plt.ylabel('Frekans')
    plt.tight_layout()
    plt.savefig('kelime_sayisi_dagilimi.png')
    
    # Kategori başına ortalama kelime sayısı
    plt.figure(figsize=(12, 6))
    category_word_counts = df.groupby('ETIKET')['kelime_sayisi'].mean().sort_values(ascending=False)
    sns.barplot(x=category_word_counts.values, y=category_word_counts.index)
    plt.title('Kategori Başına Ortalama Kelime Sayısı')
    plt.xlabel('Ortalama Kelime Sayısı')
    plt.ylabel('Kategoriler')
    plt.tight_layout()
    plt.savefig('kategori_kelime_sayisi.png')
    
    return category_word_counts

# Model eğitimi
def model_egitimi(X_train, y_train, X_test, y_test):
    """Naive Bayes ve Lojistik Regresyon modelleri eğitme"""
    # TF-IDF vektörleştirme
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Naive Bayes modeli
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    
    print("Naive Bayes Sınıflandırma Raporu:")
    print(classification_report(y_test, nb_pred))
    print(f"Doğruluk: {accuracy_score(y_test, nb_pred):.4f}")
    
    # Lojistik Regresyon modeli
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    
    print("\nLojistik Regresyon Sınıflandırma Raporu:")
    print(classification_report(y_test, lr_pred))
    print(f"Doğruluk: {accuracy_score(y_test, lr_pred):.4f}")
    
    # Confusion matrix görselleştirme
    plt.figure(figsize=(12, 10))
    conf_mat = confusion_matrix(y_test, lr_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title('Lojistik Regresyon Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return tfidf_vectorizer, nb_model, lr_model

def main():
    """Ana işlev"""
    # Veri yükleme
    df = veri_yukle('veri/TurkishHeadlines.csv')
    
    # Veri inceleme
    print("Veri seti şekli:", df.shape)
    print("\nÖrnek veriler:")
    print(df.head())
    
    # Kategori dağılımı
    print("\nKategori dağılımı:")
    print(df['ETIKET'].value_counts())
    
    # Metin ön işleme
    print("\nMetin temizleme ve durak kelimeleri kaldırma işlemi başlıyor...")
    df['temiz_haber'] = df['HABERLER'].apply(metin_temizle)
    df['islenmiş_haber'] = df['temiz_haber'].apply(durak_kelimeleri_kaldir)
    
    # Veri analizi
    print("\nVeri analizi yapılıyor...")
    category_word_counts = veri_analizi(df)
    print("\nKategori başına ortalama kelime sayısı:")
    print(category_word_counts)
    
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        df['islenmiş_haber'], df['ETIKET'], test_size=0.2, random_state=42)
    
    print("\nEğitim seti boyutu:", X_train.shape[0])
    print("Test seti boyutu:", X_test.shape[0])
    
    # Model eğitimi
    print("\nModel eğitimi başlıyor...")
    vectorizer, nb_model, lr_model = model_egitimi(X_train, y_train, X_test, y_test)
    
    # Modelleri kaydetme
    print("\nModelleri kaydediliyor...")
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    print("\nİşlem tamamlandı!")
    
    # Örnek tahmin
    ornek_haberler = [
        "Türkiyenin ekonomik büyümesi beklentilerin üzerinde gerçekleşti",
        "Galatasaray deplasmanda Fenerbahçeyi 2-1 mağlup etti",
        "Sağlık Bakanlığı yeni Covid-19 tedbirlerini açıkladı"
    ]
    
    print("\nÖrnek tahminler:")
    for haber in ornek_haberler:
        temiz_haber = durak_kelimeleri_kaldir(metin_temizle(haber))
        tfidf_haber = vectorizer.transform([temiz_haber])
        nb_tahmin = nb_model.predict(tfidf_haber)[0]
        lr_tahmin = lr_model.predict(tfidf_haber)[0]
        
        print(f"\nHaber: {haber}")
        print(f"Naive Bayes tahmini: {nb_tahmin}")
        print(f"Lojistik Regresyon tahmini: {lr_tahmin}")

if __name__ == "__main__":
    main() 
---
title: Türkçe Haber Sınıflandırma
emoji: 📰
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: web_arayuz.py
pinned: false
---

# Türkçe Haber Sınıflandırma Projesi

Bu proje, Türkçe haber başlıklarını kategorilerine göre sınıflandıran bir makine öğrenmesi uygulamasıdır.

## Özellikler

- Klasik makine öğrenmesi modelleri (Naive Bayes, Lojistik Regresyon) ile sınıflandırma
- BERT tabanlı derin öğrenme modeli ile sınıflandırma
- Kullanıcı dostu Streamlit web arayüzü
- Çeşitli performans metrikleri ve görselleştirmeler

## Kullanım

Web arayüzündeki metin kutusuna bir haber başlığı girin ve "Tahmin Et" butonuna tıklayın. Sistem, seçilen modele göre başlığın hangi kategoriye ait olduğunu tahmin edecektir.

## Teknik Detaylar

- Python 3.10 ile geliştirilmiştir
- NLP için BERT modelini kullanır
- Streamlit ile web arayüzü oluşturulmuştur

## Sınıflandırma Kategorileri

- Dünya
- Ekonomi
- Kültür-Sanat
- Politika
- Spor
- Teknoloji
- ve diğer kategoriler

## Docker ile Çalıştırma

Projeyi Docker ile çalıştırmak için:

```bash
docker build -t haber-siniflandirma .
docker run -p 8501:8501 haber-siniflandirma
```

## Proje Hakkında

Bu proje, metin sınıflandırma teknikleri kullanılarak Türkçe haber başlıklarını farklı kategorilere (Ekonomi, Spor, Siyaset vb.) sınıflandırmayı amaçlamaktadır. Hem geleneksel makine öğrenmesi hem de derin öğrenme tabanlı modeller kullanılmıştır.

## Veri Seti

Projede kullanılan veri seti, Türkçe haber başlıklarını ve kategorilerini içeren `TurkishHeadlines.csv` dosyasıdır.

### Veri Özellikleri:
- **HABERLER**: Türkçe haber başlıkları
- **ETIKET**: Haberlerin kategorisi (ör. Ekonomi, Spor, Siyaset)

## Kurulum

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. NLTK için gerekli veri setlerini indirin:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Kullanım

### Geleneksel Makine Öğrenmesi Modeli

Geleneksel makine öğrenmesi yaklaşımı (Naive Bayes, Lojistik Regresyon) için:

```bash
python haber_siniflandirma.py
```

Bu, veri setini işleyecek, modelleri eğitecek ve sonuçları gösterecektir.

### BERT Tabanlı Model

Derin öğrenme tabanlı, Türkçe BERT kullanarak sınıflandırma için:

```bash
python bert_siniflandirma.py
```

Bu, BERT modelini kullanarak daha gelişmiş bir sınıflandırma yapacaktır.

## Proje Çıktıları

Her iki modellemede de şu çıktılar elde edilir:

1. **Görselleştirmeler**:
   - Kategori dağılımı
   - Kelime sayısı dağılımı
   - Kategorilerin ortalama kelime sayıları
   - Confusion matrix
   - BERT eğitim geçmişi grafikleri

2. **Model Performans Metrikleri**:
   - Doğruluk (Accuracy)
   - Hassasiyet (Precision)
   - Duyarlılık (Recall)
   - F1 skoru

3. **Eğitilmiş Modeller**:
   - `bert_haber_siniflandirici.pth`: Eğitilmiş BERT modeli

## Sınıflandırma Örneği

Her iki model de aşağıdaki örnek cümleler üzerinde test edilmektedir:

1. "Türkiyenin ekonomik büyümesi beklentilerin üzerinde gerçekleşti"
2. "Galatasaray deplasmanda Fenerbahçeyi 2-1 mağlup etti"
3. "Sağlık Bakanlığı yeni Covid-19 tedbirlerini açıkladı"

## Proje Genişletme

Bu projeyi genişletmek için aşağıdaki adımları deneyebilirsiniz:

1. **Daha Fazla Veri**: Veri setini genişleterek daha iyi performans elde edebilirsiniz.
2. **Farklı Modeller**: Diğer transformers modelleri (XLM-RoBERTa gibi) denenebilir.
3. **Hiperparametre Optimizasyonu**: GridSearch veya Bayesian optimizasyon ile model parametreleri iyileştirilebilir.
4. **Web Uygulaması**: Model sonuçlarını gösteren bir web uygulaması geliştirilebilir.

## Katkıda Bulunma

1. Bu repo'yu forklayın
2. Yeni bir dal oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Dalınıza push yapın (`git push origin feature/amazing-feature`)
5. Pull Request gönderin

#   N e w s C a t e g o r y F o r e c a s t A I

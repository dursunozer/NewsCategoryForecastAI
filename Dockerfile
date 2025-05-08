FROM python:3.10-slim

WORKDIR /app

# Gerekli paketleri yükle
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# NLTK veri paketlerini yükle
RUN python -m nltk.downloader punkt stopwords

# Geri kalan tüm dosyaları kopyala
COPY . /app/

# Streamlit için kullanıcı oluşturma
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit
RUN chown -R streamlit:streamlit /app

# Streamlit'in kendi dizinini oluşturmayı sağlamak için HOME ayarı
ENV HOME=/app
ENV STREAMLIT_HOME_PATH=/app/.streamlit
RUN mkdir -p /app/.streamlit
RUN chown -R streamlit:streamlit /app/.streamlit

EXPOSE 8501

# Kullanıcıyı değiştir
USER streamlit

CMD ["streamlit", "run", "web_arayuz.py", "--server.port=8501", "--server.address=0.0.0.0"] 
FROM python:3.10-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "web_arayuz.py", "--server.port=8501", "--server.address=0.0.0.0"] 
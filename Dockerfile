FROM python:3.12-slim

WORKDIR /app


COPY requirements.txt .
COPY streamlit.py .
COPY pages/ ./pages/
COPY models/model_2_params/skin_cancer_model_complete2.pth ./models

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8503


ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
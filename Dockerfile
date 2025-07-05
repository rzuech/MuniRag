FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip poppler-utils
RUN pip3 install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY src/ ./src
COPY .env.example ./.env
RUN mkdir -p chroma_data ollama_models data
EXPOSE 8501
WORKDIR /app/src
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

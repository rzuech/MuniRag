# 🏛️ MuniRag – Open Source Municipal AI Assistant

An open-source Retrieval-Augmented Generation (RAG) application allowing Florida municipalities to securely upload PDFs and website content to ask AI-driven questions locally or on cloud GPU providers like RunPod.

---

## Quickstart

```bash
git clone https://github.com/<your-org>/municipal-ai-rag.git
cd munirag
cp .env.example .env
docker compose up --build
```

The included `docker-compose.yml` now uses the `gpus: all` option so both the
`munirag` and `ollama` services can access your NVIDIA GPU. Make sure the
NVIDIA Container Toolkit is installed on the host before running Docker Compose.

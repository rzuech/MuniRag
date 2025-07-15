# 🏛️ MuniRag - Quick Start Guide for Newcomers

Welcome to MuniRag! This guide will get you up and running in 5 minutes.

## 📋 What is MuniRag?

MuniRag is a **Retrieval-Augmented Generation (RAG)** system that lets municipalities:
- Upload PDF documents and crawl websites
- Ask questions about the content using AI
- Get accurate answers with source citations
- Run everything locally for privacy and security

**Key Features:**
- 🔒 **Private**: No data leaves your server
- 💰 **Free**: Uses local AI models (no API costs)
- 🎯 **Accurate**: Cites sources for all answers
- 🛡️ **Secure**: Built-in security and error handling

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB RAM, 20GB disk space

### Step 1: Get the Code
```bash
git clone https://github.com/rzuech/MuniRag.git
cd munirag
```

**Windows users:** Use Command Prompt or PowerShell  
**Linux/Mac users:** Use terminal

### Step 2: Configure Environment
```bash
# Windows
copy .env.example .env

# Linux/Mac  
cp .env.example .env

# Edit .env if needed (optional - defaults work fine)
```

### Step 3: Start the Application
```bash
docker compose up --build
```

**Wait 2-5 minutes** for:
- AI models to download (first time only)
- Services to start up
- Embedding model to load

### Step 4: Access the Interface
Open your browser to: **http://localhost:8501**

### Step 5: Add Content
1. **Upload PDFs**: Use the sidebar to upload municipal documents
2. **Crawl Website**: Enter a city website URL to crawl
3. **Ask Questions**: Type questions in the chat interface

---

## 🔧 How It Works (Technical Overview)

### The RAG Pipeline
```
1. INGESTION: PDF/Web → Text Chunks → Embeddings → Vector DB
2. RETRIEVAL: Question → Similar Chunks → Context
3. GENERATION: Context + Question → AI Response
```

### Key Components
- **Frontend**: Streamlit web interface (`app.py`)
- **Document Processing**: PDF/web ingestion (`ingest.py`)
- **Embeddings**: Convert text to vectors (`embedder.py`)
- **Vector Database**: ChromaDB for similarity search (`retriever.py`)
- **AI Generation**: Ollama for responses (`llm.py`)
- **Configuration**: Environment-based settings (`config.py`)

### Data Flow
1. **Upload/Crawl** → Documents are processed and stored
2. **Question** → Similar content is found using embeddings
3. **Context + Question** → AI generates response with citations

---

## 📁 File Structure Guide

```
munirag/
├── src/                    # Main application code
│   ├── app.py             # Streamlit web interface
│   ├── config.py          # Configuration management
│   ├── embedder.py        # Text-to-vector conversion
│   ├── ingest.py          # Document processing
│   ├── llm.py             # AI model interface
│   ├── retriever.py       # Vector search
│   ├── utils.py           # Helper functions
│   └── logger.py          # Logging system
├── docker-compose.yml     # Service orchestration
├── Dockerfile            # Application container
├── requirements.txt      # Python dependencies
├── .env.example         # Configuration template
└── CLAUDE.md           # Developer documentation
```

---

## ⚙️ Configuration Options

Edit `.env` to customize behavior:

```bash
# AI Models
EMBEDDING_MODEL=intfloat/e5-large-v2  # Text-to-vector model
LLM_MODEL=llama3:8b                   # Response generation model

# Processing
MAX_CHUNK_TOKENS=300                  # Text chunk size
TOP_K=4                              # Number of sources to retrieve

# Security
MAX_FILE_SIZE_MB=50                  # Upload limit
MAX_PAGES_CRAWL=20                   # Website crawling limit

# Performance
REQUEST_TIMEOUT=30                   # Network timeout
LOG_LEVEL=INFO                       # Logging verbosity
```

---

## 🔍 Understanding the Interface

### Sidebar (Left)
- **📂 Add Content**: Upload PDFs or crawl websites
- **🗑️ Reset Data**: Clear all stored documents
- **🔧 System Status**: Check service health

### Main Area
- **💬 Chat Interface**: Ask questions about your documents
- **📚 Sources**: See which documents were used for answers
- **⚡ Streaming**: Responses appear in real-time

---

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB free space
- **Network**: Internet for initial model downloads

### Recommended Setup
- **GPU**: NVIDIA with 8GB+ VRAM
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD

### Performance Tips
- **GPU**: Enables much faster embedding generation
- **SSD**: Improves database performance
- **RAM**: Allows larger models and more documents

---

## 🛠️ Common Commands

```bash
# Start services
docker compose up

# Start in background
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs munirag
docker compose logs ollama

# Rebuild after changes
docker compose up --build

# Reset everything
docker compose down -v
docker compose up --build
```

---

## 🔧 Troubleshooting

### "Cannot connect to Ollama"
- Wait 2-3 minutes for Ollama to start
- Check: `docker compose logs ollama`
- Restart: `docker compose restart ollama`

### "No GPU detected"
- Install NVIDIA Container Toolkit
- Restart Docker daemon
- Check: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

### "Embedding model failed"
- Check available disk space (models are large)
- Restart: `docker compose restart munirag`
- Check logs: `docker compose logs munirag`

### "Website crawling failed"
- Check URL is valid and publicly accessible
- Some sites block automated crawling
- Try smaller sites first

---

## 🎯 Best Practices

### Document Preparation
- **PDFs**: Ensure text is selectable (not scanned images)
- **Websites**: Start with official municipal sites
- **Size**: Break large documents into smaller files if possible

### Question Asking
- **Specific**: "What is the 2024 road maintenance budget?" vs "Tell me about roads"
- **Context**: Reference document types: "According to the city budget..."
- **Follow-up**: Ask for specific page numbers or sections

### Performance
- **Start small**: Test with a few documents first
- **Monitor**: Check system status in sidebar
- **GPU**: Enable for much better performance

---

## 🔒 Security Notes

- **Local only**: No data sent to external APIs
- **File validation**: Automatic size and type checking
- **URL filtering**: Prevents crawling private/localhost addresses
- **Error handling**: Graceful failure without crashes

---

## 📚 Next Steps

1. **Upload municipal documents** (budgets, ordinances, plans)
2. **Test with sample questions** to verify accuracy
3. **Explore configuration options** in `.env`
4. **Monitor logs** for performance insights
5. **Scale up** with more documents and users

---

## 🆘 Getting Help

- **Logs**: Check `docker compose logs` for errors
- **Configuration**: Review `.env.example` for options
- **Performance**: Monitor system status in sidebar
- **Issues**: Check GitHub issues or create new ones

---

**Congratulations!** 🎉 You now have a working AI assistant for municipal documents. Start by uploading a few PDFs and asking questions about their content!
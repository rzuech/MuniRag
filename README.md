# 🏛️ MuniRAG v0.90 – Open Source Municipal AI Assistant

An open-source Retrieval-Augmented Generation (RAG) application allowing municipalities to securely upload PDFs and website content to ask AI-driven questions locally or on cloud GPU providers like RunPod.

---

## Quickstart

```bash
git clone https://github.com/rzuech/MuniRag.git
cd munirag
cp .env.example .env
docker compose up --build
```

The included `docker-compose.yml` now uses the `gpus: all` option so both the
`munirag` and `ollama` services can access your NVIDIA GPU. Make sure the
NVIDIA Container Toolkit is installed on the host before running Docker Compose.

## Accessing the Application

- **FastAPI (REST API)**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI (Legacy)**: http://localhost:8501

## Website Integration Options

MuniRag provides multiple integration methods for municipal websites:

### 1. Iframe Embed (Easiest - 1 hour)
```html
<iframe src="https://your-munirag-domain.com" width="100%" height="600px"></iframe>
```
- **Difficulty**: ⭐ (1/5) - Copy & paste HTML
- **Time**: 1 hour (just deployment)
- **Pros**: No coding required, works immediately
- **Cons**: Limited styling, iframe restrictions

### 2. JavaScript Widget (Recommended - 1-2 days)
```javascript
<script src="https://your-domain.com/munirag-widget.js"></script>
<div id="munirag-chat"></div>
```
- **Difficulty**: ⭐⭐⭐ (3/5) - Moderate frontend work
- **Time**: 1-2 days to build widget
- **Pros**: Native look, customizable, lightweight
- **Cons**: Requires JavaScript development

### 3. Direct API Integration (Most Flexible - 2-4 hours)
```javascript
fetch('https://your-domain.com/api/query', {
  method: 'POST',
  headers: {'Authorization': 'Bearer YOUR_API_KEY'},
  body: JSON.stringify({query: "What are park hours?"})
})
```
- **Difficulty**: ⭐⭐ (2/5) - Basic API calls
- **Time**: 2-4 hours for basic integration
- **Pros**: Complete control, integrate anywhere
- **Cons**: Requires frontend development

### 4. WordPress/CMS Plugin (Most User-Friendly - 3-5 days)
- **Difficulty**: ⭐⭐⭐⭐ (4/5) - Plugin architecture knowledge
- **Time**: 3-5 days for full plugin
- **Pros**: One-click install, auto-updates, admin UI
- **Cons**: Platform-specific development

### Implementation Requirements

Before integration, you'll need:
1. **CORS Configuration** - Allow cross-origin requests (30 minutes)
2. **Authentication** - API keys or JWT tokens (2-4 hours)
3. **Rate Limiting** - Prevent abuse (1-2 hours)
4. **SSL/HTTPS** - Secure connection (varies by host)
5. **Domain Setup** - Public URL for your instance (1 hour)

### Recommended Integration Path

1. **Start with iframe** - Get it live quickly
2. **Add API authentication** - Secure your endpoints
3. **Build JavaScript widget** - Better user experience
4. **Create CMS plugins** - For non-technical users

Total time for all integrations: ~1 week with one developer

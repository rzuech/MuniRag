# LLM Architecture and Model Support

## 🎯 Overview

MuniRAG v2.0 supports multiple LLM providers and models with a focus on:
- Local deployment for data privacy
- GPU optimization for RTX 4090
- Cloud options for larger models
- API integration for future expansion

## 🚀 Supported Models

### Local Models (RTX 4090 Optimized)

#### Currently Implemented
- **Llama3.1:8b** - Primary model
  - Size: ~5GB (16-bit)
  - Speed: Excellent on RTX 4090
  - Context: 8192 tokens
  - Use case: General municipal Q&A

#### Planned Local Models

1. **Llama 2 13B** (Q8 Quantized)
   ```bash
   ollama pull llama2:13b-chat-q8_0
   ```
   - Size: ~13GB (8-bit quantized)
   - Speed: Good on RTX 4090
   - Context: 4096 tokens
   - Use case: More complex reasoning

2. **Llama 2 30B** (Q4 Quantized)
   ```bash
   ollama pull llama2:30b-chat-q4_0
   ```
   - Size: ~17GB (4-bit quantized)
   - Speed: Moderate (may compete with embeddings)
   - Context: 4096 tokens
   - Use case: Advanced analysis

3. **Mistral 7B Instruct**
   ```bash
   ollama pull mistral:7b-instruct-v0.2
   ```
   - Size: ~4GB
   - Speed: Excellent
   - Context: 32768 tokens (!)
   - Use case: Long document analysis

4. **Mixtral 8x7B** (MoE, Q4 Quantized)
   ```bash
   ollama pull mixtral:8x7b-instruct-q4_0
   ```
   - Size: ~26GB (pushes 4090 limits)
   - Speed: Moderate
   - Context: 32768 tokens
   - Use case: Expert-level responses

### Cloud Models (RunPod/Remote)

1. **Llama 2 70B** (16-bit)
   - Size: ~140GB
   - Requirements: A100 80GB or multiple GPUs
   - Context: 4096 tokens
   - Use case: Maximum capability

2. **Mixtral 8x22B** (Large MoE)
   - Size: ~300GB+
   - Requirements: Multiple A100s
   - Context: 65536 tokens
   - Use case: Research-grade analysis

### Future API Integrations

```python
# Planned provider support
PROVIDERS = {
    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    "anthropic": ["claude-3-sonnet", "claude-3-opus"],
    "google": ["gemini-pro", "gemini-ultra"],
    "cohere": ["command", "command-light"]
}
```

## 📊 Model Selection Matrix

| Model | Size | RTX 4090 | Speed | Quality | Context | Best For |
|-------|------|----------|-------|---------|---------|----------|
| Llama3.1:8b | 5GB | ✅ Perfect | ⚡⚡⚡ | ⭐⭐⭐ | 8K | Daily use |
| Llama2:13b-q8 | 13GB | ✅ Good | ⚡⚡ | ⭐⭐⭐⭐ | 4K | Complex queries |
| Llama2:30b-q4 | 17GB | ⚠️ Tight | ⚡ | ⭐⭐⭐⭐ | 4K | Advanced analysis |
| Mistral:7b | 4GB | ✅ Perfect | ⚡⚡⚡ | ⭐⭐⭐ | 32K | Long documents |
| Mixtral:8x7b-q4 | 26GB | ❌ Too big | 🐌 | ⭐⭐⭐⭐⭐ | 32K | Expert tasks |

## 🎛️ User-Exposed Parameters

### Currently Implemented
```python
# In settings.py
LLM_TEMPERATURE: float = 0.1  # 0.0-1.0 (creativity)
LLM_MAX_TOKENS: int = 2048    # Response length
LLM_TOP_P: float = 0.9        # Nucleus sampling
```

### Planned UI Controls
```python
# Streamlit sidebar controls
temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.1)
max_tokens = st.slider("Max Response Length", 256, 4096, 2048)
top_p = st.slider("Top-p (Focus)", 0.5, 1.0, 0.9)
model = st.selectbox("Model", ["llama3.1:8b", "mistral:7b", "llama2:13b"])
```

## 💬 System Message Architecture

### Default System Message
```python
SYSTEM_MESSAGE = """You are MuniRAG, a helpful AI assistant for municipal government information. 

Your primary purpose is to answer questions using ONLY the information provided in the context. Follow these guidelines:

1. Base your answers strictly on the provided context/documents
2. If the context doesn't contain relevant information, say "I don't have information about that in the provided documents"
3. Never use information from your training data - only use the context provided
4. Be precise and factual in your responses
5. If you're uncertain, express that uncertainty rather than guessing
6. Cite specific sections or documents when possible
7. Keep answers concise and relevant to municipal operations

Remember: It's better to say you don't know than to provide incorrect information."""
```

### Customizable System Messages
```python
# Department-specific examples
SYSTEM_MESSAGES = {
    "planning": "Focus on zoning, permits, and development regulations...",
    "finance": "Emphasize budgets, taxes, and financial procedures...",
    "public_works": "Prioritize infrastructure and maintenance topics...",
    "legal": "Ensure accuracy on ordinances and legal procedures..."
}
```

## 🔄 GPU Resource Management

### Priority System
```
1. User LLM Queries (HIGHEST) ⬆️
2. Admin Embeddings (LOWER) ⬇️
```

### Implementation Strategy
```python
class GPUResourceManager:
    def should_defer_embeddings(self) -> bool:
        """Check if embeddings should yield to LLM."""
        # Check active LLM requests
        if self.active_llm_requests > 0:
            return True
        
        # Check GPU utilization
        if self.gpu_utilization > 0.8:
            return True
            
        # Check memory pressure
        if self.gpu_memory_usage > 0.7:
            return True
            
        return False
```

### Resource Allocation Rules
1. **Single GPU Strategy**:
   - LLM gets priority during business hours
   - Embeddings can run overnight
   - Automatic pause/resume for embeddings

2. **Memory Management**:
   - Reserve 8GB for LLM inference
   - Use remaining for embeddings
   - Switch to CPU if memory tight

3. **Queue Management**:
   ```python
   # Planned implementation
   embedding_queue = PriorityQueue()
   llm_queue = PriorityQueue()
   
   # LLM requests get higher priority
   llm_priority = 10
   embedding_priority = 5
   ```

## 📈 Performance Considerations

### Model Loading Times
- First load: 10-30 seconds (download)
- Subsequent: 2-5 seconds (from cache)
- Keep models loaded in production

### Inference Speed (RTX 4090)
- Llama3.1:8b: ~150 tokens/sec
- Llama2:13b-q8: ~80 tokens/sec
- Mistral:7b: ~200 tokens/sec
- Mixtral:8x7b-q4: ~30 tokens/sec

### Memory Requirements
```python
# Rough estimates for RTX 4090 (24GB)
MEMORY_MAP = {
    "llama3.1:8b": 5,      # GB
    "embeddings": 2,       # GB
    "system": 2,           # GB
    "buffer": 3,           # GB
    # Total: 12GB, leaving 12GB free
}
```

## 🛠️ Implementation Phases

### Phase 1: Current ✅
- Llama3.1:8b via Ollama
- Basic temperature control
- System message integration

### Phase 2: Multi-Model Support
- Model selector in UI
- Per-model optimization
- Automatic model downloads

### Phase 3: Advanced Features
- Streaming improvements
- Token counting/display
- Response caching
- Conversation memory

### Phase 4: Cloud Integration
- RunPod deployment scripts
- API key management
- Hybrid local/cloud routing

## 🔍 Monitoring and Debugging

### Metrics to Track
```python
# Per request
- Model used
- Token count (input/output)
- Generation time
- GPU memory before/after

# System level
- Concurrent requests
- Queue depths
- Model switching frequency
- Error rates
```

### Common Issues
1. **OOM Errors**: Model too large for GPU
   - Solution: Use quantized versions
   
2. **Slow Generation**: GPU contention
   - Solution: Check embedding jobs
   
3. **Wrong Model Responses**: Model not following system message
   - Solution: Adjust temperature, check prompt format

## 📚 Best Practices

1. **Model Selection**:
   - Start with Llama3.1:8b
   - Upgrade only if needed
   - Consider context length requirements

2. **Resource Management**:
   - Monitor GPU usage continuously
   - Set up alerts for high usage
   - Plan maintenance windows

3. **User Experience**:
   - Show model being used
   - Display generation progress
   - Provide model switching options

4. **Stability**:
   - Test each model thoroughly
   - Have fallback options
   - Document model quirks
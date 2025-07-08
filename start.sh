#!/bin/bash
export HF_HOME=/app/.cache/huggingface

# Start FastAPI in background
cd /app && python3 main.py &

# Give FastAPI time to start
sleep 5

# Start Streamlit in foreground
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
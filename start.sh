#!/bin/bash
# CosyVoice2 RunPod Pod startup script
# This runs ON the RunPod GPU pod — downloads model + starts API server
set -e

echo "=== CosyVoice2 Voice Clone Setup ==="

cd /workspace

# Clone CosyVoice if not already present
if [ ! -d "CosyVoice" ]; then
    echo "Cloning CosyVoice..."
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    pip install -r requirements.txt
    cd /workspace
fi

# Download model if not cached
MODEL_DIR="/workspace/pretrained_models/CosyVoice2-0.5B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading CosyVoice2-0.5B model..."
    pip install modelscope
    python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='$MODEL_DIR')
print('Model download complete')
"
fi

# Install handler dependencies
pip install fastapi uvicorn requests

# Copy handler if uploaded, otherwise use default
if [ -f "/workspace/handler.py" ]; then
    echo "Using /workspace/handler.py"
else
    echo "ERROR: handler.py not found in /workspace"
    exit 1
fi

echo "=== Starting CosyVoice2 API server on port 8000 ==="
cd /workspace
python3 handler.py

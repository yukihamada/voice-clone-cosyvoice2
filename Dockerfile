FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg sox libsox-dev wget ca-certificates \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Pre-install build deps that CosyVoice requirements need
RUN pip install --no-cache-dir cython setuptools

# Clone CosyVoice and install its deps
# openai-whisper's setup.py needs pkg_resources at build time, so install it without build isolation first
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 && \
    cd /app/CosyVoice && pip install --no-cache-dir -r requirements.txt

# Install handler deps (use huggingface_hub instead of modelscope to avoid torchvision conflicts)
RUN pip install --no-cache-dir runpod requests huggingface_hub

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# Download CosyVoice2-0.5B model from HuggingFace (baked in for fast cold start)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/app/pretrained_models/CosyVoice2-0.5B')"

ENV MODEL_DIR=/app/pretrained_models/CosyVoice2-0.5B

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

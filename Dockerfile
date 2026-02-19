FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg sox libsox-dev wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Clone CosyVoice and install its deps
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    cd /app/CosyVoice && pip install --no-cache-dir -r requirements.txt

# Install handler deps
RUN pip install --no-cache-dir runpod requests modelscope

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# Download CosyVoice2-0.5B model (baked in for fast cold start)
RUN python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='/app/pretrained_models/CosyVoice2-0.5B')"

ENV MODEL_DIR=/app/pretrained_models/CosyVoice2-0.5B

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

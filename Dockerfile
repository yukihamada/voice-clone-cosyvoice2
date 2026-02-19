FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg sox libsox-dev wget ca-certificates \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Pre-install build deps that CosyVoice requirements need
RUN pip install --no-cache-dir cython setuptools

# Clone CosyVoice and install its deps
# Remove torch/torchaudio pins to keep the pre-installed 2.5.1 (transformers 4.51.3 needs >=2.5)
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    sed -i '/^torch==/d; /^torchaudio==/d' /app/CosyVoice/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 && \
    cd /app/CosyVoice && pip install --no-cache-dir -r requirements.txt

# Fix ABI: CosyVoice deps may install wrong torchaudio; reinstall matching versions
# Remove torchvision (uses register_fake which needs torch>=2.6)
RUN pip uninstall -y torchvision 2>/dev/null; true && \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install handler deps
RUN pip install --no-cache-dir runpod requests huggingface_hub

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# Model will be downloaded at startup if not present (keeps image small)
ENV MODEL_DIR=/app/pretrained_models/CosyVoice2-0.5B

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

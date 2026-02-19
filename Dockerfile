FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg sox libsox-dev wget ca-certificates \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Pre-install build deps that CosyVoice requirements need
RUN pip install --no-cache-dir cython setuptools

# Clone CosyVoice and install its deps
# Remove torch/torchaudio pins to keep the pre-installed 2.5.1 (transformers 4.51.3 needs >=2.5)
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    sed -i '/^torch==/d; /^torchaudio==/d' /app/CosyVoice/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 && \
    cd /app/CosyVoice && pip install --no-cache-dir -r requirements.txt

# Install handler deps
RUN pip install --no-cache-dir runpod requests modelscope

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# Download CosyVoice2-0.5B model via git-lfs (avoids Python import issues)
RUN git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git /app/pretrained_models/CosyVoice2-0.5B

ENV MODEL_DIR=/app/pretrained_models/CosyVoice2-0.5B

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

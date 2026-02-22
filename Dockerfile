FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg sox libsox-dev wget ca-certificates \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Pre-install build deps that CosyVoice requirements need
RUN pip install --no-cache-dir cython setuptools numpy

# Clone CosyVoice (with Matcha-TTS submodule) and install deps
# Remove torch/torchaudio pins to keep the pre-installed 2.5.1
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    sed -i '/^torch==/d; /^torchaudio==/d' /app/CosyVoice/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20231117 && \
    cd /app/CosyVoice && pip install --no-cache-dir -r requirements.txt

# Fix ABI: CosyVoice deps may install wrong torchaudio; reinstall matching CUDA versions
# --no-deps prevents pulling in CPU-only dependencies
RUN pip uninstall -y torchvision 2>/dev/null; true && \
    pip install --no-cache-dir --no-deps torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install CUDA nvcc AFTER requirements (so setup.py doesn't try to compile CUDA ops at build time)
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-1 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Install Matcha-TTS AFTER torch is finalized (has Cython extension that links against torch)
RUN cd /app/CosyVoice/third_party/Matcha-TTS && \
    pip install --no-cache-dir --no-build-isolation .

# Install handler deps
RUN pip install --no-cache-dir runpod requests huggingface_hub

# Patch huggingface_hub: add cached_download shim (removed in >=0.26, needed by modelscope)
COPY patch_hf_hub.py /app/patch_hf_hub.py
RUN python /app/patch_hf_hub.py

# Verify imports
RUN python -c "import matcha; print(f'matcha OK: {matcha.__file__}')" && \
    python -c "import transformers; print(f'transformers OK: {transformers.__version__}')"

ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS"

# Bake model into image to avoid 2-5min cold start download
ENV MODEL_DIR=/app/pretrained_models/CosyVoice2-0.5B
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('FunAudioLLM/CosyVoice2-0.5B', \
    local_dir='/app/pretrained_models/CosyVoice2-0.5B')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

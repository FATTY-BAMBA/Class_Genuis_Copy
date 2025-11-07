# syntax=docker/dockerfile:1
# =========================================================
# Base: NVIDIA CUDA 11.8 + cuDNN 8.7
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# -------------------- Install Python 3.10 --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# -------------------- Environment --------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Taipei \
    HF_HOME=/workspace/models \
    WHISPER_CACHE=/workspace/models \
    CTRANSLATE2_CACHE=/workspace/models \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    GLOG_minloglevel=2 \
    GLOG_logtostderr=0 \
    FLAGS_fraction_of_gpu_memory_to_use=0.9

# -------------------- System deps --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config libcairo2-dev \
      libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
      libavfilter-dev libswscale-dev libswresample-dev \
      ffmpeg redis-server redis-tools \
      libsndfile1 libgl1 libgomp1 libglib2.0-0 \
      libsm6 libxext6 libxrender1 libcairo2 \
      curl aria2 netcat-openbsd procps net-tools lsof patchelf \
      libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip==24.0 setuptools wheel

# -------------------- Workdir & app files --------------------
WORKDIR /app

COPY requirements.txt constraints.txt /app/
COPY . .

# -------------------- Python deps --------------------
RUN python -m pip install --no-cache-dir numpy==1.26.4

# Install requirements (PyTorch will be installed separately)
RUN python -m pip install --no-cache-dir \
    -r /app/requirements.txt -c /app/constraints.txt || true

# -------------------- Install PyTorch 2.3.1 (CUDA 11.8) --------------------
# PyTorch 2.3 officially supports CUDA 11.8 + cuDNN 8.7.0.84
RUN pip3 install --no-cache-dir --force-reinstall \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

# -------------------- Whisper stack (CUDA 11.8 + cuDNN 8) --------------------
# Install PyAV with pre-built wheel
RUN python -m pip install --no-cache-dir --only-binary=:all: av==12.3.0

# CRITICAL: For CUDA 11 + cuDNN 8, use ctranslate2 3.24.0 (per faster-whisper docs)
RUN python -m pip install --no-cache-dir ctranslate2==3.24.0

# Install faster-whisper 0.10.1 (compatible with ctranslate2 3.24.0)
RUN python -m pip install --no-cache-dir faster-whisper==0.10.1

RUN python -m pip install --no-cache-dir "tokenizers>=0.14,<0.15"

RUN python -m pip install --no-cache-dir \
    "transformers==4.36.2" -c /app/constraints.txt

# EasyOCR
RUN python -m pip install --no-cache-dir easyocr==1.7.1

# Verify installations
RUN python -c "import torch; print('✅ PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())" && \
    python -c "import av; print('✅ PyAV:', av.__version__)" && \
    python -c "import ctranslate2; print('✅ ctranslate2:', ctranslate2.__version__)" && \
    python -c "import easyocr; print('✅ EasyOCR:', easyocr.__version__)" && \
    python -c "import faster_whisper; print('✅ faster-whisper:', faster_whisper.__version__)"

# -------------------- Optional: legacy numpy.int shim --------------------
RUN python -c "import sys, pathlib, site; \
    site_dir = pathlib.Path(site.getsitepackages()[0]); \
    (site_dir/'numpy_patch.py').write_text('import numpy as np; np.int = int if not hasattr(np, \"int\") else np.int'); \
    sitecustomize = site_dir/'sitecustomize.py'; \
    sitecustomize.write_text((sitecustomize.read_text() if sitecustomize.exists() else '') + '\ntry: import numpy_patch\nexcept: pass\n')"

# -------------------- Non-root user & dirs --------------------
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

USER appuser

EXPOSE 5000 8888

CMD ["./start.sh"]

# syntax=docker/dockerfile:1

# ==================== BASE: CUDA 11.8 + cuDNN 8 on Ubuntu 20.04 ====================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

# ==================== ENVIRONMENT VARIABLES ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ==================== SYSTEM DEPENDENCIES ====================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential cmake git curl ca-certificates wget \
        libcairo2-dev libjpeg-dev libgif-dev pkg-config \
        libopenblas-dev libssl-dev patchelf \
        gnupg lsb-release && \
    wget -O /tmp/python.tar.xz "https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-x86_64-unknown-linux-gnu-install_only.tar.gz" && \
    tar -xf /tmp/python.tar.xz -C /usr/local --strip-components=1 && \
    rm /tmp/python.tar.xz && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    /usr/local/bin/python3.10 -m ensurepip && \
    /usr/local/bin/python3.10 -m pip install --upgrade pip && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# ==================== PYTHON SETUP ====================
RUN python -m pip install --upgrade pip==24.0 setuptools wheel

# ==================== BUILD DEPENDENCIES ====================
WORKDIR /build

# Copy dependency lists
COPY requirements.txt constraints.txt /build/

# Remove conflicting packages from requirements (including lanms-neo which fails to build)
RUN grep -v "ctranslate2\|faster-whisper\|tokenizers\|transformers\|numpy\|lanms-neo\|Polygon3" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt

# Install core dependencies
RUN python -m pip install --no-cache-dir \
    packaging>=20.0 \
    Cython==3.0.10 \
    pybind11==2.12.0 \
    meson==1.2.3 \
    meson-python==0.15.0 \
    ninja==1.11.1

# Install NumPy 1.x FIRST (before PyTorch and other packages)
RUN python -m pip install --no-cache-dir numpy==1.26.4

# Install PyTorch with CUDA 11.8 support (will use existing NumPy)
RUN python -m pip install \
    torch==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    torchaudio==2.2.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install other requirements (without numpy and lanms-neo since we already have numpy and lanms-neo fails)
RUN python -m pip install --no-cache-dir -r requirements_filtered.txt

# Install Polygon3
#RUN python -m pip install --no-cache-dir "Polygon3==3.0.9.1"

# Install compatible versions for whisper
RUN python -m pip install --no-cache-dir \
    faster-whisper==0.10.1 \
    ctranslate2==3.24.0 \
    transformers==4.36.2

# Install PaddlePaddle and PaddleOCR (compatible versions)
RUN python -m pip install --no-cache-dir \
    paddlepaddle-gpu==2.5.1 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    python -m pip install --no-cache-dir paddleocr==2.7.0

# Install VisualDL
RUN python -m pip install --no-cache-dir visualdl==2.5.3

# CRITICAL: Force NumPy 1.x as the final step to override any package that installed 2.x
RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 && \
    python -c "import numpy; print(f'NumPy version locked at: {numpy.__version__}')"

# ==================== RUNTIME STAGE ====================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS final

# ==================== ENVIRONMENT VARIABLES ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    HF_HOME=/workspace/models \
    WHISPER_CACHE=/workspace/models \
    CTRANSLATE2_CACHE=/workspace/models \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    GLOG_minloglevel=2 \
    GLOG_logtostderr=0 \
    FLAGS_fraction_of_gpu_memory_to_use=0.9 \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

# ==================== RUNTIME DEPENDENCIES ====================
# The runtime image already has cuDNN and CUDA libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        gnupg \
        lsb-release && \
    wget -O /tmp/python.tar.xz "https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-x86_64-unknown-linux-gnu-install_only.tar.gz" && \
    tar -xf /tmp/python.tar.xz -C /usr/local --strip-components=1 && \
    rm /tmp/python.tar.xz && \
    apt-get install -y --no-install-recommends \
        ffmpeg redis-server redis-tools \
        libsndfile1 libgl1 libgomp1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 libcairo2 \
        curl aria2 netcat-openbsd procps net-tools lsof \
        patchelf && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    /usr/local/bin/python3.10 -m ensurepip && \
    /usr/local/bin/python3.10 -m pip install --upgrade pip && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# ==================== COPY PYTHON ENVIRONMENT ====================
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# ==================== FIX LIBRARY PATHS ====================
# Ensure CUDA libraries are accessible
RUN ldconfig /usr/local/cuda/lib64

# ==================== APPLICATION ====================
WORKDIR /app
COPY . .

# Fix numpy.int deprecation (keep for compatibility with old code)
RUN echo "import numpy as np; np.int = int if not hasattr(np, 'int') else np.int" > /usr/local/lib/python3.10/dist-packages/numpy_patch.py || true && \
    echo "try: import numpy_patch\nexcept: pass" >> /usr/local/lib/python3.10/dist-packages/sitecustomize.py || true

# ==================== CREATE NON-ROOT USER ====================
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

# ==================== SWITCH TO NON-ROOT ====================
USER appuser

EXPOSE 5000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

CMD ["./start.sh"]

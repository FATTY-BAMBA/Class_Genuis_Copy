# syntax=docker/dockerfile:1

# ==================== BASE: CUDA 11.8 + cuDNN 8 on Ubuntu 20.04 ====================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

# ==================== ENVIRONMENT VARIABLES ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Taipei \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ==================== SYSTEM DEPENDENCIES (BUILDER) ====================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential cmake git curl ca-certificates wget \
        libcairo2-dev libjpeg-dev libgif-dev pkg-config \
        libopenblas-dev libssl-dev patchelf g++ gcc && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip==24.0 setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# ==================== BUILD WORKDIR ====================
WORKDIR /build

# Copy dependency lists early for layer caching
COPY requirements.txt constraints.txt /build/

# (Optional) filter packages you’ll install separately/pinned below
RUN grep -v "ctranslate2\|faster-whisper\|tokenizers\|transformers\|numpy" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt

# ==================== LOW-LEVEL BUILD TOOLS ====================
RUN python -m pip install --no-cache-dir \
      "packaging>=20.0" \
      Cython==3.0.10 \
      pybind11==2.12.0 \
      meson==1.2.3 meson-python==0.15.0 ninja==1.11.1

# ==================== NUMPY FIRST (STAY ON 1.x) ====================
RUN python -m pip install --no-cache-dir numpy==1.26.4

# ==================== PYTORCH CUDA 11.8 ====================
RUN python -m pip install \
      torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118

# ==================== PROJECT REQUIREMENTS (RESPECT CONSTRAINTS) ====================
RUN python -m pip install --no-cache-dir -r requirements_filtered.txt -c constraints.txt

# ==================== WHISPER STACK (split so wheels are used) ====================
# CTranslate2 + faster-whisper
RUN python -m pip install --no-cache-dir \
      ctranslate2==3.24.0 faster-whisper==0.10.1

# Tokenizers from wheels only to avoid Rust build; then Transformers
RUN python -m pip install --no-cache-dir "tokenizers>=0.14,<0.15" --only-binary=:all:
RUN python -m pip install --no-cache-dir "transformers==4.36.2" -c constraints.txt

# ==================== PADDLE + OCR ====================
RUN python -m pip install --no-cache-dir \
      paddlepaddle-gpu==2.5.1 \
      -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    python -m pip install --no-cache-dir paddleocr==2.7.0 -c constraints.txt

# ==================== VISUALDL ====================
RUN python -m pip install --no-cache-dir visualdl==2.5.3

# (Optional) Guardrail to ensure NumPy stays 1.x
# RUN python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4

# ==================== RUNTIME STAGE ====================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS final

# ==================== ENVIRONMENT VARIABLES ====================
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
    FLAGS_fraction_of_gpu_memory_to_use=0.9 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

# ==================== RUNTIME DEPENDENCIES ====================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-distutils python3-pip \
        ffmpeg redis-server redis-tools \
        libsndfile1 libgl1 libgomp1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 libcairo2 \
        curl aria2 netcat-openbsd procps net-tools lsof \
        patchelf \
        # tiny safety net if any wheel expects system BLAS
        libopenblas0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip==24.0 \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# ==================== COPY PYTHON ENVIRONMENT FROM BUILDER ====================
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Ensure loader sees copied libs
RUN ldconfig

# ==================== APPLICATION ====================
WORKDIR /app
COPY . .

# Legacy numpy.int shim (if you have old code that needs it)
RUN python - <<'PY'
import pathlib
site = pathlib.Path('/usr/local/lib/python3.10/dist-packages')
site.mkdir(parents=True, exist_ok=True)
(site/'numpy_patch.py').write_text("import numpy as np; np.int = int if not hasattr(np, 'int') else np.int")
with (site/'sitecustomize.py').open('a') as f: f.write("\ntry: import numpy_patch\nexcept: pass\n")
PY

# ==================== NON-ROOT USER ====================
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

USER appuser

EXPOSE 5000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

CMD ["./start.sh"]
